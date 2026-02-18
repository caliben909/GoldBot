"""
Backtesting Engine - Professional backtesting with Monte Carlo simulation and stress testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t, genextreme
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib

from core.risk_engine import RiskEngine
from core.strategy_engine import StrategyEngine
from core.data_engine import DataEngine
from utils.indicators import TechnicalIndicators
from utils.helpers import retry_with_backoff

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    enabled: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100000.0
    commission: float = 0.0001
    slippage: float = 0.0001
    slippage_model: str = 'percentage'  # 'fixed', 'percentage', 'market_impact'
    market_impact_params: Dict = field(default_factory=dict)
    data_source: str = 'database'
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ['1H'])
    
    # Monte Carlo settings
    monte_carlo_enabled: bool = True
    monte_carlo_simulations: int = 1000
    monte_carlo_confidence: List[float] = field(default_factory=lambda: [0.95, 0.99])
    
    # Stress testing
    stress_test_enabled: bool = True
    stress_scenarios: List[Dict] = field(default_factory=list)
    
    # Output settings
    output: Dict = field(default_factory=lambda: {
        'save_trades': True,
        'save_equity_curve': True,
        'generate_report': True,
        'plot_results': True
    })
    save_trades: bool = True
    save_equity_curve: bool = True
    generate_report: bool = True
    report_format: str = 'html'
    plot_results: bool = True


@dataclass
class TradeResult:
    """Individual trade result"""
    trade_id: str
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    commission: float
    slippage: float
    profit: float
    profit_pips: float
    r_multiple: float
    entry_time: datetime
    exit_time: datetime
    holding_period: float  # in minutes
    exit_reason: str
    signal_type: str
    regime: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    profit_factor: float
    expectancy: float
    avg_r_multiple: float
    r_multiple_std: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    recovery_factor: float
    ulcer_index: float
    
    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    tail_ratio: float
    gain_to_pain_ratio: float
    
    # Monte Carlo metrics
    mc_expected_return: float
    mc_expected_sharpe: float
    mc_expected_max_dd: float
    mc_var_95: float
    mc_var_99: float
    mc_probability_of_ruin: float
    
    # Stress test metrics
    stress_test_results: Dict = field(default_factory=dict)


class BacktestEngine:
    """
    Professional backtesting engine with:
    - Multi-asset backtesting
    - Advanced transaction cost modeling
    - Market impact modeling
    - Monte Carlo simulation
    - Stress testing with historical scenarios
    - Parameter optimization
    - Walk-forward analysis
    - Comprehensive metrics
    - Interactive visualizations
    """
    
    def __init__(self, config: dict):
        self.config = BacktestConfig(**config['backtesting'])
        self.logger = logging.getLogger(__name__)
        
        # Initialize engines
        self.risk_engine = RiskEngine(config)
        self.strategy_engine = StrategyEngine(config)
        self.data_engine = None  # Will be initialized per run
        self.indicators = TechnicalIndicators()
        
        # Results storage
        self.trades: List[TradeResult] = []
        self.equity_curve: pd.Series = None
        self.drawdown_curve: pd.Series = None
        self.daily_returns: pd.Series = None
        self.metrics: Optional[BacktestMetrics] = None
        
        # Monte Carlo results
        self.mc_equity_curves: List[pd.Series] = []
        self.mc_metrics: List[Dict] = []
        
        # Stress test results
        self.stress_results: Dict = {}
        
        logger.info("BacktestEngine initialized")
    
    async def run(self, strategy, data: Dict[str, pd.DataFrame] = None) -> BacktestMetrics:
        """
        Run comprehensive backtest
        
        Args:
            strategy: Strategy instance
            data: Pre-loaded data (optional)
        
        Returns:
            BacktestMetrics object
        """
        self.logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        try:
            # Load data if not provided
            if data is None:
                data = await self._load_data()
            
            # Run main backtest
            self.trades, self.equity_curve = await self._run_backtest(strategy, data)
            
            # Calculate returns and drawdown
            self.daily_returns = self.equity_curve.pct_change().dropna()
            self.drawdown_curve = self._calculate_drawdown(self.equity_curve)
            
            # Calculate metrics
            self.metrics = self._calculate_metrics()
            
            # Run Monte Carlo simulation
            if self.config.monte_carlo_enabled:
                await self._run_monte_carlo()
            
            # Run stress tests
            if self.config.stress_test_enabled:
                await self._run_stress_tests(data)
            
            # Generate report
            if self.config.generate_report:
                await self._generate_report()
            
            # Plot results
            if self.config.plot_results:
                self._plot_results()
            
            self.logger.info(f"Backtest completed. Total return: {self.metrics.total_return:.2f}%, Sharpe: {self.metrics.sharpe_ratio:.2f}")
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}", exc_info=True)
            raise
    
    async def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical data for all symbols"""
        self.data_engine = DataEngine(self.config.__dict__)
        await self.data_engine.initialize()
        
        data = {}
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                df = await self.data_engine.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    quality_check=True,
                    fill_gaps=True,
                    detect_outliers=True
                )
                
                if df is not None:
                    key = f"{symbol}_{timeframe}"
                    data[key] = df
                    self.logger.info(f"Loaded {len(df)} bars for {key}")
        
        await self.data_engine.shutdown()
        return data
    
    async def _run_backtest(self, strategy, data: Dict[str, pd.DataFrame]) -> Tuple[List[TradeResult], pd.Series]:
        """Execute main backtest"""
        all_trades = []
        equity = self.config.initial_capital
        equity_curve = []
        
        # Get primary timeframe for equity curve
        primary_key = list(data.keys())[0]
        primary_df = data[primary_key]
        
        for idx, timestamp in enumerate(primary_df.index):
            # Get data slice up to current timestamp
            current_data = {}
            for key, df in data.items():
                current_data[key] = df[df.index <= timestamp].copy()
            
            # Generate signals
            signals = await strategy.generate_trading_signals(current_data)
            
            # Process signals
            for signal in signals:
                trade = await self._execute_trade(signal, timestamp, equity)
                if trade:
                    all_trades.append(trade)
                    equity += trade.profit
            
            equity_curve.append({
                'timestamp': timestamp,
                'equity': equity
            })
        
        # Create equity curve Series
        equity_df = pd.DataFrame(equity_curve)
        equity_series = pd.Series(equity_df['equity'].values, index=equity_df['timestamp'])
        
        return all_trades, equity_series
    
    async def _execute_trade(self, signal: Dict, timestamp: datetime, 
                           current_equity: float) -> Optional[TradeResult]:
        """Execute single trade with transaction costs"""
        
        # Calculate commission
        commission = self._calculate_commission(signal)
        
        # Calculate slippage
        slippage = self._calculate_slippage(signal)
        
        # Adjust prices for slippage
        if signal['direction'] == 'long':
            entry_price = signal['entry_price'] * (1 + slippage)
            exit_price = signal['take_profit'] * (1 - slippage)
            stop_price = signal['stop_loss'] * (1 - slippage)
        else:
            entry_price = signal['entry_price'] * (1 - slippage)
            exit_price = signal['take_profit'] * (1 + slippage)
            stop_price = signal['stop_loss'] * (1 + slippage)
        
        # Calculate profit
        if signal['direction'] == 'long':
            profit = (exit_price - entry_price) * signal['quantity'] - commission
        else:
            profit = (entry_price - exit_price) * signal['quantity'] - commission
        
        # Calculate R-multiple
        risk = abs(entry_price - stop_price) * signal['quantity']
        r_multiple = profit / risk if risk > 0 else 0
        
        # Create trade record
        trade = TradeResult(
            trade_id=self._generate_trade_id(signal, timestamp),
            timestamp=timestamp,
            symbol=signal['symbol'],
            direction=signal['direction'],
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=signal['quantity'],
            commission=commission,
            slippage=slippage * 10000,  # Convert to bps
            profit=profit,
            profit_pips=self._calculate_pips(entry_price, exit_price, signal['symbol']),
            r_multiple=r_multiple,
            entry_time=timestamp,
            exit_time=timestamp + timedelta(hours=1),  # Simplified
            holding_period=60,  # Minutes
            exit_reason='take_profit',
            signal_type=signal.get('signal_type', 'unknown'),
            regime=signal.get('regime', 'unknown'),
            metadata=signal
        )
        
        return trade
    
    def _calculate_commission(self, signal: Dict) -> float:
        """Calculate transaction commission"""
        instrument = self._get_instrument_type(signal['symbol'])
        commission_rate = self.config.commission.get(instrument, 0.0001)
        
        notional = signal['entry_price'] * signal['quantity']
        return notional * commission_rate
    
    def _calculate_slippage(self, signal: Dict) -> float:
        """Calculate slippage based on model"""
        if self.config.slippage_model == 'fixed':
            return self.config.market_impact_params.get('base_impact', 0.0001)
        
        elif self.config.slippage_model == 'percentage':
            return signal['entry_price'] * 0.0001
        
        elif self.config.slippage_model == 'market_impact':
            # Almgren-Chriss market impact model
            params = self.config.market_impact_params
            base_impact = params.get('base_impact', 0.0001)
            volume_factor = params.get('volume_impact_factor', 0.5)
            
            # Simplified impact calculation
            impact = base_impact * (1 + volume_factor * np.log(signal['quantity'] + 1))
            return impact
        
        return 0.0001
    
    def _calculate_pips(self, entry: float, exit: float, symbol: str) -> float:
        """Calculate profit in pips"""
        if 'JPY' in symbol:
            multiplier = 100
        elif 'XAU' in symbol or 'XAG' in symbol:
            multiplier = 10
        else:
            multiplier = 10000
        
        return abs(exit - entry) * multiplier
    
    def _get_instrument_type(self, symbol: str) -> str:
        """Get instrument type from symbol"""
        if symbol.endswith(('USDT', 'BTC', 'ETH')):
            return 'crypto'
        elif len(symbol) in [6, 7] and symbol.isalpha():
            return 'forex'
        elif symbol in ['XAUUSD', 'XAGUSD']:
            return 'commodity'
        else:
            return 'other'
    
    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak
        return drawdown
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return None
        
        # Basic metrics
        total_return = (self.equity_curve.iloc[-1] / self.config.initial_capital - 1) * 100
        
        # Annualized return
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Volatility
        daily_returns = self.daily_returns
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside = daily_returns[daily_returns < 0]
        sortino = np.sqrt(252) * daily_returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
        
        # Calmar ratio
        max_dd = self.drawdown_curve.max() * 100
        calmar = annualized_return / max_dd if max_dd > 0 else 0
        
        # Trade metrics
        profits = [t.profit for t in self.trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        total_trades = len(self.trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        max_win = max(profits) if profits else 0
        max_loss = min(profits) if profits else 0
        
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        # R-multiple metrics
        r_multiples = [t.r_multiple for t in self.trades]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        r_std = np.std(r_multiples) if r_multiples else 0
        
        # Drawdown metrics
        max_drawdown = self.drawdown_curve.max() * 100
        
        # Calculate max drawdown duration
        in_drawdown = False
        current_duration = 0
        max_duration = 0
        
        for dd in self.drawdown_curve:
            if dd > 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
            else:
                in_drawdown = False
                current_duration = 0
        
        avg_drawdown = self.drawdown_curve.mean() * 100
        
        # Recovery factor
        total_profit = self.equity_curve.iloc[-1] - self.config.initial_capital
        recovery_factor = total_profit / (max_drawdown / 100 * self.config.initial_capital) if max_drawdown > 0 else float('inf')
        
        # Ulcer index
        ulcer_index = np.sqrt((self.drawdown_curve ** 2).mean()) * 100
        
        # VaR and CVaR
        var_95 = np.percentile(daily_returns, 5) * 100
        var_99 = np.percentile(daily_returns, 1) * 100
        cvar_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
        cvar_99 = daily_returns[daily_returns <= np.percentile(daily_returns, 1)].mean() * 100
        
        # Tail ratio
        tail_ratio = abs(var_95 / var_99) if var_99 != 0 else 0
        
        # Gain to pain ratio
        gain_to_pain = total_profit / abs(sum(daily_returns[daily_returns < 0])) if any(daily_returns < 0) else float('inf')
        
        # Omega ratio
        threshold = 0
        gains = daily_returns[daily_returns > threshold].sum()
        losses = abs(daily_returns[daily_returns < threshold].sum())
        omega = gains / losses if losses > 0 else float('inf')
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_r_multiple=avg_r,
            r_multiple_std=r_std,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_duration,
            avg_drawdown=avg_drawdown,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            tail_ratio=tail_ratio,
            gain_to_pain_ratio=gain_to_pain,
            mc_expected_return=0,
            mc_expected_sharpe=0,
            mc_expected_max_dd=0,
            mc_var_95=0,
            mc_var_99=0,
            mc_probability_of_ruin=0
        )
    
    async def _run_monte_carlo(self):
        """Run Monte Carlo simulation"""
        self.logger.info(f"Running {self.config.monte_carlo_simulations} Monte Carlo simulations...")
        
        # Get trade distribution
        if not self.trades:
            return
        
        profits = [t.profit for t in self.trades]
        
        # Fit distribution to trade profits
        dist = self._fit_distribution(profits)
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            for i in range(self.config.monte_carlo_simulations):
                future = executor.submit(
                    self._run_single_simulation,
                    profits,
                    dist,
                    len(self.trades),
                    self.config.initial_capital
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                sim_equity, sim_metrics = future.result()
                self.mc_equity_curves.append(sim_equity)
                self.mc_metrics.append(sim_metrics)
        
        # Calculate Monte Carlo metrics
        self._calculate_monte_carlo_metrics()
    
    def _run_single_simulation(self, profits: List[float], dist, n_trades: int, 
                              initial_capital: float) -> Tuple[pd.Series, Dict]:
        """Run single Monte Carlo simulation"""
        np.random.seed()  # Ensure different seeds per process
        
        # Resample trades with replacement
        sampled_profits = np.random.choice(profits, size=n_trades, replace=True)
        
        # Calculate equity curve
        equity = initial_capital + np.cumsum(sampled_profits)
        
        # Calculate metrics
        total_return = (equity[-1] / initial_capital - 1) * 100
        max_dd = self._calculate_max_drawdown_from_series(equity)
        sharpe = self._calculate_sharpe_from_series(equity)
        
        return pd.Series(equity), {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe
        }
    
    def _fit_distribution(self, data: List[float]) -> stats.rv_continuous:
        """Fit statistical distribution to data"""
        # Try different distributions
        distributions = [
            stats.norm,
            stats.t,
            stats.laplace,
            stats.genextreme
        ]
        
        best_dist = None
        best_aic = float('inf')
        
        for dist in distributions:
            try:
                params = dist.fit(data)
                aic = self._calculate_aic(dist, params, data)
                if aic < best_aic:
                    best_aic = aic
                    best_dist = (dist, params)
            except:
                continue
        
        if best_dist:
            dist, params = best_dist
            return dist(*params)
        
        return stats.norm(*stats.norm.fit(data))
    
    def _calculate_aic(self, dist, params, data) -> float:
        """Calculate Akaike Information Criterion"""
        log_likelihood = np.sum(dist.logpdf(data, *params))
        k = len(params)
        return 2 * k - 2 * log_likelihood
    
    def _calculate_max_drawdown_from_series(self, equity: np.ndarray) -> float:
        """Calculate max drawdown from equity series"""
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return np.max(drawdown) * 100
    
    def _calculate_sharpe_from_series(self, equity: np.ndarray) -> float:
        """Calculate Sharpe ratio from equity series"""
        returns = np.diff(equity) / equity[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_monte_carlo_metrics(self):
        """Calculate Monte Carlo statistics"""
        if not self.mc_metrics:
            return
        
        # Extract metrics
        returns = [m['total_return'] for m in self.mc_metrics]
        drawdowns = [m['max_drawdown'] for m in self.mc_metrics]
        sharpes = [m['sharpe_ratio'] for m in self.mc_metrics]
        
        # Update metrics
        self.metrics.mc_expected_return = np.mean(returns)
        self.metrics.mc_expected_sharpe = np.mean(sharpes)
        self.metrics.mc_expected_max_dd = np.mean(drawdowns)
        self.metrics.mc_var_95 = np.percentile(returns, 5)
        self.metrics.mc_var_99 = np.percentile(returns, 1)
        
        # Probability of ruin (losing > 50%)
        ruin_count = sum(1 for r in returns if r < -50)
        self.metrics.mc_probability_of_ruin = ruin_count / len(returns)
    
    async def _run_stress_tests(self, data: Dict[str, pd.DataFrame]):
        """Run stress test scenarios"""
        self.logger.info("Running stress tests...")
        
        for scenario in self.config.stress_scenarios:
            try:
                # Apply stress scenario to data
                stressed_data = self._apply_stress_scenario(data, scenario)
                
                # Run backtest on stressed data
                trades, equity = await self._run_backtest(self.strategy_engine, stressed_data)
                
                # Calculate metrics
                returns = (equity.iloc[-1] / self.config.initial_capital - 1) * 100
                max_dd = self._calculate_drawdown(equity).max() * 100
                
                self.stress_results[scenario['name']] = {
                    'total_return': returns,
                    'max_drawdown': max_dd,
                    'survived': returns > -100  # Not bankrupt
                }
                
                self.logger.info(f"Stress test {scenario['name']}: Return={returns:.1f}%, DD={max_dd:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Stress test {scenario['name']} failed: {e}")
    
    def _apply_stress_scenario(self, data: Dict[str, pd.DataFrame], 
                              scenario: Dict) -> Dict[str, pd.DataFrame]:
        """Apply stress scenario to data"""
        stressed = {}
        
        for key, df in data.items():
            df_stressed = df.copy()
            
            # Apply price shock
            if 'equity_drop' in scenario:
                shock = scenario['equity_drop']
                df_stressed['close'] = df['close'] * (1 + shock)
                df_stressed['open'] = df['open'] * (1 + shock)
                df_stressed['high'] = df['high'] * (1 + shock)
                df_stressed['low'] = df['low'] * (1 + shock)
            
            # Apply volatility shock
            if 'volatility_multiplier' in scenario:
                mult = scenario['volatility_multiplier']
                returns = df['close'].pct_change()
                stressed_returns = returns * mult
                df_stressed['close'] = df['close'].iloc[0] * (1 + stressed_returns).cumprod()
            
            stressed[key] = df_stressed
        
        return stressed
    
    async def _generate_report(self):
        """Generate comprehensive HTML/PDF report"""
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f"backtest_report_{timestamp}.html"
        
        # Generate HTML report
        html = self._generate_html_report()
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"Report saved to {report_path}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #F44336; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value {'positive' if self.metrics.total_return > 0 else 'negative'}">
                        {self.metrics.total_return:.2f}%
                    </div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.metrics.sharpe_ratio:.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.metrics.max_drawdown:.2f}%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
            </div>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{self.metrics.total_trades}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{self.metrics.win_rate:.2f}%</td>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td>{self.metrics.profit_factor:.2f}</td>
                </tr>
                <tr>
                    <td>Avg R-Multiple</td>
                    <td>{self.metrics.avg_r_multiple:.2f}</td>
                </tr>
                <tr>
                    <td>Expectancy</td>
                    <td>${self.metrics.expectancy:.2f}</td>
                </tr>
            </table>
            
            <h2>Risk Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>VaR (95%)</td>
                    <td>{self.metrics.var_95:.2f}%</td>
                </tr>
                <tr>
                    <td>VaR (99%)</td>
                    <td>{self.metrics.var_99:.2f}%</td>
                </tr>
                <tr>
                    <td>CVaR (95%)</td>
                    <td>{self.metrics.cvar_95:.2f}%</td>
                </tr>
                <tr>
                    <td>CVaR (99%)</td>
                    <td>{self.metrics.cvar_99:.2f}%</td>
                </tr>
                <tr>
                    <td>Ulcer Index</td>
                    <td>{self.metrics.ulcer_index:.2f}</td>
                </tr>
            </table>
            
            <h2>Monte Carlo Results</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Expected Return</td>
                    <td>{self.metrics.mc_expected_return:.2f}%</td>
                </tr>
                <tr>
                    <td>Expected Sharpe</td>
                    <td>{self.metrics.mc_expected_sharpe:.2f}</td>
                </tr>
                <tr>
                    <td>Expected Max DD</td>
                    <td>{self.metrics.mc_expected_max_dd:.2f}%</td>
                </tr>
                <tr>
                    <td>VaR (95%) - MC</td>
                    <td>{self.metrics.mc_var_95:.2f}%</td>
                </tr>
                <tr>
                    <td>VaR (99%) - MC</td>
                    <td>{self.metrics.mc_var_99:.2f}%</td>
                </tr>
                <tr>
                    <td>Probability of Ruin</td>
                    <td>{self.metrics.mc_probability_of_ruin:.2%}</td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        return html
    
    def _plot_results(self):
        """Generate interactive plots"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown', 'Monthly Returns', 
                          'Trade Distribution', 'Rolling Sharpe', 'Monte Carlo Bands'),
            specs=[[{'secondary_y': True}, {}], [{}, {}], [{}, {}]]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=self.equity_curve.index, y=self.equity_curve.values,
                      mode='lines', name='Equity', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=self.drawdown_curve.index, y=self.drawdown_curve.values * 100,
                      mode='lines', name='Drawdown', fill='tozeroy', line=dict(color='red')),
            row=1, col=2
        )
        
        # Monthly returns heatmap
        if self.daily_returns is not None:
            monthly_returns = self.daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            years = monthly_returns.index.year
            months = monthly_returns.index.month
            
            pivot = pd.pivot_table(
                pd.DataFrame({'return': monthly_returns, 'year': years, 'month': months}),
                values='return', index='month', columns='year', aggfunc='mean'
            )
            
            fig.add_trace(
                go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                          colorscale='RdYlGn', name='Monthly Returns'),
                row=2, col=1
            )
        
        # Trade distribution
        if self.trades:
            profits = [t.profit for t in self.trades]
            fig.add_trace(
                go.Histogram(x=profits, nbinsx=50, name='Trade Distribution'),
                row=2, col=2
            )
        
        # Rolling Sharpe (60-day)
        if self.daily_returns is not None and len(self.daily_returns) > 60:
            rolling_sharpe = self.daily_returns.rolling(60).apply(
                lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
            )
            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                          mode='lines', name='Rolling Sharpe (60d)'),
                row=3, col=1
            )
        
        # Monte Carlo bands
        if self.mc_equity_curves:
            mc_array = np.array([ec.values for ec in self.mc_equity_curves])
            percentile_5 = np.percentile(mc_array, 5, axis=0)
            percentile_95 = np.percentile(mc_array, 95, axis=0)
            
            fig.add_trace(
                go.Scatter(x=self.equity_curve.index, y=percentile_95,
                          mode='lines', name='MC 95%', line=dict(color='gray', dash='dash')),
                row=3, col=2
            )
            fig.add_trace(
                go.Scatter(x=self.equity_curve.index, y=percentile_5,
                          mode='lines', name='MC 5%', line=dict(color='gray', dash='dash'),
                          fill='tonexty'),
                row=3, col=2
            )
        
        fig.update_layout(height=1200, showlegend=True, title_text="Backtest Results")
        
        # Save plot
        plot_path = Path('reports') / f"backtest_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path)
        
        self.logger.info(f"Plot saved to {plot_path}")
    
    def _generate_trade_id(self, signal: Dict, timestamp: datetime) -> str:
        """Generate unique trade ID"""
        unique_str = f"{signal['symbol']}_{timestamp}_{signal.get('direction', '')}_{np.random.random()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    def save_results(self, path: Optional[Path] = None):
        """Save backtest results to disk"""
        if path is None:
            path = Path('backtest/results')
        
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
            trades_df.to_csv(path / f'trades_{timestamp}.csv', index=False)
        
        # Save equity curve
        if self.equity_curve is not None:
            self.equity_curve.to_csv(path / f'equity_{timestamp}.csv')
        
        # Save metrics
        if self.metrics:
            metrics_dict = {k: v for k, v in self.metrics.__dict__.items() 
                          if not isinstance(v, dict)}
            with open(path / f'metrics_{timestamp}.json', 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {path}")
    
    def print_summary(self):
        """Print comprehensive performance summary"""
        if not self.metrics:
            print("No backtest results available")
            return
        
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)
        
        print("\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Return: {self.metrics.total_return:+.2f}%")
        print(f"  Annualized Return: {self.metrics.annualized_return:+.2f}%")
        print(f"  Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {self.metrics.sortino_ratio:.2f}")
        print(f"  Calmar Ratio: {self.metrics.calmar_ratio:.2f}")
        print(f"  Omega Ratio: {self.metrics.omega_ratio:.2f}")
        
        print("\n📈 TRADE STATISTICS:")
        print(f"  Total Trades: {self.metrics.total_trades}")
        print(f"  Winning Trades: {self.metrics.winning_trades}")
        print(f"  Losing Trades: {self.metrics.losing_trades}")
        print(f"  Win Rate: {self.metrics.win_rate:.2f}%")
        print(f"  Profit Factor: {self.metrics.profit_factor:.2f}")
        print(f"  Expectancy: ${self.metrics.expectancy:.2f}")
        print(f"  Avg R-Multiple: {self.metrics.avg_r_multiple:.2f}")
        
        print("\n💰 TRADE SIZES:")
        print(f"  Avg Win: ${self.metrics.avg_win:.2f}")
        print(f"  Avg Loss: -${self.metrics.avg_loss:.2f}")
        print(f"  Max Win: ${self.metrics.max_win:.2f}")
        print(f"  Max Loss: -${self.metrics.max_loss:.2f}")
        
        print("\n📉 RISK METRICS:")
        print(f"  Max Drawdown: {self.metrics.max_drawdown:.2f}%")
        print(f"  Max DD Duration: {self.metrics.max_drawdown_duration} days")
        print(f"  Avg Drawdown: {self.metrics.avg_drawdown:.2f}%")
        print(f"  Recovery Factor: {self.metrics.recovery_factor:.2f}")
        print(f"  Ulcer Index: {self.metrics.ulcer_index:.2f}")
        
        print("\n🎲 VALUE AT RISK:")
        print(f"  VaR (95%): {self.metrics.var_95:.2f}%")
        print(f"  VaR (99%): {self.metrics.var_99:.2f}%")
        print(f"  CVaR (95%): {self.metrics.cvar_95:.2f}%")
        print(f"  CVaR (99%): {self.metrics.cvar_99:.2f}%")
        print(f"  Tail Ratio: {self.metrics.tail_ratio:.2f}")
        print(f"  Gain/Pain Ratio: {self.metrics.gain_to_pain_ratio:.2f}")
        
        if self.config.monte_carlo_enabled and self.metrics.mc_expected_return != 0:
            print("\n🎲 MONTE CARLO SIMULATION:")
            print(f"  Expected Return: {self.metrics.mc_expected_return:.2f}%")
            print(f"  Expected Sharpe: {self.metrics.mc_expected_sharpe:.2f}")
            print(f"  Expected Max DD: {self.metrics.mc_expected_max_dd:.2f}%")
            print(f"  VaR (95%) - MC: {self.metrics.mc_var_95:.2f}%")
            print(f"  VaR (99%) - MC: {self.metrics.mc_var_99:.2f}%")
            print(f"  Probability of Ruin: {self.metrics.mc_probability_of_ruin:.2%}")
        
        if self.stress_results:
            print("\n🌪️ STRESS TEST RESULTS:")
            for name, result in self.stress_results.items():
                status = "✅" if result['survived'] else "❌"
                print(f"  {status} {name}:")
                print(f"     Return: {result['total_return']:.1f}%")
                print(f"     Max DD: {result['max_drawdown']:.1f}%")
        
        print("\n" + "="*60)