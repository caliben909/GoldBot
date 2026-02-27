"""
Backtesting Engine - Professional backtesting with Monte Carlo simulation and stress testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
import logging
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t, genextreme, kurtosis, skew
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import pickle
import time
from enum import Enum
import traceback

# Import local modules
from core.risk_engine import RiskEngine
from core.strategy_engine import StrategyEngine
from core.data_engine import DataEngine
from utils.indicators import TechnicalIndicators
from utils.helpers import retry_with_backoff, safe_divide

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class Direction(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"
    BULLISH = "bullish"
    BEARISH = "bearish"


class ExitReason(Enum):
    """Exit reason for trades"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    END_OF_BACKTEST = "end_of_backtest"
    SIGNAL = "signal"
    TIME_EXIT = "time_exit"


class SlippageModel(Enum):
    """Slippage calculation models"""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    MARKET_IMPACT = "market_impact"


class DataSource(Enum):
    """Data source types"""
    YFINANCE = "yfinance"
    CSV = "csv"
    MT5 = "mt5"
    BINANCE = "binance"
    DATABASE = "database"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    enabled: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100000.0
    commission: Union[float, Dict[str, float]] = 0.0001
    slippage: float = 0.0001
    slippage_model: str = 'percentage'
    market_impact_params: Dict[str, Any] = field(default_factory=lambda: {
        'base_impact': 0.0001,
        'volume_impact_factor': 0.5,
        'spread_cost': 0.0001
    })
    data_source: str = 'database'
    data_quality: Dict[str, Any] = field(default_factory=lambda: {
        'sources': {
            'yfinance': {'enabled': True, 'priority': 1},
            'csv': {'enabled': False, 'priority': 2},
            'mt5': {'enabled': False, 'priority': 3},
            'binance': {'enabled': False, 'priority': 4}
        },
        'fill_gaps': True,
        'detect_outliers': True,
        'min_data_points': 100
    })
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ['1D'])
    
    # Risk management
    risk_management: Dict[str, Any] = field(default_factory=lambda: {
        'risk_per_trade': 0.25,  # percentage
        'max_position_size': 5.0,
        'min_position_size': 0.005,
        'max_correlation': 0.7,
        'max_drawdown': 0.25,
        'max_leverage': 2.0
    })
    
    # Monte Carlo settings
    monte_carlo_enabled: bool = True
    monte_carlo_simulations: int = 1000
    monte_carlo_confidence: List[float] = field(default_factory=lambda: [0.95, 0.99])
    
    # Stress testing
    stress_test_enabled: bool = True
    stress_scenarios: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'name': '2008 Crisis', 'equity_drop': -0.5, 'volatility_multiplier': 3.0},
        {'name': 'COVID Crash', 'equity_drop': -0.35, 'volatility_multiplier': 2.5},
        {'name': 'Flash Crash', 'equity_drop': -0.1, 'volatility_multiplier': 5.0}
    ])
    
    # Output settings
    output: Dict[str, Any] = field(default_factory=lambda: {
        'save_trades': True,
        'save_equity_curve': True,
        'generate_report': True,
        'plot_results': True,
        'report_format': 'html'
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime objects to strings
        result['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        result['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        result['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        return result


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
    stress_test_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def create_empty(cls) -> 'BacktestMetrics':
        """Create empty metrics object"""
        return cls(
            total_return=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            omega_ratio=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            max_win=0,
            max_loss=0,
            profit_factor=0,
            expectancy=0,
            avg_r_multiple=0,
            r_multiple_std=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            avg_drawdown=0,
            recovery_factor=0,
            ulcer_index=0,
            var_95=0,
            var_99=0,
            cvar_95=0,
            cvar_99=0,
            tail_ratio=0,
            gain_to_pain_ratio=0,
            mc_expected_return=0,
            mc_expected_sharpe=0,
            mc_expected_max_dd=0,
            mc_var_95=0,
            mc_var_99=0,
            mc_probability_of_ruin=0
        )


@dataclass
class TradingSignal:
    """Trading signal from strategy"""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float = 0.0
    signal_type: str = "standard"
    regime: str = "neutral"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


# =============================================================================
# Signal Object Adapter
# =============================================================================

class SignalAdapter:
    """Adapter to handle both dict and object signal types"""
    
    @staticmethod
    def to_object(signal: Union[Dict[str, Any], TradingSignal, Any]) -> TradingSignal:
        """Convert signal to TradingSignal object"""
        if isinstance(signal, TradingSignal):
            return signal
        
        if isinstance(signal, dict):
            return TradingSignal(
                symbol=signal.get('symbol', ''),
                direction=signal.get('direction', 'long'),
                entry_price=signal.get('entry_price', 0.0),
                stop_loss=signal.get('stop_loss', 0.0),
                take_profit=signal.get('take_profit', 0.0),
                quantity=signal.get('quantity', 0.0),
                signal_type=signal.get('signal_type', 'standard'),
                regime=signal.get('regime', 'neutral'),
                confidence=signal.get('confidence', 1.0),
                metadata=signal.get('metadata', {}),
                timestamp=signal.get('timestamp')
            )
        
        # Try to access attributes
        try:
            return TradingSignal(
                symbol=getattr(signal, 'symbol', ''),
                direction=getattr(signal, 'direction', 'long'),
                entry_price=getattr(signal, 'entry_price', 0.0),
                stop_loss=getattr(signal, 'stop_loss', 0.0),
                take_profit=getattr(signal, 'take_profit', 0.0),
                quantity=getattr(signal, 'quantity', 0.0),
                signal_type=getattr(signal, 'signal_type', 'standard'),
                regime=getattr(signal, 'regime', 'neutral'),
                confidence=getattr(signal, 'confidence', 1.0),
                metadata=getattr(signal, 'metadata', {}),
                timestamp=getattr(signal, 'timestamp', None)
            )
        except:
            raise TypeError(f"Cannot convert {type(signal)} to TradingSignal")


# =============================================================================
# Backtest Engine
# =============================================================================

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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtest engine
        
        Args:
            config: Configuration dictionary
        """
        # Initialize backtest config
        self.config = self._create_config(config)
        
        # Extract symbols from assets config
        self._extract_symbols(config)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize engines (lazy loading)
        self._risk_engine = None
        self._strategy_engine = None
        self._data_engine = None
        self._indicators = None
        
        # Results storage
        self.trades: List[TradeResult] = []
        self.equity_curve: Optional[pd.Series] = None
        self.drawdown_curve: Optional[pd.Series] = None
        self.daily_returns: Optional[pd.Series] = None
        self.metrics: Optional[BacktestMetrics] = None
        
        # Monte Carlo results
        self.mc_equity_curves: List[pd.Series] = []
        self.mc_metrics: List[Dict[str, float]] = []
        
        # Stress test results
        self.stress_results: Dict[str, Dict[str, Any]] = {}
        
        # Signal adapter
        self.signal_adapter = SignalAdapter()
        
        self.logger.info(f"BacktestEngine initialized with symbols: {', '.join(self.config.symbols)}")
    
    def _create_config(self, config: Dict[str, Any]) -> BacktestConfig:
        """Create BacktestConfig from dictionary"""
        backtest_config = config.get('backtesting', {})
        
        # Convert string dates to datetime
        if 'start_date' in backtest_config and isinstance(backtest_config['start_date'], str):
            backtest_config['start_date'] = pd.to_datetime(backtest_config['start_date'])
        if 'end_date' in backtest_config and isinstance(backtest_config['end_date'], str):
            backtest_config['end_date'] = pd.to_datetime(backtest_config['end_date'])
        
        return BacktestConfig(**backtest_config)
    
    def _extract_symbols(self, config: Dict[str, Any]) -> None:
        """Extract symbols from assets configuration"""
        if self.config.symbols is None or len(self.config.symbols) == 0:
            self.config.symbols = []
            
            # Add forex symbols
            if config.get('assets', {}).get('forex', {}).get('enabled', False):
                forex_symbols = config.get('assets', {}).get('forex', {}).get('symbols', [])
                self.config.symbols.extend(forex_symbols)
            
            # Add crypto symbols
            if config.get('assets', {}).get('crypto', {}).get('enabled', False):
                crypto_symbols = config.get('assets', {}).get('crypto', {}).get('symbols', [])
                self.config.symbols.extend(crypto_symbols)
            
            # Add indices symbols
            if config.get('assets', {}).get('indices', {}).get('enabled', False):
                indices_symbols = config.get('assets', {}).get('indices', {}).get('symbols', [])
                self.config.symbols.extend(indices_symbols)
    
    @property
    def risk_engine(self) -> RiskEngine:
        """Lazy load risk engine"""
        if self._risk_engine is None:
            self._risk_engine = RiskEngine({})
        return self._risk_engine
    
    @property
    def strategy_engine(self) -> StrategyEngine:
        """Lazy load strategy engine"""
        if self._strategy_engine is None:
            self._strategy_engine = StrategyEngine({})
        return self._strategy_engine
    
    @property
    def data_engine(self) -> DataEngine:
        """Lazy load data engine"""
        if self._data_engine is None:
            from core.data_engine import DataEngine
            data_config = {
                'data_quality': self.config.data_quality,
                'execution': {
                    'mt5': {'enabled': False},
                    'binance': {'enabled': False}
                }
            }
            self._data_engine = DataEngine(data_config)
        return self._data_engine
    
    @property
    def indicators(self) -> 'TechnicalIndicators':
        """Lazy load indicators"""
        if self._indicators is None:
            from utils.indicators import TechnicalIndicators
            self._indicators = TechnicalIndicators()
        return self._indicators
    
    async def run(self, strategy: Any, data: Optional[Dict[str, pd.DataFrame]] = None) -> BacktestMetrics:
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
            # Initialize data engine if needed
            if data is None:
                await self.data_engine.initialize()
                data = await self._load_data()
            
            # Run main backtest
            self.trades, self.equity_curve = await self._run_backtest(strategy, data)
            
            if not self.trades:
                self.logger.warning("No trades were executed during backtest")
                self.metrics = BacktestMetrics.create_empty()
                return self.metrics
            
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
            
            # Save results
            if self.config.save_trades or self.config.save_equity_curve:
                self.save_results()
            
            self.logger.info(
                f"Backtest completed. Total return: {self.metrics.total_return:.2f}%, "
                f"Sharpe: {self.metrics.sharpe_ratio:.2f}, "
                f"Trades: {self.metrics.total_trades}"
            )
            
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            if data is None and hasattr(self.data_engine, 'shutdown'):
                await self.data_engine.shutdown()
    
    async def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical data for all symbols"""
        data = {}
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                try:
                    df = await self.data_engine.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=self.config.start_date,
                        end=self.config.end_date,
                        quality_check=self.config.data_quality.get('fill_gaps', True),
                        fill_gaps=self.config.data_quality.get('fill_gaps', True),
                        detect_outliers=self.config.data_quality.get('detect_outliers', True)
                    )
                    
                    if df is not None and len(df) >= self.config.data_quality.get('min_data_points', 100):
                        key = f"{symbol}_{timeframe}"
                        data[key] = df
                        self.logger.info(f"Loaded {len(df)} bars for {key}")
                    else:
                        self.logger.warning(f"Insufficient data for {symbol} {timeframe}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to load data for {symbol} {timeframe}: {e}")
        
        if not data:
            raise ValueError("No data loaded for any symbol")
        
        return data
    
    async def _run_backtest(self, strategy: Any, data: Dict[str, pd.DataFrame]) -> Tuple[List[TradeResult], pd.Series]:
        """Execute main backtest"""
        all_trades = []
        equity = self.config.initial_capital
        equity_history = []
        open_positions = {}  # Track open positions by symbol
        
        # Get primary timeframe for equity curve
        primary_key = list(data.keys())[0]
        primary_df = data[primary_key]
        
        total_bars = len(primary_df)
        progress_interval = max(1, total_bars // 10)  # 10% progress updates
        
        for idx, timestamp in enumerate(primary_df.index):
            # Progress logging
            if idx % progress_interval == 0:
                progress = (idx / total_bars) * 100
                self.logger.debug(f"Backtest progress: {progress:.1f}%")
            
            # Get data slice up to current timestamp
            current_data = {}
            for key, df in data.items():
                current_data[key] = df[df.index <= timestamp].copy()
            
            # Generate signals for each symbol
            signals = []
            for key, df in current_data.items():
                if df.empty or len(df) < 50:  # Need minimum history
                    continue
                    
                # Extract symbol from key
                symbol = key.split('_')[0] if '_' in key else key
                
                # Generate signals for this symbol
                try:
                    symbol_signals = await strategy.generate_trading_signals(df, symbol)
                    if symbol_signals:
                        # Convert to list if single signal
                        if not isinstance(symbol_signals, list):
                            symbol_signals = [symbol_signals]
                        
                        # Add timestamp if missing
                        for sig in symbol_signals:
                            if hasattr(sig, 'timestamp') and sig.timestamp is None:
                                sig.timestamp = timestamp
                            elif isinstance(sig, dict) and 'timestamp' not in sig:
                                sig['timestamp'] = timestamp
                        
                        signals.extend(symbol_signals)
                except Exception as e:
                    self.logger.error(f"Failed to generate signals for {symbol}: {e}")
            
            # Process signals
            for signal_obj in signals:
                # Convert to TradingSignal object
                try:
                    signal = self.signal_adapter.to_object(signal_obj)
                except Exception as e:
                    self.logger.error(f"Failed to convert signal: {e}")
                    continue
                
                # Check if we already have an open position for this symbol
                if signal.symbol in open_positions:
                    # Check for exit signal
                    if signal.direction == 'exit' or signal.signal_type == 'exit':
                        trade = await self._close_position(signal.symbol, timestamp, current_data[primary_key])
                        if trade:
                            all_trades.append(trade)
                            equity += trade.profit
                            del open_positions[signal.symbol]
                    continue
                
                # Execute new trade
                trade = await self._execute_trade(signal, timestamp, equity, current_data[primary_key])
                if trade:
                    all_trades.append(trade)
                    equity += trade.profit
                    open_positions[signal.symbol] = trade
            
            # Check for stop loss / take profit on open positions
            for symbol in list(open_positions.keys()):
                trade = await self._check_position_exit(symbol, timestamp, current_data[primary_key])
                if trade:
                    all_trades.append(trade)
                    equity += trade.profit
                    del open_positions[symbol]
            
            equity_history.append({
                'timestamp': timestamp,
                'equity': equity
            })
        
        # Close any remaining positions at end of backtest
        for symbol, position in open_positions.items():
            trade = await self._force_close_position(symbol, primary_df.index[-1], primary_df)
            if trade:
                all_trades.append(trade)
                equity += trade.profit
        
        # Create equity curve Series
        equity_df = pd.DataFrame(equity_history)
        equity_series = pd.Series(equity_df['equity'].values, index=equity_df['timestamp'])
        
        return all_trades, equity_series
    
    def _calculate_position_size(self, signal: TradingSignal, current_equity: float) -> float:
        """Calculate position size based on risk management parameters"""
        # Get risk per trade from config (percentage)
        risk_per_trade = self.config.risk_management.get('risk_per_trade', 0.25) / 100
        
        # Calculate risk amount in currency
        risk_amount = current_equity * risk_per_trade
        
        # Calculate risk per unit (distance from entry to stop loss)
        risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        
        if risk_per_unit == 0 or np.isnan(risk_per_unit):
            return 0.0
            
        # Calculate position size
        quantity = risk_amount / risk_per_unit
        
        # Apply position size limits
        min_size = self.config.risk_management.get('min_position_size', 0.005)
        max_size = self.config.risk_management.get('max_position_size', 5.0)
        
        return max(min_size, min(quantity, max_size))
    
    async def _execute_trade(self, signal: TradingSignal, timestamp: datetime, 
                           current_equity: float, data: pd.DataFrame) -> Optional[TradeResult]:
        """Execute single trade with transaction costs"""
        
        # Check for multi-tier exit strategy
        if signal.metadata and 'multi_tier_exit' in signal.metadata:
            return await self._execute_enhanced_trade(signal, timestamp, current_equity, data)
        
        # Calculate position size
        signal.quantity = self._calculate_position_size(signal, current_equity)
        
        if signal.quantity <= 0:
            return None
        
        # Calculate commission
        commission = self._calculate_commission(signal)
        
        # Calculate slippage
        slippage_pct = self._calculate_slippage(signal)
        
        # Adjust prices for slippage
        if signal.direction in ['long', 'bullish']:
            entry_price = signal.entry_price * (1 + slippage_pct)
            tp_price = signal.take_profit * (1 + slippage_pct)
            sl_price = signal.stop_loss * (1 - slippage_pct)
        else:  # short or bearish
            entry_price = signal.entry_price * (1 - slippage_pct)
            tp_price = signal.take_profit * (1 - slippage_pct)
            sl_price = signal.stop_loss * (1 + slippage_pct)
        
        self.logger.debug(f"Executing {signal.direction} trade at {timestamp}: "
                         f"Entry={entry_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}")
        
        # Find exit
        exit_result = self._find_exit(
            data, timestamp, signal.direction, entry_price, sl_price, tp_price
        )
        
        if not exit_result:
            return None
        
        exit_price, exit_time, exit_reason = exit_result
        
        # Calculate profit
        if signal.direction in ['long', 'bullish']:
            profit = (exit_price - entry_price) * signal.quantity - commission
        else:
            profit = (entry_price - exit_price) * signal.quantity - commission
        
        # Calculate R-multiple
        risk = abs(entry_price - sl_price) * signal.quantity
        r_multiple = safe_divide(profit, risk, 0.0)
        
        # Calculate holding period
        holding_period = (exit_time - timestamp).total_seconds() / 60  # Minutes
        
        # Calculate pips
        profit_pips = self._calculate_pips(entry_price, exit_price, signal.symbol)
        
        # Create trade record
        trade = TradeResult(
            trade_id=self._generate_trade_id(signal, timestamp),
            timestamp=timestamp,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=signal.quantity,
            commission=commission,
            slippage=slippage_pct * 10000,  # Convert to bps
            profit=profit,
            profit_pips=profit_pips,
            r_multiple=r_multiple,
            entry_time=timestamp,
            exit_time=exit_time,
            holding_period=holding_period,
            exit_reason=exit_reason,
            signal_type=signal.signal_type,
            regime=signal.regime,
            metadata=signal.metadata
        )
        
        return trade
    
    async def _execute_enhanced_trade(self, signal: TradingSignal, timestamp: datetime,
                                     current_equity: float, data: pd.DataFrame) -> Optional[TradeResult]:
        """Execute enhanced trade with trailing stop and scaling-in logic"""
        
        # Calculate initial position size (50%)
        base_size = self._calculate_position_size(signal, current_equity)
        initial_quantity = base_size * 0.5
        signal.quantity = initial_quantity
        
        if initial_quantity <= 0:
            return None
        
        # Calculate commission for initial position
        initial_commission = self._calculate_commission(signal)
        
        # Calculate slippage
        slippage_pct = self._calculate_slippage(signal)
        
        # Adjust prices for slippage
        if signal.direction in ['long', 'bullish']:
            entry_price = signal.entry_price * (1 + slippage_pct)
            sl_price = signal.stop_loss * (1 - slippage_pct)
        else:  # short or bearish
            entry_price = signal.entry_price * (1 - slippage_pct)
            sl_price = signal.stop_loss * (1 + slippage_pct)
        
        self.logger.debug(f"Executing enhanced {signal.direction} trade at {timestamp}")
        
        # Get trailing stop parameters
        metadata = signal.metadata
        activate_threshold = metadata.get('trailing_stop', {}).get('activate_threshold', 0.03)
        trailing_percent = metadata.get('trailing_stop', {}).get('trailing_percent', 0.5)
        scale_in_price = metadata.get('scaling', {}).get('scale_in_price', None)
        
        # Find exit with enhanced logic
        exit_result = self._find_enhanced_exit(
            data, timestamp, signal.direction, entry_price, sl_price,
            activate_threshold, trailing_percent, scale_in_price,
            base_size, signal
        )
        
        if not exit_result:
            return None
        
        exit_price, exit_time, exit_reason, final_quantity, total_commission = exit_result
        
        # Calculate profit
        if signal.direction in ['long', 'bullish']:
            profit = (exit_price - entry_price) * final_quantity - total_commission
        else:
            profit = (entry_price - exit_price) * final_quantity - total_commission
        
        # Calculate R-multiple
        risk = abs(entry_price - sl_price) * final_quantity
        r_multiple = safe_divide(profit, risk, 0.0)
        
        # Calculate holding period
        holding_period = (exit_time - timestamp).total_seconds() / 60  # Minutes
        
        # Calculate pips
        profit_pips = self._calculate_pips(entry_price, exit_price, signal.symbol)
        
        # Create trade record
        trade = TradeResult(
            trade_id=self._generate_trade_id(signal, timestamp),
            timestamp=timestamp,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=final_quantity,
            commission=total_commission,
            slippage=slippage_pct * 10000,  # Convert to bps
            profit=profit,
            profit_pips=profit_pips,
            r_multiple=r_multiple,
            entry_time=timestamp,
            exit_time=exit_time,
            holding_period=holding_period,
            exit_reason=exit_reason,
            signal_type=signal.signal_type,
            regime=signal.regime,
            metadata=signal.metadata
        )
        
        return trade
    
    def _find_exit(self, data: pd.DataFrame, entry_time: datetime, direction: str,
                  entry_price: float, sl_price: float, tp_price: float) -> Optional[Tuple[float, datetime, str]]:
        """Find exit price and time by scanning future candles"""
        
        # Get index of current timestamp
        try:
            current_idx = data.index.get_loc(entry_time)
        except KeyError:
            return None
        
        trade_executed = False
        exit_price = 0
        exit_time = entry_time
        exit_reason = ''
        
        # Look into future to find where SL or TP is hit
        for i in range(current_idx + 1, len(data)):
            next_timestamp = data.index[i]
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            
            if direction in ['long', 'bullish']:
                # Long trade: TP is above entry, SL is below entry
                if high >= tp_price:
                    exit_price = tp_price
                    exit_time = next_timestamp
                    exit_reason = ExitReason.TAKE_PROFIT.value
                    trade_executed = True
                    break
                elif low <= sl_price:
                    exit_price = sl_price
                    exit_time = next_timestamp
                    exit_reason = ExitReason.STOP_LOSS.value
                    trade_executed = True
                    break
            else:
                # Short trade: TP is below entry, SL is above entry
                if low <= tp_price:
                    exit_price = tp_price
                    exit_time = next_timestamp
                    exit_reason = ExitReason.TAKE_PROFIT.value
                    trade_executed = True
                    break
                elif high >= sl_price:
                    exit_price = sl_price
                    exit_time = next_timestamp
                    exit_reason = ExitReason.STOP_LOSS.value
                    trade_executed = True
                    break
        
        # If no SL/TP hit, close at last available price
        if not trade_executed:
            exit_price = data['close'].iloc[-1]
            exit_time = data.index[-1]
            exit_reason = ExitReason.END_OF_BACKTEST.value
        
        return exit_price, exit_time, exit_reason
    
    def _find_enhanced_exit(self, data: pd.DataFrame, entry_time: datetime, direction: str,
                          entry_price: float, sl_price: float, activate_threshold: float,
                          trailing_percent: float, scale_in_price: Optional[float],
                          base_size: float, signal: TradingSignal) -> Optional[Tuple[float, datetime, str, float, float]]:
        """Find exit with trailing stop and scaling logic"""
        
        try:
            current_idx = data.index.get_loc(entry_time)
        except KeyError:
            return None
        
        trade_executed = False
        exit_price = 0
        exit_time = entry_time
        exit_reason = ''
        final_quantity = base_size * 0.5
        total_commission = self._calculate_commission(signal)
        trailing_stop = None
        scaled_in = False
        
        # Multi-tier take profit levels
        tp_levels = []
        if signal.metadata and 'multi_tier_exit' in signal.metadata:
            tp_levels = sorted(
                signal.metadata['multi_tier_exit'],
                key=lambda x: x['percentage']
            )
        
        for i in range(current_idx + 1, len(data)):
            next_timestamp = data.index[i]
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            
            # Check for scaling-in opportunity
            if not scaled_in and scale_in_price is not None:
                if direction in ['long', 'bullish'] and low <= scale_in_price:
                    # Add scaling-in position
                    signal.quantity = base_size * 0.5
                    scale_commission = self._calculate_commission(signal)
                    total_commission += scale_commission
                    final_quantity += base_size * 0.5
                    scaled_in = True
                    self.logger.debug(f"Scaled in at {next_timestamp}, new size: {final_quantity}")
                    
                elif direction in ['short', 'bearish'] and high >= scale_in_price:
                    signal.quantity = base_size * 0.5
                    scale_commission = self._calculate_commission(signal)
                    total_commission += scale_commission
                    final_quantity += base_size * 0.5
                    scaled_in = True
                    self.logger.debug(f"Scaled in at {next_timestamp}, new size: {final_quantity}")
            
            # Handle trailing stop activation
            if trailing_stop is None:
                if direction in ['long', 'bullish'] and high >= entry_price * (1 + activate_threshold):
                    trailing_stop = high * (1 - trailing_percent)
                    self.logger.debug(f"Trailing stop activated at {trailing_stop:.2f}")
                elif direction in ['short', 'bearish'] and low <= entry_price * (1 - activate_threshold):
                    trailing_stop = low * (1 + trailing_percent)
                    self.logger.debug(f"Trailing stop activated at {trailing_stop:.2f}")
            else:
                # Update trailing stop
                if direction in ['long', 'bullish'] and high > entry_price * (1 + activate_threshold):
                    new_trailing = high * (1 - trailing_percent)
                    if new_trailing > trailing_stop:
                        trailing_stop = new_trailing
                        self.logger.debug(f"Trailing stop updated to {trailing_stop:.2f}")
                elif direction in ['short', 'bearish'] and low < entry_price * (1 - activate_threshold):
                    new_trailing = low * (1 + trailing_percent)
                    if new_trailing < trailing_stop:
                        trailing_stop = new_trailing
                        self.logger.debug(f"Trailing stop updated to {trailing_stop:.2f}")
            
            # Check multi-tier take profit
            for tp_level in tp_levels:
                if direction in ['long', 'bullish'] and high >= tp_level['level']:
                    exit_price = tp_level['level']
                    exit_time = next_timestamp
                    exit_reason = f"{ExitReason.TAKE_PROFIT.value}_{int(tp_level['percentage']*100)}%"
                    trade_executed = True
                    self.logger.debug(f"TP hit at {exit_price:.2f} ({exit_reason})")
                    break
                elif direction in ['short', 'bearish'] and low <= tp_level['level']:
                    exit_price = tp_level['level']
                    exit_time = next_timestamp
                    exit_reason = f"{ExitReason.TAKE_PROFIT.value}_{int(tp_level['percentage']*100)}%"
                    trade_executed = True
                    self.logger.debug(f"TP hit at {exit_price:.2f} ({exit_reason})")
                    break
            
            if trade_executed:
                break
            
            # Check stop loss or trailing stop
            if trailing_stop is not None:
                if direction in ['long', 'bullish'] and low <= trailing_stop:
                    exit_price = trailing_stop
                    exit_time = next_timestamp
                    exit_reason = ExitReason.TRAILING_STOP.value
                    trade_executed = True
                    self.logger.debug(f"Trailing stop hit at {exit_price:.2f}")
                    break
                elif direction in ['short', 'bearish'] and high >= trailing_stop:
                    exit_price = trailing_stop
                    exit_time = next_timestamp
                    exit_reason = ExitReason.TRAILING_STOP.value
                    trade_executed = True
                    self.logger.debug(f"Trailing stop hit at {exit_price:.2f}")
                    break
            else:
                if direction in ['long', 'bullish'] and low <= sl_price:
                    exit_price = sl_price
                    exit_time = next_timestamp
                    exit_reason = ExitReason.STOP_LOSS.value
                    trade_executed = True
                    self.logger.debug(f"Stop loss hit at {exit_price:.2f}")
                    break
                elif direction in ['short', 'bearish'] and high >= sl_price:
                    exit_price = sl_price
                    exit_time = next_timestamp
                    exit_reason = ExitReason.STOP_LOSS.value
                    trade_executed = True
                    self.logger.debug(f"Stop loss hit at {exit_price:.2f}")
                    break
        
        # If no exit found, close at last price
        if not trade_executed:
            exit_price = data['close'].iloc[-1]
            exit_time = data.index[-1]
            exit_reason = ExitReason.END_OF_BACKTEST.value
        
        return exit_price, exit_time, exit_reason, final_quantity, total_commission
    
    async def _check_position_exit(self, symbol: str, timestamp: datetime, 
                                  data: pd.DataFrame) -> Optional[TradeResult]:
        """Check if existing position should be exited due to SL/TP"""
        # This would require tracking open positions
        # Simplified version - actual implementation would need position tracking
        return None
    
    async def _close_position(self, symbol: str, timestamp: datetime, 
                             data: pd.DataFrame) -> Optional[TradeResult]:
        """Close position based on exit signal"""
        # Simplified - actual implementation would need position data
        return None
    
    async def _force_close_position(self, symbol: str, timestamp: datetime,
                                   data: pd.DataFrame) -> Optional[TradeResult]:
        """Force close position at end of backtest"""
        # Simplified - actual implementation would need position data
        return None
    
    def _calculate_commission(self, signal: TradingSignal) -> float:
        """Calculate transaction commission"""
        instrument = self._get_instrument_type(signal.symbol)
        
        # Get commission rate
        if isinstance(self.config.commission, float):
            commission_rate = self.config.commission
        elif isinstance(self.config.commission, dict):
            commission_rate = self.config.commission.get(instrument, 0.0001)
        else:
            commission_rate = 0.0001
        
        notional = signal.entry_price * signal.quantity
        return notional * commission_rate
    
    def _calculate_slippage(self, signal: TradingSignal) -> float:
        """Calculate slippage based on model"""
        model = SlippageModel(self.config.slippage_model)
        
        if model == SlippageModel.FIXED:
            return self.config.market_impact_params.get('base_impact', 0.0001)
        
        elif model == SlippageModel.PERCENTAGE:
            return self.config.slippage
        
        elif model == SlippageModel.MARKET_IMPACT:
            # Almgren-Chriss market impact model
            params = self.config.market_impact_params
            base_impact = params.get('base_impact', 0.0001)
            volume_factor = params.get('volume_impact_factor', 0.5)
            spread_cost = params.get('spread_cost', 0.0001)
            
            # Simplified impact calculation
            impact = base_impact * (1 + volume_factor * np.log(signal.quantity + 1))
            return impact + spread_cost
        
        return self.config.slippage
    
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
            return BacktestMetrics.create_empty()
        
        # Basic metrics
        total_return = (self.equity_curve.iloc[-1] / self.config.initial_capital - 1) * 100
        
        # Annualized return
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = max(days / 365.25, 0.01)  # Avoid division by zero
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        
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
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        max_win = max(profits) if profits else 0
        max_loss = min(profits) if profits else 0
        
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        profit_factor = safe_divide(gross_profit, gross_loss, float('inf'))
        
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
        recovery_factor = safe_divide(
            total_profit,
            (max_drawdown / 100 * self.config.initial_capital),
            float('inf')
        )
        
        # Ulcer index
        ulcer_index = np.sqrt((self.drawdown_curve ** 2).mean()) * 100
        
        # VaR and CVaR
        var_95 = np.percentile(daily_returns, 5) * 100
        var_99 = np.percentile(daily_returns, 1) * 100
        cvar_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
        cvar_99 = daily_returns[daily_returns <= np.percentile(daily_returns, 1)].mean() * 100
        
        # Tail ratio
        tail_ratio = safe_divide(abs(var_95), abs(var_99), 0)
        
        # Gain to pain ratio
        gain_to_pain = safe_divide(
            total_profit,
            abs(sum(daily_returns[daily_returns < 0])),
            float('inf')
        )
        
        # Omega ratio
        threshold = 0
        gains = daily_returns[daily_returns > threshold].sum()
        losses = abs(daily_returns[daily_returns < threshold].sum())
        omega = safe_divide(gains, losses, float('inf'))
        
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
        
        if not self.trades:
            self.logger.warning("No trades to run Monte Carlo simulation")
            return
        
        # Get trade distribution
        profits = [t.profit for t in self.trades]
        
        # Fit distribution to trade profits
        dist = self._fit_distribution(profits)
        
        # Run simulations in parallel
        n_workers = min(mp.cpu_count(), self.config.monte_carlo_simulations)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
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
                try:
                    sim_equity, sim_metrics = future.result(timeout=60)
                    self.mc_equity_curves.append(pd.Series(sim_equity))
                    self.mc_metrics.append(sim_metrics)
                except Exception as e:
                    self.logger.error(f"Monte Carlo simulation failed: {e}")
        
        # Calculate Monte Carlo metrics
        self._calculate_monte_carlo_metrics()
    
    def _run_single_simulation(self, profits: List[float], dist, n_trades: int,
                              initial_capital: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run single Monte Carlo simulation"""
        np.random.seed()  # Ensure different seeds per process
        
        # Resample trades with replacement
        sampled_profits = np.random.choice(profits, size=n_trades, replace=True)
        
        # Calculate equity curve
        equity = initial_capital + np.cumsum(sampled_profits)
        
        # Calculate metrics
        total_return = (equity[-1] / initial_capital - 1) * 100
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown) * 100
        
        # Sharpe ratio
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        return equity, {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe
        }
    
    def _fit_distribution(self, data: List[float]) -> Any:
        """Fit statistical distribution to data"""
        # Try different distributions
        distributions = [
            (stats.norm, stats.norm.fit(data)),
            (stats.t, self._fit_t_distribution(data)),
            (stats.laplace, stats.laplace.fit(data)),
            (stats.genextreme, stats.genextreme.fit(data))
        ]
        
        best_dist = None
        best_aic = float('inf')
        
        for dist, params in distributions:
            try:
                aic = self._calculate_aic(dist, params, data)
                if aic < best_aic:
                    best_aic = aic
                    best_dist = dist(*params)
            except:
                continue
        
        return best_dist or stats.norm(*stats.norm.fit(data))
    
    def _fit_t_distribution(self, data: List[float]) -> Tuple[float, float, float]:
        """Fit t-distribution to data"""
        # Try different degrees of freedom
        best_df = 5
        best_params = None
        best_aic = float('inf')
        
        for df in [3, 4, 5, 6, 8, 10, 15]:
            try:
                params = stats.t.fit(data, f