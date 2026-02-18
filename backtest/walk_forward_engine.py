"""
Walk-forward Analysis Engine for strategy optimization
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import warnings
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import itertools
import concurrent.futures
from functools import partial

from backtest.backtest_engine import BacktestEngine, BacktestMetrics, TradeResult
from core.strategy_engine import StrategyEngine
from core.data_engine import DataEngine
from core.risk_engine import RiskEngine
from utils.indicators import TechnicalIndicators
from utils.helpers import retry_with_backoff

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    # Test period settings
    test_period_days: int = 60
    training_period_days: int = 180
    lookback_period_days: int = 30
    reoptimization_frequency: int = 30  # days
    
    # Optimization settings
    parameter_grid: Dict[str, List[Any]] = field(default_factory=dict)
    max_workers: int = 4
    timeout: int = 3600
    
    # Performance metrics for optimization
    optimization_metric: str = 'sharpe_ratio'
    secondary_metric: str = 'profit_factor'
    
    # Risk constraints
    max_drawdown_constraint: float = 0.25
    min_profit_factor: float = 1.1
    min_win_rate: float = 0.45
    
    # Output settings
    save_results: bool = True
    save_optimal_params: bool = True
    generate_report: bool = True
    report_format: str = 'html'
    plot_results: bool = True


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis"""
    # Overall performance
    overall_metrics: BacktestMetrics
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    
    # Per-period results
    period_results: List[Dict]
    optimal_parameters: List[Dict]
    
    # Best parameters over time
    best_parameters_over_time: List[Dict]
    
    # Parameter stability
    parameter_stability: Dict[str, float]
    
    # Performance metrics per parameter set
    parameter_performance: Dict[str, BacktestMetrics]
    
    # Trading statistics
    total_trades: int
    avg_trades_per_period: float
    period_win_rate: float
    
    # Optimization history
    optimization_history: List[Dict]


class WalkForwardEngine:
    """
    Walk-forward analysis engine for robust strategy optimization
    
    Features:
    - Rolling window optimization
    - Parameter stability analysis
    - Out-of-sample performance testing
    - Parallel execution for speed
    - Comprehensive performance metrics
    - Risk-aware parameter selection
    """
    
    def __init__(self, config: dict):
        self.config = WalkForwardConfig(**config['walk_forward'])
        self.logger = logging.getLogger(__name__)
        
        # Initialize engines
        self.backtest_engine = BacktestEngine(config)
        self.risk_engine = RiskEngine(config)
        self.indicators = TechnicalIndicators()
        
        # Results storage
        self.results: Optional[WalkForwardResult] = None
        self.all_trades: List[TradeResult] = []
        
        logger.info("WalkForwardEngine initialized")
    
    async def run(self, strategy: StrategyEngine, 
                 data: Dict[str, pd.DataFrame]) -> WalkForwardResult:
        """
        Run walk-forward analysis
        
        Args:
            strategy: Strategy instance
            data: Pre-loaded data
            
        Returns:
            WalkForwardResult object
        """
        self.logger.info(f"Starting walk-forward analysis with {len(data)} symbols")
        
        try:
            # Split data into walk-forward periods
            periods = await self._split_data_into_periods(data)
            
            # Run optimization for each period
            period_results = await self._run_period_optimization(strategy, periods)
            
            # Calculate overall performance
            overall_metrics, equity_curve = await self._calculate_overall_performance(period_results)
            
            # Analyze parameter stability
            parameter_stability = await self._analyze_parameter_stability(period_results)
            
            # Calculate parameter performance
            parameter_performance = await self._calculate_parameter_performance(period_results)
            
            # Generate results object
            self.results = WalkForwardResult(
                overall_metrics=overall_metrics,
                equity_curve=equity_curve,
                drawdown_curve=self._calculate_drawdown(equity_curve),
                period_results=period_results,
                optimal_parameters=[r['best_params'] for r in period_results],
                best_parameters_over_time=[r['best_params'] for r in period_results],
                parameter_stability=parameter_stability,
                parameter_performance=parameter_performance,
                total_trades=sum(r['metrics'].total_trades for r in period_results),
                avg_trades_per_period=np.mean([r['metrics'].total_trades for r in period_results]),
                period_win_rate=sum(1 for r in period_results if r['metrics'].win_rate > 50) / len(period_results),
                optimization_history=self._extract_optimization_history(period_results)
            )
            
            # Generate report
            if self.config.generate_report:
                await self._generate_report()
                
            # Plot results
            if self.config.plot_results:
                await self._plot_results()
                
            self.logger.info(f"Walk-forward analysis completed. Total return: {overall_metrics.total_return:.2f}%")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis failed: {e}", exc_info=True)
            raise
    
    async def _split_data_into_periods(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Split data into walk-forward periods"""
        periods = []
        
        # Find global date range
        all_dates = []
        for symbol_data in data.values():
            all_dates.extend(symbol_data.index.tolist())
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Calculate period boundaries
        current_period_start = min_date
        while current_period_start + timedelta(days=self.training_period_days + self.test_period_days) <= max_date:
            training_start = current_period_start
            training_end = current_period_start + timedelta(days=self.training_period_days)
            test_start = training_end
            test_end = training_end + timedelta(days=self.test_period_days)
            
            # Split data for period
            period_data = {
                'training_data': {},
                'test_data': {}
            }
            
            for symbol, df in data.items():
                # Filter training data
                training_mask = (df.index >= training_start) & (df.index < training_end)
                period_data['training_data'][symbol] = df[training_mask].copy()
                
                # Filter test data
                test_mask = (df.index >= test_start) & (df.index < test_end)
                period_data['test_data'][symbol] = df[test_mask].copy()
            
            periods.append({
                'training_start': training_start,
                'training_end': training_end,
                'test_start': test_start,
                'test_end': test_end,
                'data': period_data
            })
            
            current_period_start += timedelta(days=self.reoptimization_frequency)
        
        self.logger.info(f"Split data into {len(periods)} walk-forward periods")
        
        return periods
    
    async def _run_period_optimization(self, strategy: StrategyEngine, 
                                     periods: List[Dict]) -> List[Dict]:
        """Run optimization for each period in parallel"""
        period_results = []
        
        # Use ProcessPoolExecutor for parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for period in periods:
                future = executor.submit(
                    self._optimize_period,
                    strategy,
                    period
                )
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures, timeout=self.config.timeout):
                try:
                    result = future.result()
                    period_results.append(result)
                except Exception as e:
                    self.logger.error(f"Period optimization failed: {e}")
        
        # Sort results by date
        period_results.sort(key=lambda x: x['training_start'])
        
        return period_results
    
    def _optimize_period(self, strategy: StrategyEngine, period: Dict) -> Dict:
        """Optimize strategy parameters for a single period"""
        # Generate parameter combinations
        parameter_combinations = list(itertools.product(*self.config.parameter_grid.values()))
        parameter_names = list(self.config.parameter_grid.keys())
        
        best_metrics = None
        best_params = None
        best_backtest = None
        
        # Evaluate each parameter combination
        for params_tuple in parameter_combinations:
            params = dict(zip(parameter_names, params_tuple))
            
            # Update strategy parameters
            strategy.update_parameters(params)
            
            # Run backtest on training data
            backtest_result = self.backtest_engine.run(
                strategy,
                period['data']['training_data']
            )
            
            if backtest_result:
                metrics = backtest_result['metrics']
                
                # Apply constraints
                if (metrics.max_drawdown < self.config.max_drawdown_constraint and
                    metrics.profit_factor > self.config.min_profit_factor and
                    metrics.win_rate > self.config.min_win_rate):
                    
                    # Compare with best result
                    if best_metrics is None or self._is_better(metrics, best_metrics):
                        best_metrics = metrics
                        best_params = params
                        best_backtest = backtest_result
        
        # Test best parameters on out-of-sample data
        strategy.update_parameters(best_params)
        test_result = self.backtest_engine.run(
            strategy,
            period['data']['test_data']
        )
        
        return {
            'training_start': period['training_start'],
            'training_end': period['training_end'],
            'test_start': period['test_start'],
            'test_end': period['test_end'],
            'best_params': best_params,
            'training_metrics': best_metrics,
            'test_metrics': test_result['metrics'],
            'backtest_result': test_result
        }
    
    def _is_better(self, metrics1: BacktestMetrics, metrics2: BacktestMetrics) -> bool:
        """Determine if metrics1 is better than metrics2 based on optimization criteria"""
        # First check primary metric
        if self.config.optimization_metric == 'sharpe_ratio':
            if metrics1.sharpe_ratio > metrics2.sharpe_ratio:
                return True
        elif self.config.optimization_metric == 'profit_factor':
            if metrics1.profit_factor > metrics2.profit_factor:
                return True
        elif self.config.optimization_metric == 'total_return':
            if metrics1.total_return > metrics2.total_return:
                return True
        elif self.config.optimization_metric == 'calmar_ratio':
            if metrics1.calmar_ratio > metrics2.calmar_ratio:
                return True
        
        # If primary metric is equal, check secondary metric
        if self.config.secondary_metric == 'sharpe_ratio':
            return metrics1.sharpe_ratio > metrics2.sharpe_ratio
        elif self.config.secondary_metric == 'profit_factor':
            return metrics1.profit_factor > metrics2.profit_factor
        elif self.config.secondary_metric == 'max_drawdown':
            return metrics1.max_drawdown < metrics2.max_drawdown
        
        return False
    
    async def _calculate_overall_performance(self, period_results: List[Dict]) -> Tuple[BacktestMetrics, pd.Series]:
        """Calculate overall performance across all periods"""
        # Collect all trades
        all_trades = []
        for result in period_results:
            if 'backtest_result' in result and 'trades' in result['backtest_result']:
                all_trades.extend(result['backtest_result']['trades'])
        
        # Create equity curve
        equity_curve = self._create_equity_curve(all_trades)
        
        # Calculate metrics
        metrics = await self._calculate_metrics(all_trades, equity_curve)
        
        return metrics, equity_curve
    
    def _create_equity_curve(self, trades: List[TradeResult]) -> pd.Series:
        """Create equity curve from trades"""
        equity = [100000]
        timestamps = []
        
        # Sort trades by time
        sorted_trades = sorted(trades, key=lambda x: x.timestamp)
        
        for trade in sorted_trades:
            equity.append(equity[-1] + trade.profit)
            timestamps.append(trade.timestamp)
        
        return pd.Series(equity, index=timestamps)
    
    async def _calculate_metrics(self, trades: List[TradeResult], 
                               equity_curve: pd.Series) -> BacktestMetrics:
        """Calculate overall performance metrics"""
        if not trades:
            return BacktestMetrics(
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
                mc_probability_of_ruin=0,
                stress_test_results={}
            )
        
        # Calculate metrics using backtest engine
        self.backtest_engine.trades = trades
        self.backtest_engine.equity_curve = equity_curve
        self.backtest_engine.daily_returns = equity_curve.pct_change().dropna()
        self.backtest_engine.drawdown_curve = self._calculate_drawdown(equity_curve)
        
        return self.backtest_engine._calculate_metrics()
    
    async def _analyze_parameter_stability(self, period_results: List[Dict]) -> Dict[str, float]:
        """Analyze parameter stability over time"""
        parameter_stability = {}
        
        # Get all parameter names
        param_names = []
        for result in period_results:
            if 'best_params' in result:
                param_names.extend(list(result['best_params'].keys()))
        
        param_names = list(set(param_names))
        
        # Calculate stability for each parameter
        for param_name in param_names:
            # Collect values across periods
            values = []
            
            for result in period_results:
                if 'best_params' in result and param_name in result['best_params']:
                    values.append(result['best_params'][param_name])
            
            if len(values) >= 2:
                # Calculate coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else 0
                
                # Calculate correlation between consecutive values
                consecutive_diff = np.diff(values)
                correlation = np.corrcoef(values[:-1], values[1:])[0, 1]
                
                parameter_stability[param_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,
                    'correlation': correlation,
                    'values': values
                }
        
        return parameter_stability
    
    async def _calculate_parameter_performance(self, period_results: List[Dict]) -> Dict[str, BacktestMetrics]:
        """Calculate performance per parameter set"""
        param_performance = {}
        
        for result in period_results:
            params = tuple(sorted(result['best_params'].items()))
            params_str = str(params)
            
            if params_str not in param_performance:
                param_performance[params_str] = {
                    'parameters': dict(params),
                    'periods': 0,
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'profit_factor': 0,
                    'win_rate': 0
                }
            
            param_performance[params_str]['periods'] += 1
            param_performance[params_str]['total_return'] += result['test_metrics'].total_return
            param_performance[params_str]['sharpe_ratio'] += result['test_metrics'].sharpe_ratio
            param_performance[params_str]['profit_factor'] += result['test_metrics'].profit_factor
            param_performance[params_str]['win_rate'] += result['test_metrics'].win_rate
        
        # Calculate averages
        for params_str, performance in param_performance.items():
            performance['total_return'] /= performance['periods']
            performance['sharpe_ratio'] /= performance['periods']
            performance['profit_factor'] /= performance['periods']
            performance['win_rate'] /= performance['periods']
        
        return param_performance
    
    def _extract_optimization_history(self, period_results: List[Dict]) -> List[Dict]:
        """Extract optimization history from period results"""
        history = []
        
        for result in period_results:
            history.append({
                'training_start': result['training_start'],
                'training_end': result['training_end'],
                'test_start': result['test_start'],
                'test_end': result['test_end'],
                'parameters': result['best_params'],
                'training_return': result['training_metrics'].total_return,
                'test_return': result['test_metrics'].total_return,
                'sharpe_ratio': result['test_metrics'].sharpe_ratio,
                'profit_factor': result['test_metrics'].profit_factor,
                'win_rate': result['test_metrics'].win_rate,
                'max_drawdown': result['test_metrics'].max_drawdown
            })
        
        return history
    
    async def _generate_report(self):
        """Generate comprehensive report"""
        from jinja2 import Template
        
        # Load template
        template_path = Path(__file__).parent / 'templates' / 'walk_forward_report.html'
        if template_path.exists():
            with open(template_path, 'r') as f:
                template = Template(f.read())
        
        # Render report
        report_content = template.render(
            analysis_start=self.results.period_results[0]['training_start'],
            analysis_end=self.results.period_results[-1]['test_end'],
            periods_count=len(self.results.period_results),
            metrics=self.results.overall_metrics,
            parameter_stability=self.results.parameter_stability,
            parameter_performance=self.results.parameter_performance,
            equity_curve=self._plot_equity_curve(),
            drawdown_curve=self._plot_drawdown_curve()
        )
        
        # Save report
        report_path = Path(__file__).parent / 'results'
        report_path.mkdir(exist_ok=True)
        
        filename = f"walk_forward_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path / filename, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Report generated: {filename}")
    
    def _plot_equity_curve(self) -> str:
        """Plot equity curve and return HTML"""
        import plotly.graph_objects as go
        from io import StringIO
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.results.equity_curve.index,
            y=self.results.equity_curve.values,
            mode='lines',
            name='Equity Curve'
        ))
        
        # Add period boundaries
        for period in self.results.period_results:
            fig.add_vline(x=period['test_start'], line_dash='dash', line_color='gray')
        
        fig.update_layout(
            title='Equity Curve During Walk-Forward Analysis',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            height=400
        )
        
        return fig.to_html(full_html=False)
    
    def _plot_drawdown_curve(self) -> str:
        """Plot drawdown curve and return HTML"""
        import plotly.graph_objects as go
        from io import StringIO
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.results.drawdown_curve.index,
            y=self.results.drawdown_curve.values * 100,
            mode='lines',
            name='Drawdown (%)'
        ))
        
        fig.update_layout(
            title='Drawdown During Walk-Forward Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=300
        )
        
        return fig.to_html(full_html=False)
    
    async def _plot_results(self):
        """Plot comprehensive results"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                'Equity Curve',
                'Drawdown',
                'Parameter Stability'
            ),
            vertical_spacing=0.1
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.results.equity_curve.index,
                y=self.results.equity_curve.values,
                name='Equity'
            ),
            row=1,
            col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=self.results.drawdown_curve.index,
                y=self.results.drawdown_curve.values * 100,
                name='Drawdown (%)'
            ),
            row=2,
            col=1
        )
        
        # Parameter stability (heatmap of best parameter values)
        if self.results.parameter_stability:
            param_names = list(self.results.parameter_stability.keys())
            param_values = []
            
            for param in param_names:
                param_values.append(self.results.parameter_stability[param]['values'])
            
            fig.add_trace(
                go.Heatmap(
                    z=param_values,
                    x=self.results.period_results,
                    y=param_names,
                    colorscale='RdYlGn'
                ),
                row=3,
                col=1
            )
        
        fig.update_layout(height=800, title_text='Walk-Forward Analysis Results')
        
        # Save plot
        plot_path = Path(__file__).parent / 'results'
        plot_path.mkdir(exist_ok=True)
        
        plot_filename = f"walk_forward_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path / plot_filename)
        
        self.logger.info(f"Plot saved: {plot_filename}")
    
    def save_results(self, filename: Optional[str] = None):
        """Save walk-forward results to file"""
        if filename is None:
            filename = f"walk_forward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_dict = {
            'overall_metrics': self._metrics_to_dict(self.results.overall_metrics),
            'period_results': self._period_results_to_dict(self.results.period_results),
            'parameter_stability': self.results.parameter_stability,
            'parameter_performance': self.results.parameter_performance,
            'total_trades': self.results.total_trades,
            'avg_trades_per_period': self.results.avg_trades_per_period,
            'period_win_rate': self.results.period_win_rate
        }
        
        results_path = Path(__file__).parent / 'results'
        results_path.mkdir(exist_ok=True)
        
        with open(results_path / filename, 'w') as f:
            json.dump(results_dict, f, default=str)
        
        self.logger.info(f"Results saved: {filename}")
    
    def _metrics_to_dict(self, metrics: BacktestMetrics) -> Dict:
        """Convert metrics object to dict"""
        return {
            'total_return': metrics.total_return,
            'annualized_return': metrics.annualized_return,
            'volatility': metrics.volatility,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'omega_ratio': metrics.omega_ratio,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': metrics.win_rate,
            'avg_win': metrics.avg_win,
            'avg_loss': metrics.avg_loss,
            'max_win': metrics.max_win,
            'max_loss': metrics.max_loss,
            'profit_factor': metrics.profit_factor,
            'expectancy': metrics.expectancy,
            'avg_r_multiple': metrics.avg_r_multiple,
            'r_multiple_std': metrics.r_multiple_std,
            'max_drawdown': metrics.max_drawdown,
            'max_drawdown_duration': metrics.max_drawdown_duration,
            'avg_drawdown': metrics.avg_drawdown,
            'recovery_factor': metrics.recovery_factor,
            'ulcer_index': metrics.ulcer_index,
            'var_95': metrics.var_95,
            'var_99': metrics.var_99,
            'cvar_95': metrics.cvar_95,
            'cvar_99': metrics.cvar_99,
            'tail_ratio': metrics.tail_ratio,
            'gain_to_pain_ratio': metrics.gain_to_pain_ratio
        }
    
    def _period_results_to_dict(self, period_results: List[Dict]) -> List[Dict]:
        """Convert period results to dict"""
        result_dicts = []
        
        for result in period_results:
            result_dicts.append({
                'training_start': str(result['training_start']),
                'training_end': str(result['training_end']),
                'test_start': str(result['test_start']),
                'test_end': str(result['test_end']),
                'best_params': result['best_params'],
                'training_metrics': self._metrics_to_dict(result['training_metrics']),
                'test_metrics': self._metrics_to_dict(result['test_metrics'])
            })
        
        return result_dicts
