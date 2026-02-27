"""
Walk-forward Analysis Engine for strategy optimization
Enhanced version with better performance, memory optimization, and advanced features
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
import logging
from pathlib import Path
import json
import warnings
from scipy import stats, optimize
from sklearn.model_selection import TimeSeriesSplit
import itertools
import concurrent.futures
from functools import partial, lru_cache
import hashlib
import pickle
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from enum import Enum
import psutil
import gc

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Parameter type enumeration"""
    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'
    CATEGORICAL = 'categorical'
    BOOLEAN = 'boolean'


@dataclass
class ParameterDefinition:
    """Definition of a strategy parameter"""
    name: str
    type: ParameterType
    values: Union[List[Any], Tuple[float, float]]  # List for discrete/categorical, (min,max) for continuous
    description: str = ''
    step: Optional[float] = None  # For continuous parameters
    
    def generate_values(self, n: int = 10) -> List[Any]:
        """Generate parameter values for optimization"""
        if self.type == ParameterType.CONTINUOUS:
            min_val, max_val = self.values
            if self.step:
                return np.arange(min_val, max_val + self.step, self.step).tolist()
            else:
                return np.linspace(min_val, max_val, n).tolist()
        else:
            return self.values


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    # Test period settings
    test_period_days: int = 60
    training_period_days: int = 180
    lookback_period_days: int = 30
    reoptimization_frequency: int = 30  # days
    min_periods: int = 3  # Minimum number of periods
    
    # Optimization settings
    parameters: Dict[str, ParameterDefinition] = field(default_factory=dict)
    optimization_method: str = 'grid'  # 'grid', 'random', 'bayesian'
    random_iterations: int = 100  # For random search
    max_workers: int = 4
    timeout: int = 7200
    cache_results: bool = True
    cache_dir: str = 'cache/walk_forward'
    
    # Performance metrics for optimization
    optimization_metric: str = 'sharpe_ratio'
    secondary_metric: str = 'profit_factor'
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'sharpe_ratio': 0.3,
        'profit_factor': 0.2,
        'win_rate': 0.15,
        'max_drawdown': -0.15,  # Negative weight for minimization
        'calmar_ratio': 0.2
    })
    
    # Risk constraints
    max_drawdown_constraint: float = 0.25
    min_profit_factor: float = 1.1
    min_win_rate: float = 0.45
    min_trades_per_period: int = 10
    
    # Walk-forward validation
    validation_method: str = 'anchored'  # 'anchored', 'rolling', 'expanding'
    oos_percentage: float = 0.3  # Out-of-sample percentage for validation
    
    # Output settings
    save_results: bool = True
    save_optimal_params: bool = True
    generate_report: bool = True
    report_format: str = 'html'
    plot_results: bool = True
    verbose: bool = True


@dataclass
class PeriodResult:
    """Result of a single walk-forward period"""
    training_start: datetime
    training_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict[str, Any]
    training_metrics: 'BacktestMetrics'
    test_metrics: 'BacktestMetrics'
    all_params_tested: List[Dict[str, Any]]
    optimization_time: float
    convergence: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'training_start': str(self.training_start),
            'training_end': str(self.training_end),
            'test_start': str(self.test_start),
            'test_end': str(self.test_end),
            'best_params': self.best_params,
            'training_metrics': self._metrics_to_dict(self.training_metrics),
            'test_metrics': self._metrics_to_dict(self.test_metrics),
            'optimization_time': self.optimization_time,
            'convergence': self.convergence
        }
    
    def _metrics_to_dict(self, metrics):
        """Convert metrics to dict"""
        if hasattr(metrics, 'to_dict'):
            return metrics.to_dict()
        return {}


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis"""
    # Overall performance
    overall_metrics: 'BacktestMetrics'
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    
    # Per-period results
    period_results: List[PeriodResult]
    
    # Parameter stability
    parameter_stability: Dict[str, Dict[str, float]]
    parameter_importance: Dict[str, float]
    
    # Walk-forward metrics
    wfa_score: float  # Walk-forward analysis score
    robustness_score: float
    consistency_score: float
    
    # Trading statistics
    total_trades: int
    avg_trades_per_period: float
    period_win_rate: float
    
    # Optimization history
    optimization_history: List[Dict]
    
    # Configuration
    config: WalkForwardConfig
    
    def save(self, filepath: str):
        """Save results to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'WalkForwardResult':
        """Load results from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class WalkForwardEngine:
    """
    Walk-forward analysis engine for robust strategy optimization
    
    Features:
    - Rolling window optimization with multiple validation methods
    - Parameter stability and importance analysis
    - Out-of-sample performance testing
    - Parallel execution with caching
    - Comprehensive performance metrics
    - Risk-aware parameter selection
    - Bayesian optimization support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize walk-forward engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = self._create_config(config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize engines (lazy loading)
        self._backtest_engine = None
        self._risk_engine = None
        self._indicators = None
        
        # Results storage
        self.results: Optional[WalkForwardResult] = None
        self._cache = {}
        
        # Setup cache directory
        if self.config.cache_results:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"WalkForwardEngine initialized with {len(self.config.parameters)} parameters")
    
    def _create_config(self, config: Dict) -> WalkForwardConfig:
        """Create configuration from dict"""
        # Convert parameter definitions
        param_defs = {}
        if 'parameters' in config.get('walk_forward', {}):
            for name, param_config in config['walk_forward']['parameters'].items():
                param_type = ParameterType(param_config.get('type', 'continuous'))
                param_defs[name] = ParameterDefinition(
                    name=name,
                    type=param_type,
                    values=param_config['values'],
                    description=param_config.get('description', ''),
                    step=param_config.get('step')
                )
        
        base_config = config.get('walk_forward', {})
        base_config['parameters'] = param_defs
        
        return WalkForwardConfig(**base_config)
    
    @property
    def backtest_engine(self):
        """Lazy load backtest engine"""
        if self._backtest_engine is None:
            from backtest.backtest_engine import BacktestEngine
            self._backtest_engine = BacktestEngine({})
        return self._backtest_engine
    
    @property
    def risk_engine(self):
        """Lazy load risk engine"""
        if self._risk_engine is None:
            from core.risk_engine import RiskEngine
            self._risk_engine = RiskEngine({})
        return self._risk_engine
    
    @property
    def indicators(self):
        """Lazy load indicators"""
        if self._indicators is None:
            from utils.indicators import TechnicalIndicators
            self._indicators = TechnicalIndicators()
        return self._indicators
    
    def _get_cache_key(self, strategy, data_hash: str, period: Dict) -> str:
        """Generate cache key for optimization results"""
        strategy_hash = hashlib.md5(str(strategy.__class__).encode()).hexdigest()[:8]
        param_hash = hashlib.md5(str(self.config.parameters).encode()).hexdigest()[:8]
        
        key = f"{strategy_hash}_{param_hash}_{data_hash}_{period['training_start']}_{period['training_end']}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if results are cached"""
        if not self.config.cache_results:
            return None
        
        cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_cache(self, cache_key: str, results: Dict):
        """Save results to cache"""
        if not self.config.cache_results:
            return
        
        try:
            cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    async def run(self, strategy: 'StrategyEngine', 
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
            # Generate data hash for caching
            data_hash = hashlib.md5(str(len(data)).encode()).hexdigest()
            
            # Split data into walk-forward periods
            periods = await self._split_data_into_periods(data)
            
            if len(periods) < self.config.min_periods:
                raise ValueError(f"Only {len(periods)} periods generated, need at least {self.config.min_periods}")
            
            # Run optimization for each period
            period_results = await self._run_period_optimization(strategy, periods, data_hash)
            
            # Calculate overall performance
            overall_metrics, equity_curve = await self._calculate_overall_performance(period_results)
            
            # Calculate drawdown curve
            drawdown_curve = self._calculate_drawdown(equity_curve)
            
            # Analyze parameter stability and importance
            parameter_stability = await self._analyze_parameter_stability(period_results)
            parameter_importance = await self._calculate_parameter_importance(period_results)
            
            # Calculate walk-forward metrics
            wfa_score = self._calculate_wfa_score(period_results)
            robustness_score = self._calculate_robustness_score(period_results)
            consistency_score = self._calculate_consistency_score(period_results)
            
            # Generate results object
            self.results = WalkForwardResult(
                overall_metrics=overall_metrics,
                equity_curve=equity_curve,
                drawdown_curve=drawdown_curve,
                period_results=period_results,
                parameter_stability=parameter_stability,
                parameter_importance=parameter_importance,
                wfa_score=wfa_score,
                robustness_score=robustness_score,
                consistency_score=consistency_score,
                total_trades=sum(r.test_metrics.total_trades for r in period_results),
                avg_trades_per_period=np.mean([r.test_metrics.total_trades for r in period_results]),
                period_win_rate=np.mean([1 for r in period_results if r.test_metrics.win_rate > 0.5]),
                optimization_history=self._extract_optimization_history(period_results),
                config=self.config
            )
            
            # Generate report
            if self.config.generate_report:
                await self._generate_report()
                
            # Plot results
            if self.config.plot_results:
                await self._plot_results()
                
            # Save results
            if self.config.save_results:
                self.save_results()
            
            self.logger.info(f"Walk-forward analysis completed. "
                           f"WFA Score: {wfa_score:.2f}, "
                           f"Return: {overall_metrics.total_return:.2%}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis failed: {e}", exc_info=True)
            raise
    
    async def _split_data_into_periods(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Split data into walk-forward periods with validation"""
        periods = []
        
        # Find global date range
        all_dates = pd.DatetimeIndex([])
        for symbol_data in data.values():
            all_dates = all_dates.union(symbol_data.index)
        
        if len(all_dates) == 0:
            raise ValueError("No data available")
        
        min_date = all_dates.min()
        max_date = all_dates.max()
        
        # Calculate period boundaries based on validation method
        if self.config.validation_method == 'anchored':
            # Training period starts from beginning and expands
            current_test_start = min_date + timedelta(days=self.config.training_period_days)
            
            while current_test_start + timedelta(days=self.config.test_period_days) <= max_date:
                training_start = min_date
                training_end = current_test_start
                test_start = current_test_start
                test_end = current_test_start + timedelta(days=self.config.test_period_days)
                
                periods.append(self._create_period_data(data, training_start, training_end, test_start, test_end))
                current_test_start += timedelta(days=self.config.reoptimization_frequency)
                
        elif self.config.validation_method == 'rolling':
            # Rolling window (fixed training period)
            current_period_start = min_date
            
            while current_period_start + timedelta(days=self.config.training_period_days + self.config.test_period_days) <= max_date:
                training_start = current_period_start
                training_end = current_period_start + timedelta(days=self.config.training_period_days)
                test_start = training_end
                test_end = training_end + timedelta(days=self.config.test_period_days)
                
                periods.append(self._create_period_data(data, training_start, training_end, test_start, test_end))
                current_period_start += timedelta(days=self.config.reoptimization_frequency)
                
        elif self.config.validation_method == 'expanding':
            # Expanding window
            training_start = min_date
            current_test_start = min_date + timedelta(days=self.config.training_period_days)
            
            while current_test_start + timedelta(days=self.config.test_period_days) <= max_date:
                training_end = current_test_start
                test_start = current_test_start
                test_end = current_test_start + timedelta(days=self.config.test_period_days)
                
                periods.append(self._create_period_data(data, training_start, training_end, test_start, test_end))
                current_test_start += timedelta(days=self.config.reoptimization_frequency)
        
        self.logger.info(f"Split data into {len(periods)} walk-forward periods")
        
        return periods
    
    def _create_period_data(self, data: Dict[str, pd.DataFrame], 
                           training_start: datetime, training_end: datetime,
                           test_start: datetime, test_end: datetime) -> Dict:
        """Create period data dictionary"""
        period_data = {
            'training_data': {},
            'test_data': {},
            'training_start': training_start,
            'training_end': training_end,
            'test_start': test_start,
            'test_end': test_end
        }
        
        for symbol, df in data.items():
            # Training data
            training_mask = (df.index >= training_start) & (df.index < training_end)
            period_data['training_data'][symbol] = df[training_mask].copy()
            
            # Test data
            test_mask = (df.index >= test_start) & (df.index < test_end)
            period_data['test_data'][symbol] = df[test_mask].copy()
        
        return period_data
    
    async def _run_period_optimization(self, strategy: 'StrategyEngine',
                                     periods: List[Dict], 
                                     data_hash: str) -> List[PeriodResult]:
        """Run optimization for each period with parallel execution"""
        period_results = []
        
        # Check memory availability
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
        max_workers = min(self.config.max_workers, int(available_memory * 2))
        
        # Create progress bar
        pbar = tqdm(total=len(periods), desc="Optimizing periods", disable=not self.config.verbose)
        
        # Use ProcessPoolExecutor for parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for i, period in enumerate(periods):
                # Check cache
                cache_key = self._get_cache_key(strategy, data_hash, period)
                cached_result = self._check_cache(cache_key)
                
                if cached_result:
                    period_results.append(cached_result)
                    pbar.update(1)
                else:
                    future = executor.submit(
                        self._optimize_period,
                        strategy,
                        period,
                        i
                    )
                    futures[future] = (period, cache_key)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures, timeout=self.config.timeout):
                period, cache_key = futures[future]
                try:
                    result = future.result()
                    period_results.append(result)
                    
                    # Save to cache
                    if self.config.cache_results:
                        self._save_cache(cache_key, result)
                    
                except Exception as e:
                    self.logger.error(f"Period optimization failed: {e}")
                finally:
                    pbar.update(1)
                    # Force garbage collection
                    gc.collect()
        
        pbar.close()
        
        # Sort results by date
        period_results.sort(key=lambda x: x.training_start)
        
        return period_results
    
    def _optimize_period(self, strategy: 'StrategyEngine', period: Dict, period_idx: int) -> PeriodResult:
        """Optimize strategy parameters for a single period"""
        start_time = datetime.now()
        
        try:
            # Generate parameter combinations based on optimization method
            if self.config.optimization_method == 'grid':
                param_combinations = self._generate_grid_combinations()
            elif self.config.optimization_method == 'random':
                param_combinations = self._generate_random_combinations()
            elif self.config.optimization_method == 'bayesian':
                param_combinations = self._generate_bayesian_combinations(strategy, period)
            else:
                param_combinations = self._generate_grid_combinations()
            
            # Evaluate parameter combinations
            best_score = -float('inf')
            best_params = None
            best_training_metrics = None
            all_tested = []
            
            for params in param_combinations:
                # Update strategy parameters
                strategy.update_parameters(params)
                
                # Run backtest on training data
                backtest_result = self._run_backtest_with_timeout(
                    strategy,
                    period['training_data']
                )
                
                if backtest_result:
                    metrics = backtest_result['metrics']
                    
                    # Apply constraints
                    if self._meets_constraints(metrics):
                        score = self._calculate_composite_score(metrics)
                        
                        all_tested.append({
                            'params': params,
                            'score': score,
                            'metrics': metrics
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_training_metrics = metrics
            
            if best_params is None:
                # If no params meet constraints, use best available
                if all_tested:
                    best_item = max(all_tested, key=lambda x: x['score'])
                    best_params = best_item['params']
                    best_training_metrics = best_item['metrics']
                else:
                    # Use default parameters
                    best_params = self._get_default_parameters()
                    strategy.update_parameters(best_params)
                    backtest_result = self._run_backtest_with_timeout(
                        strategy,
                        period['training_data']
                    )
                    best_training_metrics = backtest_result['metrics'] if backtest_result else None
            
            # Test best parameters on out-of-sample data
            if best_params:
                strategy.update_parameters(best_params)
                test_result = self._run_backtest_with_timeout(
                    strategy,
                    period['test_data']
                )
                test_metrics = test_result['metrics'] if test_result else None
            else:
                test_metrics = None
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate convergence (if applicable)
            convergence = self._calculate_convergence(all_tested) if all_tested else None
            
            return PeriodResult(
                training_start=period['training_start'],
                training_end=period['training_end'],
                test_start=period['test_start'],
                test_end=period['test_end'],
                best_params=best_params or {},
                training_metrics=best_training_metrics,
                test_metrics=test_metrics,
                all_params_tested=all_tested,
                optimization_time=optimization_time,
                convergence=convergence
            )
            
        except Exception as e:
            self.logger.error(f"Error in period optimization: {e}")
            raise
    
    def _run_backtest_with_timeout(self, strategy, data, timeout: int = 300):
        """Run backtest with timeout"""
        # Simplified version - in production use asyncio with timeout
        try:
            return self.backtest_engine.run(strategy, data)
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return None
    
    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search"""
        param_values = []
        param_names = []
        
        for name, param_def in self.config.parameters.items():
            param_names.append(name)
            param_values.append(param_def.generate_values())
        
        combinations = []
        for values_tuple in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, values_tuple)))
        
        return combinations
    
    def _generate_random_combinations(self) -> List[Dict[str, Any]]:
        """Generate random parameter combinations"""
        combinations = []
        
        for _ in range(self.config.random_iterations):
            params = {}
            for name, param_def in self.config.parameters.items():
                if param_def.type == ParameterType.CONTINUOUS:
                    min_val, max_val = param_def.values
                    if param_def.step:
                        values = np.arange(min_val, max_val + param_def.step, param_def.step)
                        params[name] = np.random.choice(values)
                    else:
                        params[name] = np.random.uniform(min_val, max_val)
                elif param_def.type == ParameterType.DISCRETE:
                    params[name] = np.random.choice(param_def.values)
                elif param_def.type == ParameterType.CATEGORICAL:
                    params[name] = np.random.choice(param_def.values)
                elif param_def.type == ParameterType.BOOLEAN:
                    params[name] = np.random.choice([True, False])
            
            combinations.append(params)
        
        return combinations
    
    def _generate_bayesian_combinations(self, strategy, period) -> List[Dict[str, Any]]:
        """Generate parameter combinations using Bayesian optimization"""
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        
        # Define search space
        dimensions = []
        for name, param_def in self.config.parameters.items():
            if param_def.type == ParameterType.CONTINUOUS:
                dimensions.append(Real(*param_def.values, name=name))
            elif param_def.type == ParameterType.DISCRETE:
                dimensions.append(Integer(param_def.values[0], param_def.values[-1], name=name))
            elif param_def.type == ParameterType.CATEGORICAL:
                dimensions.append(Categorical(param_def.values, name=name))
        
        # Define objective function
        def objective(params):
            param_dict = dict(zip([d.name for d in dimensions], params))
            strategy.update_parameters(param_dict)
            
            result = self._run_backtest_with_timeout(strategy, period['training_data'])
            if result:
                metrics = result['metrics']
                if self._meets_constraints(metrics):
                    return -self._calculate_composite_score(metrics)  # Minimize negative score
            return 1e6  # Penalty for invalid
        
        # Run Bayesian optimization
        try:
            result = gp_minimize(
                objective,
                dimensions,
                n_calls=self.config.random_iterations,
                n_initial_points=10,
                acq_func='EI',
                random_state=42
            )
            
            # Generate combinations from optimization path
            combinations = []
            for x in result.x_iters:
                param_dict = dict(zip([d.name for d in dimensions], x))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {e}")
            return self._generate_random_combinations()
    
    def _meets_constraints(self, metrics) -> bool:
        """Check if metrics meet constraints"""
        if not metrics:
            return False
        
        return (metrics.max_drawdown < self.config.max_drawdown_constraint and
                metrics.profit_factor > self.config.min_profit_factor and
                metrics.win_rate > self.config.min_win_rate and
                metrics.total_trades >= self.config.min_trades_per_period)
    
    def _calculate_composite_score(self, metrics) -> float:
        """Calculate composite score from multiple metrics"""
        score = 0
        
        for metric, weight in self.config.metric_weights.items():
            if hasattr(metrics, metric):
                value = getattr(metrics, metric)
                if np.isfinite(value):
                    score += weight * value
        
        return score
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters"""
        defaults = {}
        for name, param_def in self.config.parameters.items():
            if param_def.type == ParameterType.CONTINUOUS:
                min_val, max_val = param_def.values
                defaults[name] = (min_val + max_val) / 2
            elif param_def.values:
                defaults[name] = param_def.values[0]
            else:
                defaults[name] = None
        return defaults
    
    def _calculate_convergence(self, tested_params: List[Dict]) -> float:
        """Calculate optimization convergence score"""
        if len(tested_params) < 10:
            return 0.0
        
        scores = [p['score'] for p in tested_params]
        
        # Calculate improvement over last 20% of iterations
        split_idx = int(len(scores) * 0.8)
        early_scores = scores[:split_idx]
        late_scores = scores[split_idx:]
        
        if early_scores and late_scores:
            early_best = max(early_scores)
            late_best = max(late_scores)
            
            if early_best > 0:
                improvement = (late_best - early_best) / abs(early_best)
                return max(0, improvement)
        
        return 0.0
    
    async def _calculate_overall_performance(self, period_results: List[PeriodResult]) -> Tuple:
        """Calculate overall performance across all periods"""
        # Collect all trades
        all_trades = []
        for result in period_results:
            if hasattr(result.test_metrics, 'trades'):
                all_trades.extend(result.test_metrics.trades)
        
        # Create equity curve
        equity_curve = self._create_equity_curve(all_trades)
        
        # Calculate metrics
        if all_trades:
            self.backtest_engine.trades = all_trades
            self.backtest_engine.equity_curve = equity_curve
            self.backtest_engine.daily_returns = equity_curve.pct_change().dropna()
            metrics = self.backtest_engine._calculate_metrics()
        else:
            from backtest.backtest_engine import BacktestMetrics
            metrics = BacktestMetrics.create_empty()
        
        return metrics, equity_curve
    
    def _create_equity_curve(self, trades: List) -> pd.Series:
        """Create equity curve from trades"""
        if not trades:
            return pd.Series([100000])
        
        equity = [100000]
        timestamps = []
        
        # Sort trades by time
        sorted_trades = sorted(trades, key=lambda x: x.timestamp if hasattr(x, 'timestamp') else datetime.now())
        
        for trade in sorted_trades:
            profit = trade.profit if hasattr(trade, 'profit') else 0
            equity.append(equity[-1] + profit)
            timestamps.append(trade.timestamp if hasattr(trade, 'timestamp') else datetime.now())
        
        return pd.Series(equity, index=[datetime.now()] + timestamps)
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown curve"""
        if equity_curve.empty:
            return pd.Series()
        
        rolling_max = equity_curve.expanding().max()
        drawdown = (rolling_max - equity_curve) / rolling_max
        return drawdown
    
    async def _analyze_parameter_stability(self, period_results: List[PeriodResult]) -> Dict[str, Dict[str, float]]:
        """Analyze parameter stability over time"""
        parameter_stability = {}
        
        if not period_results:
            return parameter_stability
        
        # Get all parameter names
        param_names = set()
        for result in period_results:
            param_names.update(result.best_params.keys())
        
        for param_name in param_names:
            # Collect values across periods
            values = []
            periods_with_param = []
            
            for i, result in enumerate(period_results):
                if param_name in result.best_params:
                    values.append(result.best_params[param_name])
                    periods_with_param.append(i)
            
            if len(values) >= 3:
                # Convert to numpy array
                values_array = np.array(values, dtype=float)
                
                # Calculate statistics
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                cv = std_val / abs(mean_val) if abs(mean_val) > 1e-6 else 0
                
                # Calculate trend
                if len(values) >= 3:
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values_array)
                else:
                    slope, r_value, p_value = 0, 0, 1
                
                # Calculate autocorrelation
                if len(values) >= 4:
                    autocorr = pd.Series(values).autocorr()
                else:
                    autocorr = 0
                
                parameter_stability[param_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'cv': float(cv),
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'autocorrelation': float(autocorr),
                    'values': values,
                    'periods': periods_with_param
                }
        
        return parameter_stability
    
    async def _calculate_parameter_importance(self, period_results: List[PeriodResult]) -> Dict[str, float]:
        """Calculate parameter importance based on performance impact"""
        importance = {}
        
        if not period_results:
            return importance
        
        # Collect all tested parameters and their scores
        all_tested = []
        for result in period_results:
            all_tested.extend(result.all_params_tested)
        
        if len(all_tested) < 10:
            return importance
        
        # Create DataFrame for analysis
        df_data = []
        for test in all_tested:
            row = test['params'].copy()
            row['score'] = test['score']
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Calculate correlation with score for each parameter
        for col in df.columns:
            if col != 'score' and df[col].dtype in [np.float64, np.int64]:
                corr = df[col].corr(df['score'])
                if pd.notna(corr):
                    importance[col] = abs(corr)
        
        # Normalize importance scores
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def _calculate_wfa_score(self, period_results: List[PeriodResult]) -> float:
        """Calculate Walk-Forward Analysis score"""
        if len(period_results) < 2:
            return 0.0
        
        scores = []
        
        for result in period_results:
            if result.training_metrics and result.test_metrics:
                # Compare training and test performance
                train_return = result.training_metrics.total_return
                test_return = result.test_metrics.total_return
                
                if train_return > 0:
                    ratio = test_return / train_return
                    scores.append(min(ratio, 2.0))  # Cap at 2.0
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_robustness_score(self, period_results: List[PeriodResult]) -> float:
        """Calculate robustness score based on consistency across periods"""
        if len(period_results) < 3:
            return 0.0
        
        test_returns = [r.test_metrics.total_return for r in period_results if r.test_metrics]
        
        if not test_returns:
            return 0.0
        
        # Calculate metrics
        mean_return = np.mean(test_returns)
        std_return = np.std(test_returns)
        positive_periods = sum(1 for r in test_returns if r > 0) / len(test_returns)
        
        # Robustness score components
        if std_return > 0:
            sharpe_across_periods = mean_return / std_return
        else:
            sharpe_across_periods = 0
        
        # Combine components
        score = (0.5 * min(sharpe_across_periods, 3) / 3 +  # Normalize to ~0-0.5
                 0.5 * positive_periods)  # 0-0.5
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_consistency_score(self, period_results: List[PeriodResult]) -> float:
        """Calculate consistency score based on parameter stability"""
        if not period_results:
            return 0.0
        
        # Check if parameters change too much
        param_changes = 0
        total_params = 0
        
        for i in range(1, len(period_results)):
            prev_params = period_results[i-1].best_params
            curr_params = period_results[i].best_params
            
            for key in set(prev_params.keys()) & set(curr_params.keys()):
                total_params += 1
                if prev_params[key] != curr_params[key]:
                    param_changes += 1
        
        if total_params > 0:
            stability = 1 - (param_changes / total_params)
        else:
            stability = 1.0
        
        # Check return consistency
        returns = [r.test_metrics.total_return for r in period_results if r.test_metrics]
        if len(returns) >= 3:
            # Check for outliers
            q1 = np.percentile(returns, 25)
            q3 = np.percentile(returns, 75)
            iqr = q3 - q1
            
            outliers = sum(1 for r in returns if r < q1 - 1.5 * iqr or r > q3 + 1.5 * iqr)
            outlier_ratio = 1 - (outliers / len(returns))
        else:
            outlier_ratio = 1.0
        
        # Combine scores
        score = 0.6 * stability + 0.4 * outlier_ratio
        
        return score
    
    def _extract_optimization_history(self, period_results: List[PeriodResult]) -> List[Dict]:
        """Extract optimization history from period results"""
        history = []
        
        for result in period_results:
            history.append({
                'training_start': result.training_start,
                'training_end': result.training_end,
                'test_start': result.test_start,
                'test_end': result.test_end,
                'parameters': result.best_params,
                'training_return': result.training_metrics.total_return if result.training_metrics else 0,
                'test_return': result.test_metrics.total_return if result.test_metrics else 0,
                'sharpe_ratio': result.test_metrics.sharpe_ratio if result.test_metrics else 0,
                'profit_factor': result.test_metrics.profit_factor if result.test_metrics else 0,
                'win_rate': result.test_metrics.win_rate if result.test_metrics else 0,
                'max_drawdown': result.test_metrics.max_drawdown if result.test_metrics else 0,
                'optimization_time': result.optimization_time,
                'convergence': result.convergence
            })
        
        return history
    
    async def _generate_report(self):
        """Generate comprehensive HTML report"""
        from jinja2 import Template, Environment, FileSystemLoader
        
        try:
            # Setup template environment
            template_dir = Path(__file__).parent / 'templates'
            if not template_dir.exists():
                template_dir.mkdir(parents=True)
                self._create_default_template(template_dir)
            
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template('walk_forward_report.html')
            
            # Prepare data for template
            report_data = {
                'analysis_start': self.results.period_results[0].training_start,
                'analysis_end': self.results.period_results[-1].test_end,
                'periods_count': len(self.results.period_results),
                'metrics': self._metrics_to_dict(self.results.overall_metrics),
                'parameter_stability': self.results.parameter_stability,
                'parameter_importance': self.results.parameter_importance,
                'wfa_score': self.results.wfa_score,
                'robustness_score': self.results.robustness_score,
                'consistency_score': self.results.consistency_score,
                'total_trades': self.results.total_trades,
                'avg_trades_per_period': self.results.avg_trades_per_period,
                'period_win_rate': self.results.period_win_rate,
                'equity_curve_html': self._plot_equity_curve_html(),
                'drawdown_curve_html': self._plot_drawdown_curve_html(),
                'parameter_stability_html': self._plot_parameter_stability_html(),
                'period_performance_html': self._plot_period_performance_html(),
                'optimization_history': self.results.optimization_history,
                'config': asdict(self.config)
            }
            
            # Render report
            report_content = template.render(**report_data)
            
            # Save report
            report_path = Path(__file__).parent / 'results'
            report_path.mkdir(exist_ok=True)
            
            filename = f"walk_forward_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path / filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"Report generated: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
    
    def _create_default_template(self, template_dir: Path):
        """Create default HTML template"""
        template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Walk-Forward Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
        .metric-card { 
            background: #f5f5f5; 
            border-radius: 5px; 
            padding: 15px; 
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 14px; color: #666; margin-top: 5px; }
        .chart-container { margin: 30px 0; border: 1px solid #ddd; border-radius: 5px; padding: 20px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <h1>Walk-Forward Analysis Report</h1>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f%%"|format(metrics.total_return * 100) }}</div>
            <div class="metric-label">Total Return</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f"|format(metrics.sharpe_ratio) }}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f"|format(wfa_score * 100) }}%</div>
            <div class="metric-label">WFA Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ "%.2f"|format(robustness_score * 100) }}%</div>
            <div class="metric-label">Robustness</div>
        </div>
    </div>
    
    <div class="chart-container">
        <h2>Equity Curve</h2>
        {{ equity_curve_html|safe }}
    </div>
    
    <div class="chart-container">
        <h2>Drawdown</h2>
        {{ drawdown_curve_html|safe }}
    </div>
    
    <div class="chart-container">
        <h2>Parameter Stability</h2>
        {{ parameter_stability_html|safe }}
    </div>
    
    <div class="chart-container">
        <h2>Period Performance</h2>
        {{ period_performance_html|safe }}
    </div>
    
    <h2>Parameter Importance</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Importance</th>
        </tr>
        {% for param, importance in parameter_importance.items() %}
        <tr>
            <td>{{ param }}</td>
            <td>{{ "%.2f"|format(importance * 100) }}%</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Period Results</h2>
    <table>
        <tr>
            <th>Period</th>
            <th>Test Return</th>
            <th>Sharpe</th>
            <th>Win Rate</th>
            <th>Max DD</th>
        </tr>
        {% for period in optimization_history %}
        <tr>
            <td>{{ period.test_start.strftime('%Y-%m-%d') }}</td>
            <td class="{{ 'positive' if period.test_return > 0 else 'negative' }}">
                {{ "%.2f%%"|format(period.test_return * 100) }}
            </td>
            <td>{{ "%.2f"|format(period.sharpe_ratio) }}</td>
            <td>{{ "%.1f%%"|format(period.win_rate) }}</td>
            <td>{{ "%.2f%%"|format(period.max_drawdown * 100) }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
        """
        
        with open(template_dir / 'walk_forward_report.html', 'w') as f:
            f.write(template_content)
    
    def _plot_equity_curve_html(self) -> str:
        """Plot equity curve and return HTML"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.results.equity_curve.index,
            y=self.results.equity_curve.values,
            mode='lines',
            name='Equity Curve',
            line=dict(color='blue', width=2)
        ))
        
        # Add period boundaries
        for result in self.results.period_results:
            fig.add_vline(
                x=result.test_start,
                line_dash='dash',
                line_color='gray',
                opacity=0.5
            )
        
        fig.update_layout(
            title='Equity Curve During Walk-Forward Analysis',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _plot_drawdown_curve_html(self) -> str:
        """Plot drawdown curve and return HTML"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.results.drawdown_curve.index,
            y=self.results.drawdown_curve.values * 100,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='Drawdown During Walk-Forward Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=300,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _plot_parameter_stability_html(self) -> str:
        """Plot parameter stability heatmap"""
        if not self.results.parameter_stability:
            return "<p>No parameter stability data available</p>"
        
        # Prepare data for heatmap
        param_names = []
        param_values = []
        periods = []
        
        for param_name, stability in self.results.parameter_stability.items():
            if 'values' in stability:
                param_names.append(param_name)
                param_values.append(stability['values'])
                if len(periods) < len(stability['values']):
                    periods = list(range(len(stability['values'])))
        
        if not param_values:
            return "<p>No parameter values available</p>"
        
        fig = go.Figure(data=go.Heatmap(
            z=param_values,
            x=periods,
            y=param_names,
            colorscale='RdYlGn',
            colorbar=dict(title='Parameter Value')
        ))
        
        fig.update_layout(
            title='Parameter Values Across Periods',
            xaxis_title='Period',
            yaxis_title='Parameter',
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _plot_period_performance_html(self) -> str:
        """Plot period performance comparison"""
        fig = go.Figure()
        
        periods = list(range(len(self.results.period_results)))
        train_returns = []
        test_returns = []
        
        for result in self.results.period_results:
            train_returns.append(result.training_metrics.total_return if result.training_metrics else 0)
            test_returns.append(result.test_metrics.total_return if result.test_metrics else 0)
        
        fig.add_trace(go.Bar(
            x=periods,
            y=train_returns,
            name='Training',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=periods,
            y=test_returns,
            name='Test',
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title='Training vs Test Performance by Period',
            xaxis_title='Period',
            yaxis_title='Return',
            barmode='group',
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    async def _plot_results(self):
        """Plot comprehensive results"""
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                'Equity Curve',
                'Drawdown',
                'Period Performance',
                'Parameter Stability',
                'Parameter Importance',
                'WFA Metrics'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.results.equity_curve.index,
                y=self.results.equity_curve.values,
                mode='lines',
                name='Equity'
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=self.results.drawdown_curve.index,
                y=self.results.drawdown_curve.values * 100,
                mode='lines',
                name='Drawdown',
                fill='tozeroy'
            ),
            row=1, col=2
        )
        
        # Period performance
        periods = list(range(len(self.results.period_results)))
        test_returns = [r.test_metrics.total_return if r.test_metrics else 0 
                       for r in self.results.period_results]
        
        fig.add_trace(
            go.Bar(
                x=periods,
                y=test_returns,
                name='Period Return',
                marker_color=['green' if r > 0 else 'red' for r in test_returns]
            ),
            row=2, col=1
        )
        
        # Parameter importance
        if self.results.parameter_importance:
            params = list(self.results.parameter_importance.keys())
            importance = list(self.results.parameter_importance.values())
            
            fig.add_trace(
                go.Bar(
                    x=importance,
                    y=params,
                    orientation='h',
                    name='Importance'
                ),
                row=2, col=2
            )
        
        # WFA Metrics
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=self.results.wfa_score * 100,
                title={'text': 'WFA Score'},
                gauge={'axis': {'range': [0, 100]}}
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=self.results.robustness_score * 100,
                title={'text': 'Robustness'},
                gauge={'axis': {'range': [0, 100]}}
            ),
            row=3, col=2
        )
        
        fig.update_layout(height=900, title_text='Walk-Forward Analysis Results')
        
        # Save plot
        plot_path = Path(__file__).parent / 'results'
        plot_path.mkdir(exist_ok=True)
        
        plot_filename = f"walk_forward_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path / plot_filename)
        
        self.logger.info(f"Plot saved: {plot_filename}")
    
    def save_results(self, filename: Optional[str] = None):
        """Save walk-forward results to file"""
        if filename is None:
            filename = f"walk_forward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        results_path = Path(__file__).parent / 'results'
        results_path.mkdir(exist_ok=True)
        
        filepath = results_path / filename
        self.results.save(str(filepath))
        
        # Also save as JSON for easy viewing
        json_filename = filename.replace('.pkl', '.json')
        json_filepath = results_path / json_filename
        
        # Convert to JSON-serializable dict
        json_data = {
            'wfa_score': self.results.wfa_score,
            'robustness_score': self.results.robustness_score,
            'consistency_score': self.results.consistency_score,
            'total_trades': self.results.total_trades,
            'avg_trades_per_period': self.results.avg_trades_per_period,
            'period_win_rate': self.results.period_win_rate,
            'parameter_stability': self.results.parameter_stability,
            'parameter_importance': self.results.parameter_importance,
            'overall_metrics': self._metrics_to_dict(self.results.overall_metrics),
            'period_results': [r.to_dict() for r in self.results.period_results],
            'optimization_history': self.results.optimization_history
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, default=str, indent=2)
        
        self.logger.info(f"Results saved: {filename} and {json_filename}")
    
    def _metrics_to_dict(self, metrics) -> Dict:
        """Convert metrics object to dict"""
        if hasattr(metrics, 'to_dict'):
            return metrics.to_dict()
        
        # Fallback
        return {
            'total_return': getattr(metrics, 'total_return', 0),
            'sharpe_ratio': getattr(metrics, 'sharpe_ratio', 0),
            'profit_factor': getattr(metrics, 'profit_factor', 0),
            'win_rate': getattr(metrics, 'win_rate', 0),
            'max_drawdown': getattr(metrics, 'max_drawdown', 0),
            'total_trades': getattr(metrics, 'total_trades', 0)
        }
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the most successful parameters across all periods"""
        if not self.results or not self.results.period_results:
            return {}
        
        # Weight parameters by period performance
        param_scores = {}
        
        for result in self.results.period_results:
            if result.test_metrics and result.best_params:
                weight = max(0, result.test_metrics.sharpe_ratio)
                
                for param, value in result.best_params.items():
                    if param not in param_scores:
                        param_scores[param] = {}
                    
                    str_value = str(value)
                    if str_value not in param_scores[param]:
                        param_scores[param][str_value] = 0
                    
                    param_scores[param][str_value] += weight
        
        # Select most common weighted value for each parameter
        best_params = {}
        for param, value_scores in param_scores.items():
            if value_scores:
                best_value = max(value_scores.items(), key=lambda x: x[1])[0]
                # Convert back to original type
                for result in self.results.period_results:
                    if param in result.best_params:
                        original_type = type(result.best_params[param])
                        try:
                            best_params[param] = original_type(best_value)
                        except:
                            best_params[param] = result.best_params[param]
                        break
        
        return best_params
    
    def get_parameter_range(self, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Get confidence interval for each parameter"""
        ranges = {}
        
        if not self.results or not self.results.parameter_stability:
            return ranges
        
        for param, stability in self.results.parameter_stability.items():
            if 'values' in stability and len(stability['values']) >= 3:
                values = stability['values']
                
                # Calculate confidence interval
                mean = np.mean(values)
                std = np.std(values)
                
                if std > 0:
                    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
                    margin = z_score * std / np.sqrt(len(values))
                    
                    ranges[param] = (mean - margin, mean + margin)
                else:
                    ranges[param] = (mean, mean)
        
        return ranges