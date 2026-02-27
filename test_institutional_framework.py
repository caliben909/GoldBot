"""
Test script for Gold Institutional Quant Framework
Comprehensive testing with multiple scenarios, statistical validation, and performance analysis.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional, Any
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import argparse
import sys
import traceback
from dataclasses import dataclass, field
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from institutional.institutional_backtest import InstitutionalBacktestEngine
from utils.logger import setup_logger
from utils.helpers import format_currency, format_percentage, safe_divide

warnings.filterwarnings('ignore')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestScenario:
    """Test scenario configuration"""
    name: str
    description: str
    initial_equity: float = 10000
    commission: float = 0.0005
    spread: float = 0.3
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    volatility_multiplier: float = 1.0
    trend_strength: float = 1.0
    correlation_strength: float = 1.0
    regime: str = 'normal'  # normal, high_volatility, low_volatility, trending, ranging


@dataclass
class TestResult:
    """Test result container"""
    scenario: TestScenario
    backtest_results: Dict[str, Any]
    session_stats: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Any]
    benchmark_comparison: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scenario': self.scenario.__dict__,
            'backtest_results': self._serialize_results(self.backtest_results),
            'session_stats': self.session_stats,
            'statistical_tests': self.statistical_tests,
            'benchmark_comparison': self.benchmark_comparison,
            'confidence_intervals': self.confidence_intervals
        }
    
    def _serialize_results(self, obj: Any) -> Any:
        """Serialize results for JSON output"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_results(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_results(v) for v in obj]
        else:
            return obj


# =============================================================================
# Test Scenarios
# =============================================================================

TEST_SCENARIOS = [
    TestScenario(
        name="baseline",
        description="Standard market conditions with normal volatility",
        initial_equity=10000,
        commission=0.0005,
        spread=0.3,
        regime='normal'
    ),
    TestScenario(
        name="high_volatility",
        description="High volatility environment (e.g., during crises)",
        initial_equity=10000,
        commission=0.0005,
        spread=0.5,
        volatility_multiplier=2.0,
        regime='high_volatility'
    ),
    TestScenario(
        name="low_volatility",
        description="Low volatility environment (e.g., calm markets)",
        initial_equity=10000,
        commission=0.0005,
        spread=0.2,
        volatility_multiplier=0.5,
        regime='low_volatility'
    ),
    TestScenario(
        name="strong_trend",
        description="Strong trending market",
        initial_equity=10000,
        commission=0.0005,
        spread=0.3,
        trend_strength=2.0,
        regime='trending'
    ),
    TestScenario(
        name="ranging",
        description="Ranging/consolidating market",
        initial_equity=10000,
        commission=0.0005,
        spread=0.3,
        trend_strength=0.2,
        regime='ranging'
    ),
    TestScenario(
        name="institutional",
        description="Institutional trading with higher capital",
        initial_equity=1000000,
        commission=0.0002,
        spread=0.2,
        regime='normal'
    ),
    TestScenario(
        name="retail",
        description="Retail trading with smaller capital",
        initial_equity=1000,
        commission=0.001,
        spread=0.5,
        regime='normal'
    ),
    TestScenario(
        name="crisis_2020",
        description="COVID-19 crisis simulation (March 2020)",
        initial_equity=10000,
        commission=0.0005,
        spread=0.8,
        volatility_multiplier=3.0,
        trend_strength=1.5,
        regime='high_volatility',
        start_date=datetime(2020, 3, 1),
        end_date=datetime(2020, 6, 1)
    )
]


# =============================================================================
# Data Generation
# =============================================================================

def generate_realistic_gold_data(
    start_date: datetime = datetime(2024, 1, 1),
    end_date: datetime = datetime(2025, 12, 31),
    freq: str = '15T',
    scenario: TestScenario = None
) -> pd.DataFrame:
    """
    Generate realistic synthetic Gold data with configurable market conditions
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency
        scenario: Test scenario for market conditions
    
    Returns:
        DataFrame with OHLCV data
    """
    if scenario is None:
        scenario = TEST_SCENARIOS[0]
    
    dates = pd.date_range(start_date, end_date, freq=freq)
    period_length = len(dates)
    
    # Base parameters
    base_price = 2000
    target_price = 2400
    base_volatility = 25  # Annualized volatility in price terms
    
    # Adjust for scenario
    volatility = base_volatility * scenario.volatility_multiplier
    trend_strength = scenario.trend_strength
    
    # Create trend component
    trend = np.linspace(base_price, target_price, period_length) * trend_strength
    trend = trend + (1 - trend_strength) * base_price
    
    # Create volatility component (random walk)
    daily_vol = volatility / np.sqrt(252 * 24 * 4)  # Convert to 15-min volatility
    random_returns = np.random.randn(period_length) * daily_vol
    random_walk = np.cumsum(random_returns)
    
    # Create intraday patterns
    hour_of_day = dates.hour
    minute_of_hour = dates.minute
    
    # London session (more volatility)
    london_boost = np.where((hour_of_day >= 8) & (hour_of_day < 16), 1.2, 1.0)
    
    # New York session (most volatility)
    ny_boost = np.where((hour_of_day >= 13) & (hour_of_day < 22), 1.5, 1.0)
    
    # Lunch hour (lower volatility)
    lunch_dip = np.where((hour_of_day == 12) & (minute_of_hour >= 30), 0.7, 1.0)
    
    # Combine session effects
    session_effect = london_boost * ny_boost * lunch_dip
    
    # Generate prices
    base = trend + random_walk * session_effect
    
    # Add mean reversion
    mean_reversion = 0.01 * (base_price - base)
    base = base + mean_reversion
    
    # Ensure no negative prices
    base = np.maximum(base, base_price * 0.5)
    
    # Generate OHLC with realistic spreads and patterns
    spreads = 0.3 + np.random.exponential(0.2, period_length)  # Spread in price terms
    spreads = spreads * session_effect  # Wider spreads during active sessions
    
    # Generate random price movements with autocorrelation
    returns = np.random.randn(period_length) * daily_vol * session_effect
    
    # Add autocorrelation (momentum)
    for i in range(1, len(returns)):
        returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
    
    # Calculate OHLC
    closes = base + np.cumsum(returns)
    
    # High and low based on volatility
    highs = closes + np.abs(returns) * 2 + spreads/2
    lows = closes - np.abs(returns) * 2 - spreads/2
    
    # Open is previous close with some variation
    opens = np.roll(closes, 1)
    opens[0] = base[0]
    
    # Volume with patterns
    base_volume = 30000
    volume_pattern = 1 + 0.5 * np.sin(2 * np.pi * np.arange(period_length) / (24 * 4))  # Daily pattern
    volume_noise = np.random.lognormal(0, 0.3, period_length)
    volumes = base_volume * volume_pattern * volume_noise * session_effect
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes.astype(int)
    }, index=dates)
    
    # Clean up any inconsistencies
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def generate_correlated_data(
    gold_data: pd.DataFrame,
    scenario: TestScenario = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate DXY and Yield data correlated with Gold
    
    Args:
        gold_data: Gold price data
        scenario: Test scenario
    
    Returns:
        Tuple of (dxy_data, yield_data)
    """
    if scenario is None:
        scenario = TEST_SCENARIOS[0]
    
    dates = gold_data.index
    period_length = len(dates)
    
    # Correlation strengths
    gold_dxy_corr = -0.7 * scenario.correlation_strength  # Negative correlation
    gold_yield_corr = -0.5 * scenario.correlation_strength  # Negative correlation
    
    # Generate DXY data
    dxy_base = 103
    dxy_trend = np.linspace(0, 2, period_length) * scenario.trend_strength
    
    # Generate returns correlated with Gold
    gold_returns = gold_data['close'].pct_change().fillna(0)
    
    # DXY returns (inverse correlation)
    dxy_returns = -gold_returns * abs(gold_dxy_corr) + \
                  np.random.randn(period_length) * 0.2 * scenario.volatility_multiplier
    
    # Build DXY series
    dxy_close = dxy_base + dxy_trend + np.cumsum(dxy_returns)
    dxy_close = np.maximum(dxy_close, 90)  # Floor
    
    dxy_data = pd.DataFrame({
        'close': dxy_close
    }, index=dates)
    
    # Generate Yield data
    yield_base = 4.2
    yield_trend = np.linspace(0, 0.3, period_length) * scenario.trend_strength
    
    # Yield returns (inverse correlation)
    yield_returns = -gold_returns * abs(gold_yield_corr) + \
                    np.random.randn(period_length) * 0.03 * scenario.volatility_multiplier
    
    # Build Yield series
    yield_close = yield_base + yield_trend + np.cumsum(yield_returns)
    yield_close = np.maximum(yield_close, 0.5)  # Floor
    
    yield_data = pd.DataFrame({
        'close': yield_close
    }, index=dates)
    
    return dxy_data, yield_data


def load_real_data(scenario: TestScenario = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load historical data or generate synthetic data
    
    Args:
        scenario: Test scenario
    
    Returns:
        Tuple of (gold_data, dxy_data, yield_data)
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading test data...")
    
    if scenario is None:
        scenario = TEST_SCENARIOS[0]
    
    try:
        # Try to load from existing CSV files if available
        data_dir = Path('data')
        gold_file = data_dir / 'XAUUSD.csv'
        dxy_file = data_dir / 'DXY.csv'
        yield_file = data_dir / 'US10Y.csv'
        
        if all(f.exists() for f in [gold_file, dxy_file, yield_file]):
            gold_data = pd.read_csv(gold_file, index_col='Date', parse_dates=True)
            dxy_data = pd.read_csv(dxy_file, index_col='Date', parse_dates=True)
            yield_data = pd.read_csv(yield_file, index_col='Date', parse_dates=True)
            
            # Filter by date range if specified
            if scenario.start_date:
                gold_data = gold_data[gold_data.index >= scenario.start_date]
                dxy_data = dxy_data[dxy_data.index >= scenario.start_date]
                yield_data = yield_data[yield_data.index >= scenario.start_date]
            
            if scenario.end_date:
                gold_data = gold_data[gold_data.index <= scenario.end_date]
                dxy_data = dxy_data[dxy_data.index <= scenario.end_date]
                yield_data = yield_data[yield_data.index <= scenario.end_date]
            
            logger.info(f"Loaded real data from CSV files: {len(gold_data)} periods")
            return gold_data, dxy_data, yield_data
    except Exception as e:
        logger.warning(f"Error loading real data: {e}")
    
    # Generate synthetic data
    logger.info(f"Generating synthetic data for scenario: {scenario.name}")
    
    # Determine date range
    if scenario.start_date and scenario.end_date:
        start_date = scenario.start_date
        end_date = scenario.end_date
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
    
    # Generate data
    gold_data = generate_realistic_gold_data(start_date, end_date, scenario=scenario)
    dxy_data, yield_data = generate_correlated_data(gold_data, scenario)
    
    logger.info(f"Generated {len(gold_data)} periods of synthetic data")
    
    return gold_data, dxy_data, yield_data


# =============================================================================
# Statistical Analysis
# =============================================================================

def calculate_confidence_intervals(
    results: Dict[str, Any],
    confidence: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for key metrics
    
    Args:
        results: Backtest results
        confidence: Confidence level (0.95 = 95%)
    
    Returns:
        Dictionary with confidence intervals
    """
    intervals = {}
    
    if 'trades' in results and len(results['trades']) > 0:
        trades_df = results['trades']
        profits = trades_df['profit'].values
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_means = []
        bootstrap_win_rates = []
        
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(profits, size=len(profits), replace=True)
            bootstrap_means.append(np.mean(sample))
            bootstrap_win_rates.append(np.mean(sample > 0))
        
        # Calculate percentiles
        alpha = 1 - confidence
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100
        
        intervals['avg_trade'] = (
            float(np.percentile(bootstrap_means, lower_pct)),
            float(np.percentile(bootstrap_means, upper_pct))
        )
        
        intervals['win_rate'] = (
            float(np.percentile(bootstrap_win_rates, lower_pct)),
            float(np.percentile(bootstrap_win_rates, upper_pct))
        )
    
    return intervals


def perform_statistical_tests(
    results: Dict[str, Any],
    benchmark_return: float = 0.0
) -> Dict[str, Any]:
    """
    Perform statistical tests on backtest results
    
    Args:
        results: Backtest results
        benchmark_return: Benchmark return for comparison
    
    Returns:
        Dictionary with test results
    """
    tests = {}
    
    if 'trades' not in results or len(results['trades']) == 0:
        return tests
    
    trades_df = results['trades']
    profits = trades_df['profit'].values
    
    # Test if mean profit is significantly different from zero
    if len(profits) > 1:
        t_stat, p_value = stats.ttest_1samp(profits, 0)
        tests['t_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
        
        # Non-parametric test (Wilcoxon)
        if np.any(profits != 0):
            w_stat, w_p_value = stats.wilcoxon(profits[profits != 0])
            tests['wilcoxon'] = {
                'statistic': float(w_stat),
                'p_value': float(w_p_value),
                'significant': w_p_value < 0.05
            }
    
    # Test for normality of returns
    if len(profits) >= 8:
        shapiro_stat, shapiro_p = stats.shapiro(profits[:5000] if len(profits) > 5000 else profits)
        tests['normality'] = {
            'shapiro_statistic': float(shapiro_stat),
            'p_value': float(shapiro_p),
            'is_normal': shapiro_p > 0.05
        }
    
    # Test for autocorrelation
    if len(profits) > 20:
        from statsmodels.tsa.stattools import acf, q_stat
        try:
            acf_values = acf(profits, nlags=10, fft=True)
            q_stat, p_values = q_stat(acf_values[1:], len(profits))
            tests['autocorrelation'] = {
                'q_statistic': float(q_stat[-1]),
                'p_value': float(p_values[-1]),
                'has_autocorrelation': p_values[-1] < 0.05
            }
        except:
            pass
    
    # Compare with benchmark
    if 'total_return' in results:
        tests['vs_benchmark'] = {
            'excess_return': results['total_return'] - benchmark_return,
            'information_ratio': safe_divide(
                results['total_return'] - benchmark_return,
                results.get('volatility', 1.0)
            )
        }
    
    return tests


def compare_with_benchmark(
    results: Dict[str, Any],
    equity_curve: pd.Series
) -> Dict[str, float]:
    """
    Compare strategy performance with benchmarks
    
    Args:
        results: Backtest results
        equity_curve: Strategy equity curve
    
    Returns:
        Dictionary with benchmark comparisons
    """
    benchmarks = {}
    
    # Buy and hold benchmark
    if 'trades' in results and len(results['trades']) > 0:
        first_trade = results['trades'].iloc[0]
        last_trade = results['trades'].iloc[-1]
        
        # Simple buy and hold return (simplified)
        buy_hold_return = (last_trade['exit_price'] - first_trade['entry_price']) / first_trade['entry_price']
        benchmarks['buy_hold_return'] = buy_hold_return * 100
    
    # Sharpe ratio comparison
    benchmarks['sharpe_ratio'] = results.get('sharpe_ratio', 0)
    
    # Risk-adjusted metrics
    if results.get('max_drawdown', 0) > 0:
        benchmarks['return_over_drawdown'] = results['total_return'] / results['max_drawdown']
    
    # Alpha and Beta (simplified)
    if 'total_return' in results and 'volatility' in results:
        benchmarks['alpha'] = results['total_return'] - 2.0  # Assuming 2% risk-free
        benchmarks['beta'] = results['volatility'] / 15.0  # Assuming 15% market volatility
    
    return benchmarks


# =============================================================================
# Visualization
# =============================================================================

def plot_comprehensive_results(
    results: Dict[str, Any],
    equity_curve: List[float],
    drawdown_curve: List[float],
    scenario: TestScenario,
    output_dir: Path
):
    """
    Create comprehensive visualization of backtest results
    
    Args:
        results: Backtest results
        equity_curve: Equity curve values
        drawdown_curve: Drawdown curve values
        scenario: Test scenario
        output_dir: Output directory
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Backtest Results: {scenario.name.upper()} Scenario\n{scenario.description}", 
                 fontsize=14, fontweight='bold')
    
    # 1. Equity Curve
    ax1 = axes[0, 0]
    ax1.plot(equity_curve, color='blue', linewidth=1.5)
    ax1.axhline(y=scenario.initial_equity, color='gray', linestyle='--', alpha=0.7)
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    drawdown_pct = [d * 100 for d in drawdown_curve]
    ax2.fill_between(range(len(drawdown_pct)), drawdown_pct, color='red', alpha=0.3)
    ax2.plot(drawdown_pct, color='red', linewidth=1)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Trade Profit Distribution
    ax3 = axes[0, 2]
    if 'trades' in results and len(results['trades']) > 0:
        trades_df = results['trades']
        profits = trades_df['profit'].values
        colors = ['green' if p > 0 else 'red' for p in profits]
        ax3.bar(range(len(profits)), profits, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title(f'Trade Distribution (Win Rate: {results["win_rate"]:.1%})')
        ax3.set_ylabel('Profit ($)')
        ax3.set_xlabel('Trade Number')
        ax3.grid(True, alpha=0.3)
    
    # 4. Monthly Returns Heatmap
    ax4 = axes[1, 0]
    if 'trades' in results and len(results['trades']) > 0:
        trades_df = results['trades']
        trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.month
        trades_df['year'] = pd.to_datetime(trades_df['timestamp']).dt.year
        
        monthly_returns = trades_df.groupby(['year', 'month'])['profit'].sum()
        
        if len(monthly_returns) > 0:
            pivot = monthly_returns.unstack(level=0)
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax4)
            ax4.set_title('Monthly Returns ($)')
    
    # 5. Win Rate by Session
    ax5 = axes[1, 1]
    if 'trades' in results and len(results['trades']) > 0:
        trades_df = results['trades']
        trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
        trades_df['session'] = trades_df['hour'].apply(lambda x: 
            'Asian' if 0 <= x < 8 else 'London' if 8 <= x < 16 else 'New York')
        
        session_win_rates = trades_df.groupby('session').apply(
            lambda x: (x['profit'] > 0).mean()
        )
        
        colors = ['gold', 'lightblue', 'lightgreen']
        session_win_rates.plot(kind='bar', color=colors, ax=ax5)
        ax5.set_title('Win Rate by Session')
        ax5.set_ylabel('Win Rate')
        ax5.set_ylim([0, 1])
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(session_win_rates):
            ax5.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    
    # 6. R-Multiple Distribution
    ax6 = axes[1, 2]
    if 'trades' in results and len(results['trades']) > 0 and 'r_multiple' in trades_df.columns:
        r_multiples = trades_df['r_multiple'].dropna()
        ax6.hist(r_multiples, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax6.axvline(x=r_multiples.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {r_multiples.mean():.2f}')
        ax6.set_title('R-Multiple Distribution')
        ax6.set_xlabel('R-Multiple')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Cumulative Profit
    ax7 = axes[2, 0]
    if 'trades' in results and len(results['trades']) > 0:
        cumulative_profit = results['trades']['profit'].cumsum()
        ax7.plot(cumulative_profit, color='green', linewidth=2)
        ax7.fill_between(range(len(cumulative_profit)), 0, cumulative_profit, 
                        alpha=0.3, color='green', where=cumulative_profit > 0)
        ax7.fill_between(range(len(cumulative_profit)), cumulative_profit, 0, 
                        alpha=0.3, color='red', where=cumulative_profit < 0)
        ax7.set_title('Cumulative Profit')
        ax7.set_ylabel('Cumulative Profit ($)')
        ax7.set_xlabel('Trade Number')
        ax7.grid(True, alpha=0.3)
    
    # 8. Key Metrics Bar Chart
    ax8 = axes[2, 1]
    metrics = {
        'Total Return': results.get('total_return', 0) * 100,
        'Win Rate': results.get('win_rate', 0) * 100,
        'Sharpe': results.get('sharpe_ratio', 0),
        'Max DD': results.get('max_drawdown', 0) * 100
    }
    
    colors = ['green' if v > 0 else 'red' if 'DD' in k else 'blue' for k, v in metrics.items()]
    bars = ax8.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7)
    ax8.set_title('Key Performance Metrics')
    ax8.set_ylabel('Value')
    ax8.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, (_, value) in zip(bars, metrics.items()):
        height = bar.get_height()
        label = f'{value:.1f}' + ('%' if any(x in bar.get_label() for x in ['Return', 'Rate', 'DD']) else '')
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                label, ha='center', va='bottom', fontweight='bold')
    
    # 9. Risk-Reward Scatter
    ax9 = axes[2, 2]
    if 'trades' in results and len(results['trades']) > 0:
        trades_df = results['trades']
        winning = trades_df[trades_df['profit'] > 0]
        losing = trades_df[trades_df['profit'] <= 0]
        
        if len(winning) > 0:
            ax9.scatter(winning.index, winning['profit'], color='green', 
                       alpha=0.6, label='Winning', s=30)
        if len(losing) > 0:
            ax9.scatter(losing.index, losing['profit'], color='red', 
                       alpha=0.6, label='Losing', s=30)
        
        ax9.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax9.set_title('Risk-Reward Scatter')
        ax9.set_xlabel('Trade Number')
        ax9.set_ylabel('Profit ($)')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'comprehensive_results_{scenario.name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.getLogger(__name__).info(f"Comprehensive plot saved to {output_file}")


# =============================================================================
# Main Test Function
# =============================================================================

def run_scenario_test(
    scenario: TestScenario,
    output_dir: Path,
    save_plots: bool = True,
    save_data: bool = True
) -> Optional[TestResult]:
    """
    Run backtest for a specific scenario
    
    Args:
        scenario: Test scenario
        output_dir: Output directory
        save_plots: Whether to save plots
        save_data: Whether to save data
    
    Returns:
        TestResult object
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Scenario: {scenario.name}")
    logger.info(f"Description: {scenario.description}")
    logger.info(f"{'='*60}")
    
    try:
        # Load test data
        gold_data, dxy_data, yield_data = load_real_data(scenario)
        
        # Initialize backtest engine
        backtest = InstitutionalBacktestEngine(
            initial_equity=scenario.initial_equity,
            commission=scenario.commission
        )
        
        # Run backtest
        results = backtest.run_backtest(
            gold_data,
            dxy_data,
            yield_data,
            spread=scenario.spread,
            symbol="XAUUSD"
        )
        
        # Calculate additional analytics
        session_stats = {}
        statistical_tests = {}
        benchmark_comparison = {}
        confidence_intervals = {}
        
        if results['total_trades'] > 0:
            trades_df = results['trades']
            
            # Session analysis
            trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
            trades_df['session'] = trades_df['hour'].apply(lambda x: 
                'Asian' if 0 <= x < 8 else 
                'London' if 8 <= x < 16 else 
                'New York')
            
            for session in ['Asian', 'London', 'New York']:
                session_trades = trades_df[trades_df['session'] == session]
                if len(session_trades) > 0:
                    session_win_rate = len(session_trades[session_trades['profit'] > 0]) / len(session_trades)
                    session_avg_profit = session_trades['profit'].mean()
                    session_stats[session] = {
                        'trades': len(session_trades),
                        'win_rate': session_win_rate,
                        'avg_profit': session_avg_profit,
                        'total_profit': session_trades['profit'].sum()
                    }
            
            # Statistical tests
            statistical_tests = perform_statistical_tests(results)
            
            # Benchmark comparison
            equity_series = pd.Series(backtest.equity_curve)
            benchmark_comparison = compare_with_benchmark(results, equity_series)
            
            # Confidence intervals
            confidence_intervals = calculate_confidence_intervals(results)
        
        # Create test result
        test_result = TestResult(
            scenario=scenario,
            backtest_results=results,
            session_stats=session_stats,
            statistical_tests=statistical_tests,
            benchmark_comparison=benchmark_comparison,
            confidence_intervals=confidence_intervals
        )
        
        # Print results
        print_results(test_result)
        
        # Generate plots
        if save_plots and results['total_trades'] > 0:
            scenario_dir = output_dir / scenario.name
            scenario_dir.mkdir(exist_ok=True)
            
            plot_comprehensive_results(
                results,
                backtest.equity_curve,
                backtest.drawdown_curve,
                scenario,
                scenario_dir
            )
            
            # Also generate standard report
            backtest.generate_report(results, output_dir=str(scenario_dir))
        
        # Save data
        if save_data:
            save_test_results(test_result, output_dir)
        
        return test_result
        
    except Exception as e:
        logger.error(f"Error in scenario {scenario.name}: {e}")
        logger.debug(traceback.format_exc())
        return None


def print_results(result: TestResult):
    """Print formatted test results"""
    results = result.backtest_results
    scenario = result.scenario
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {scenario.name.upper()}")
    print(f"{'='*60}")
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Final Equity: {format_currency(results['final_equity'])}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    
    print(f"\n📈 TRADE STATISTICS:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate']:.2%}")
    print(f"  Avg Trade: {format_currency(results['average_trade'])}")
    print(f"  Risk-Reward: {results['risk_reward']:.2f}")
    
    # Statistical significance
    if result.statistical_tests:
        print(f"\n🔬 STATISTICAL TESTS:")
        if 't_test' in result.statistical_tests:
            t_test = result.statistical_tests['t_test']
            sig = "✅" if t_test['significant'] else "❌"
            print(f"  {sig} T-Test: p={t_test['p_value']:.4f}")
        
        if 'wilcoxon' in result.statistical_tests:
            wilcox = result.statistical_tests['wilcoxon']
            sig = "✅" if wilcox['significant'] else "❌"
            print(f"  {sig} Wilcoxon: p={wilcox['p_value']:.4f}")
    
    # Confidence intervals
    if result.confidence_intervals:
        print(f"\n📊 CONFIDENCE INTERVALS (95%):")
        if 'avg_trade' in result.confidence_intervals:
            ci = result.confidence_intervals['avg_trade']
            print(f"  Avg Trade: [{format_currency(ci[0])}, {format_currency(ci[1])}]")
        if 'win_rate' in result.confidence_intervals:
            ci = result.confidence_intervals['win_rate']
            print(f"  Win Rate: [{ci[0]:.1%}, {ci[1]:.1%}]")
    
    # Session performance
    if result.session_stats:
        print(f"\n🌍 SESSION PERFORMANCE:")
        for session, stats in result.session_stats.items():
            print(f"  {session}: {stats['trades']} trades, "
                  f"{stats['win_rate']:.1%} win rate, "
                  f"total {format_currency(stats['total_profit'])}")
    
    print(f"\n{'='*60}\n")


def save_test_results(result: TestResult, output_dir: Path):
    """Save test results to files"""
    scenario_dir = output_dir / result.scenario.name
    scenario_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_file = scenario_dir / 'results.json'
    with open(json_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    
    # Save trades CSV
    if 'trades' in result.backtest_results and len(result.backtest_results['trades']) > 0:
        trades_df = result.backtest_results['trades']
        trades_file = scenario_dir / 'trades.csv'
        trades_df.to_csv(trades_file, index=False)
    
    logging.getLogger(__name__).info(f"Results saved to {scenario_dir}")


def run_all_scenarios(
    scenarios: List[TestScenario] = None,
    output_dir: Path = None,
    parallel: bool = False
) -> Dict[str, TestResult]:
    """
    Run all test scenarios
    
    Args:
        scenarios: List of scenarios to run (default: all)
        output_dir: Output directory
        parallel: Run in parallel
    
    Returns:
        Dictionary of scenario results
    """
    if scenarios is None:
        scenarios = TEST_SCENARIOS
    
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('test_results') / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Running {len(scenarios)} test scenarios")
    logger.info(f"Results will be saved to: {output_dir}")
    
    results = {}
    
    if parallel:
        # Run in parallel
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_scenario = {
                executor.submit(run_scenario_test, scenario, output_dir, True, True): scenario
                for scenario in scenarios
            }
            
            for future in concurrent.futures.as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    result = future.result(timeout=300)
                    if result:
                        results[scenario.name] = result
                except Exception as e:
                    logger.error(f"Scenario {scenario.name} failed: {e}")
    else:
        # Run sequentially
        for scenario in scenarios:
            result = run_scenario_test(scenario, output_dir, True, True)
            if result:
                results[scenario.name] = result
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    return results


def generate_summary_report(results: Dict[str, TestResult], output_dir: Path):
    """Generate summary report comparing all scenarios"""
    
    summary_data = []
    
    for name, result in results.items():
        res = result.backtest_results
        summary_data.append({
            'Scenario': name,
            'Return': f"{res['total_return']:.2%}",
            'Sharpe': f"{res['sharpe_ratio']:.2f}",
            'Win Rate': f"{res['win_rate']:.1%}",
            'Max DD': f"{res['max_drawdown']:.1%}",
            'Trades': res['total_trades'],
            'Avg Trade': f"${res['average_trade']:.2f}",
            'RR Ratio': f"{res['risk_reward']:.2f}"
        })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_file = output_dir / 'summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Scenario Comparison Summary', fontsize=14, fontweight='bold')
    
    # Returns comparison
    ax1 = axes[0, 0]
    returns = [float(r['Return'].strip('%')) / 100 for r in summary_data]
    scenarios = [r['Scenario'] for r in summary_data]
    colors = ['green' if r > 0 else 'red' for r in returns]
    bars = ax1.bar(scenarios, returns, color=colors, alpha=0.7)
    ax1.set_title('Total Return by Scenario')
    ax1.set_ylabel('Return')
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Sharpe ratio comparison
    ax2 = axes[0, 1]
    sharpes = [float(r['Sharpe']) for r in summary_data]
    bars = ax2.bar(scenarios, sharpes, color='blue', alpha=0.7)
    ax2.set_title('Sharpe Ratio by Scenario')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Good')
    ax2.axhline(y=2.0, color='gold', linestyle='--', alpha=0.7, label='Excellent')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Win rate vs max drawdown
    ax3 = axes[1, 0]
    win_rates = [float(r['Win Rate'].strip('%')) / 100 for r in summary_data]
    max_dds = [float(r['Max DD'].strip('%')) / 100 for r in summary_data]
    
    scatter = ax3.scatter(win_rates, max_dds, c=returns, s=200, cmap='RdYlGn', alpha=0.7)
    ax3.set_title('Win Rate vs Max Drawdown')
    ax3.set_xlabel('Win Rate')
    ax3.set_ylabel('Max Drawdown')
    ax3.grid(True, alpha=0.3)
    
    # Add labels
    for i, scenario in enumerate(scenarios):
        ax3.annotate(scenario[:3], (win_rates[i], max_dds[i]), 
                    fontsize=8, ha='center', va='center')
    
    # Trade count
    ax4 = axes[1, 1]
    trades = [r['Trades'] for r in summary_data]
    bars = ax4.bar(scenarios, trades, color='purple', alpha=0.7)
    ax4.set_title('Number of Trades by Scenario')
    ax4.set_ylabel('Trades')
    ax4.set_xticklabels(scenarios, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary chart
    summary_chart = output_dir / 'summary_comparison.png'
    plt.savefig(summary_chart, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.getLogger(__name__).info(f"Summary report saved to {output_dir}")
    print(f"\n📊 Summary report saved to {output_dir}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test Gold Institutional Quant Framework')
    parser.add_argument('--scenario', type=str, default='all',
                       help='Test scenario to run (all, baseline, high_volatility, etc.)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run scenarios in parallel')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable saving results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    print("\n" + "="*70)
    print(" GOLD INSTITUTIONAL QUANT FRAMEWORK - COMPREHENSIVE TESTING")
    print("="*70)
    
    # Select scenarios
    if args.scenario == 'all':
        scenarios = TEST_SCENARIOS
    else:
        scenarios = [s for s in TEST_SCENARIOS if s.name == args.scenario]
        if not scenarios:
            print(f"❌ Unknown scenario: {args.scenario}")
            print(f"Available: all, {', '.join([s.name for s in TEST_SCENARIOS])}")
            return 1
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('test_results') / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    results = run_all_scenarios(
        scenarios=scenarios,
        output_dir=output_dir,
        parallel=args.parallel and len(scenarios) > 1
    )
    
    # Print final summary
    print("\n" + "="*70)
    print(" TESTING COMPLETE")
    print("="*70)
    print(f"\n✅ Successfully tested: {len(results)} scenarios")
    print(f"❌ Failed: {len(scenarios) - len(results)} scenarios")
    print(f"\n📁 Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())