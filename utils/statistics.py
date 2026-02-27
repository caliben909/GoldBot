"""
Statistics - Trading performance and risk analysis utilities
Comprehensive statistics and performance metrics for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# =============================================================================
# Risk Metrics
# =============================================================================

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, 
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sharpe ratio
    """
    try:
        excess_returns = returns - (risk_free_rate / periods_per_year)
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    except Exception as e:
        logger.error(f"Failed to calculate Sharpe ratio: {e}")
        return 0.0


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation)
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sortino ratio
    """
    try:
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
            
        downside_deviation = downside_returns.std()
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation
    except Exception as e:
        logger.error(f"Failed to calculate Sortino ratio: {e}")
        return 0.0


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (average return / max drawdown)
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
    
    Returns:
        Calmar ratio
    """
    try:
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = calculate_drawdown_statistics(cumulative_returns)['max_drawdown']
        
        if max_drawdown == 0:
            return 0.0
            
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1
        return annualized_return / (max_drawdown / 100)
    except Exception as e:
        logger.error(f"Failed to calculate Calmar ratio: {e}")
        return 0.0


def calculate_omega_ratio(returns: pd.Series, target_return: float = 0.0,
                        periods_per_year: int = 252) -> float:
    """
    Calculate Omega ratio (gain-loss ratio)
    
    Args:
        returns: Series of returns
        target_return: Target return per period
        periods_per_year: Number of periods per year
    
    Returns:
        Omega ratio
    """
    try:
        excess_returns = returns - (target_return / periods_per_year)
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()
        
        if losses == 0:
            return 0.0
            
        return gains / losses
    except Exception as e:
        logger.error(f"Failed to calculate Omega ratio: {e}")
        return 0.0


def calculate_win_loss_statistics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate win/loss statistics
    
    Args:
        returns: Series of returns
    
    Returns:
        Dictionary with win/loss statistics
    """
    try:
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        return {
            'win_rate': len(winning_trades) / len(returns) * 100 if len(returns) > 0 else 0,
            'average_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'average_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else 0,
            'max_win': winning_trades.max() if len(winning_trades) > 0 else 0,
            'max_loss': losing_trades.min() if len(losing_trades) > 0 else 0,
            'win_loss_ratio': len(winning_trades) / len(losing_trades) if len(losing_trades) > 0 else 0
        }
    except Exception as e:
        logger.error(f"Failed to calculate win/loss statistics: {e}")
        return {}


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor
    
    Args:
        returns: Series of returns
    
    Returns:
        Profit factor
    """
    try:
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        if len(losing_trades) == 0:
            return 0.0
            
        gross_profit = winning_trades.sum()
        gross_loss = -losing_trades.sum()
        
        return gross_profit / gross_loss
    except Exception as e:
        logger.error(f"Failed to calculate profit factor: {e}")
        return 0.0


def calculate_expectancy(returns: pd.Series) -> float:
    """
    Calculate expectancy (average return per trade)
    
    Args:
        returns: Series of returns
    
    Returns:
        Expectancy
    """
    try:
        return returns.mean()
    except Exception as e:
        logger.error(f"Failed to calculate expectancy: {e}")
        return 0.0


def calculate_r_multiple_distribution(returns: pd.Series, risk_per_trade: float = 0.01) -> pd.Series:
    """
    Calculate R-multiples for each trade
    
    Args:
        returns: Series of returns
        risk_per_trade: Risk per trade (fraction)
    
    Returns:
        Series of R-multiples
    """
    try:
        return returns / risk_per_trade
    except Exception as e:
        logger.error(f"Failed to calculate R-multiples: {e}")
        return pd.Series()


def calculate_rolling_sharpe(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling Sharpe ratio
    
    Args:
        returns: Series of returns
        window: Rolling window size
    
    Returns:
        Series of rolling Sharpe ratios
    """
    try:
        return returns.rolling(window=window).apply(
            lambda x: calculate_sharpe_ratio(x), raw=False
        )
    except Exception as e:
        logger.error(f"Failed to calculate rolling Sharpe: {e}")
        return pd.Series()


def calculate_drawdown_statistics(cumulative_returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive drawdown statistics
    
    Args:
        cumulative_returns: Series of cumulative returns
    
    Returns:
        Dictionary with drawdown statistics
    """
    try:
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (rolling_max - cumulative_returns) / rolling_max
        
        max_drawdown = drawdown.max()
        max_drawdown_end = drawdown.idxmax()
        
        # Find start of max drawdown
        max_drawdown_start = None
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown.iloc[i] == 0:
                max_drawdown_start = drawdown.index[i + 1] if i + 1 < len(drawdown) else drawdown.index[i]
                break
        
        # Calculate duration
        if max_drawdown_start and max_drawdown_end:
            duration = (max_drawdown_end - max_drawdown_start).days
        else:
            duration = 0
        
        return {
            'max_drawdown': max_drawdown * 100,
            'max_drawdown_start': max_drawdown_start,
            'max_drawdown_end': max_drawdown_end,
            'max_drawdown_duration': duration,
            'average_drawdown': drawdown.mean() * 100,
            'drawdown_std': drawdown.std() * 100
        }
    except Exception as e:
        logger.error(f"Failed to calculate drawdown statistics: {e}")
        return {}


def calculate_risk_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
    
    Returns:
        Dictionary with risk metrics
    """
    try:
        cumulative_returns = (1 + returns).cumprod()
        drawdown_stats = calculate_drawdown_statistics(cumulative_returns)
        
        return {
            'sharpe_ratio': calculate_sharpe_ratio(returns, periods_per_year=periods_per_year),
            'sortino_ratio': calculate_sortino_ratio(returns, periods_per_year=periods_per_year),
            'calmar_ratio': calculate_calmar_ratio(returns, periods_per_year=periods_per_year),
            'omega_ratio': calculate_omega_ratio(returns, periods_per_year=periods_per_year),
            'max_drawdown': drawdown_stats['max_drawdown'],
            'profit_factor': calculate_profit_factor(returns),
            'win_rate': calculate_win_loss_statistics(returns)['win_rate'],
            'expectancy': calculate_expectancy(returns)
        }
    except Exception as e:
        logger.error(f"Failed to calculate risk metrics: {e}")
        return {}


def calculate_vaR(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Series of returns
        confidence: Confidence level (0.95 = 95%)
    
    Returns:
        VaR value
    """
    try:
        return np.percentile(returns, (1 - confidence) * 100)
    except Exception as e:
        logger.error(f"Failed to calculate VaR: {e}")
        return 0.0


def calculate_cVaR(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (cVaR)
    
    Args:
        returns: Series of returns
        confidence: Confidence level (0.95 = 95%)
    
    Returns:
        cVaR value
    """
    try:
        var = calculate_vaR(returns, confidence)
        tail_returns = returns[returns <= var]
        return tail_returns.mean()
    except Exception as e:
        logger.error(f"Failed to calculate cVaR: {e}")
        return 0.0


def calculate_tail_ratio(returns: pd.Series) -> float:
    """
    Calculate tail ratio (upper tail / lower tail)
    
    Args:
        returns: Series of returns
    
    Returns:
        Tail ratio
    """
    try:
        upper_tail = returns.quantile(0.95)
        lower_tail = returns.quantile(0.05)
        return abs(upper_tail / lower_tail) if lower_tail != 0 else 0.0
    except Exception as e:
        logger.error(f"Failed to calculate tail ratio: {e}")
        return 0.0


def calculate_gain_to_pain_ratio(returns: pd.Series) -> float:
    """
    Calculate Gain-to-Pain ratio
    
    Args:
        returns: Series of returns
    
    Returns:
        Gain-to-Pain ratio
    """
    try:
        total_gain = returns[returns > 0].sum()
        total_pain = -returns[returns < 0].sum()
        return total_gain / total_pain if total_pain != 0 else 0.0
    except Exception as e:
        logger.error(f"Failed to calculate Gain-to-Pain ratio: {e}")
        return 0.0


def calculate_recovery_factor(returns: pd.Series) -> float:
    """
    Calculate Recovery factor
    
    Args:
        returns: Series of returns
    
    Returns:
        Recovery factor
    """
    try:
        cumulative_returns = (1 + returns).cumprod()
        drawdown_stats = calculate_drawdown_statistics(cumulative_returns)
        
        total_return = cumulative_returns.iloc[-1] - 1
        max_drawdown = drawdown_stats['max_drawdown'] / 100
        
        return total_return / max_drawdown if max_drawdown != 0 else 0.0
    except Exception as e:
        logger.error(f"Failed to calculate recovery factor: {e}")
        return 0.0


def calculate_ulcer_index(returns: pd.Series) -> float:
    """
    Calculate Ulcer Index
    
    Args:
        returns: Series of returns
    
    Returns:
        Ulcer Index
    """
    try:
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (rolling_max - cumulative_returns) / rolling_max
        return np.sqrt((drawdown ** 2).mean())
    except Exception as e:
        logger.error(f"Failed to calculate ulcer index: {e}")
        return 0.0


def calculate_upi_index(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Ulcer Performance Index
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
    
    Returns:
        Ulcer Performance Index
    """
    try:
        ulcer_index = calculate_ulcer_index(returns)
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1
        return annualized_return / ulcer_index if ulcer_index != 0 else 0.0
    except Exception as e:
        logger.error(f"Failed to calculate UPI index: {e}")
        return 0.0


# =============================================================================
# Volatility Metrics
# =============================================================================

def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
    
    Returns:
        Annualized volatility
    """
    try:
        return returns.std() * np.sqrt(periods_per_year)
    except Exception as e:
        logger.error(f"Failed to calculate volatility: {e}")
        return 0.0


def calculate_rolling_volatility(returns: pd.Series, window: int = 20,
                               periods_per_year: int = 252) -> pd.Series:
    """
    Calculate rolling volatility
    
    Args:
        returns: Series of returns
        window: Rolling window size
        periods_per_year: Number of periods per year
    
    Returns:
        Series of rolling volatility
    """
    try:
        return returns.rolling(window=window).std() * np.sqrt(periods_per_year)
    except Exception as e:
        logger.error(f"Failed to calculate rolling volatility: {e}")
        return pd.Series()


def calculate_skewness(returns: pd.Series) -> float:
    """
    Calculate skewness
    
    Args:
        returns: Series of returns
    
    Returns:
        Skewness
    """
    try:
        return returns.skew()
    except Exception as e:
        logger.error(f"Failed to calculate skewness: {e}")
        return 0.0


def calculate_kurtosis(returns: pd.Series) -> float:
    """
    Calculate kurtosis
    
    Args:
        returns: Series of returns
    
    Returns:
        Kurtosis
    """
    try:
        return returns.kurt()
    except Exception as e:
        logger.error(f"Failed to calculate kurtosis: {e}")
        return 0.0


# =============================================================================
# Return Metrics
# =============================================================================

def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
    
    Returns:
        Annualized return
    """
    try:
        if len(returns) == 0:
            return 0.0
        return (1 + returns.mean()) ** periods_per_year - 1
    except Exception as e:
        logger.error(f"Failed to calculate annualized return: {e}")
        return 0.0


def calculate_cumulative_return(returns: pd.Series) -> float:
    """
    Calculate cumulative return
    
    Args:
        returns: Series of returns
    
    Returns:
        Cumulative return
    """
    try:
        return (1 + returns).cumprod().iloc[-1] - 1
    except Exception as e:
        logger.error(f"Failed to calculate cumulative return: {e}")
        return 0.0


def calculate_monthly_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate monthly returns
    
    Args:
        returns: Series of returns with datetime index
    
    Returns:
        Series of monthly returns
    """
    try:
        if not isinstance(returns.index, pd.DatetimeIndex):
            return pd.Series()
            
        return returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    except Exception as e:
        logger.error(f"Failed to calculate monthly returns: {e}")
        return pd.Series()


def calculate_yearly_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate yearly returns
    
    Args:
        returns: Series of returns with datetime index
    
    Returns:
        Series of yearly returns
    """
    try:
        if not isinstance(returns.index, pd.DatetimeIndex):
            return pd.Series()
            
        return returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    except Exception as e:
        logger.error(f"Failed to calculate yearly returns: {e}")
        return pd.Series()


# =============================================================================
# Statistical Tests
# =============================================================================

def calculate_normality_test(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate normality tests
    
    Args:
        returns: Series of returns
    
    Returns:
        Dictionary with test results
    """
    try:
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(returns)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(returns, 'norm', args=(returns.mean(), returns.std()))
        
        return {
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'is_normal': shapiro_p > 0.05
        }
    except Exception as e:
        logger.error(f"Failed to calculate normality tests: {e}")
        return {}


def calculate_autocorrelation(returns: pd.Series, lag: int = 1) -> float:
    """
    Calculate autocorrelation at specified lag
    
    Args:
        returns: Series of returns
        lag: Time lag
    
    Returns:
        Autocorrelation coefficient
    """
    try:
        return returns.autocorr(lag=lag)
    except Exception as e:
        logger.error(f"Failed to calculate autocorrelation: {e}")
        return 0.0


# =============================================================================
# Portfolio Metrics
# =============================================================================

def calculate_portfolio_returns(weights: np.array, returns: pd.DataFrame) -> pd.Series:
    """
    Calculate portfolio returns from individual asset returns
    
    Args:
        weights: Portfolio weights
        returns: DataFrame of asset returns
    
    Returns:
        Series of portfolio returns
    """
    try:
        return (returns * weights).sum(axis=1)
    except Exception as e:
        logger.error(f"Failed to calculate portfolio returns: {e}")
        return pd.Series()


def calculate_portfolio_volatility(weights: np.array, returns: pd.DataFrame,
                                  periods_per_year: int = 252) -> float:
    """
    Calculate portfolio volatility
    
    Args:
        weights: Portfolio weights
        returns: DataFrame of asset returns
        periods_per_year: Number of periods per year
    
    Returns:
        Portfolio volatility
    """
    try:
        covariance = returns.cov()
        volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        return volatility * np.sqrt(periods_per_year)
    except Exception as e:
        logger.error(f"Failed to calculate portfolio volatility: {e}")
        return 0.0


def calculate_maximum_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related statistics
    
    Args:
        equity_curve: Series of equity values
    
    Returns:
        Dictionary with drawdown statistics
    """
    return calculate_drawdown_statistics(equity_curve)


__all__ = [
    # Risk Metrics
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_omega_ratio',
    'calculate_win_loss_statistics',
    'calculate_profit_factor',
    'calculate_expectancy',
    'calculate_r_multiple_distribution',
    'calculate_rolling_sharpe',
    'calculate_drawdown_statistics',
    'calculate_risk_metrics',
    'calculate_vaR',
    'calculate_cVaR',
    'calculate_tail_ratio',
    'calculate_gain_to_pain_ratio',
    'calculate_recovery_factor',
    'calculate_ulcer_index',
    'calculate_upi_index',
    
    # Volatility Metrics
    'calculate_volatility',
    'calculate_rolling_volatility',
    'calculate_skewness',
    'calculate_kurtosis',
    
    # Return Metrics
    'calculate_annualized_return',
    'calculate_cumulative_return',
    'calculate_monthly_returns',
    'calculate_yearly_returns',
    
    # Statistical Tests
    'calculate_normality_test',
    'calculate_autocorrelation',
    
    # Portfolio Metrics
    'calculate_portfolio_returns',
    'calculate_portfolio_volatility',
    'calculate_maximum_drawdown'
]
