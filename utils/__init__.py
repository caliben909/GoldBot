"""
Utils Module - Utility functions and helpers for the trading bot
Provides comprehensive utilities for data processing, technical indicators,
and common trading operations with type safety and error handling.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import pickle
import time
from contextlib import contextmanager
import warnings

# Define decorators class first
class decorators:
    """Collection of utility decorators"""
    
    @staticmethod
    def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0,
                          backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
        """
        Retry decorator with exponential backoff
        
        Args:
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            backoff_factor: Multiplier for delay after each retry
            exceptions: Tuple of exceptions to catch
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                delay = initial_delay
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_retries:
                            raise
                        time.sleep(delay)
                        delay *= backoff_factor
                return None
            return wrapper
        return decorator
    
    @staticmethod
    def memoize(ttl: Optional[int] = None):
        """
        Memoization decorator with optional TTL
        
        Args:
            ttl: Time to live in seconds (None for infinite)
        """
        def decorator(func):
            cache = {}
            timestamps = {}
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = hashlib.md5(
                    str((args, kwargs)).encode()
                ).hexdigest()
                
                now = time.time()
                
                if key in cache:
                    if ttl is None or (now - timestamps[key]) < ttl:
                        return cache[key]
                
                result = func(*args, **kwargs)
                cache[key] = result
                timestamps[key] = now
                
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def timing(func):
        """Measure and log function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logging.debug(f"{func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def singleton(cls):
        """Singleton decorator for classes"""
        instances = {}
        
        @wraps(cls)
        def get_instance(*args, **kwargs):
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
        
        return get_instance
    
    @staticmethod
    def rate_limit(max_calls: int, period: float):
        """
        Rate limiting decorator
        
        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        def decorator(func):
            calls = []
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                now = time.time()
                
                # Remove old calls
                calls[:] = [call for call in calls if call > now - period]
                
                if len(calls) >= max_calls:
                    wait_time = calls[0] + period - now
                    if wait_time > 0:
                        time.sleep(wait_time)
                
                calls.append(now)
                return func(*args, **kwargs)
            return wrapper
        return decorator

# Re-export from submodules
from .logging_config import setup_logging
retry_with_backoff = decorators.retry_with_backoff
memoize = decorators.memoize
timing = decorators.timing
singleton = decorators.singleton
rate_limit = decorators.rate_limit


# Import from indicators module (implementations below)
from .indicators import (
    TechnicalIndicators,
    IndicatorError,
    InsufficientDataError,
    InvalidParameterError,
    Trend,
    Signal,
    Divergence,
    IndicatorResult
)

# Import from data_loader module
from .data_loader import (
    load_csv_data,
    save_csv_data,
    resample_data,
    clean_data,
    filter_time_range,
    detect_outliers,
    fill_missing_values,
    calculate_return_statistics,
    calculate_drawdown,
    calculate_time_based_features,
    merge_dataframes,
    load_from_multiple_sources,
    load_parquet_data,
    save_parquet_data,
    load_hdf5_data,
    save_hdf5_data,
    load_from_database
)

# Import from helpers module
from .helpers import (
    load_config,
    save_config,
    create_directories,
    format_currency,
    format_percentage,
    format_timestamp,
    calculate_pips,
    get_pip_multiplier,
    calculate_point_value,
    validate_symbol,
    normalize_symbol,
    get_instrument_type,
    safe_divide,
    round_to_tick,
    clamp,
    calculate_risk_reward,
    calculate_position_size,
    time_to_next_interval,
    get_interval_minutes,
    flatten_dict,
    nested_get,
    nested_set,
    deep_merge,
    parse_timedelta,
    format_timedelta,
    get_date_range,
    is_market_open,
    get_next_market_open,
    get_previous_market_close
)

# New utility functions
from .statistics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_omega_ratio,
    calculate_profit_factor,
    calculate_expectancy,
    calculate_r_multiple_distribution,
    calculate_rolling_sharpe,
    calculate_drawdown_statistics,
    calculate_win_loss_statistics,
    calculate_risk_metrics,
    calculate_vaR,
    calculate_cVaR,
    calculate_tail_ratio,
    calculate_gain_to_pain_ratio,
    calculate_recovery_factor,
    calculate_ulcer_index,
    calculate_upi_index
)

from .validation import (
    validate_dataframe,
    validate_columns,
    validate_positive,
    validate_non_negative,
    validate_range,
    validate_in_list,
    validate_datetime_index,
    validate_frequency,
    validate_trade_parameters,
    validate_risk_parameters,
    ValidationResult
)

from .transform import (
    normalize_data,
    standardize_data,
    winsorize_data,
    trim_outliers,
    winsorize_series,
    boxcox_transform,
    log_transform,
    difference_transform,
    percentage_change,
    rolling_normalize,
    rolling_standardize,
    create_lagged_features,
    create_rolling_features,
    create_expanding_features
)

# Version information
__version__ = "1.0.0"
__author__ = "Trading Bot Team"


# =============================================================================
# INDICATORS IMPLEMENTATION
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: Series of prices
        period: RSI period
        
    Returns:
        Series of RSI values
    """
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        raise IndicatorError(f"Failed to calculate RSI: {e}")


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        Series of ATR values
    """
    try:
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(window=period).mean()
        return atr
    except Exception as e:
        raise IndicatorError(f"Failed to calculate ATR: {e}")


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Series of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    try:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': histogram
        })
    except Exception as e:
        raise IndicatorError(f"Failed to calculate MACD: {e}")


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Series of prices
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        DataFrame with Upper, Middle, Lower bands
    """
    try:
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return pd.DataFrame({
            'Upper': upper,
            'Middle': middle,
            'Lower': lower
        })
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Bollinger Bands: {e}")


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period
        
    Returns:
        DataFrame with %K and %D values
    """
    try:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            '%K': k,
            '%D': d
        })
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Stochastic: {e}")


def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Momentum indicator
    
    Args:
        prices: Series of prices
        period: Momentum period
        
    Returns:
        Series of momentum values
    """
    try:
        return prices.diff(period)
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Momentum: {e}")


def calculate_volume_profile(df: pd.DataFrame, bins: int = 50) -> Dict[str, Any]:
    """
    Calculate Volume Profile (Market Profile)
    
    Args:
        df: DataFrame with 'price' and 'volume' columns
        bins: Number of price bins
        
    Returns:
        Dictionary with volume profile statistics
    """
    try:
        price_min = df['price'].min()
        price_max = df['price'].max()
        price_bins = np.linspace(price_min, price_max, bins)
        
        volume_profile = pd.cut(df['price'], bins=price_bins).value_counts()
        volume_by_price = df.groupby(pd.cut(df['price'], bins=price_bins))['volume'].sum()
        
        poc_idx = volume_by_price.idxmax()
        poc_price = (poc_idx.left + poc_idx.right) / 2
        
        vah = price_max  # Value Area High
        val = price_min  # Value Area Low
        
        # Calculate Value Area (70% of volume)
        total_volume = volume_by_price.sum()
        sorted_by_volume = volume_by_price.sort_values(ascending=False)
        
        cumulative_volume = 0
        value_area_prices = []
        
        for price_bin, volume in sorted_by_volume.items():
            cumulative_volume += volume
            value_area_prices.append(price_bin)
            if cumulative_volume >= total_volume * 0.7:
                break
        
        if value_area_prices:
            all_bins = [b.left for b in value_area_prices] + [b.right for b in value_area_prices]
            val = min(all_bins)
            vah = max(all_bins)
        
        return {
            'volume_profile': volume_by_price,
            'poc_price': poc_price,
            'value_area_high': vah,
            'value_area_low': val,
            'total_volume': total_volume,
            'bins': price_bins.tolist()
        }
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Volume Profile: {e}")


def calculate_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate market structure features (swing highs/lows, trends)
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with structure features
    """
    try:
        result = df.copy()
        
        # Swing highs and lows
        result['swing_high'] = (
            (result['high'] > result['high'].shift(1)) & 
            (result['high'] > result['high'].shift(-1))
        )
        result['swing_low'] = (
            (result['low'] < result['low'].shift(1)) & 
            (result['low'] < result['low'].shift(-1))
        )
        
        # Higher highs / lower lows pattern
        result['higher_high'] = (
            result['swing_high'] & 
            (result['high'] > result['high'].shift(2))
        )
        result['lower_low'] = (
            result['swing_low'] & 
            (result['low'] < result['low'].shift(2))
        )
        
        # Trend direction
        result['sma_20'] = result['close'].rolling(20).mean()
        result['sma_50'] = result['close'].rolling(50).mean()
        result['trend'] = np.where(
            result['sma_20'] > result['sma_50'], 1,
            np.where(result['sma_20'] < result['sma_50'], -1, 0)
        )
        
        # Price position relative to recent range
        result['highest_20'] = result['high'].rolling(20).max()
        result['lowest_20'] = result['low'].rolling(20).min()
        result['range_position'] = (
            (result['close'] - result['lowest_20']) / 
            (result['highest_20'] - result['lowest_20'])
        )
        
        # Volatility regime
        result['atr_20'] = calculate_atr(result['high'], result['low'], result['close'], 20)
        result['atr_ratio'] = result['atr_20'] / result['atr_20'].rolling(50).mean()
        
        return result
    except Exception as e:
        raise IndicatorError(f"Failed to calculate structure features: {e}")


def calculate_volatility(prices: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    """
    Calculate volatility (standard deviation of returns)
    
    Args:
        prices: Series of prices
        period: Rolling window period
        annualize: Whether to annualize the volatility
        
    Returns:
        Series of volatility values
    """
    try:
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=period).std()
        
        if annualize:
            volatility = volatility * np.sqrt(252)  # Assuming daily data
            
        return volatility
    except Exception as e:
        raise IndicatorError(f"Failed to calculate volatility: {e}")


def calculate_correlation(series1: pd.Series, series2: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate rolling correlation between two series
    
    Args:
        series1: First series
        series2: Second series
        period: Rolling window period
        
    Returns:
        Series of correlation values
    """
    try:
        return series1.rolling(window=period).corr(series2)
    except Exception as e:
        raise IndicatorError(f"Failed to calculate correlation: {e}")


def calculate_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                       tenkan_period: int = 9, kijun_period: int = 26,
                       senkou_period: int = 52) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud indicators
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        tenkan_period: Tenkan-sen period
        kijun_period: Kijun-sen period
        senkou_period: Senkou Span B period
        
    Returns:
        DataFrame with Ichimoku components
    """
    try:
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=senkou_period).max()
        senkou_b_low = low.rolling(window=senkou_period).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span)
        chikou = close.shift(-kijun_period)
        
        return pd.DataFrame({
            'Tenkan': tenkan,
            'Kijun': kijun,
            'Senkou_A': senkou_a,
            'Senkou_B': senkou_b,
            'Chikou': chikou
        })
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Ichimoku: {e}")


def calculate_parabolic_sar(high: pd.Series, low: pd.Series, 
                           acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """
    Calculate Parabolic SAR
    
    Args:
        high: High prices
        low: Low prices
        acceleration: Acceleration factor
        maximum: Maximum acceleration factor
        
    Returns:
        Series of Parabolic SAR values
    """
    try:
        length = len(high)
        sar = np.zeros(length)
        ep = np.zeros(length)
        af = acceleration
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        # Initialize
        sar[0] = low[0] if trend == 1 else high[0]
        ep[0] = high[0] if trend == 1 else low[0]
        
        for i in range(1, length):
            if trend == 1:  # Uptrend
                sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
                
                if low[i] < sar[i]:  # Reversal
                    trend = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af = acceleration
                else:
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af = min(af + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        
            else:  # Downtrend
                sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
                
                if high[i] > sar[i]:  # Reversal
                    trend = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af = acceleration
                else:
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af = min(af + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
        
        return pd.Series(sar, index=high.index)
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Parabolic SAR: {e}")


def calculate_awesome_oscillator(high: pd.Series, low: pd.Series, 
                                fast: int = 5, slow: int = 34) -> pd.Series:
    """
    Calculate Awesome Oscillator
    """
    try:
        median = (high + low) / 2
        fast_sma = median.rolling(window=fast).mean()
        slow_sma = median.rolling(window=slow).mean()
        return fast_sma - slow_sma
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Awesome Oscillator: {e}")


def calculate_chaikin_money_flow(high: pd.Series, low: pd.Series, 
                                close: pd.Series, volume: pd.Series, 
                                period: int = 20) -> pd.Series:
    """
    Calculate Chaikin Money Flow
    """
    try:
        mfm = ((close - low) - (high - close)) / (high - low)
        mfv = mfm * volume
        return mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Chaikin Money Flow: {e}")


def calculate_donchian_channels(high: pd.Series, low: pd.Series, 
                               period: int = 20) -> pd.DataFrame:
    """
    Calculate Donchian Channels
    """
    try:
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        return pd.DataFrame({
            'Upper': upper,
            'Middle': middle,
            'Lower': lower
        })
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Donchian Channels: {e}")


def calculate_keltner_channels(high: pd.Series, low: pd.Series, 
                              close: pd.Series, period: int = 20, 
                              multiplier: float = 2.0) -> pd.DataFrame:
    """
    Calculate Keltner Channels
    """
    try:
        ema = close.ewm(span=period, adjust=False).mean()
        atr = calculate_atr(high, low, close, period)
        upper = ema + (atr * multiplier)
        lower = ema - (atr * multiplier)
        return pd.DataFrame({
            'Upper': upper,
            'Middle': ema,
            'Lower': lower
        })
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Keltner Channels: {e}")


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, 
                 period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index
    """
    try:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=low.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            '+DI': plus_di,
            '-DI': minus_di
        })
    except Exception as e:
        raise IndicatorError(f"Failed to calculate ADX: {e}")


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, 
                 period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index
    """
    try:
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return (tp - sma) / (0.015 * mad)
    except Exception as e:
        raise IndicatorError(f"Failed to calculate CCI: {e}")


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 14) -> pd.Series:
    """
    Calculate Williams %R
    """
    try:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Williams %R: {e}")


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume
    """
    try:
        obv = volume.copy()
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    except Exception as e:
        raise IndicatorError(f"Failed to calculate OBV: {e}")


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                 volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index
    """
    try:
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_sum / negative_sum.replace(0, np.nan)
        return 100 - (100 / (1 + money_ratio))
    except Exception as e:
        raise IndicatorError(f"Failed to calculate MFI: {e}")


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
                  volume: pd.Series, period: Optional[int] = None) -> pd.Series:
    """
    Calculate Volume Weighted Average Price
    """
    try:
        typical_price = (high + low + close) / 3
        
        if period is None:
            cum_vol_price = (typical_price * volume).cumsum()
            cum_volume = volume.cumsum()
            vwap = cum_vol_price / cum_volume
        else:
            cum_vol_price = (typical_price * volume).rolling(window=period).sum()
            cum_volume = volume.rolling(window=period).sum()
            vwap = cum_vol_price / cum_volume
        
        return vwap
    except Exception as e:
        raise IndicatorError(f"Failed to calculate VWAP: {e}")


def calculate_rolling_stats(prices: pd.Series, period: int = 20) -> pd.DataFrame:
    """
    Calculate rolling statistics
    """
    try:
        result = pd.DataFrame()
        result['mean'] = prices.rolling(window=period).mean()
        result['std'] = prices.rolling(window=period).std()
        result['min'] = prices.rolling(window=period).min()
        result['max'] = prices.rolling(window=period).max()
        result['skew'] = prices.rolling(window=period).skew()
        result['kurt'] = prices.rolling(window=period).kurt()
        return result
    except Exception as e:
        raise IndicatorError(f"Failed to calculate rolling stats: {e}")


def calculate_z_score(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Z-Score
    """
    try:
        mean = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return (prices - mean) / std.replace(0, np.nan)
    except Exception as e:
        raise IndicatorError(f"Failed to calculate Z-Score: {e}")


# =============================================================================
# DECORATORS
# =============================================================================

class decorators:
    """Collection of utility decorators"""
    
    @staticmethod
    def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0,
                          backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
        """
        Retry decorator with exponential backoff
        
        Args:
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            backoff_factor: Multiplier for delay after each retry
            exceptions: Tuple of exceptions to catch
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                delay = initial_delay
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_retries:
                            raise
                        time.sleep(delay)
                        delay *= backoff_factor
                return None
            return wrapper
        return decorator
    
    @staticmethod
    def memoize(ttl: Optional[int] = None):
        """
        Memoization decorator with optional TTL
        
        Args:
            ttl: Time to live in seconds (None for infinite)
        """
        def decorator(func):
            cache = {}
            timestamps = {}
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = hashlib.md5(
                    str((args, kwargs)).encode()
                ).hexdigest()
                
                now = time.time()
                
                if key in cache:
                    if ttl is None or (now - timestamps[key]) < ttl:
                        return cache[key]
                
                result = func(*args, **kwargs)
                cache[key] = result
                timestamps[key] = now
                
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def timing(func):
        """Measure and log function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logging.debug(f"{func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def singleton(cls):
        """Singleton decorator for classes"""
        instances = {}
        
        @wraps(cls)
        def get_instance(*args, **kwargs):
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
        
        return get_instance
    
    @staticmethod
    def rate_limit(max_calls: int, period: float):
        """
        Rate limiting decorator
        
        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        def decorator(func):
            calls = []
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                now = time.time()
                
                # Remove old calls
                calls[:] = [call for call in calls if call > now - period]
                
                if len(calls) >= max_calls:
                    wait_time = calls[0] + period - now
                    if wait_time > 0:
                        time.sleep(wait_time)
                
                calls.append(now)
                return func(*args, **kwargs)
            return wrapper
        return decorator


# =============================================================================
# EXCEPTIONS
# =============================================================================

class UtilsError(Exception):
    """Base exception for utils module"""
    pass


class DataError(UtilsError):
    """Exception raised for data-related errors"""
    pass


class IndicatorError(UtilsError):
    """Exception raised for indicator calculation errors"""
    pass


class ConfigurationError(UtilsError):
    """Exception raised for configuration errors"""
    pass


class ValidationError(UtilsError):
    """Exception raised for validation errors"""
    pass


class CalculationError(UtilsError):
    """Exception raised for calculation errors"""
    pass


# =============================================================================
# STATISTICS FUNCTIONS
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
        raise CalculationError(f"Failed to calculate Sharpe ratio: {e}")


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
        raise CalculationError(f"Failed to calculate Sortino ratio: {e}")


def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related statistics
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Dictionary with drawdown statistics
    """
    try:
        rolling_max = equity_curve.expanding().max()
        drawdown = (rolling_max - equity_curve) / rolling_max
        
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
            'max_drawdown': max_drawdown,
            'max_drawdown_start': max_drawdown_start,
            'max_drawdown_end': max_drawdown_end,
            'max_drawdown_duration': duration
        }
    except Exception as e:
        raise CalculationError(f"Failed to calculate max drawdown: {e}")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

class ValidationResult:
    """Result of validation operation"""
    
    def __init__(self, is_valid: bool = True, errors: List[str] = None,
                 warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result"""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def __str__(self) -> str:
        parts = []
        if not self.is_valid:
            parts.append(f"Errors: {', '.join(self.errors)}")
        if self.warnings:
            parts.append(f"Warnings: {', '.join(self.warnings)}")
        return ' | '.join(parts)


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1,
                      required_columns: List[str] = None) -> ValidationResult:
    """
    Validate DataFrame
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        required_columns: List of required column names
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    if not isinstance(df, pd.DataFrame):
        result.add_error(f"Expected DataFrame, got {type(df)}")
        return result
    
    if len(df) < min_rows:
        result.add_error(f"DataFrame has {len(df)} rows, need at least {min_rows}")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            result.add_error(f"Missing columns: {missing}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isin([np.inf, -np.inf]).any():
            result.add_warning(f"Column {col} contains infinite values")
    
    return result


def validate_positive(value: float, name: str = "value") -> ValidationResult:
    """Validate that a value is positive"""
    result = ValidationResult()
    
    if not isinstance(value, (int, float)):
        result.add_error(f"{name} must be a number, got {type(value)}")
    elif value <= 0:
        result.add_error(f"{name} must be positive, got {value}")
    
    return result


def validate_range(value: float, min_val: float, max_val: float,
                  name: str = "value") -> ValidationResult:
    """Validate that a value is within range"""
    result = ValidationResult()
    
    if not isinstance(value, (int, float)):
        result.add_error(f"{name} must be a number, got {type(value)}")
    elif value < min_val or value > max_val:
        result.add_error(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    return result


def validate_symbol(symbol: str) -> ValidationResult:
    """Validate trading symbol"""
    result = ValidationResult()
    
    if not isinstance(symbol, str):
        result.add_error(f"Symbol must be string, got {type(symbol)}")
    elif len(symbol) < 2:
        result.add_error(f"Symbol too short: {symbol}")
    elif not symbol.replace('/', '').isalnum():
        result.add_warning(f"Symbol contains non-alphanumeric characters: {symbol}")
    
    return result


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def normalize_data(data: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize data using various methods
    
    Args:
        data: Series to normalize
        method: 'minmax', 'zscore', 'robust', 'maxabs'
        
    Returns:
        Normalized series
    """
    try:
        if method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'robust':
            median = data.median()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            return (data - median) / iqr
        elif method == 'maxabs':
            return data / data.abs().max()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    except Exception as e:
        raise CalculationError(f"Failed to normalize data: {e}")


def winsorize_data(data: pd.Series, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    """
    Winsorize data (cap extreme values)
    
    Args:
        data: Series to winsorize
        limits: Tuple of (lower_percentile, upper_percentile)
        
    Returns:
        Winsorized series
    """
    try:
        lower_limit = data.quantile(limits[0])
        upper_limit = data.quantile(1 - limits[1])
        
        return data.clip(lower_limit, upper_limit)
    except Exception as e:
        raise CalculationError(f"Failed to winsorize data: {e}")


def create_lagged_features(data: pd.Series, lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features from series
    
    Args:
        data: Input series
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features
    """
    try:
        result = pd.DataFrame(index=data.index)
        
        for lag in lags:
            result[f'lag_{lag}'] = data.shift(lag)
        
        return result
    except Exception as e:
        raise CalculationError(f"Failed to create lagged features: {e}")


def create_rolling_features(data: pd.Series, windows: List[int],
                           functions: List[str] = None) -> pd.DataFrame:
    """
    Create rolling window features
    
    Args:
        data: Input series
        windows: List of window sizes
        functions: List of functions to apply ('mean', 'std', 'min', 'max', 'skew', 'kurt')
        
    Returns:
        DataFrame with rolling features
    """
    if functions is None:
        functions = ['mean', 'std']
    
    try:
        result = pd.DataFrame(index=data.index)
        
        for window in windows:
            rolling = data.rolling(window=window)
            
            if 'mean' in functions:
                result[f'rolling_mean_{window}'] = rolling.mean()
            if 'std' in functions:
                result[f'rolling_std_{window}'] = rolling.std()
            if 'min' in functions:
                result[f'rolling_min_{window}'] = rolling.min()
            if 'max' in functions:
                result[f'rolling_max_{window}'] = rolling.max()
            if 'skew' in functions:
                result[f'rolling_skew_{window}'] = rolling.skew()
            if 'kurt' in functions:
                result[f'rolling_kurt_{window}'] = rolling.kurt()
        
        return result
    except Exception as e:
        raise CalculationError(f"Failed to create rolling features: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_path: Path to config file (JSON or YAML)
        
    Returns:
        Configuration dictionary
    """
    try:
        path = Path(config_path)
        
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix == '.json':
                return json.load(f)
            else:
                raise ConfigurationError(f"Unsupported config format: {path.suffix}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load config: {e}")


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False)
            elif path.suffix == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported config format: {path.suffix}")
    except Exception as e:
        raise ConfigurationError(f"Failed to save config: {e}")


def create_directories(dirs: List[Union[str, Path]]):
    """
    Create directories if they don't exist
    
    Args:
        dirs: List of directory paths
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def format_currency(value: float, symbol: str = '$', decimals: int = 2) -> str:
    """
    Format value as currency
    
    Args:
        value: Numeric value
        symbol: Currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    try:
        if pd.isna(value) or np.isinf(value):
            return f"{symbol}NaN"
        
        rounded = Decimal(str(value)).quantize(
            Decimal(f"1e-{decimals}"), rounding=ROUND_HALF_UP
        )
        
        return f"{symbol}{rounded:,.{decimals}f}"
    except:
        return f"{symbol}{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Numeric value (0.15 = 15%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    try:
        if pd.isna(value) or np.isinf(value):
            return "NaN%"
        
        return f"{value * 100:.{decimals}f}%"
    except:
        return f"{value * 100:.{decimals}f}%"


def calculate_position_size(account_balance: float, risk_percent: float,
                           stop_loss_pips: float, pip_value: float) -> float:
    """
    Calculate position size based on risk
    
    Args:
        account_balance: Account balance
        risk_percent: Risk percentage per trade (0.01 = 1%)
        stop_loss_pips: Stop loss in pips
        pip_value: Value per pip
        
    Returns:
        Position size in units
    """
    risk_amount = account_balance * risk_percent
    position_size = risk_amount / (stop_loss_pips * pip_value)
    
    return max(0, position_size)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is 0
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        return numerator / denominator
    except:
        return default


def round_to_tick(value: float, tick_size: float) -> float:
    """
    Round value to nearest tick size
    
    Args:
        value: Value to round
        tick_size: Tick size
        
    Returns:
        Rounded value
    """
    return round(value / tick_size) * tick_size


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def parse_timedelta(time_str: str) -> timedelta:
    """
    Parse time string to timedelta
    
    Args:
        time_str: Time string (e.g., '1d', '2h', '30m', '15s')
        
    Returns:
        timedelta object
    """
    units = {
        'd': 'days',
        'h': 'hours',
        'm': 'minutes',
        's': 'seconds',
        'w': 'weeks'
    }
    
    try:
        value = int(time_str[:-1])
        unit = time_str[-1].lower()
        
        if unit not in units:
            raise ValueError(f"Unknown unit: {unit}")
        
        return timedelta(**{units[unit]: value})
    except Exception as e:
        raise ValueError(f"Failed to parse timedelta '{time_str}': {e}")


def get_date_range(start_date: Union[str, datetime], 
                  end_date: Union[str, datetime],
                  freq: str = 'D') -> List[datetime]:
    """
    Generate date range
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency ('D', 'H', '15T', etc.)
        
    Returns:
        List of dates
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    return pd.date_range(start=start_date, end=end_date, freq=freq).tolist()


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager for timing code blocks
    
    Args:
        name: Name of the operation
    """
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        logging.info(f"{name} took {end - start:.4f} seconds")


@contextmanager
def suppress_output():
    """Context manager to suppress stdout/stderr"""
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextmanager
def change_directory(path: Union[str, Path]):
    """
    Context manager to temporarily change directory
    
    Args:
        path: Directory path
    """
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Logging
    'setup_logging',
    
    # Decorators
    'decorators',
    'retry_with_backoff',
    'memoize',
    'timing',
    'singleton',
    'rate_limit',
    
    # Exceptions
    'UtilsError',
    'DataError',
    'IndicatorError',
    'ConfigurationError',
    'ValidationError',
    'CalculationError',
    
    # Technical Indicators
    'calculate_rsi',
    'calculate_atr',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_stochastic',
    'calculate_momentum',
    'calculate_volume_profile',
    'calculate_structure_features',
    'calculate_volatility',
    'calculate_correlation',
    'calculate_ichimoku',
    'calculate_parabolic_sar',
    'calculate_awesome_oscillator',
    'calculate_chaikin_money_flow',
    'calculate_donchian_channels',
    'calculate_keltner_channels',
    'calculate_adx',
    'calculate_cci',
    'calculate_williams_r',
    'calculate_obv',
    'calculate_mfi',
    'calculate_vwap',
    'calculate_rolling_stats',
    'calculate_z_score',
    
    # Data Loading
    'load_csv_data',
    'save_csv_data',
    'load_parquet_data',
    'save_parquet_data',
    'load_hdf5_data',
    'save_hdf5_data',
    'load_from_multiple_sources',
    'load_from_database',
    'save_to_database',
    
    # Data Processing
    'resample_data',
    'clean_data',
    'filter_time_range',
    'detect_outliers',
    'fill_missing_values',
    'merge_dataframes',
    
    # Statistics
    'calculate_return_statistics',
    'calculate_drawdown',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_omega_ratio',
    'calculate_profit_factor',
    'calculate_expectancy',
    'calculate_r_multiple_distribution',
    'calculate_rolling_sharpe',
    'calculate_drawdown_statistics',
    'calculate_win_loss_statistics',
    'calculate_risk_metrics',
    'calculate_vaR',
    'calculate_cVaR',
    'calculate_tail_ratio',
    'calculate_gain_to_pain_ratio',
    'calculate_recovery_factor',
    'calculate_ulcer_index',
    'calculate_upi_index',
    
    # Feature Engineering
    'calculate_time_based_features',
    'create_lagged_features',
    'create_rolling_features',
    'create_expanding_features',
    
    # Data Transformation
    'normalize_data',
    'standardize_data',
    'winsorize_data',
    'trim_outliers',
    'winsorize_series',
    'boxcox_transform',
    'log_transform',
    'difference_transform',
    'percentage_change',
    'rolling_normalize',
    'rolling_standardize',
    
    # Validation
    'validate_dataframe',
    'validate_columns',
    'validate_positive',
    'validate_non_negative',
    'validate_range',
    'validate_in_list',
    'validate_datetime_index',
    'validate_frequency',
    'validate_trade_parameters',
    'validate_risk_parameters',
    'validate_symbol',
    'ValidationResult',
    
    # Helpers
    'load_config',
    'save_config',
    'create_directories',
    'format_currency',
    'format_percentage',
    'format_timestamp',
    'calculate_pips',
    'get_pip_multiplier',
    'calculate_point_value',
    'normalize_symbol',
    'get_instrument_type',
    'safe_divide',
    'round_to_tick',
    'clamp',
    'calculate_risk_reward',
    'calculate_position_size',
    'time_to_next_interval',
    'get_interval_minutes',
    'flatten_dict',
    'nested_get',
    'nested_set',
    'deep_merge',
    'parse_timedelta',
    'format_timedelta',
    'get_date_range',
    'is_market_open',
    'get_next_market_open',
    'get_previous_market_close',
    
    # Context Managers
    'timer',
    'suppress_output',
    'change_directory',
    
    # Version
    '__version__',
    '__author__'
]