"""
Transform - Data transformation and preprocessing utilities
Comprehensive data transformation functions for feature engineering and analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any, Callable
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# =============================================================================
# Normalization and Standardization
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
        logger.error(f"Failed to normalize data: {e}")
        return data


def standardize_data(data: pd.Series) -> pd.Series:
    """
    Standardize data to have mean 0 and standard deviation 1
    
    Args:
        data: Series to standardize
    
    Returns:
        Standardized series
    """
    return normalize_data(data, 'zscore')


def rolling_normalize(data: pd.Series, window: int = 20,
                     method: str = 'minmax') -> pd.Series:
    """
    Rolling normalization
    
    Args:
        data: Series to normalize
        window: Rolling window size
        method: Normalization method
    
    Returns:
        Series of normalized values
    """
    try:
        return data.rolling(window=window).apply(
            lambda x: normalize_data(x, method)
        )
    except Exception as e:
        logger.error(f"Failed to calculate rolling normalization: {e}")
        return pd.Series()


def rolling_standardize(data: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling standardization
    
    Args:
        data: Series to standardize
        window: Rolling window size
    
    Returns:
        Series of standardized values
    """
    return rolling_normalize(data, window, 'zscore')


# =============================================================================
# Outlier Handling
# =============================================================================

def winsorize_data(data: pd.Series, limits: tuple = (0.05, 0.05)) -> pd.Series:
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
        logger.error(f"Failed to winsorize data: {e}")
        return data


def winsorize_series(data: pd.Series, limits: tuple = (0.05, 0.05)) -> pd.Series:
    """
    Alias for winsorize_data
    """
    return winsorize_data(data, limits)


def trim_outliers(data: pd.Series, method: str = 'iqr',
                  threshold: float = 3.0) -> pd.Series:
    """
    Trim outliers from data
    
    Args:
        data: Series to trim
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
    
    Returns:
        Series with outliers removed
    """
    try:
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            lower_bound = data.mean() - threshold * data.std()
            upper_bound = data.mean() + threshold * data.std()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return data[(data >= lower_bound) & (data <= upper_bound)]
    except Exception as e:
        logger.error(f"Failed to trim outliers: {e}")
        return data


# =============================================================================
# Mathematical Transformations
# =============================================================================

def log_transform(data: pd.Series) -> pd.Series:
    """
    Log transform data
    
    Args:
        data: Series to transform
    
    Returns:
        Log-transformed series
    """
    try:
        return np.log(data)
    except Exception as e:
        logger.error(f"Failed to log transform: {e}")
        return data


def boxcox_transform(data: pd.Series, lmbda: Optional[float] = None) -> pd.Series:
    """
    Box-Cox transform data
    
    Args:
        data: Series to transform
        lmbda: Lambda parameter (None for automatic calculation)
    
    Returns:
        Box-Cox transformed series
    """
    try:
        if lmbda is None:
            transformed, lmbda = stats.boxcox(data)
            return pd.Series(transformed, index=data.index)
        else:
            return pd.Series(stats.boxcox(data, lmbda), index=data.index)
    except Exception as e:
        logger.error(f"Failed to Box-Cox transform: {e}")
        return data


def difference_transform(data: pd.Series, periods: int = 1) -> pd.Series:
    """
    Difference transform data
    
    Args:
        data: Series to transform
        periods: Number of periods to difference
    
    Returns:
        Differenced series
    """
    try:
        return data.diff(periods)
    except Exception as e:
        logger.error(f"Failed to difference transform: {e}")
        return data


def percentage_change(data: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate percentage change
    
    Args:
        data: Series to transform
        periods: Number of periods to compare
    
    Returns:
        Percentage change series
    """
    try:
        return data.pct_change(periods) * 100
    except Exception as e:
        logger.error(f"Failed to calculate percentage change: {e}")
        return data


# =============================================================================
# Feature Engineering
# =============================================================================

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
        logger.error(f"Failed to create lagged features: {e}")
        return pd.DataFrame()


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
        logger.error(f"Failed to create rolling features: {e}")
        return pd.DataFrame()


def create_expanding_features(data: pd.Series, functions: List[str] = None) -> pd.DataFrame:
    """
    Create expanding window features
    
    Args:
        data: Input series
        functions: List of functions to apply ('mean', 'std', 'min', 'max', 'skew', 'kurt')
    
    Returns:
        DataFrame with expanding features
    """
    if functions is None:
        functions = ['mean', 'std']
    
    try:
        result = pd.DataFrame(index=data.index)
        expanding = data.expanding()
        
        if 'mean' in functions:
            result['expanding_mean'] = expanding.mean()
        if 'std' in functions:
            result['expanding_std'] = expanding.std()
        if 'min' in functions:
            result['expanding_min'] = expanding.min()
        if 'max' in functions:
            result['expanding_max'] = expanding.max()
        if 'skew' in functions:
            result['expanding_skew'] = expanding.skew()
        if 'kurt' in functions:
            result['expanding_kurt'] = expanding.kurt()
        
        return result
    except Exception as e:
        logger.error(f"Failed to create expanding features: {e}")
        return pd.DataFrame()


# =============================================================================
# Seasonal and Cyclical Transformations
# =============================================================================

def create_cyclical_features(data: pd.Series, period: int = 24,
                           normalize: bool = True) -> pd.DataFrame:
    """
    Create cyclical features from time series
    
    Args:
        data: Input series
        period: Cycle period
        normalize: Whether to normalize values
    
    Returns:
        DataFrame with sin and cos components
    """
    try:
        result = pd.DataFrame(index=data.index)
        
        if isinstance(data.index, pd.DatetimeIndex):
            # If datetime index, use hour as base for daily cycle
            x = data.index.hour
        else:
            # If numeric index, use as-is
            x = data.index
        
        sin_component = np.sin(2 * np.pi * x / period)
        cos_component = np.cos(2 * np.pi * x / period)
        
        if normalize:
            sin_component = (sin_component - sin_component.min()) / (sin_component.max() - sin_component.min())
            cos_component = (cos_component - cos_component.min()) / (cos_component.max() - cos_component.min())
        
        result['sin_component'] = sin_component
        result['cos_component'] = cos_component
        
        return result
    except Exception as e:
        logger.error(f"Failed to create cyclical features: {e}")
        return pd.DataFrame()


# =============================================================================
# Technical Indicator Transformations
# =============================================================================

def calculate_technical_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calculate technical indicator features
    
    Args:
        df: DataFrame with OHLC data
        features: List of features to calculate
    
    Returns:
        DataFrame with technical features
    """
    try:
        from .indicators import (
            calculate_rsi,
            calculate_macd,
            calculate_bollinger_bands,
            calculate_stochastic,
            calculate_atr
        )
        
        result = pd.DataFrame(index=df.index)
        
        for feature in features:
            if feature == 'rsi':
                result['rsi'] = calculate_rsi(df['close'])
            elif feature == 'macd':
                macd_result = calculate_macd(df['close'])
                result['macd'] = macd_result['MACD']
                result['macd_signal'] = macd_result['Signal']
                result['macd_hist'] = macd_result['Histogram']
            elif feature == 'bollinger':
                bollinger_result = calculate_bollinger_bands(df['close'])
                result['bollinger_mid'] = bollinger_result['Middle']
                result['bollinger_upper'] = bollinger_result['Upper']
                result['bollinger_lower'] = bollinger_result['Lower']
                result['bollinger_width'] = (bollinger_result['Upper'] - bollinger_result['Lower']) / bollinger_result['Middle']
            elif feature == 'stochastic':
                stochastic_result = calculate_stochastic(df['high'], df['low'], df['close'])
                result['stoch_k'] = stochastic_result['%K']
                result['stoch_d'] = stochastic_result['%D']
            elif feature == 'atr':
                result['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        
        return result
    except Exception as e:
        logger.error(f"Failed to calculate technical features: {e}")
        return pd.DataFrame()


# =============================================================================
# Feature Scaling
# =============================================================================

def scale_features(df: pd.DataFrame, scaler: Callable = None) -> pd.DataFrame:
    """
    Scale features using specified scaler
    
    Args:
        df: DataFrame to scale
        scaler: Scaler instance (default: StandardScaler)
    
    Returns:
        Scaled DataFrame
    """
    try:
        if scaler is None:
            scaler = StandardScaler()
        
        scaled_data = scaler.fit_transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    except Exception as e:
        logger.error(f"Failed to scale features: {e}")
        return df


def minmax_scale_features(df: pd.DataFrame, feature_range: tuple = (0, 1)) -> pd.DataFrame:
    """
    Min-max scale features
    
    Args:
        df: DataFrame to scale
        feature_range: Target range
    
    Returns:
        Scaled DataFrame
    """
    try:
        scaler = MinMaxScaler(feature_range=feature_range)
        return scale_features(df, scaler)
    except Exception as e:
        logger.error(f"Failed to min-max scale features: {e}")
        return df


# =============================================================================
# Data Cleaning Transformations
# =============================================================================

def fill_missing_values(df: pd.DataFrame, method: str = 'ffill',
                      limit: int = None) -> pd.DataFrame:
    """
    Fill missing values
    
    Args:
        df: DataFrame to process
        method: 'ffill', 'bfill', 'interpolate', 'mean', 'median'
        limit: Maximum number of consecutive NaNs to fill
    
    Returns:
        DataFrame with filled values
    """
    try:
        df_filled = df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                if method == 'mean':
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                elif method == 'median':
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
                elif method == 'interpolate':
                    df_filled[col] = df_filled[col].interpolate(limit=limit)
                else:
                    df_filled[col] = df_filled[col].fillna(method=method, limit=limit)
        
        return df_filled
    except Exception as e:
        logger.error(f"Failed to fill missing values: {e}")
        return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows
    
    Args:
        df: DataFrame to process
    
    Returns:
        DataFrame without duplicates
    """
    try:
        return df[~df.duplicated()]
    except Exception as e:
        logger.error(f"Failed to remove duplicates: {e}")
        return df


# =============================================================================
# Time Series Transformations
# =============================================================================

def resample_time_series(df: pd.DataFrame, rule: str,
                        agg_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Resample time series data
    
    Args:
        df: DataFrame with datetime index
        rule: Resampling rule
        agg_dict: Aggregation dictionary for columns
    
    Returns:
        Resampled DataFrame
    """
    try:
        if agg_dict is None:
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        
        # Filter to only include existing columns
        existing_cols = [col for col in agg_dict if col in df.columns]
        agg_dict = {col: agg_dict[col] for col in existing_cols}
        
        return df.resample(rule).agg(agg_dict)
    except Exception as e:
        logger.error(f"Failed to resample time series: {e}")
        return df


def align_time_series(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """
    Align two time series
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
    
    Returns:
        Tuple of aligned DataFrames
    """
    try:
        common_index = df1.index.intersection(df2.index)
        return df1.loc[common_index], df2.loc[common_index]
    except Exception as e:
        logger.error(f"Failed to align time series: {e}")
        return df1, df2


__all__ = [
    'normalize_data',
    'standardize_data',
    'rolling_normalize',
    'rolling_standardize',
    'winsorize_data',
    'winsorize_series',
    'trim_outliers',
    'log_transform',
    'boxcox_transform',
    'difference_transform',
    'percentage_change',
    'create_lagged_features',
    'create_rolling_features',
    'create_expanding_features',
    'create_cyclical_features',
    'calculate_technical_features',
    'scale_features',
    'minmax_scale_features',
    'fill_missing_values',
    'remove_duplicates',
    'resample_time_series',
    'align_time_series'
]
