"""
Data Loader - Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def load_csv_data(filepath: Union[str, Path], date_column: str = 'timestamp', 
                 date_format: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        date_format: Date format string
    
    Returns:
        DataFrame with datetime index
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        
        if date_column in df.columns:
            if date_format:
                df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            else:
                df[date_column] = pd.to_datetime(df[date_column])
            
            df.set_index(date_column, inplace=True)
            df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


def save_csv_data(df: pd.DataFrame, filepath: Union[str, Path], 
                 date_format: str = '%Y-%m-%d %H:%M:%S') -> bool:
    """
    Save data to CSV file
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        date_format: Date format for index
    
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df_copy = df.copy()
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = df_copy.index.strftime(date_format)
        
        df_copy.to_csv(filepath)
        logger.info(f"Saved {len(df_copy)} rows to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        return False


def resample_data(df: pd.DataFrame, rule: str, agg_dict: Optional[Dict] = None) -> pd.DataFrame:
    """
    Resample data to different timeframe
    
    Args:
        df: DataFrame with datetime index
        rule: Resampling rule (e.g., '5T', '1H', '1D')
        agg_dict: Aggregation dictionary for each column
    
    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame index must be DatetimeIndex")
        return df
    
    if agg_dict is None:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    try:
        resampled = df.resample(rule).agg(agg_dict)
        resampled.dropna(inplace=True)
        logger.info(f"Resampled from {len(df)} to {len(resampled)} rows")
        return resampled
        
    except Exception as e:
        logger.error(f"Error resampling data: {e}")
        return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by removing duplicates and handling missing values
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove duplicates
    df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
    
    # Sort index
    df_clean.sort_index(inplace=True)
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            # Forward fill then backward fill for numeric columns
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"Cleaned data: {len(df_clean)} rows remaining")
    return df_clean


def filter_time_range(df: pd.DataFrame, start: Union[str, datetime], 
                      end: Union[str, datetime]) -> pd.DataFrame:
    """
    Filter data to specific time range
    
    Args:
        df: DataFrame with datetime index
        start: Start time
        end: End time
    
    Returns:
        Filtered DataFrame
    """
    if isinstance(start, str):
        start = pd.to_datetime(start)
    if isinstance(end, str):
        end = pd.to_datetime(end)
    
    mask = (df.index >= start) & (df.index <= end)
    filtered = df[mask]
    
    logger.info(f"Filtered to {len(filtered)} rows between {start} and {end}")
    return filtered


def detect_outliers(df: pd.DataFrame, column: str, n_std: float = 3.0) -> pd.Series:
    """
    Detect outliers using standard deviation method
    
    Args:
        df: DataFrame
        column: Column name to check
        n_std: Number of standard deviations
    
    Returns:
        Boolean series where True indicates outlier
    """
    mean = df[column].mean()
    std = df[column].std()
    
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    logger.info(f"Detected {outliers.sum()} outliers in {column}")
    return outliers


def fill_missing_values(df: pd.DataFrame, method: str = 'ffill', 
                        limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame
    
    Args:
        df: DataFrame
        method: Fill method ('ffill', 'bfill', 'interpolate')
        limit: Maximum number of consecutive NaN to fill
    
    Returns:
        DataFrame with filled values
    """
    df_filled = df.copy()
    
    if method == 'ffill':
        df_filled = df_filled.fillna(method='ffill', limit=limit)
        df_filled = df_filled.fillna(method='bfill', limit=limit)
    elif method == 'bfill':
        df_filled = df_filled.fillna(method='bfill', limit=limit)
        df_filled = df_filled.fillna(method='ffill', limit=limit)
    elif method == 'interpolate':
        df_filled = df_filled.interpolate(method='linear', limit=limit, limit_direction='both')
    
    return df_filled


def calculate_return_statistics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate return statistics
    
    Args:
        returns: Series of returns
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'mean': returns.mean(),
        'std': returns.std(),
        'skew': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'min': returns.min(),
        'max': returns.max(),
        'positive_pct': (returns > 0).mean() * 100,
        'negative_pct': (returns < 0).mean() * 100,
        'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        'var_95': returns.quantile(0.05),
        'var_99': returns.quantile(0.01),
        'cvar_95': returns[returns <= returns.quantile(0.05)].mean()
    }
    
    return stats


def calculate_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate drawdown statistics
    
    Args:
        equity_curve: Series of equity values
    
    Returns:
        Dictionary with drawdown statistics
    """
    peak = equity_curve.expanding().max()
    drawdown = (peak - equity_curve) / peak
    
    stats = {
        'max_drawdown': drawdown.max() * 100,
        'avg_drawdown': drawdown.mean() * 100,
        'current_drawdown': drawdown.iloc[-1] * 100,
        'max_drawdown_duration': _calculate_drawdown_duration(drawdown),
        'drawdown_std': drawdown.std() * 100
    }
    
    return stats


def _calculate_drawdown_duration(drawdown: pd.Series) -> int:
    """Calculate maximum drawdown duration in periods"""
    in_drawdown = False
    current_duration = 0
    max_duration = 0
    
    for value in drawdown:
        if value > 0:
            if not in_drawdown:
                in_drawdown = True
                current_duration = 1
            else:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
        else:
            in_drawdown = False
            current_duration = 0
    
    return max_duration


def calculate_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based features from datetime index
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with additional time features
    """
    features = pd.DataFrame(index=df.index)
    
    features['hour'] = df.index.hour
    features['minute'] = df.index.minute
    features['day_of_week'] = df.index.dayofweek
    features['day_of_month'] = df.index.day
    features['week_of_year'] = df.index.isocalendar().week
    features['month'] = df.index.month
    features['quarter'] = df.index.quarter
    features['year'] = df.index.year
    
    # Session indicators
    features['is_asia'] = ((features['hour'] >= 0) & (features['hour'] < 9)).astype(int)
    features['is_london'] = ((features['hour'] >= 8) & (features['hour'] < 17)).astype(int)
    features['is_ny'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)
    features['is_overlap'] = ((features['hour'] >= 13) & (features['hour'] < 17)).astype(int)
    
    return features


def merge_dataframes(dfs: List[pd.DataFrame], how: str = 'inner', 
                    on: Optional[str] = None) -> pd.DataFrame:
    """
    Merge multiple DataFrames
    
    Args:
        dfs: List of DataFrames to merge
        how: Merge method ('inner', 'outer', 'left', 'right')
        on: Column to merge on (if None, merge on index)
    
    Returns:
        Merged DataFrame
    """
    if len(dfs) == 0:
        return pd.DataFrame()
    
    if len(dfs) == 1:
        return dfs[0]
    
    result = dfs[0]
    
    for df in dfs[1:]:
        if on:
            result = pd.merge(result, df, on=on, how=how)
        else:
            result = pd.merge(result, df, left_index=True, right_index=True, how=how)
    
    logger.info(f"Merged {len(dfs)} DataFrames into {len(result)} rows")
    return result