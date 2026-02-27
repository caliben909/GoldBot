"""
Data Loader - Data loading and preprocessing utilities
Comprehensive data handling with support for multiple formats, validation,
and advanced preprocessing features.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
import json
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import time
import gc
from contextlib import contextmanager
import re

# Optional imports for additional formats
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import sqlalchemy
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False

try:
    import pymongo
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class DataLoaderError(Exception):
    """Base exception for data loader"""
    pass


class DataNotFoundError(DataLoaderError):
    """Raised when data file is not found"""
    pass


class DataFormatError(DataLoaderError):
    """Raised when data format is invalid"""
    pass


class DataValidationError(DataLoaderError):
    """Raised when data validation fails"""
    pass


class DataProcessingError(DataLoaderError):
    """Raised when data processing fails"""
    pass


# =============================================================================
# Configuration and Constants
# =============================================================================

class DataLoaderConfig:
    """Configuration for data loader"""
    
    DEFAULT_AGG_DICT = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'bid': 'last',
        'ask': 'last',
        'spread': 'last'
    }
    
    SESSION_HOURS = {
        'asia': (0, 9),
        'london': (8, 17),
        'ny': (13, 22),
        'overlap': (13, 17)
    }
    
    SUPPORTED_FORMATS = ['csv', 'parquet', 'hdf5', 'feather', 'pickle', 'json', 'excel']
    
    @classmethod
    def get_session_hours(cls, session: str) -> Tuple[int, int]:
        """Get session hours"""
        return cls.SESSION_HOURS.get(session.lower(), (0, 24))


# =============================================================================
# Decorators
# =============================================================================

def handle_data_errors(func: Callable) -> Callable:
    """Decorator for handling data loader errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise DataNotFoundError(f"File not found: {e}")
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data: {e}")
            raise DataFormatError(f"Empty data: {e}")
        except Exception as e:
            logger.error(f"Data processing error in {func.__name__}: {e}")
            raise DataProcessingError(f"Error in {func.__name__}: {e}")
    return wrapper


def cache_data(ttl: int = 3600):
    """Cache decorator for data loading functions"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key_parts = [str(arg) for arg in args] + [f"{k}:{v}" for k, v in sorted(kwargs.items())]
            key = hashlib.md5(''.join(key_parts).encode()).hexdigest()
            
            now = time.time()
            if key in cache:
                data, timestamp = cache[key]
                if now - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return data
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


# =============================================================================
# Data Loading Functions
# =============================================================================

@handle_data_errors
@cache_data(ttl=300)
@log_execution_time
def load_csv_data(filepath: Union[str, Path], 
                 date_column: str = 'timestamp',
                 date_format: Optional[str] = None,
                 parse_dates: bool = True,
                 columns: Optional[List[str]] = None,
                 dtype: Optional[Dict] = None,
                 encoding: str = 'utf-8',
                 chunksize: Optional[int] = None,
                 nrows: Optional[int] = None,
                 skiprows: Optional[List[int]] = None,
                 low_memory: bool = False) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
    """
    Load data from CSV file with advanced options
    
    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        date_format: Date format string
        parse_dates: Whether to parse dates
        columns: List of columns to load
        dtype: Dictionary of column data types
        encoding: File encoding
        chunksize: Number of rows per chunk for chunked reading
        nrows: Number of rows to read
        skiprows: Rows to skip
        low_memory: Internally process file in low memory
    
    Returns:
        DataFrame or TextFileReader if chunksize specified
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise DataNotFoundError(f"File not found: {filepath}")
    
    # Check file size for optimization
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    if file_size_mb > 1000 and not chunksize:
        logger.warning(f"Large file ({file_size_mb:.1f}MB). Consider using chunksize parameter.")
    
    try:
        # Read CSV with optimizations
        df = pd.read_csv(
            filepath,
            parse_dates=[date_column] if parse_dates and date_column else False,
            date_format=date_format,
            usecols=columns,
            dtype=dtype,
            encoding=encoding,
            chunksize=chunksize,
            nrows=nrows,
            skiprows=skiprows,
            low_memory=low_memory
        )
        
        # If chunksize is specified, return iterator
        if chunksize:
            return df
        
        # Set datetime index if column exists
        if date_column in df.columns and parse_dates:
            if date_column != df.index.name:
                df.set_index(date_column, inplace=True)
            df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(df)} rows from {filepath} ({file_size_mb:.1f}MB)")
        return df
        
    except Exception as e:
        raise DataFormatError(f"Error loading CSV: {e}")


@handle_data_errors
def load_parquet_data(filepath: Union[str, Path],
                     columns: Optional[List[str]] = None,
                     filters: Optional[List] = None) -> pd.DataFrame:
    """
    Load data from Parquet file (efficient columnar storage)
    
    Args:
        filepath: Path to Parquet file
        columns: List of columns to load
        filters: PyArrow filters for row filtering
    
    Returns:
        DataFrame
    """
    if not PARQUET_AVAILABLE:
        raise ImportError("pyarrow is required for Parquet support")
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise DataNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_parquet(filepath, columns=columns, filters=filters)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        raise DataFormatError(f"Error loading Parquet: {e}")


@handle_data_errors
def load_hdf5_data(filepath: Union[str, Path],
                  key: str = 'data',
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load data from HDF5 file
    
    Args:
        filepath: Path to HDF5 file
        key: Key in HDF5 file
        columns: List of columns to load
    
    Returns:
        DataFrame
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 support")
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise DataNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_hdf(filepath, key=key, columns=columns)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        raise DataFormatError(f"Error loading HDF5: {e}")


@handle_data_errors
def load_feather_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from Feather file (fast binary format)
    
    Args:
        filepath: Path to Feather file
    
    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise DataNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_feather(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        raise DataFormatError(f"Error loading Feather: {e}")


@handle_data_errors
def load_pickle_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from Pickle file
    
    Args:
        filepath: Path to Pickle file
    
    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise DataNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_pickle(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        raise DataFormatError(f"Error loading Pickle: {e}")


@handle_data_errors
def load_json_data(filepath: Union[str, Path],
                  orient: str = 'records',
                  date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        orient: JSON orientation
        date_column: Column to parse as datetime
    
    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise DataNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_json(filepath, orient=orient)
        
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        raise DataFormatError(f"Error loading JSON: {e}")


@handle_data_errors
def load_excel_data(filepath: Union[str, Path],
                   sheet_name: Union[str, int] = 0,
                   date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from Excel file
    
    Args:
        filepath: Path to Excel file
        sheet_name: Sheet name or index
        date_column: Column to parse as datetime
    
    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise DataNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        raise DataFormatError(f"Error loading Excel: {e}")


@handle_data_errors
def load_from_database(connection_string: str,
                      query: str,
                      params: Optional[Dict] = None,
                      date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from SQL database
    
    Args:
        connection_string: SQLAlchemy connection string
        query: SQL query
        params: Query parameters
        date_column: Column to use as datetime index
    
    Returns:
        DataFrame
    """
    if not SQL_AVAILABLE:
        raise ImportError("sqlalchemy is required for database support")
    
    try:
        engine = sqlalchemy.create_engine(connection_string)
        df = pd.read_sql(query, engine, params=params)
        
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        
        logger.info(f"Loaded {len(df)} rows from database")
        return df
    except Exception as e:
        raise DataProcessingError(f"Error loading from database: {e}")


def load_from_multiple_sources(filepaths: List[Union[str, Path]],
                              concat_axis: int = 0,
                              ignore_index: bool = True,
                              **kwargs) -> pd.DataFrame:
    """
    Load and concatenate data from multiple files
    
    Args:
        filepaths: List of file paths
        concat_axis: Axis to concatenate on (0 for rows, 1 for columns)
        ignore_index: Ignore index when concatenating
        **kwargs: Additional arguments for load functions
    
    Returns:
        Concatenated DataFrame
    """
    dfs = []
    
    for filepath in filepaths:
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue
        
        # Determine file type from extension
        suffix = filepath.suffix.lower()
        
        if suffix == '.csv':
            df = load_csv_data(filepath, **kwargs)
        elif suffix == '.parquet':
            df = load_parquet_data(filepath, **kwargs)
        elif suffix == '.h5':
            df = load_hdf5_data(filepath, **kwargs)
        elif suffix == '.feather':
            df = load_feather_data(filepath)
        elif suffix == '.pkl':
            df = load_pickle_data(filepath)
        elif suffix == '.json':
            df = load_json_data(filepath, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            df = load_excel_data(filepath, **kwargs)
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            continue
        
        dfs.append(df)
    
    if not dfs:
        raise DataNotFoundError("No valid data files found")
    
    result = pd.concat(dfs, axis=concat_axis, ignore_index=ignore_index)
    logger.info(f"Combined {len(dfs)} files into {len(result)} rows")
    
    return result


# =============================================================================
# Data Saving Functions
# =============================================================================

@handle_data_errors
def save_csv_data(df: pd.DataFrame, 
                 filepath: Union[str, Path],
                 date_format: str = '%Y-%m-%d %H:%M:%S',
                 index: bool = True,
                 compression: Optional[str] = None,
                 **kwargs) -> bool:
    """
    Save data to CSV file
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        date_format: Date format for index
        index: Whether to write index
        compression: Compression type ('gzip', 'bz2', 'zip')
        **kwargs: Additional arguments for to_csv
    
    Returns:
        True if successful
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df_copy = df.copy()
        
        if index and isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = df_copy.index.strftime(date_format)
        
        df_copy.to_csv(filepath, index=index, compression=compression, **kwargs)
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df_copy)} rows to {filepath} ({file_size_mb:.1f}MB)")
        return True
        
    except Exception as e:
        raise DataProcessingError(f"Error saving CSV: {e}")


@handle_data_errors
def save_parquet_data(df: pd.DataFrame,
                     filepath: Union[str, Path],
                     compression: str = 'snappy',
                     index: bool = True) -> bool:
    """
    Save data to Parquet file
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        compression: Compression algorithm
        index: Whether to write index
    
    Returns:
        True if successful
    """
    if not PARQUET_AVAILABLE:
        raise ImportError("pyarrow is required for Parquet support")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_parquet(filepath, compression=compression, index=index)
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df)} rows to {filepath} ({file_size_mb:.1f}MB)")
        return True
    except Exception as e:
        raise DataProcessingError(f"Error saving Parquet: {e}")


@handle_data_errors
def save_hdf5_data(df: pd.DataFrame,
                  filepath: Union[str, Path],
                  key: str = 'data',
                  mode: str = 'a',
                  complevel: int = 0,
                  complib: str = 'zlib') -> bool:
    """
    Save data to HDF5 file
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        key: Key in HDF5 file
        mode: File mode ('w', 'a')
        complevel: Compression level (0-9)
        complib: Compression library
    
    Returns:
        True if successful
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 support")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_hdf(filepath, key=key, mode=mode, complevel=complevel, complib=complib)
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df)} rows to {filepath} ({file_size_mb:.1f}MB)")
        return True
    except Exception as e:
        raise DataProcessingError(f"Error saving HDF5: {e}")


# =============================================================================
# Data Validation Functions
# =============================================================================

def validate_dataframe(df: pd.DataFrame,
                      required_columns: Optional[List[str]] = None,
                      min_rows: int = 1,
                      check_monotonic_index: bool = True,
                      check_time_gaps: bool = False,
                      max_gap_seconds: Optional[int] = None) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame for trading data requirements
    
    Args:
        df: DataFrame to validate
        required_columns: List of required columns
        min_rows: Minimum number of rows required
        check_monotonic_index: Check if index is monotonic increasing
        check_time_gaps: Check for time gaps
        max_gap_seconds: Maximum allowed gap in seconds
    
    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []
    
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    if len(df) < min_rows:
        errors.append(f"DataFrame has {len(df)} rows, minimum required is {min_rows}")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
    
    if check_monotonic_index and isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            errors.append("Index is not monotonic increasing")
    
    if check_time_gaps and isinstance(df.index, pd.DatetimeIndex) and max_gap_seconds:
        time_diffs = df.index.to_series().diff()
        max_gap = time_diffs.max()
        
        if pd.notna(max_gap) and max_gap.total_seconds() > max_gap_seconds:
            errors.append(f"Maximum time gap ({max_gap}) exceeds {max_gap_seconds} seconds")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isin([np.inf, -np.inf]).any():
            errors.append(f"Column '{col}' contains infinite values")
    
    return len(errors) == 0, errors


def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect data quality issues and generate report
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with quality metrics and issues
    """
    issues = {
        'missing_values': {},
        'outliers': {},
        'duplicates': 0,
        'gaps': None,
        'invalid_values': {}
    }
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col in missing[missing > 0].index:
        issues['missing_values'][col] = {
            'count': int(missing[col]),
            'percentage': float(missing_pct[col])
        }
    
    # Duplicates
    issues['duplicates'] = df.duplicated().sum()
    
    # Time gaps (if datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        time_diffs = df.index.to_series().diff()
        max_gap = time_diffs.max()
        if pd.notna(max_gap):
            expected_freq = pd.infer_freq(df.index)
            issues['gaps'] = {
                'max_gap_seconds': max_gap.total_seconds(),
                'expected_frequency': expected_freq
            }
    
    # Outliers using IQR method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            issues['outliers'][col] = {
                'count': int(outliers),
                'percentage': float(outliers / len(df) * 100),
                'lower_bound': float(Q1 - 1.5 * IQR),
                'upper_bound': float(Q3 + 1.5 * IQR)
            }
    
    return issues


# =============================================================================
# Data Processing Functions
# =============================================================================

@handle_data_errors
def resample_data(df: pd.DataFrame, 
                 rule: str,
                 agg_dict: Optional[Dict] = None,
                 fill_method: Optional[str] = None,
                 limit: Optional[int] = None) -> pd.DataFrame:
    """
    Resample data to different timeframe with advanced options
    
    Args:
        df: DataFrame with datetime index
        rule: Resampling rule (e.g., '5T', '1H', '1D')
        agg_dict: Aggregation dictionary for each column
        fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate')
        limit: Maximum number of consecutive NaN to fill
    
    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("DataFrame index must be DatetimeIndex")
    
    if agg_dict is None:
        agg_dict = DataLoaderConfig.DEFAULT_AGG_DICT.copy()
    
    # Filter agg_dict to only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    try:
        # Handle OHLC data specially
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            agg_dict.update(ohlc_dict)
        
        resampled = df.resample(rule).agg(agg_dict)
        
        # Fill missing values if requested
        if fill_method:
            if fill_method == 'ffill':
                resampled = resampled.fillna(method='ffill', limit=limit)
                resampled = resampled.fillna(method='bfill', limit=limit)
            elif fill_method == 'interpolate':
                resampled = resampled.interpolate(method='time', limit=limit)
        
        resampled.dropna(inplace=True)
        
        logger.info(f"Resampled from {len(df)} to {len(resampled)} rows (rule: {rule})")
        return resampled
        
    except Exception as e:
        raise DataProcessingError(f"Error resampling data: {e}")


@handle_data_errors
def clean_data(df: pd.DataFrame,
              remove_duplicates: bool = True,
              handle_missing: bool = True,
              sort_index: bool = True,
              remove_outliers: bool = False,
              outlier_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning
    
    Args:
        df: Input DataFrame
        remove_duplicates: Whether to remove duplicate indices
        handle_missing: Whether to handle missing values
        sort_index: Whether to sort index
        remove_outliers: Whether to remove outliers
        outlier_config: Configuration for outlier detection
            {'method': 'iqr'/'zscore', 'threshold': 3.0, 'columns': [...]}
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    original_rows = len(df_clean)
    
    # Remove duplicates
    if remove_duplicates:
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
    
    # Sort index
    if sort_index:
        df_clean.sort_index(inplace=True)
    
    # Handle missing values
    if handle_missing:
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                # Forward fill then backward fill for numeric columns
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            elif df_clean[col].dtype == 'object':
                # Fill with mode for categorical data
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
    
    # Remove outliers
    if remove_outliers and outlier_config:
        method = outlier_config.get('method', 'iqr')
        threshold = outlier_config.get('threshold', 3.0)
        columns = outlier_config.get('columns', df_clean.select_dtypes(include=[np.number]).columns)
        
        for col in columns:
            if col in df_clean.columns:
                outlier_mask = detect_outliers(df_clean, col, method=method, threshold=threshold)
                df_clean = df_clean[~outlier_mask]
    
    rows_removed = original_rows - len(df_clean)
    logger.info(f"Cleaned data: {rows_removed} rows removed, {len(df_clean)} rows remaining")
    
    return df_clean


@handle_data_errors
def filter_time_range(df: pd.DataFrame, 
                     start: Union[str, datetime, None] = None,
                     end: Union[str, datetime, None] = None,
                     inclusive: str = 'both') -> pd.DataFrame:
    """
    Filter data to specific time range
    
    Args:
        df: DataFrame with datetime index
        start: Start time (None for no lower bound)
        end: End time (None for no upper bound)
        inclusive: Include boundaries ('both', 'left', 'right', 'neither')
    
    Returns:
        Filtered DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("DataFrame index must be DatetimeIndex")
    
    if start:
        if isinstance(start, str):
            start = pd.to_datetime(start)
    else:
        start = df.index.min()
    
    if end:
        if isinstance(end, str):
            end = pd.to_datetime(end)
    else:
        end = df.index.max()
    
    mask = pd.Series(True, index=df.index)
    
    if inclusive == 'both':
        mask = (df.index >= start) & (df.index <= end)
    elif inclusive == 'left':
        mask = (df.index >= start) & (df.index < end)
    elif inclusive == 'right':
        mask = (df.index > start) & (df.index <= end)
    elif inclusive == 'neither':
        mask = (df.index > start) & (df.index < end)
    
    filtered = df[mask]
    
    logger.info(f"Filtered to {len(filtered)} rows between {start} and {end}")
    return filtered


def detect_outliers(df: pd.DataFrame,
                   column: str,
                   method: str = 'iqr',
                   threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using various methods
    
    Args:
        df: DataFrame
        column: Column name to check
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean series where True indicates outlier
    """
    if column not in df.columns:
        raise DataValidationError(f"Column '{column}' not found")
    
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        outliers = z_scores > threshold
        
    elif method == 'modified_zscore':
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Map back to original index
    outlier_series = pd.Series(False, index=df.index)
    outlier_series[outliers.index] = outliers
    
    return outlier_series


def fill_missing_values(df: pd.DataFrame,
                       method: str = 'ffill',
                       limit: Optional[int] = None,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame with advanced options
    
    Args:
        df: DataFrame
        method: Fill method 
            ('ffill', 'bfill', 'interpolate', 'mean', 'median', 'mode')
        limit: Maximum number of consecutive NaN to fill
        columns: Columns to fill (if None, fill all)
    
    Returns:
        DataFrame with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue
        
        if method in ['ffill', 'bfill']:
            df_filled[col] = df_filled[col].fillna(method=method, limit=limit)
            # Fill remaining with opposite method
            opposite = 'bfill' if method == 'ffill' else 'ffill'
            df_filled[col] = df_filled[col].fillna(method=opposite, limit=limit)
            
        elif method == 'interpolate':
            df_filled[col] = df_filled[col].interpolate(
                method='time', limit=limit, limit_direction='both'
            )
            
        elif method == 'mean':
            df_filled[col] = df_filled[col].fillna(df[col].mean())
            
        elif method == 'median':
            df_filled[col] = df_filled[col].fillna(df[col].median())
            
        elif method == 'mode':
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df_filled[col] = df_filled[col].fillna(mode_val[0])
        else:
            raise ValueError(f"Unknown fill method: {method}")
    
    return df_filled


def calculate_return_statistics(returns: pd.Series,
                               periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate comprehensive return statistics
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
    
    Returns:
        Dictionary with statistics
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    stats = {
        'mean': float(returns.mean()),
        'std': float(returns.std()),
        'skew': float(returns.skew()),
        'kurtosis': float(returns.kurtosis()),
        'min': float(returns.min()),
        'max': float(returns.max()),
        'median': float(returns.median()),
        'positive_pct': float((returns > 0).mean() * 100),
        'negative_pct': float((returns < 0).mean() * 100),
        'sharpe': float(returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0,
        'sortino': float(calculate_sortino_ratio(returns, periods_per_year)),
        'var_95': float(returns.quantile(0.05)),
        'var_99': float(returns.quantile(0.01)),
        'cvar_95': float(returns[returns <= returns.quantile(0.05)].mean()) if any(returns <= returns.quantile(0.05)) else 0,
        'cvar_99': float(returns[returns <= returns.quantile(0.01)].mean()) if any(returns <= returns.quantile(0.01)) else 0,
        'calmar': float(calculate_calmar_ratio(returns, periods_per_year))
    }
    
    return stats


def calculate_sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Sortino ratio"""
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    return returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio"""
    if len(returns) < periods_per_year:
        return 0.0
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    max_drawdown = calculate_drawdown(cum_returns)['max_drawdown'] / 100
    
    if max_drawdown == 0:
        return 0.0
    
    annualized_return = (1 + returns.mean()) ** periods_per_year - 1
    return annualized_return / max_drawdown


def calculate_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive drawdown statistics
    
    Args:
        equity_curve: Series of equity values
    
    Returns:
        Dictionary with drawdown statistics
    """
    peak = equity_curve.expanding().max()
    drawdown = (peak - equity_curve) / peak
    
    # Find all drawdown periods
    in_drawdown = False
    current_dd_start = None
    drawdown_periods = []
    
    for i, (idx, dd) in enumerate(drawdown.items()):
        if dd > 0 and not in_drawdown:
            in_drawdown = True
            current_dd_start = idx
        elif dd == 0 and in_drawdown:
            in_drawdown = False
            duration = (idx - current_dd_start).total_seconds() / (24 * 3600)  # in days
            drawdown_periods.append({
                'start': current_dd_start,
                'end': idx,
                'max_dd': drawdown.loc[current_dd_start:idx].max(),
                'duration_days': duration
            })
    
    stats = {
        'max_drawdown': float(drawdown.max() * 100),
        'avg_drawdown': float(drawdown.mean() * 100),
        'current_drawdown': float(drawdown.iloc[-1] * 100) if len(drawdown) > 0 else 0,
        'max_drawdown_duration': _calculate_max_drawdown_duration(drawdown),
        'avg_drawdown_duration': float(np.mean([p['duration_days'] for p in drawdown_periods])) if drawdown_periods else 0,
        'drawdown_std': float(drawdown.std() * 100),
        'drawdown_periods': len(drawdown_periods)
    }
    
    return stats


def _calculate_max_drawdown_duration(drawdown: pd.Series) -> int:
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


def calculate_time_based_features(df: pd.DataFrame,
                                 include_sessions: bool = True,
                                 include_cycles: bool = True) -> pd.DataFrame:
    """
    Calculate comprehensive time-based features from datetime index
    
    Args:
        df: DataFrame with datetime index
        include_sessions: Include market session indicators
        include_cycles: Include cyclical features (sin/cos transformations)
    
    Returns:
        DataFrame with additional time features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("DataFrame index must be DatetimeIndex")
    
    features = pd.DataFrame(index=df.index)
    
    # Basic time features
    features['hour'] = df.index.hour
    features['minute'] = df.index.minute
    features['second'] = df.index.second
    features['day_of_week'] = df.index.dayofweek
    features['day_of_month'] = df.index.day
    features['day_of_year'] = df.index.dayofyear
    features['week_of_year'] = df.index.isocalendar().week
    features['month'] = df.index.month
    features['quarter'] = df.index.quarter
    features['year'] = df.index.year
    features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    features['is_month_start'] = df.index.is_month_start.astype(int)
    features['is_month_end'] = df.index.is_month_end.astype(int)
    features['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    features['is_year_start'] = df.index.is_year_start.astype(int)
    features['is_year_end'] = df.index.is_year_end.astype(int)
    
    # Market sessions
    if include_sessions:
        for session, (start, end) in DataLoaderConfig.SESSION_HOURS.items():
            if start <= end:
                features[f'is_{session}'] = (
                    (features['hour'] >= start) & (features['hour'] < end)
                ).astype(int)
            else:  # Overnight session
                features[f'is_{session}'] = (
                    (features['hour'] >= start) | (features['hour'] < end)
                ).astype(int)
    
    # Cyclical features (for ML models)
    if include_cycles:
        # Hour of day (0-23)
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Day of week (0-6)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Month (1-12)
        features['month_sin'] = np.sin(2 * np.pi * (features['month'] - 1) / 12)
        features['month_cos'] = np.cos(2 * np.pi * (features['month'] - 1) / 12)
    
    return features


def merge_dataframes(dfs: List[pd.DataFrame],
                    how: str = 'inner',
                    on: Optional[Union[str, List[str]]] = None,
                    suffixes: Tuple[str, str] = ('_x', '_y'),
                    validate: Optional[str] = None) -> pd.DataFrame:
    """
    Merge multiple DataFrames with advanced options
    
    Args:
        dfs: List of DataFrames to merge
        how: Merge method ('inner', 'outer', 'left', 'right', 'cross')
        on: Column(s) to merge on (if None, merge on index)
        suffixes: Suffixes for overlapping columns
        validate: Validation type ('1:1', '1:m', 'm:1', 'm:m')
    
    Returns:
        Merged DataFrame
    """
    if len(dfs) == 0:
        return pd.DataFrame()
    
    if len(dfs) == 1:
        return dfs[0]
    
    result = dfs[0]
    
    for i, df in enumerate(dfs[1:], start=1):
        # Check for empty DataFrames
        if df.empty:
            logger.warning(f"DataFrame {i} is empty, skipping")
            continue
        
        try:
            if on:
                result = pd.merge(
                    result, df, on=on, how=how,
                    suffixes=(f'_{i-1}', f'_{i}') if suffixes else suffixes,
                    validate=validate
                )
            else:
                result = pd.merge(
                    result, df, left_index=True, right_index=True,
                    how=how, suffixes=(f'_{i-1}', f'_{i}') if suffixes else suffixes,
                    validate=validate
                )
        except Exception as e:
            logger.error(f"Error merging DataFrame {i}: {e}")
            continue
    
    logger.info(f"Merged {len(dfs)} DataFrames into {len(result)} rows")
    return result


# =============================================================================
# Data Streaming and Chunking
# =============================================================================

class DataStreamer:
    """Stream data in chunks for memory-efficient processing"""
    
    def __init__(self, filepath: Union[str, Path], chunksize: int = 10000, **kwargs):
        self.filepath = Path(filepath)
        self.chunksize = chunksize
        self.kwargs = kwargs
        self._iterator = None
        self._total_chunks = None
        
    def __iter__(self):
        """Iterate over chunks"""
        if self.filepath.suffix == '.csv':
            self._iterator = pd.read_csv(
                self.filepath,
                chunksize=self.chunksize,
                **self.kwargs
            )
        elif self.filepath.suffix == '.parquet':
            # Parquet doesn't support chunked reading directly
            df = pd.read_parquet(self.filepath, **self.kwargs)
            self._iterator = (df[i:i+self.chunksize] 
                            for i in range(0, len(df), self.chunksize))
        else:
            raise DataFormatError(f"Unsupported format for streaming: {self.filepath.suffix}")
        
        return self._iterator
    
    def process(self, processor: Callable[[pd.DataFrame], Any]) -> List[Any]:
        """
        Process data in chunks
        
        Args:
            processor: Function to apply to each chunk
        
        Returns:
            List of results from each chunk
        """
        results = []
        for chunk in self:
            result = processor(chunk)
            results.append(result)
        return results
    
    def parallel_process(self, processor: Callable[[pd.DataFrame], Any],
                        max_workers: int = 4) -> List[Any]:
        """
        Process chunks in parallel
        
        Args:
            processor: Function to apply to each chunk
            max_workers: Maximum number of worker threads
        
        Returns:
            List of results from each chunk
        """
        chunks = list(self)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(processor, chunks))
        
        return results


# =============================================================================
# Data Cache Manager
# =============================================================================

class DataCache:
    """Simple cache manager for frequently accessed data"""
    
    def __init__(self, cache_dir: Union[str, Path] = 'cache/data', ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._memory_cache = {}
    
    def get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
    
    def get(self, key: str, use_memory: bool = True) -> Optional[pd.DataFrame]:
        """Get data from cache"""
        # Check memory cache first
        if use_memory and key in self._memory_cache:
            data, timestamp = self._memory_cache[key]
            if time.time() - timestamp < self.ttl:
                logger.debug(f"Memory cache hit for {key}")
                return data
        
        # Check disk cache
        cache_path = self.get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Disk cache hit for {key}")
                
                # Update memory cache
                if use_memory:
                    self._memory_cache[key] = (data, time.time())
                
                return data
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        
        return None
    
    def set(self, key: str, data: pd.DataFrame, use_memory: bool = True):
        """Save data to cache"""
        # Memory cache
        if use_memory:
            self._memory_cache[key] = (data, time.time())
        
        # Disk cache
        try:
            cache_path = self.get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached data for {key}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def clear(self, pattern: Optional[str] = None):
        """Clear cache"""
        if pattern:
            # Clear matching files
            for cache_file in self.cache_dir.glob(f"*{pattern}*"):
                cache_file.unlink()
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        
        # Clear memory cache
        self._memory_cache.clear()
        logger.info("Cache cleared")


# =============================================================================
# Context Manager for Data Operations
# =============================================================================

@contextmanager
def data_operation_context(operation_name: str):
    """Context manager for data operations with timing and error handling"""
    start_time = time.time()
    logger.info(f"Starting {operation_name}")
    
    try:
        yield
        elapsed = time.time() - start_time
        logger.info(f"Completed {operation_name} in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed {operation_name} after {elapsed:.2f}s: {e}")
        raise


# =============================================================================
# Export all functions
# =============================================================================

__all__ = [
    # Main loading functions
    'load_csv_data',
    'load_parquet_data',
    'load_hdf5_data',
    'load_feather_data',
    'load_pickle_data',
    'load_json_data',
    'load_excel_data',
    'load_from_database',
    'load_from_multiple_sources',
    
    # Main saving functions
    'save_csv_data',
    'save_parquet_data',
    'save_hdf5_data',
    
    # Validation
    'validate_dataframe',
    'detect_data_quality_issues',
    
    # Processing
    'resample_data',
    'clean_data',
    'filter_time_range',
    'detect_outliers',
    'fill_missing_values',
    'merge_dataframes',
    
    # Statistics
    'calculate_return_statistics',
    'calculate_drawdown',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    
    # Feature engineering
    'calculate_time_based_features',
    
    # Streaming and caching
    'DataStreamer',
    'DataCache',
    
    # Context manager
    'data_operation_context',
    
    # Exceptions
    'DataLoaderError',
    'DataNotFoundError',
    'DataFormatError',
    'DataValidationError',
    'DataProcessingError',
    
    # Configuration
    'DataLoaderConfig'
]