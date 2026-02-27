"""
Validation - Data and parameter validation utilities
Comprehensive validation functions for data quality and trading parameters.
"""

import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Union
import logging
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Result Class
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


# =============================================================================
# DataFrame Validation
# =============================================================================

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


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> ValidationResult:
    """
    Validate DataFrame columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    missing = set(required_columns) - set(df.columns)
    if missing:
        result.add_error(f"Missing columns: {missing}")
    
    return result


def validate_datetime_index(df: pd.DataFrame) -> ValidationResult:
    """
    Validate that DataFrame has datetime index
    
    Args:
        df: DataFrame to validate
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        result.add_error(f"Expected DatetimeIndex, got {type(df.index)}")
    
    return result


def validate_frequency(df: pd.DataFrame, expected_freq: str) -> ValidationResult:
    """
    Validate DataFrame frequency
    
    Args:
        df: DataFrame with datetime index
        expected_freq: Expected frequency (e.g., 'T', 'H', 'D')
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        result.add_error("DataFrame must have datetime index for frequency validation")
        return result
    
    inferred_freq = pd.infer_freq(df.index)
    
    if inferred_freq != expected_freq:
        result.add_warning(f"Expected frequency {expected_freq}, got {inferred_freq}")
    
    return result


# =============================================================================
# Numeric Validation
# =============================================================================

def validate_positive(value: float, name: str = "value") -> ValidationResult:
    """Validate that a value is positive"""
    result = ValidationResult()
    
    if not isinstance(value, (int, float)):
        result.add_error(f"{name} must be a number, got {type(value)}")
    elif value <= 0:
        result.add_error(f"{name} must be positive, got {value}")
    
    return result


def validate_non_negative(value: float, name: str = "value") -> ValidationResult:
    """Validate that a value is non-negative"""
    result = ValidationResult()
    
    if not isinstance(value, (int, float)):
        result.add_error(f"{name} must be a number, got {type(value)}")
    elif value < 0:
        result.add_error(f"{name} must be non-negative, got {value}")
    
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


def validate_in_list(value: Any, valid_values: List[Any],
                   name: str = "value") -> ValidationResult:
    """Validate that a value is in list of valid values"""
    result = ValidationResult()
    
    if value not in valid_values:
        result.add_error(f"{name} must be in {valid_values}, got {value}")
    
    return result


# =============================================================================
# Trading Parameter Validation
# =============================================================================

def validate_trade_parameters(parameters: Dict[str, Any]) -> ValidationResult:
    """
    Validate trade parameters
    
    Args:
        parameters: Dictionary of trade parameters
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    required_params = ['symbol', 'quantity', 'entry_price', 'stop_loss', 'take_profit']
    
    for param in required_params:
        if param not in parameters:
            result.add_error(f"Missing parameter: {param}")
    
    if 'quantity' in parameters:
        result.merge(validate_positive(parameters['quantity'], 'quantity'))
    
    if 'entry_price' in parameters:
        result.merge(validate_positive(parameters['entry_price'], 'entry_price'))
    
    if 'stop_loss' in parameters and 'entry_price' in parameters:
        if parameters['stop_loss'] >= parameters['entry_price']:
            result.add_error("Stop loss must be less than entry price")
    
    if 'take_profit' in parameters and 'entry_price' in parameters:
        if parameters['take_profit'] <= parameters['entry_price']:
            result.add_error("Take profit must be greater than entry price")
    
    if 'risk_percent' in parameters:
        result.merge(validate_range(parameters['risk_percent'], 0, 100, 'risk_percent'))
    
    return result


def validate_risk_parameters(parameters: Dict[str, Any]) -> ValidationResult:
    """
    Validate risk parameters
    
    Args:
        parameters: Dictionary of risk parameters
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    required_params = ['max_risk_per_trade', 'max_drawdown', 'max_positions']
    
    for param in required_params:
        if param not in parameters:
            result.add_error(f"Missing parameter: {param}")
    
    if 'max_risk_per_trade' in parameters:
        result.merge(validate_range(parameters['max_risk_per_trade'], 0, 100, 'max_risk_per_trade'))
    
    if 'max_drawdown' in parameters:
        result.merge(validate_range(parameters['max_drawdown'], 0, 100, 'max_drawdown'))
    
    if 'max_positions' in parameters:
        result.merge(validate_positive(parameters['max_positions'], 'max_positions'))
    
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
# Data Quality Validation
# =============================================================================

def validate_duplicate_index(df: pd.DataFrame) -> ValidationResult:
    """Validate that index has no duplicates"""
    result = ValidationResult()
    
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        result.add_error(f"Index has {duplicates} duplicate values")
    
    return result


def validate_missing_values(df: pd.DataFrame, max_missing: float = 0.05) -> ValidationResult:
    """
    Validate missing values
    
    Args:
        df: DataFrame to validate
        max_missing: Maximum allowed missing values as fraction (0-1)
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    missing_percent = df.isnull().mean() * 100
    
    for col, percent in missing_percent.items():
        if percent > max_missing * 100:
            result.add_warning(f"Column {col} has {percent:.1f}% missing values")
    
    return result


def validate_data_range(df: pd.DataFrame, start_date: datetime = None,
                       end_date: datetime = None) -> ValidationResult:
    """
    Validate data date range
    
    Args:
        df: DataFrame with datetime index
        start_date: Minimum allowed date
        end_date: Maximum allowed date
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        result.add_error("DataFrame must have datetime index for date range validation")
        return result
    
    min_date = df.index.min()
    max_date = df.index.max()
    
    if start_date and min_date < start_date:
        result.add_warning(f"Data starts before {start_date}, got {min_date}")
    
    if end_date and max_date > end_date:
        result.add_warning(f"Data ends after {end_date}, got {max_date}")
    
    return result


# =============================================================================
# Price and Time Series Validation
# =============================================================================

def validate_price_series(prices: pd.Series) -> ValidationResult:
    """
    Validate price series
    
    Args:
        prices: Series of prices
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    if prices.isnull().any():
        result.add_warning("Price series contains NaN values")
    
    if prices.duplicated().any():
        result.add_warning("Price series contains duplicate values")
    
    # Check for reasonable price range (simplified)
    if prices.min() < 0.0001 or prices.max() > 1e6:
        result.add_warning("Price values seem unrealistic")
    
    return result


def validate_volume_series(volume: pd.Series) -> ValidationResult:
    """
    Validate volume series
    
    Args:
        volume: Series of volume
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    if volume.isnull().any():
        result.add_warning("Volume series contains NaN values")
    
    if (volume < 0).any():
        result.add_error("Volume series contains negative values")
    
    if volume.duplicated().any():
        result.add_warning("Volume series contains duplicate values")
    
    return result


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> ValidationResult:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    for key in required_keys:
        if key not in config:
            result.add_error(f"Missing configuration key: {key}")
    
    return result


def validate_interval(interval: str) -> ValidationResult:
    """
    Validate trading interval
    
    Args:
        interval: Interval string (e.g., '1m', '5m', '1h', '1d')
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    
    if interval not in valid_intervals:
        result.add_error(f"Invalid interval: {interval}. Valid: {valid_intervals}")
    
    return result


def validate_timeframe(timeframe: str) -> ValidationResult:
    """
    Validate timeframe
    
    Args:
        timeframe: Timeframe string
    
    Returns:
        ValidationResult
    """
    return validate_interval(timeframe)


# =============================================================================
# Performance Metrics Validation
# =============================================================================

def validate_win_rate(win_rate: float) -> ValidationResult:
    """
    Validate win rate
    
    Args:
        win_rate: Win rate percentage (0-100)
    
    Returns:
        ValidationResult
    """
    return validate_range(win_rate, 0, 100, 'win_rate')


def validate_risk_reward(risk_reward: float) -> ValidationResult:
    """
    Validate risk-reward ratio
    
    Args:
        risk_reward: Risk-reward ratio
    
    Returns:
        ValidationResult
    """
    return validate_positive(risk_reward, 'risk_reward')


def validate_max_drawdown(max_drawdown: float) -> ValidationResult:
    """
    Validate maximum drawdown
    
    Args:
        max_drawdown: Maximum drawdown percentage (0-100)
    
    Returns:
        ValidationResult
    """
    return validate_range(max_drawdown, 0, 100, 'max_drawdown')


# =============================================================================
# Currency and Financial Validation
# =============================================================================

def validate_currency(currency: str) -> ValidationResult:
    """
    Validate currency code
    
    Args:
        currency: Currency code (e.g., 'USD', 'EUR')
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
    
    if currency not in valid_currencies:
        result.add_warning(f"Unknown currency: {currency}")
    
    return result


def validate_amount(amount: float, min_amount: float = 0.01) -> ValidationResult:
    """
    Validate financial amount
    
    Args:
        amount: Amount to validate
        min_amount: Minimum acceptable amount
    
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    if amount < min_amount:
        result.add_error(f"Amount must be at least {min_amount}, got {amount}")
    
    return result


__all__ = [
    'ValidationResult',
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
    'validate_duplicate_index',
    'validate_missing_values',
    'validate_data_range',
    'validate_price_series',
    'validate_volume_series',
    'validate_config',
    'validate_interval',
    'validate_timeframe',
    'validate_win_rate',
    'validate_risk_reward',
    'validate_max_drawdown',
    'validate_currency',
    'validate_amount'
]
