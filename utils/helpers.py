"""
Helpers - General utility functions
Comprehensive utilities for configuration, formatting, calculations, and common operations.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable, Tuple, Set
import logging
from datetime import datetime, timedelta, timezone
import asyncio
import functools
import hashlib
import time
import math
import re
import os
import pickle
import random
import string
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP
from collections.abc import MutableMapping
import inspect
import warnings

# Optional imports for advanced features
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class HelperError(Exception):
    """Base exception for helper functions"""
    pass


class ConfigurationError(HelperError):
    """Raised for configuration errors"""
    pass


class ValidationError(HelperError):
    """Raised for validation errors"""
    pass


class CalculationError(HelperError):
    """Raised for calculation errors"""
    pass


# =============================================================================
# Configuration Management
# =============================================================================

class ConfigManager:
    """Enhanced configuration manager with environment variable support"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config = {}
        self._env_prefix = "TRADING_BOT_"
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config format: {config_path.suffix}")
            
            self._config = self._process_config(config)
            self.config_path = config_path
            logger.info(f"Loaded config from {config_path}")
            return self._config
            
        except Exception as e:
            raise ConfigurationError(f"Error loading config: {e}")
    
    def save(self, config_path: Optional[Union[str, Path]] = None) -> bool:
        """Save configuration to file"""
        save_path = Path(config_path) if config_path else self.config_path
        
        if not save_path:
            raise ConfigurationError("No save path specified")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                if save_path.suffix in ['.yaml', '.yml']:
                    yaml.dump(self._config, f, default_flow_style=False)
                elif save_path.suffix == '.json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported config format: {save_path.suffix}")
            
            logger.info(f"Saved config to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def _process_config(self, config: Dict) -> Dict:
        """Process configuration, replacing environment variables"""
        if isinstance(config, dict):
            return {k: self._process_config(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._process_config(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            # Environment variable substitution
            env_var = config[2:-1]
            return os.environ.get(env_var, config)
        else:
            return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot notation key"""
        keys = key.split('.')
        target = self._config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any], prefix: str = ""):
        """Update configuration with nested dictionary"""
        for key, value in updates.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self.update(value, full_key)
            else:
                self.set(full_key, value)
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get value from environment variables"""
        env_key = f"{self._env_prefix}{key.upper().replace('.', '_')}"
        return os.environ.get(env_key, default)
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        self.set(key, value)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    manager = ConfigManager(config_path)
    return manager.all


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
    """Legacy function for backward compatibility"""
    manager = ConfigManager()
    manager.update(config)
    return manager.save(config_path)


# =============================================================================
# Directory and File Management
# =============================================================================

def create_directories(paths: List[Union[str, Path]], exist_ok: bool = True) -> bool:
    """
    Create directories if they don't exist
    
    Args:
        paths: List of directory paths
        exist_ok: If True, don't error if directory exists
    
    Returns:
        True if all directories created/exist
    """
    success = True
    
    for path in paths:
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=exist_ok)
            logger.debug(f"Directory ensured: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            success = False
    
    return success


def ensure_file_directory(filepath: Union[str, Path]) -> Path:
    """Ensure the directory for a file exists"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath


def get_project_root() -> Path:
    """Get project root directory"""
    # Start from current file's directory
    current = Path(__file__).resolve()
    
    # Look for project markers
    for parent in current.parents:
        if (parent / '.git').exists() or (parent / 'setup.py').exists() or (parent / 'pyproject.toml').exists():
            return parent
    
    # Fallback to current working directory
    return Path.cwd()


def list_files(directory: Union[str, Path], pattern: str = "*", 
               recursive: bool = False) -> List[Path]:
    """List files in directory matching pattern"""
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


# =============================================================================
# String Formatting Utilities
# =============================================================================

def format_currency(value: float, currency: str = 'USD', decimals: int = 2,
                   include_symbol: bool = True) -> str:
    """
    Format value as currency string with proper symbol and formatting
    
    Args:
        value: Numeric value
        currency: Currency code (USD, EUR, GBP, JPY, BTC, etc.)
        decimals: Number of decimal places
        include_symbol: Whether to include currency symbol
    
    Returns:
        Formatted currency string
    """
    symbols = {
        'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'CHF': 'Fr',
        'CAD': 'C$', 'AUD': 'A$', 'NZD': 'NZ$', 'CNY': '¥', 'HKD': 'HK$',
        'SGD': 'S$', 'KRW': '₩', 'INR': '₹', 'BTC': '₿', 'ETH': 'Ξ',
        'XAU': 'oz', 'XAG': 'oz'
    }
    
    symbol = symbols.get(currency, '$') if include_symbol else ''
    
    # Special formatting for different currencies
    if currency in ['JPY', 'KRW']:
        # No decimals for JPY
        formatted = f"{value:,.0f}"
    elif currency in ['BTC', 'ETH']:
        # More decimals for crypto
        formatted = f"{value:,.{max(decimals, 8)}f}"
    else:
        formatted = f"{value:,.{decimals}f}"
    
    if symbol:
        # Symbol placement (some currencies put symbol after)
        if currency in ['XAU', 'XAG']:
            return f"{formatted}{symbol}"
        else:
            return f"{symbol}{formatted}"
    else:
        return formatted


def format_percentage(value: float, decimals: int = 2, include_sign: bool = False) -> str:
    """
    Format value as percentage
    
    Args:
        value: Numeric value (0.15 = 15%)
        decimals: Number of decimal places
        include_sign: Include + sign for positive values
    
    Returns:
        Formatted percentage string
    """
    percent = value * 100
    
    if include_sign and percent > 0:
        sign = "+"
    else:
        sign = ""
    
    return f"{sign}{percent:.{decimals}f}%"


def format_number(value: float, decimals: int = 2, 
                  scientific_threshold: Optional[int] = None) -> str:
    """
    Format number with appropriate formatting
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
        scientific_threshold: Use scientific notation if abs(value) > threshold
    
    Returns:
        Formatted number string
    """
    if scientific_threshold and abs(value) > scientific_threshold:
        return f"{value:.{decimals}e}"
    elif abs(value) < 0.0001 and value != 0:
        return f"{value:.{decimals}e}"
    else:
        return f"{value:,.{decimals}f}"


def format_timestamp(timestamp: Union[datetime, str, float, int], 
                    format: str = '%Y-%m-%d %H:%M:%S',
                    tz: Optional[timezone] = None) -> str:
    """
    Format timestamp with timezone support
    
    Args:
        timestamp: Timestamp (datetime, string, or Unix timestamp)
        format: Output format
        tz: Timezone for output
    
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    elif isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            # Try common formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y%m%d %H%M%S']:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
            else:
                dt = datetime.now()
    else:
        dt = timestamp
    
    # Convert timezone if specified
    if tz and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc).astimezone(tz)
    elif tz and dt.tzinfo:
        dt = dt.astimezone(tz)
    
    return dt.strftime(format)


def format_duration(seconds: float, include_ms: bool = False) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        include_ms: Include milliseconds
    
    Returns:
        Formatted duration string
    """
    if seconds < 0:
        return f"-{format_duration(abs(seconds), include_ms)}"
    
    if seconds < 1 and include_ms:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length"""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug"""
    # Convert to lowercase and replace spaces with hyphens
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text


def random_string(length: int = 8, chars: str = string.ascii_letters + string.digits) -> str:
    """Generate random string of specified length"""
    return ''.join(random.choice(chars) for _ in range(length))


# =============================================================================
# Trading Calculations
# =============================================================================

def calculate_pips(price1: float, price2: float, symbol: str) -> float:
    """
    Calculate pips difference between two prices
    
    Args:
        price1: First price
        price2: Second price
        symbol: Trading symbol
    
    Returns:
        Pip difference
    """
    multiplier = get_pip_multiplier(symbol)
    return abs(price1 - price2) * multiplier


def get_pip_multiplier(symbol: str) -> float:
    """
    Get pip multiplier for symbol
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Pip multiplier
    """
    symbol = normalize_symbol(symbol)
    
    # Forex pairs
    if 'JPY' in symbol:
        return 100  # JPY pairs: 1 pip = 0.01
    elif any(x in symbol for x in ['EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']) and 'USD' in symbol:
        return 10000  # Forex: 1 pip = 0.0001
    
    # Metals
    elif 'XAU' in symbol or 'XAG' in symbol:
        return 10  # Metals: 1 pip = 0.1
    
    # Indices
    elif symbol in ['US30', 'SPX500', 'NAS100', 'FTSE100', 'DAX40', 'NIKKEI225']:
        return 10  # Indices: 1 pip = 0.1
    
    # Crypto
    elif any(x in symbol for x in ['BTC', 'ETH', 'BNB', 'SOL', 'XRP']):
        # Determine decimal places
        if 'BTC' in symbol:
            return 1  # BTC: 1 pip = 1 (actually 0.1, but simplified)
        elif 'ETH' in symbol:
            return 10  # ETH: 1 pip = 0.1
        else:
            return 100  # Other crypto: 1 pip = 0.01
    
    # Default
    return 10000


def calculate_point_value(symbol: str, lot_size: float = 1.0) -> float:
    """
    Calculate point value for symbol
    
    Args:
        symbol: Trading symbol
        lot_size: Lot size (standard lot = 1.0)
    
    Returns:
        Point value in quote currency
    """
    symbol = normalize_symbol(symbol)
    
    # Standard lot sizes
    if 'JPY' in symbol:
        return lot_size * 1000  # 1 lot = 1000 JPY per point
    elif 'XAU' in symbol:
        return lot_size * 100    # 1 lot = 100 USD per point
    elif 'XAG' in symbol:
        return lot_size * 50     # 1 lot = 50 USD per point
    elif 'BTC' in symbol:
        return lot_size * 1      # 1 lot = 1 USD per point
    elif 'ETH' in symbol:
        return lot_size * 0.1    # 1 lot = 0.1 USD per point
    elif any(x in symbol for x in ['US30', 'SPX500', 'NAS100']):
        return lot_size * 5      # 1 lot = 5 USD per point
    else:
        return lot_size * 10     # 1 lot = 10 USD per point (standard forex)


def calculate_risk_reward(entry: float, stop: float, target: float) -> float:
    """
    Calculate risk-reward ratio
    
    Args:
        entry: Entry price
        stop: Stop loss price
        target: Take profit price
    
    Returns:
        Risk-reward ratio
    """
    risk = abs(entry - stop)
    reward = abs(target - entry)
    
    return safe_divide(reward, risk)


def calculate_position_size(risk_amount: float, entry: float, stop: float, 
                           point_value: float, round_down: bool = True) -> float:
    """
    Calculate position size based on risk
    
    Args:
        risk_amount: Risk amount in account currency
        entry: Entry price
        stop: Stop loss price
        point_value: Point value per lot
        round_down: Round down to nearest lot size
    
    Returns:
        Position size in lots
    """
    risk_distance = abs(entry - stop)
    
    if risk_distance == 0 or point_value == 0:
        return 0
    
    position_size = risk_amount / (risk_distance * point_value)
    
    if round_down:
        # Round down to 2 decimal places (0.01 lot minimum)
        position_size = math.floor(position_size * 100) / 100
    
    return max(0, position_size)


def calculate_lot_size(account_balance: float, risk_percent: float,
                       stop_loss_pips: float, pip_value: float) -> float:
    """
    Calculate lot size based on account risk
    
    Args:
        account_balance: Account balance
        risk_percent: Risk percentage per trade (0.01 = 1%)
        stop_loss_pips: Stop loss in pips
        pip_value: Value per pip per lot
    
    Returns:
        Lot size
    """
    risk_amount = account_balance * risk_percent
    lot_size = risk_amount / (stop_loss_pips * pip_value)
    
    # Round to nearest 0.01 lot
    return round(lot_size * 100) / 100


def calculate_margin(notional: float, leverage: float) -> float:
    """
    Calculate required margin
    
    Args:
        notional: Notional position value
        leverage: Leverage (e.g., 100 for 1:100)
    
    Returns:
        Required margin
    """
    return notional / leverage


def calculate_pip_value(symbol: str, lot_size: float, quote_currency: str = 'USD') -> float:
    """
    Calculate pip value for symbol
    
    Args:
        symbol: Trading symbol
        lot_size: Lot size
        quote_currency: Account currency
    
    Returns:
        Pip value in account currency
    """
    symbol = normalize_symbol(symbol)
    
    # Base pip values
    if 'JPY' in symbol:
        pip_value = lot_size * 1000  # 1000 JPY per pip
    elif any(x in symbol for x in ['EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']) and 'USD' in symbol:
        pip_value = lot_size * 10  # $10 per pip for standard lot
    elif 'XAU' in symbol:
        pip_value = lot_size * 10  # $10 per pip for standard lot
    else:
        pip_value = lot_size * 10
    
    # Convert to account currency if needed
    # This is simplified - in practice you'd need exchange rates
    return pip_value


def calculate_swap_points(symbol: str, long_short: str, 
                         interest_rate_diff: float, days: int = 1) -> float:
    """
    Calculate swap points for overnight positions
    
    Args:
        symbol: Trading symbol
        long_short: 'long' or 'short'
        interest_rate_diff: Interest rate differential
        days: Number of days
    
    Returns:
        Swap points
    """
    # Simplified swap calculation
    multiplier = 1 if long_short == 'long' else -1
    return multiplier * interest_rate_diff * days / 365


# =============================================================================
# Symbol Utilities
# =============================================================================

def validate_symbol(symbol: str, strict: bool = False) -> bool:
    """
    Validate trading symbol format
    
    Args:
        symbol: Trading symbol
        strict: Strict validation against known symbols
    
    Returns:
        True if valid
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    symbol = symbol.upper().strip()
    
    # Check length
    if len(symbol) < 4 or len(symbol) > 20:
        return False
    
    if strict:
        # Strict validation against known patterns
        valid_patterns = [
            # Forex
            symbol.endswith('USD') and len(symbol) == 6,
            symbol.endswith(('JPY', 'GBP', 'EUR', 'CHF', 'CAD', 'AUD', 'NZD')) and len(symbol) == 6,
            # Crypto
            any(x in symbol for x in ['USDT', 'BTC', 'ETH', 'BNB', 'SOL', 'XRP']) and len(symbol) <= 10,
            # Metals
            symbol in ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'],
            # Indices
            symbol in ['US30', 'SPX500', 'NAS100', 'FTSE100', 'DAX40', 'NIKKEI225', 'HSI50'],
            # Commodities
            symbol in ['UKOIL', 'USOIL', 'NATGAS']
        ]
        
        return any(valid_patterns)
    else:
        # Loose validation - just check alphanumeric
        return symbol.replace('/', '').replace('_', '').isalnum()


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol to standard format
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Normalized symbol
    """
    if not symbol:
        return ""
    
    # Remove separators and convert to uppercase
    normalized = symbol.upper().strip().replace('/', '').replace('_', '').replace('-', '')
    
    # Handle common variations
    replacements = {
        'XAU': 'XAUUSD',
        'GOLD': 'XAUUSD',
        'XAG': 'XAGUSD',
        'SILVER': 'XAGUSD',
        'US30': 'US30',
        'SPX': 'SPX500',
        'NAS': 'NAS100',
        'FTSE': 'FTSE100',
        'DAX': 'DAX40',
        'NIKKEI': 'NIKKEI225'
    }
    
    return replacements.get(normalized, normalized)


def get_instrument_type(symbol: str) -> str:
    """
    Get instrument type from symbol
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Instrument type (forex, crypto, index, commodity, metal)
    """
    symbol = normalize_symbol(symbol)
    
    # Forex
    if len(symbol) == 6 and symbol.isalpha() and symbol.endswith(('USD', 'JPY', 'GBP', 'EUR', 'CHF', 'CAD', 'AUD', 'NZD')):
        return 'forex'
    
    # Crypto
    crypto_currencies = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOT', 'LINK', 'LTC', 'BCH']
    if any(x in symbol for x in crypto_currencies) or symbol.endswith('USDT'):
        return 'crypto'
    
    # Metals
    metals = ['XAU', 'XAG', 'XPT', 'XPD']
    if any(x in symbol for x in metals):
        return 'metal'
    
    # Indices
    indices = ['US30', 'SPX500', 'NAS100', 'FTSE100', 'DAX40', 'NIKKEI225', 'HSI50', 'CAC40']
    if symbol in indices:
        return 'index'
    
    # Commodities
    commodities = ['UKOIL', 'USOIL', 'NATGAS', 'WHEAT', 'CORN', 'SOYBEAN', 'COPPER']
    if symbol in commodities:
        return 'commodity'
    
    return 'unknown'


def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Get detailed information about a trading symbol
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Dictionary with symbol information
    """
    symbol = normalize_symbol(symbol)
    instrument_type = get_instrument_type(symbol)
    
    info = {
        'symbol': symbol,
        'normalized': symbol,
        'type': instrument_type,
        'pip_multiplier': get_pip_multiplier(symbol),
        'point_value': calculate_point_value(symbol, 1.0),
        'is_forex': instrument_type == 'forex',
        'is_crypto': instrument_type == 'crypto',
        'is_metal': instrument_type == 'metal',
        'is_index': instrument_type == 'index',
        'is_commodity': instrument_type == 'commodity'
    }
    
    # Add base and quote currencies for forex
    if instrument_type == 'forex' and len(symbol) == 6:
        info['base_currency'] = symbol[:3]
        info['quote_currency'] = symbol[3:]
    
    return info


# =============================================================================
# Mathematical Utilities
# =============================================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safe division with zero check
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value if division by zero
    
    Returns:
        Division result or default
    """
    if b == 0 or math.isnan(b) or math.isinf(b):
        return default
    try:
        return a / b
    except (ZeroDivisionError, OverflowError, ValueError):
        return default


def round_to_tick(price: float, tick_size: float, rounding: str = 'half_up') -> float:
    """
    Round price to nearest tick
    
    Args:
        price: Price to round
        tick_size: Tick size
        rounding: Rounding method ('half_up', 'down', 'up')
    
    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price
    
    # Convert to Decimal for precise rounding
    price_dec = Decimal(str(price))
    tick_dec = Decimal(str(tick_size))
    
    # Calculate number of ticks
    num_ticks = price_dec / tick_dec
    
    if rounding == 'half_up':
        rounded_ticks = num_ticks.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    elif rounding == 'down':
        rounded_ticks = num_ticks.quantize(Decimal('1'), rounding=ROUND_DOWN)
    elif rounding == 'up':
        rounded_ticks = num_ticks.quantize(Decimal('1'), rounding=ROUND_UP)
    else:
        rounded_ticks = num_ticks.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    
    return float(rounded_ticks * tick_dec)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed
        max_value: Maximum allowed
    
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def weighted_average(values: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Calculate weighted average
    
    Args:
        values: List of values
        weights: List of weights (if None, equal weights)
    
    Returns:
        Weighted average
    """
    if not values:
        return 0.0
    
    if weights is None:
        return sum(values) / len(values)
    
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def exponential_moving_average(values: List[float], alpha: float) -> float:
    """
    Calculate exponential moving average
    
    Args:
        values: List of values (most recent last)
        alpha: Smoothing factor (0 < alpha <= 1)
    
    Returns:
        EMA value
    """
    if not values:
        return 0.0
    
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1 - alpha) * ema
    
    return ema


def standard_deviation(values: List[float], ddof: int = 1) -> float:
    """
    Calculate sample standard deviation
    
    Args:
        values: List of values
        ddof: Delta degrees of freedom
    
    Returns:
        Standard deviation
    """
    if len(values) < 2:
        return 0.0
    
    if NUMPY_AVAILABLE:
        return float(np.std(values, ddof=ddof))
    
    # Pure Python implementation
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - ddof)
    return math.sqrt(variance)


def percentile(values: List[float], p: float) -> float:
    """
    Calculate percentile
    
    Args:
        values: List of values
        p: Percentile (0-100)
    
    Returns:
        Percentile value
    """
    if not values:
        return 0.0
    
    if NUMPY_AVAILABLE:
        return float(np.percentile(values, p))
    
    # Pure Python implementation
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient
    
    Args:
        x: First list of values
        y: Second list of values
    
    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    if NUMPY_AVAILABLE:
        return float(np.corrcoef(x, y)[0, 1])
    
    # Pure Python implementation
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)
    sum_y2 = sum(yi ** 2 for yi in y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


# =============================================================================
# Time Utilities
# =============================================================================

def time_to_next_interval(interval_minutes: int) -> float:
    """
    Calculate seconds to next interval
    
    Args:
        interval_minutes: Interval in minutes
    
    Returns:
        Seconds to next interval
    """
    now = datetime.now()
    current_minute = now.minute
    next_interval = ((current_minute // interval_minutes) + 1) * interval_minutes
    
    next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_interval)
    
    return (next_time - now).total_seconds()


def get_interval_minutes(timeframe: str) -> int:
    """
    Convert timeframe string to minutes
    
    Args:
        timeframe: Timeframe string (e.g., 'M5', 'H1', 'D1', 'W1')
    
    Returns:
        Minutes in interval
    """
    units = {
        'M': 1,
        'H': 60,
        'D': 1440,
        'W': 10080,
        'MN': 43200  # Month (30 days)
    }
    
    # Extract number and unit
    match = re.match(r'(\d+)([A-Z]+)', timeframe.upper())
    if not match:
        # Default to 1 if no number
        match = re.match(r'([A-Z]+)', timeframe.upper())
        if not match:
            return 60  # Default to 1H
        number = 1
        unit = match.group(1)
    else:
        number = int(match.group(1))
        unit = match.group(2)
    
    return number * units.get(unit, 60)


def get_date_range(start_date: Union[str, datetime, None] = None,
                  end_date: Union[str, datetime, None] = None,
                  periods: Optional[int] = None,
                  freq: str = 'D') -> List[datetime]:
    """
    Generate date range
    
    Args:
        start_date: Start date
        end_date: End date
        periods: Number of periods
        freq: Frequency ('D', 'H', '15T', etc.)
    
    Returns:
        List of dates
    """
    if start_date is None and end_date is None:
        raise ValueError("Must specify either start_date or end_date")
    
    # Convert string dates
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date) if 'pd' in globals() else datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date) if 'pd' in globals() else datetime.fromisoformat(end_date)
    
    # Generate range
    if 'pd' in globals():
        # Use pandas if available
        if periods:
            dates = pd.date_range(end=end_date, periods=periods, freq=freq)
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        return dates.tolist()
    else:
        # Manual generation
        dates = []
        current = start_date or (end_date - timedelta(days=periods))
        
        while current <= (end_date or current):
            dates.append(current)
            # Add based on frequency (simplified)
            if freq == 'D':
                current += timedelta(days=1)
            elif freq == 'H':
                current += timedelta(hours=1)
            elif freq == '15T':
                current += timedelta(minutes=15)
            else:
                current += timedelta(days=1)
        
        return dates


def is_market_open(symbol: str, timestamp: Optional[datetime] = None) -> bool:
    """
    Check if market is open for symbol
    
    Args:
        symbol: Trading symbol
        timestamp: Time to check (default: now)
    
    Returns:
        True if market is open
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    instrument_type = get_instrument_type(symbol)
    
    # Forex: 24/5 (closed weekends)
    if instrument_type == 'forex':
        return timestamp.weekday() < 5
    
    # Crypto: 24/7
    if instrument_type == 'crypto':
        return True
    
    # Indices: specific hours
    if instrument_type == 'index':
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        if symbol == 'US30' or symbol == 'SPX500':
            # US indices: 9:30 AM - 4:00 PM ET
            return weekday < 5 and 9 <= hour < 16
        elif symbol == 'FTSE100':
            # UK: 8:00 AM - 4:30 PM GMT
            return weekday < 5 and 8 <= hour < 16
        elif symbol == 'DAX40':
            # German: 9:00 AM - 5:30 PM CET
            return weekday < 5 and 9 <= hour < 17
    
    # Metals: 24/5 (closed weekends)
    if instrument_type == 'metal':
        return timestamp.weekday() < 5
    
    return True


def get_next_market_open(symbol: str, from_time: Optional[datetime] = None) -> datetime:
    """
    Get next market open time
    
    Args:
        symbol: Trading symbol
        from_time: Starting time (default: now)
    
    Returns:
        Next market open datetime
    """
    if from_time is None:
        from_time = datetime.now()
    
    current = from_time
    
    while not is_market_open(symbol, current):
        current += timedelta(hours=1)
    
    return current


def get_previous_market_close(symbol: str, from_time: Optional[datetime] = None) -> datetime:
    """
    Get previous market close time
    
    Args:
        symbol: Trading symbol
        from_time: Starting time (default: now)
    
    Returns:
        Previous market close datetime
    """
    if from_time is None:
        from_time = datetime.now()
    
    current = from_time
    
    while is_market_open(symbol, current):
        current -= timedelta(hours=1)
    
    return current


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


def format_timedelta(delta: timedelta) -> str:
    """
    Format timedelta to human-readable string
    
    Args:
        delta: timedelta object
    
    Returns:
        Formatted string
    """
    days = delta.days
    seconds = delta.seconds
    
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return ' '.join(parts)


# =============================================================================
# Decorators
# =============================================================================

def retry_with_backoff(retries: int = 3, backoff: float = 1.0, 
                       exceptions: tuple = (Exception,),
                       max_delay: float = 60.0) -> Callable:
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        retries: Number of retries
        backoff: Backoff factor
        exceptions: Exceptions to catch
        max_delay: Maximum delay between retries
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == retries - 1:
                        raise
                    
                    wait_time = min(backoff * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Retry {attempt + 1}/{retries} for {func.__name__} "
                        f"after {wait_time:.2f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == retries - 1:
                        raise
                    
                    wait_time = min(backoff * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Retry {attempt + 1}/{retries} for {func.__name__} "
                        f"after {wait_time:.2f}s: {e}"
                    )
                    time.sleep(wait_time)
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def memoize(ttl: Optional[int] = None, maxsize: Optional[int] = None) -> Callable:
    """
    Memoization decorator with optional TTL and size limit
    
    Args:
        ttl: Time-to-live in seconds (None for infinite)
        maxsize: Maximum cache size (None for unlimited)
    
    Returns:
        Decorated function
    """
    cache = {}
    cache_stats = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            # Include function module and name in key
            key = f"{func.__module__}.{func.__name__}:{hashlib.md5(''.join(key_parts).encode()).hexdigest()}"
            
            now = time.time()
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (now - timestamp) < ttl:
                    cache_stats[key] = cache_stats.get(key, 0) + 1
                    return result
            
            # Compute result
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            
            # Enforce maxsize
            if maxsize and len(cache) > maxsize:
                # Remove oldest entry
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
                if oldest_key in cache_stats:
                    del cache_stats[oldest_key]
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            'size': len(cache),
            'hits': sum(cache_stats.values()),
            'keys': list(cache.keys())
        }
        
        return wrapper
    
    return decorator


def timing(func: Callable) -> Callable:
    """Decorator to measure and log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        
        # Add to function attributes
        if not hasattr(func, 'total_time'):
            func.total_time = 0
            func.call_count = 0
        
        func.total_time += elapsed
        func.call_count += 1
        func.avg_time = func.total_time / func.call_count
        
        return result
    
    return wrapper


def singleton(cls):
    """Singleton decorator for classes"""
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def rate_limit(max_calls: int, period: float) -> Callable:
    """
    Rate limiting decorator
    
    Args:
        max_calls: Maximum number of calls allowed
        period: Time period in seconds
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        calls = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls
            calls[:] = [call for call in calls if call > now - period]
            
            if len(calls) >= max_calls:
                wait_time = calls[0] + period - now
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
            
            calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def deprecated(message: Optional[str] = None):
    """Decorator to mark functions as deprecated"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = message or f"{func.__name__} is deprecated"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Dictionary Utilities
# =============================================================================

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        d: Nested dictionary
        parent_key: Parent key for recursion
        sep: Separator for keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) and all(isinstance(i, MutableMapping) for i in v):
            # Handle list of dictionaries
            for i, item in enumerate(v):
                items.extend(flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten flattened dictionary
    
    Args:
        d: Flattened dictionary
        sep: Separator used in keys
    
    Returns:
        Nested dictionary
    """
    result = {}
    
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        
        for part in parts[:-1]:
            # Handle array indices
            if '[' in part and ']' in part:
                name, idx = part[:-1].split('[')
                idx = int(idx)
                
                if name not in target:
                    target[name] = []
                
                while len(target[name]) <= idx:
                    target[name].append({})
                
                target = target[name][idx]
            else:
                if part not in target:
                    target[part] = {}
                target = target[part]
        
        last_part = parts[-1]
        if '[' in last_part and ']' in last_part:
            name, idx = last_part[:-1].split('[')
            idx = int(idx)
            
            if name not in target:
                target[name] = []
            
            while len(target[name]) <= idx:
                target[name].append(None)
            
            target[name][idx] = value
        else:
            target[last_part] = value
    
    return result


def deep_merge(dict1: Dict, dict2: Dict, overwrite: bool = True) -> Dict:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        overwrite: Overwrite existing keys
    
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value, overwrite)
        elif key in result and not overwrite:
            continue
        else:
            result[key] = value
    
    return result


def nested_get(d: Dict, key: str, default: Any = None, sep: str = '.') -> Any:
    """
    Get value from nested dictionary using dot notation
    
    Args:
        d: Dictionary
        key: Dot notation key
        default: Default value
        sep: Separator
    
    Returns:
        Value or default
    """
    keys = key.split(sep)
    value = d
    
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
            if value is None:
                return default
        elif isinstance(value, list) and k.isdigit():
            idx = int(k)
            if idx < len(value):
                value = value[idx]
            else:
                return default
        else:
            return default
    
    return value


def nested_set(d: Dict, key: str, value: Any, sep: str = '.'):
    """
    Set value in nested dictionary using dot notation
    
    Args:
        d: Dictionary
        key: Dot notation key
        value: Value to set
        sep: Separator
    """
    keys = key.split(sep)
    target = d
    
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]
    
    target[keys[-1]] = value


# =============================================================================
# Export all functions
# =============================================================================

__all__ = [
    # Configuration
    'ConfigManager',
    'load_config',
    'save_config',
    
    # File management
    'create_directories',
    'ensure_file_directory',
    'get_project_root',
    'list_files',
    
    # Formatting
    'format_currency',
    'format_percentage',
    'format_number',
    'format_timestamp',
    'format_duration',
    'truncate_string',
    'slugify',
    'random_string',
    
    # Trading calculations
    'calculate_pips',
    'get_pip_multiplier',
    'calculate_point_value',
    'calculate_risk_reward',
    'calculate_position_size',
    'calculate_lot_size',
    'calculate_margin',
    'calculate_pip_value',
    'calculate_swap_points',
    
    # Symbol utilities
    'validate_symbol',
    'normalize_symbol',
    'get_instrument_type',
    'get_symbol_info',
    
    # Math utilities
    'safe_divide',
    'round_to_tick',
    'clamp',
    'weighted_average',
    'exponential_moving_average',
    'standard_deviation',
    'percentile',
    'correlation',
    
    # Time utilities
    'time_to_next_interval',
    'get_interval_minutes',
    'get_date_range',
    'is_market_open',
    'get_next_market_open',
    'get_previous_market_close',
    'parse_timedelta',
    'format_timedelta',
    
    # Decorators
    'retry_with_backoff',
    'memoize',
    'timing',
    'singleton',
    'rate_limit',
    'deprecated',
    
    # Dictionary utilities
    'flatten_dict',
    'unflatten_dict',
    'deep_merge',
    'nested_get',
    'nested_set',
    
    # Exceptions
    'HelperError',
    'ConfigurationError',
    'ValidationError',
    'CalculationError'
]