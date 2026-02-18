"""
Helpers - General utility functions
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
import logging
from datetime import datetime, timedelta
import asyncio
import functools
import hashlib
import time
import math

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                logger.error(f"Unsupported config format: {config_path.suffix}")
                return {}
        
        logger.info(f"Loaded config from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False)
            elif config_path.suffix == '.json':
                json.dump(config, f, indent=2)
            else:
                logger.error(f"Unsupported config format: {config_path.suffix}")
                return False
        
        logger.info(f"Saved config to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def create_directories(paths: List[Union[str, Path]]) -> bool:
    """
    Create directories if they don't exist
    
    Args:
        paths: List of directory paths
    
    Returns:
        True if all directories created/exist
    """
    success = True
    
    for path in paths:
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            success = False
    
    return success


def format_currency(value: float, currency: str = 'USD', decimals: int = 2) -> str:
    """
    Format value as currency string
    
    Args:
        value: Numeric value
        currency: Currency code
        decimals: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'BTC': '₿'
    }
    
    symbol = symbols.get(currency, '$')
    
    if currency == 'JPY':
        return f"{symbol}{value:,.0f}"
    else:
        return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Numeric value (0-100)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"


def format_timestamp(timestamp: Union[datetime, str, float], format: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format timestamp
    
    Args:
        timestamp: Timestamp (datetime, string, or Unix timestamp)
        format: Output format
    
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp)
    else:
        dt = timestamp
    
    return dt.strftime(format)


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
    if 'JPY' in symbol:
        return 100  # JPY pairs: 1 pip = 0.01
    elif 'XAU' in symbol or 'XAG' in symbol:
        return 10   # Metals: 1 pip = 0.1
    elif 'BTC' in symbol or 'ETH' in symbol:
        return 1    # Crypto: 1 pip = 1
    else:
        return 10000  # Forex: 1 pip = 0.0001


def calculate_point_value(symbol: str, lot_size: float = 1.0) -> float:
    """
    Calculate point value for symbol
    
    Args:
        symbol: Trading symbol
        lot_size: Lot size
    
    Returns:
        Point value in quote currency
    """
    if 'JPY' in symbol:
        return lot_size * 1000  # 1 lot = 1000 JPY per point
    elif 'XAU' in symbol:
        return lot_size * 100    # 1 lot = 100 USD per point
    elif 'BTC' in symbol:
        return lot_size * 1      # 1 lot = 1 USD per point
    else:
        return lot_size * 10     # 1 lot = 10 USD per point


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format
    
    Args:
        symbol: Trading symbol
    
    Returns:
        True if valid
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Check length
    if len(symbol) < 4 or len(symbol) > 10:
        return False
    
    # Check for common patterns
    valid_patterns = [
        symbol.endswith('USD'),
        symbol.endswith('JPY'),
        symbol.endswith('GBP'),
        symbol.endswith('EUR'),
        symbol.endswith('CHF'),
        symbol.endswith('CAD'),
        symbol.endswith('AUD'),
        symbol.endswith('NZD'),
        symbol.endswith('USDT'),
        symbol.endswith('BTC'),
        symbol.endswith('ETH')
    ]
    
    return any(valid_patterns)


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol to standard format
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Normalized symbol
    """
    return symbol.upper().strip().replace('/', '').replace('_', '')


def get_instrument_type(symbol: str) -> str:
    """
    Get instrument type from symbol
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Instrument type (forex, crypto, index, commodity)
    """
    symbol = symbol.upper()
    
    if symbol.endswith(('USDT', 'BTC', 'ETH', 'BNB', 'SOL')):
        return 'crypto'
    elif len(symbol) in [6, 7] and symbol.isalpha():
        return 'forex'
    elif symbol in ['XAUUSD', 'XAGUSD', 'XPTUSD']:
        return 'commodity'
    elif symbol in ['US30', 'SPX500', 'NAS100', 'FTSE100', 'DAX40', 'NIKKEI225']:
        return 'index'
    else:
        return 'unknown'


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
    if b == 0:
        return default
    return a / b


def round_to_tick(price: float, tick_size: float) -> float:
    """
    Round price to nearest tick
    
    Args:
        price: Price to round
        tick_size: Tick size
    
    Returns:
        Rounded price
    """
    return round(price / tick_size) * tick_size


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
                           point_value: float) -> float:
    """
    Calculate position size based on risk
    
    Args:
        risk_amount: Risk amount in account currency
        entry: Entry price
        stop: Stop loss price
        point_value: Point value per lot
    
    Returns:
        Position size in lots
    """
    risk_distance = abs(entry - stop)
    
    if risk_distance == 0:
        return 0
    
    return risk_amount / (risk_distance * point_value)


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
        timeframe: Timeframe string (e.g., 'M5', 'H1', 'D1')
    
    Returns:
        Minutes in interval
    """
    units = {
        'M': 1,
        'H': 60,
        'D': 1440,
        'W': 10080
    }
    
    number = int(''.join(filter(str.isdigit, timeframe)) or '1')
    unit = ''.join(filter(str.isalpha, timeframe))
    
    return number * units.get(unit, 1)


def retry_with_backoff(retries: int = 3, backoff: float = 1.0, 
                       exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        retries: Number of retries
        backoff: Backoff factor
        exceptions: Exceptions to catch
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == retries - 1:
                        raise
                    wait_time = backoff * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{retries} for {func.__name__} after {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
            return None
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == retries - 1:
                        raise
                    wait_time = backoff * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{retries} for {func.__name__} after {wait_time}s: {e}")
                    time.sleep(wait_time)
            return None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def memoize(ttl: Optional[int] = None) -> Callable:
    """
    Memoization decorator with optional TTL
    
    Args:
        ttl: Time-to-live in seconds
    
    Returns:
        Decorated function
    """
    cache = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(''.join(key_parts).encode()).hexdigest()
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    return result
            
            # Compute result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            # Clean old cache entries
            if ttl is not None:
                current_time = time.time()
                expired = [k for k, (_, ts) in cache.items() if (current_time - ts) > ttl]
                for k in expired:
                    del cache[k]
            
            return result
        
        return wrapper
    
    return decorator


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
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)