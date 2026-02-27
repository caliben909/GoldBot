"""
Technical Indicators - Comprehensive indicator calculations
High-performance technical indicators with vectorized operations,
comprehensive error handling, and advanced features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
from enum import Enum
from dataclasses import dataclass
import warnings
from functools import wraps, lru_cache
import math

warnings.filterwarnings('ignore')


# =============================================================================
# Custom Exceptions
# =============================================================================

class IndicatorError(Exception):
    """Base exception for indicator calculations"""
    pass


class InsufficientDataError(IndicatorError):
    """Raised when insufficient data for calculation"""
    pass


class InvalidParameterError(IndicatorError):
    """Raised when invalid parameters provided"""
    pass


# =============================================================================
# Enums and Constants
# =============================================================================

class Trend(Enum):
    """Trend direction"""
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


class Signal(Enum):
    """Trading signal"""
    BUY = 1
    SELL = -1
    HOLD = 0
    STRONG_BUY = 2
    STRONG_SELL = -2


class Divergence(Enum):
    """Divergence type"""
    BULLISH = 1
    BEARISH = -1
    NONE = 0
    HIDDEN_BULLISH = 2
    HIDDEN_BEARISH = -2


@dataclass
class IndicatorResult:
    """Container for indicator results with metadata"""
    values: Union[pd.Series, pd.DataFrame]
    signal: Optional[Signal] = None
    metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# Performance Decorators
# =============================================================================

def validate_data(min_periods: int = 1):
    """Decorator to validate input data"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(data: Union[pd.Series, pd.DataFrame], *args, **kwargs):
            if data is None or (isinstance(data, pd.Series) and data.empty) or \
               (isinstance(data, pd.DataFrame) and data.empty):
                raise InsufficientDataError(f"No data provided to {func.__name__}")
            
            if isinstance(data, pd.Series):
                if len(data) < min_periods:
                    raise InsufficientDataError(
                        f"Need at least {min_periods} periods for {func.__name__}, "
                        f"got {len(data)}"
                    )
            elif isinstance(data, pd.DataFrame):
                if len(data) < min_periods:
                    raise InsufficientDataError(
                        f"Need at least {min_periods} periods for {func.__name__}, "
                        f"got {len(data)}"
                    )
            
            return func(data, *args, **kwargs)
        return wrapper
    return decorator


def handle_nan(func: Callable) -> Callable:
    """Decorator to handle NaN values in indicator calculations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pd.Series):
            return result.fillna(method='ffill').fillna(method='bfill')
        elif isinstance(result, pd.DataFrame):
            return result.fillna(method='ffill').fillna(method='bfill')
        return result
    return wrapper


# =============================================================================
# Core Technical Indicators
# =============================================================================

class TechnicalIndicators:
    """
    Comprehensive technical indicators with vectorized operations
    
    Features:
    - Over 50 technical indicators
    - Vectorized operations for performance
    - Comprehensive error handling
    - Divergence detection
    - Pattern recognition
    - Multi-timeframe support
    """
    
    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        self._cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0}
    
    def _cache_key(self, func_name: str, data_hash: str, *args, **kwargs) -> str:
        """Generate cache key"""
        return f"{func_name}:{data_hash}:{args}:{kwargs}"
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result"""
        if key in self._cache:
            self._cache_stats['hits'] += 1
            return self._cache[key]
        self._cache_stats['misses'] += 1
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Cache result"""
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest item
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value
    
    def clear_cache(self):
        """Clear indicator cache"""
        self._cache.clear()
        self._cache_stats = {'hits': 0, 'misses': 0}
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return self._cache_stats.copy()
    
    @staticmethod
    def _validate_period(period: int, min_period: int = 1) -> int:
        """Validate and adjust period"""
        if period < min_period:
            raise InvalidParameterError(f"Period must be >= {min_period}, got {period}")
        return period
    
    @staticmethod
    def _validate_series(series: pd.Series, name: str) -> pd.Series:
        """Validate series"""
        if not isinstance(series, pd.Series):
            raise InvalidParameterError(f"{name} must be a pandas Series")
        return series
    
    # =========================================================================
    # Momentum Indicators
    # =========================================================================
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=14)
    def rsi(prices: pd.Series, period: int = 14, method: str = 'wilders') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Series of prices
            period: RSI period
            method: Calculation method ('wilders', 'ema', 'sma')
        
        Returns:
            Series of RSI values (0-100)
        """
        if period < 1:
            raise InvalidParameterError(f"Period must be >= 1, got {period}")
        
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        if method == 'wilders':
            # Wilder's smoothing (used in classic RSI)
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        elif method == 'ema':
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
        elif method == 'sma':
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
        else:
            raise InvalidParameterError(f"Unknown method: {method}")
        
        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                  k_period: int = 14, d_period: int = 3,
                  slowing: int = 3, method: str = 'fast') -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            k_period: %K period
            d_period: %D period
            slowing: Slowing period for slow stochastic
            method: 'fast', 'slow', or 'full'
        
        Returns:
            DataFrame with %K and %D columns
        """
        # Calculate lowest low and highest high over period
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # Fast %K
        fast_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        if method == 'fast':
            k = fast_k
            d = k.rolling(window=d_period).mean()
        elif method == 'slow':
            # Slow %K is SMA of fast %K
            k = fast_k.rolling(window=slowing).mean()
            d = k.rolling(window=d_period).mean()
        elif method == 'full':
            # Full stochastic uses smoothed %K and %D
            k = fast_k.rolling(window=slowing).mean()
            d = k.rolling(window=d_period).mean()
        else:
            raise InvalidParameterError(f"Unknown method: {method}")
        
        return pd.DataFrame({'%K': k, '%D': d})
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=26)
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9, ma_type: str = 'ema') -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Series of prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            ma_type: Moving average type ('ema', 'sma', 'wma')
        
        Returns:
            DataFrame with MACD, Signal, and Histogram columns
        """
        # Calculate fast and slow moving averages
        if ma_type == 'ema':
            fast_ma = prices.ewm(span=fast, adjust=False).mean()
            slow_ma = prices.ewm(span=slow, adjust=False).mean()
        elif ma_type == 'sma':
            fast_ma = prices.rolling(window=fast).mean()
            slow_ma = prices.rolling(window=slow).mean()
        elif ma_type == 'wma':
            fast_ma = prices.rolling(window=fast).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
            )
            slow_ma = prices.rolling(window=slow).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
            )
        else:
            raise InvalidParameterError(f"Unknown MA type: {ma_type}")
        
        macd_line = fast_ma - slow_ma
        
        # Signal line
        if ma_type == 'ema':
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        else:
            signal_line = macd_line.rolling(window=signal).mean()
        
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=14)
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: Lookback period
        
        Returns:
            Series of Williams %R values (-100 to 0)
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=14)
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: Lookback period
        
        Returns:
            Series of CCI values
        """
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        cci = (tp - sma) / (0.015 * mad)
        
        return cci
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def awesome_oscillator(high: pd.Series, low: pd.Series, 
                          fast: int = 5, slow: int = 34) -> pd.Series:
        """
        Calculate Awesome Oscillator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            fast: Fast period
            slow: Slow period
        
        Returns:
            Series of Awesome Oscillator values
        """
        median = (high + low) / 2
        fast_sma = median.rolling(window=fast).mean()
        slow_sma = median.rolling(window=slow).mean()
        
        return fast_sma - slow_sma
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def momentum(prices: pd.Series, period: int = 10, method: str = 'simple') -> pd.Series:
        """
        Calculate Momentum indicator
        
        Args:
            prices: Series of prices
            period: Momentum period
            method: 'simple' (price difference) or 'rate' (rate of change)
        
        Returns:
            Series of momentum values
        """
        if method == 'simple':
            return prices.diff(period)
        elif method == 'rate':
            return prices.pct_change(period) * 100
        else:
            raise InvalidParameterError(f"Unknown method: {method}")
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=14)
    def roc(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change (ROC)
        
        Args:
            prices: Series of prices
            period: ROC period
        
        Returns:
            Series of ROC values
        """
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=14)
    def tsi(prices: pd.Series, long: int = 25, short: int = 13, signal: int = 13) -> pd.DataFrame:
        """
        Calculate True Strength Index (TSI)
        
        Args:
            prices: Series of prices
            long: Long period
            short: Short period
            signal: Signal period
        
        Returns:
            DataFrame with TSI and Signal columns
        """
        diff = prices.diff()
        
        # Double smooth the momentum
        momentum = diff.ewm(span=long, adjust=False).mean()
        momentum = momentum.ewm(span=short, adjust=False).mean()
        
        # Double smooth the absolute momentum
        abs_momentum = diff.abs().ewm(span=long, adjust=False).mean()
        abs_momentum = abs_momentum.ewm(span=short, adjust=False).mean()
        
        tsi = 100 * (momentum / abs_momentum)
        signal_line = tsi.ewm(span=signal, adjust=False).mean()
        
        return pd.DataFrame({'TSI': tsi, 'Signal': signal_line})
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def uo(high: pd.Series, low: pd.Series, close: pd.Series,
           period1: int = 7, period2: int = 14, period3: int = 28,
           weight1: float = 4.0, weight2: float = 2.0, weight3: float = 1.0) -> pd.Series:
        """
        Calculate Ultimate Oscillator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period1: First period
            period2: Second period
            period3: Third period
            weight1: Weight for first period
            weight2: Weight for second period
            weight3: Weight for third period
        
        Returns:
            Series of Ultimate Oscillator values
        """
        # Calculate buying pressure
        bp = close - pd.concat([low.shift(1), close.shift(1)], axis=1).min(axis=1)
        
        # Calculate true range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # Calculate averages for each period
        avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
        avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
        avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
        
        # Calculate Ultimate Oscillator
        uo = 100 * (weight1 * avg1 + weight2 * avg2 + weight3 * avg3) / (weight1 + weight2 + weight3)
        
        return uo
    
    # =========================================================================
    # Volatility Indicators
    # =========================================================================
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=14)
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14, method: str = 'classic') -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period
            method: 'classic' or 'wilders'
        
        Returns:
            Series of ATR values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        if method == 'classic':
            atr = tr.rolling(window=period).mean()
        elif method == 'wilders':
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
        else:
            raise InvalidParameterError(f"Unknown method: {method}")
        
        return atr
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def bollinger_bands(prices: pd.Series, period: int = 20, 
                       std_dev: float = 2.0, ma_type: str = 'sma') -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Series of prices
            period: Moving average period
            std_dev: Number of standard deviations
            ma_type: Moving average type ('sma', 'ema', 'wma')
        
        Returns:
            DataFrame with Upper, Middle, Lower bands
        """
        if ma_type == 'sma':
            middle = prices.rolling(window=period).mean()
        elif ma_type == 'ema':
            middle = prices.ewm(span=period, adjust=False).mean()
        elif ma_type == 'wma':
            middle = prices.rolling(window=period).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
            )
        else:
            raise InvalidParameterError(f"Unknown MA type: {ma_type}")
        
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Bandwidth and %B
        bandwidth = (upper - lower) / middle
        percent_b = (prices - lower) / (upper - lower)
        
        return pd.DataFrame({
            'Middle': middle,
            'Upper': upper,
            'Lower': lower,
            'Bandwidth': bandwidth,
            '%B': percent_b
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 20, atr_period: int = 10,
                        multiplier: float = 2.0, ma_type: str = 'ema') -> pd.DataFrame:
        """
        Calculate Keltner Channels
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: EMA period for middle line
            atr_period: ATR period for channel width
            multiplier: ATR multiplier
            ma_type: Moving average type
        
        Returns:
            DataFrame with Upper, Middle, Lower channels
        """
        if ma_type == 'ema':
            middle = close.ewm(span=period, adjust=False).mean()
        else:
            middle = close.rolling(window=period).mean()
        
        atr = TechnicalIndicators.atr(high, low, close, atr_period)
        
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        
        return pd.DataFrame({
            'Middle': middle,
            'Upper': upper,
            'Lower': lower
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
        """
        Calculate Donchian Channels
        
        Args:
            high: Series of high prices
            low: Series of low prices
            period: Lookback period
        
        Returns:
            DataFrame with Upper, Middle, Lower channels
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return pd.DataFrame({
            'Upper': upper,
            'Middle': middle,
            'Lower': lower
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=14)
    def volatility(high: pd.Series, low: pd.Series, close: pd.Series,
                  open: Optional[pd.Series] = None, period: int = 20,
                  method: str = 'close') -> pd.Series:
        """
        Calculate various volatility metrics
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            open: Series of open prices (required for Garman-Klass)
            period: Rolling period
            method: 'close', 'parkinson', 'garman_klass', 'rogers_satchell'
        
        Returns:
            Series of volatility values
        """
        if method == 'close':
            returns = close.pct_change()
            vol = returns.rolling(period).std() * np.sqrt(252)
        
        elif method == 'parkinson':
            # Parkinson volatility (high-low)
            log_hl = np.log(high / low)
            vol = log_hl.rolling(period).std() * np.sqrt(252 * 4 * np.log(2))
        
        elif method == 'garman_klass':
            # Garman-Klass volatility
            if open is None:
                raise InvalidParameterError("Open prices required for Garman-Klass volatility")
            
            log_hl = np.log(high / low)
            log_co = np.log(close / open)
            
            vol = np.sqrt(
                0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
            ).rolling(period).mean() * np.sqrt(252)
        
        elif method == 'rogers_satchell':
            # Rogers-Satchell volatility
            if open is None:
                raise InvalidParameterError("Open prices required for Rogers-Satchell volatility")
            
            log_hc = np.log(high / close)
            log_ho = np.log(high / open)
            log_lc = np.log(low / close)
            log_lo = np.log(low / open)
            
            vol = np.sqrt(
                log_hc * log_ho + log_lc * log_lo
            ).rolling(period).mean() * np.sqrt(252)
        
        else:
            raise InvalidParameterError(f"Unknown method: {method}")
        
        return vol
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate ATR as percentage of close price
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period
        
        Returns:
            Series of ATR percentage values
        """
        atr = TechnicalIndicators.atr(high, low, close, period)
        return (atr / close) * 100
    
    # =========================================================================
    # Trend Indicators
    # =========================================================================
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ADX period
        
        Returns:
            DataFrame with ADX, +DI, -DI
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with Wilder's method
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=low.index).ewm(alpha=1/period, adjust=False).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            '+DI': plus_di,
            '-DI': minus_di
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
        """
        Calculate Aroon Indicator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            period: Lookback period
        
        Returns:
            DataFrame with Aroon Up, Aroon Down, and Oscillator
        """
        def aroon_up(x):
            return ((period - x.argmax()) / period) * 100
        
        def aroon_down(x):
            return ((period - x.argmin()) / period) * 100
        
        aroon_up = high.rolling(window=period + 1).apply(aroon_up, raw=True)
        aroon_down = low.rolling(window=period + 1).apply(aroon_down, raw=True)
        
        oscillator = aroon_up - aroon_down
        
        return pd.DataFrame({
            'Aroon_Up': aroon_up,
            'Aroon_Down': aroon_down,
            'Oscillator': oscillator
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=50)
    def parabolic_sar(high: pd.Series, low: pd.Series, 
                      acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """
        Calculate Parabolic SAR
        
        Args:
            high: Series of high prices
            low: Series of low prices
            acceleration: Acceleration factor
            maximum: Maximum acceleration factor
        
        Returns:
            Series of Parabolic SAR values
        """
        length = len(high)
        sar = np.zeros(length)
        ep = np.zeros(length)  # Extreme point
        af = acceleration  # Acceleration factor
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        # Initialize
        sar[0] = low.iloc[0] if trend == 1 else high.iloc[0]
        ep[0] = high.iloc[0] if trend == 1 else low.iloc[0]
        
        for i in range(1, length):
            if trend == 1:  # Uptrend
                sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
                
                # Ensure SAR is below low
                sar[i] = min(sar[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
                
                if low.iloc[i] < sar[i]:  # Reversal
                    trend = -1
                    sar[i] = ep[i-1]
                    ep[i] = low.iloc[i]
                    af = acceleration
                else:
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af = min(af + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
            else:  # Downtrend
                sar[i] = sar[i-1] + af * (ep[i-1] - sar[i-1])
                
                # Ensure SAR is above high
                sar[i] = max(sar[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
                
                if high.iloc[i] > sar[i]:  # Reversal
                    trend = 1
                    sar[i] = ep[i-1]
                    ep[i] = high.iloc[i]
                    af = acceleration
                else:
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af = min(af + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
        
        return pd.Series(sar, index=high.index)
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=50)
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                 tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            tenkan: Tenkan-sen period
            kijun: Kijun-sen period
            senkou: Senkou Span B period
        
        Returns:
            DataFrame with Ichimoku components
        """
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan).max()
        tenkan_low = low.rolling(window=tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun).max()
        kijun_low = low.rolling(window=kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=senkou).max()
        senkou_low = low.rolling(window=senkou).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-kijun)
        
        # Cloud color (Kumo)
        kumo = senkou_span_a > senkou_span_b
        
        return pd.DataFrame({
            'Tenkan': tenkan_sen,
            'Kijun': kijun_sen,
            'Senkou_A': senkou_span_a,
            'Senkou_B': senkou_span_b,
            'Chikou': chikou_span,
            'Kumo_Bullish': kumo
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def vortex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """
        Calculate Vortex Indicator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: Lookback period
        
        Returns:
            DataFrame with VI+ and VI-
        """
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # VM+ and VM-
        vm_plus = abs(high - low.shift()).rolling(window=period).sum()
        vm_minus = abs(low - high.shift()).rolling(window=period).sum()
        
        # TR sum
        tr_sum = tr.rolling(window=period).sum()
        
        # VI+ and VI-
        vi_plus = vm_plus / tr_sum
        vi_minus = vm_minus / tr_sum
        
        return pd.DataFrame({
            'VI+': vi_plus,
            'VI-': vi_minus
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def moving_average(prices: pd.Series, period: int = 20, 
                       ma_type: str = 'sma', **kwargs) -> pd.Series:
        """
        Calculate various moving averages
        
        Args:
            prices: Series of prices
            period: Moving average period
            ma_type: Type of MA ('sma', 'ema', 'wma', 'hma', 'dema', 'tema', 'kama')
            **kwargs: Additional parameters for specific MA types
        
        Returns:
            Series of moving average values
        """
        if ma_type == 'sma':
            return prices.rolling(window=period).mean()
        
        elif ma_type == 'ema':
            return prices.ewm(span=period, adjust=False).mean()
        
        elif ma_type == 'wma':
            weights = np.arange(1, period + 1)
            return prices.rolling(window=period).apply(
                lambda x: np.average(x, weights=weights), raw=True
            )
        
        elif ma_type == 'hma':
            # Hull Moving Average
            half_period = int(period / 2)
            sqrt_period = int(np.sqrt(period))
            
            wma_half = TechnicalIndicators.moving_average(prices, half_period, 'wma')
            wma_full = TechnicalIndicators.moving_average(prices, period, 'wma')
            
            raw_hma = 2 * wma_half - wma_full
            return TechnicalIndicators.moving_average(raw_hma, sqrt_period, 'wma')
        
        elif ma_type == 'dema':
            # Double Exponential Moving Average
            ema1 = prices.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            return 2 * ema1 - ema2
        
        elif ma_type == 'tema':
            # Triple Exponential Moving Average
            ema1 = prices.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        
        elif ma_type == 'kama':
            # Kaufman Adaptive Moving Average
            fast = kwargs.get('fast', 2)
            slow = kwargs.get('slow', 30)
            
            # Efficiency Ratio
            change = abs(prices - prices.shift(period))
            volatility = abs(prices.diff()).rolling(window=period).sum()
            er = change / volatility
            
            # Smoothing Constant
            fast_sc = 2 / (fast + 1)
            slow_sc = 2 / (slow + 1)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            
            # Calculate KAMA
            kama = prices.copy()
            for i in range(period, len(prices)):
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])
            
            return kama
        
        else:
            raise InvalidParameterError(f"Unknown MA type: {ma_type}")
    
    # =========================================================================
    # Volume Indicators
    # =========================================================================
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)
        
        Args:
            close: Series of close prices
            volume: Series of volume
        
        Returns:
            Series of OBV values
        """
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
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
            volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            volume: Series of volume
            period: Lookback period
        
        Returns:
            Series of MFI values (0-100)
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        # Sum over period
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        # Money Ratio and MFI
        money_ratio = positive_sum / negative_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            volume: Series of volume
            period: Lookback period
        
        Returns:
            Series of CMF values
        """
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # CMF
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def ad_line(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            volume: Series of volume
        
        Returns:
            Series of A/D Line values
        """
        # Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        
        # Money Flow Volume
        mfv = clv * volume
        
        # Accumulation/Distribution Line
        ad_line = mfv.cumsum()
        
        return ad_line
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def eom(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Ease of Movement (EOM)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            volume: Series of volume
            period: Lookback period
        
        Returns:
            Series of EOM values
        """
        # Distance Moved
        distance = ((high + low) / 2) - ((high.shift() + low.shift()) / 2)
        
        # Box Ratio
        box_ratio = volume / (high - low)
        
        # Ease of Movement
        eom = distance / box_ratio
        eom_smoothed = eom.rolling(window=period).mean()
        
        return eom_smoothed
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            volume: Series of volume
            period: Rolling period (None for cumulative VWAP)
        
        Returns:
            Series of VWAP values
        """
        typical_price = (high + low + close) / 3
        
        if period is None:
            # Cumulative VWAP
            cum_vol_price = (typical_price * volume).cumsum()
            cum_volume = volume.cumsum()
            vwap = cum_vol_price / cum_volume
        else:
            # Rolling VWAP
            cum_vol_price = (typical_price * volume).rolling(window=period).sum()
            cum_volume = volume.rolling(window=period).sum()
            vwap = cum_vol_price / cum_volume
        
        return vwap
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def volume_profile(df: pd.DataFrame, num_bins: int = 24, 
                       value_area: float = 0.7) -> Dict[str, Any]:
        """
        Calculate Volume Profile (Market Profile)
        
        Args:
            df: DataFrame with high, low, volume columns
            num_bins: Number of price bins
            value_area: Percentage of volume for value area (0.7 = 70%)
        
        Returns:
            Dictionary with volume profile statistics
        """
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        if price_range == 0:
            return {}
        
        bin_size = price_range / num_bins
        
        volume_profile = []
        volume_by_price = {}
        
        for i in range(num_bins):
            lower = price_min + (i * bin_size)
            upper = lower + bin_size
            
            # Find bars that intersect this price level
            mask = (df['high'] >= lower) & (df['low'] <= upper)
            
            if mask.any():
                volume_sum = 0
                for idx in df[mask].index:
                    bar = df.loc[idx]
                    overlap = min(bar['high'], upper) - max(bar['low'], lower)
                    bar_range = bar['high'] - bar['low']
                    
                    if bar_range > 0:
                        contribution = (overlap / bar_range) * bar['volume']
                        volume_sum += contribution
                
                price_level = (lower + upper) / 2
                volume_profile.append({
                    'price_level': price_level,
                    'volume': volume_sum,
                    'lower': lower,
                    'upper': upper
                })
                volume_by_price[price_level] = volume_sum
        
        profile_df = pd.DataFrame(volume_profile)
        
        if profile_df.empty:
            return {}
        
        # Find Point of Control (POC)
        poc_idx = profile_df['volume'].idxmax()
        poc = profile_df.loc[poc_idx, 'price_level']
        
        # Calculate Value Area (where x% of volume occurs)
        total_volume = profile_df['volume'].sum()
        sorted_by_volume = profile_df.sort_values('volume', ascending=False)
        
        cum_volume = 0
        value_area_prices = []
        
        for _, row in sorted_by_volume.iterrows():
            cum_volume += row['volume']
            value_area_prices.append(row['price_level'])
            if cum_volume >= total_volume * value_area:
                break
        
        value_area_high = max(value_area_prices) if value_area_prices else poc
        value_area_low = min(value_area_prices) if value_area_prices else poc
        
        return {
            'profile': profile_df,
            'poc': poc,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'total_volume': total_volume,
            'bins': num_bins,
            'bin_size': bin_size
        }
    
    # =========================================================================
    # Market Structure Indicators
    # =========================================================================
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def swing_points(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2) -> pd.DataFrame:
        """
        Identify swing highs and lows
        
        Args:
            high: Series of high prices
            low: Series of low prices
            left: Number of bars to left for confirmation
            right: Number of bars to right for confirmation
        
        Returns:
            DataFrame with swing high/low indicators
        """
        swing_high = pd.Series(False, index=high.index)
        swing_low = pd.Series(False, index=low.index)
        
        for i in range(left, len(high) - right):
            # Check swing high
            if high.iloc[i] == max(high.iloc[i-left:i+right+1]):
                swing_high.iloc[i] = True
            
            # Check swing low
            if low.iloc[i] == min(low.iloc[i-left:i+right+1]):
                swing_low.iloc[i] = True
        
        return pd.DataFrame({
            'swing_high': swing_high,
            'swing_low': swing_low,
            'swing_high_price': high.where(swing_high, np.nan),
            'swing_low_price': low.where(swing_low, np.nan)
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def fractals(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2) -> pd.DataFrame:
        """
        Identify Bill Williams Fractals
        
        Args:
            high: Series of high prices
            low: Series of low prices
            left: Number of bars to left
            right: Number of bars to right
        
        Returns:
            DataFrame with fractal indicators
        """
        up_fractal = pd.Series(False, index=high.index)
        down_fractal = pd.Series(False, index=low.index)
        
        for i in range(left, len(high) - right):
            # Up fractal (bearish reversal)
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1]:
                up_fractal.iloc[i] = True
            
            # Down fractal (bullish reversal)
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i+1]:
                down_fractal.iloc[i] = True
        
        return pd.DataFrame({
            'up_fractal': up_fractal,
            'down_fractal': down_fractal,
            'up_fractal_price': high.where(up_fractal, np.nan),
            'down_fractal_price': low.where(down_fractal, np.nan)
        })
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series,
                     method: str = 'classic') -> pd.DataFrame:
        """
        Calculate pivot points for each bar
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            method: Pivot method ('classic', 'fibonacci', 'woodie', 'camarilla', 'demark')
        
        Returns:
            DataFrame with pivot levels
        """
        pivots = pd.DataFrame(index=high.index)
        
        for i in range(1, len(high)):
            prev_high = high.iloc[i-1]
            prev_low = low.iloc[i-1]
            prev_close = close.iloc[i-1]
            
            if method == 'classic':
                pp = (prev_high + prev_low + prev_close) / 3
                r1 = 2 * pp - prev_low
                r2 = pp + (prev_high - prev_low)
                r3 = prev_high + 2 * (pp - prev_low)
                s1 = 2 * pp - prev_high
                s2 = pp - (prev_high - prev_low)
                s3 = prev_low - 2 * (prev_high - pp)
            
            elif method == 'fibonacci':
                pp = (prev_high + prev_low + prev_close) / 3
                r1 = pp + 0.382 * (prev_high - prev_low)
                r2 = pp + 0.618 * (prev_high - prev_low)
                r3 = pp + 1.0 * (prev_high - prev_low)
                s1 = pp - 0.382 * (prev_high - prev_low)
                s2 = pp - 0.618 * (prev_high - prev_low)
                s3 = pp - 1.0 * (prev_high - prev_low)
            
            elif method == 'woodie':
                pp = (prev_high + prev_low + 2 * prev_close) / 4
                r1 = 2 * pp - prev_low
                r2 = pp + (prev_high - prev_low)
                r3 = prev_high + 2 * (pp - prev_low)
                s1 = 2 * pp - prev_high
                s2 = pp - (prev_high - prev_low)
                s3 = prev_low - 2 * (prev_high - pp)
            
            elif method == 'camarilla':
                pp = (prev_high + prev_low + prev_close) / 3
                r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
                r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
                r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
                r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
                s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
                s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
                s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
                s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
            
            elif method == 'demark':
                if prev_close < prev_open:
                    x = prev_high + 2 * prev_low + prev_close
                elif prev_close > prev_open:
                    x = 2 * prev_high + prev_low + prev_close
                else:
                    x = prev_high + prev_low + 2 * prev_close
                
                pp = x / 4
                r1 = x / 2 - prev_low
                s1 = x / 2 - prev_high
                r2 = s2 = None
            
            else:
                raise InvalidParameterError(f"Unknown pivot method: {method}")
            
            pivots.iloc[i] = {
                'PP': pp,
                'R1': r1,
                'R2': r2,
                'R3': r3,
                'S1': s1,
                'S2': s2,
                'S3': s3
            }
        
        return pivots
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def support_resistance(high: pd.Series, low: pd.Series, 
                           lookback: int = 20, tolerance: float = 0.01) -> pd.DataFrame:
        """
        Identify support and resistance levels
        
        Args:
            high: Series of high prices
            low: Series of low prices
            lookback: Lookback period
            tolerance: Price tolerance for level grouping
        
        Returns:
            DataFrame with support and resistance levels
        """
        levels = pd.DataFrame(index=high.index)
        
        for i in range(lookback, len(high)):
            # Get recent price range
            recent_high = high.iloc[i-lookback:i]
            recent_low = low.iloc[i-lookback:i]
            
            # Find swing points
            swing_highs = []
            swing_lows = []
            
            for j in range(1, lookback - 1):
                idx = i - lookback + j
                if (high.iloc[idx] > high.iloc[idx-1] and 
                    high.iloc[idx] > high.iloc[idx+1]):
                    swing_highs.append(high.iloc[idx])
                if (low.iloc[idx] < low.iloc[idx-1] and 
                    low.iloc[idx] < low.iloc[idx+1]):
                    swing_lows.append(low.iloc[idx])
            
            # Group nearby levels
            resistance = TechnicalIndicators._group_levels(swing_highs, tolerance)
            support = TechnicalIndicators._group_levels(swing_lows, tolerance)
            
            # Get nearest levels
            current_price = (high.iloc[i] + low.iloc[i]) / 2
            
            nearest_resistance = min([r for r in resistance if r > current_price], 
                                     default=None) if resistance else None
            nearest_support = max([s for s in support if s < current_price], 
                                  default=None) if support else None
            
            levels.iloc[i] = {
                'resistance_levels': resistance,
                'support_levels': support,
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support
            }
        
        return levels
    
    @staticmethod
    def _group_levels(levels: List[float], tolerance: float) -> List[float]:
        """Group nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        grouped = []
        current_group = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - sum(current_group) / len(current_group)) <= tolerance:
                current_group.append(level)
            else:
                grouped.append(sum(current_group) / len(current_group))
                current_group = [level]
        
        grouped.append(sum(current_group) / len(current_group))
        return grouped
    
    # =========================================================================
    # Statistical Indicators
    # =========================================================================
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def correlation(series1: pd.Series, series2: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate rolling correlation between two series
        
        Args:
            series1: First series
            series2: Second series
            period: Rolling window period
        
        Returns:
            Series of correlation values (-1 to 1)
        """
        return series1.rolling(window=period).corr(series2)
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def z_score(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Z-Score (how many standard deviations from mean)
        
        Args:
            prices: Series of prices
            period: Rolling window period
        
        Returns:
            Series of Z-Score values
        """
        mean = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return (prices - mean) / std.replace(0, np.nan)
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def kurtosis(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate rolling kurtosis
        
        Args:
            prices: Series of prices
            period: Rolling window period
        
        Returns:
            Series of kurtosis values
        """
        return prices.rolling(window=period).kurt()
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def skew(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate rolling skewness
        
        Args:
            prices: Series of prices
            period: Rolling window period
        
        Returns:
            Series of skewness values
        """
        return prices.rolling(window=period).skew()
    
    @staticmethod
    @handle_nan
    @validate_data(min_periods=20)
    def beta(stock_returns: pd.Series, market_returns: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate rolling beta
        
        Args:
            stock_returns: Stock returns series
            market_returns: Market returns series
            period: Rolling window period
        
        Returns:
            Series of beta values
        """
        covariance = stock_returns.rolling(window=period).cov(market_returns)
        variance = market_returns.rolling(window=period).var()
        
        return covariance / variance.replace(0, np.nan)
    
    # =========================================================================
    # Advanced Features
    # =========================================================================
    
    @staticmethod
    def detect_divergence(price: pd.Series, indicator: pd.Series, 
                          lookback: int = 20) -> pd.DataFrame:
        """
        Detect divergence between price and indicator
        
        Args:
            price: Price series
            indicator: Indicator series (RSI, MACD, etc.)
            lookback: Lookback period for swing detection
        
        Returns:
            DataFrame with divergence signals
        """
        # Find swing points in price
        price_swings = TechnicalIndicators.swing_points(price, price, left=2, right=2)
        price_highs = price_swings['swing_high_price']
        price_lows = price_swings['swing_low_price']
        
        # Find swing points in indicator
        ind_swings = TechnicalIndicators.swing_points(indicator, indicator, left=2, right=2)
        ind_highs = ind_swings['swing_high_price']
        ind_lows = ind_swings['swing_low_price']
        
        divergences = pd.DataFrame(index=price.index)
        divergences['regular_bullish'] = False
        divergences['regular_bearish'] = False
        divergences['hidden_bullish'] = False
        divergences['hidden_bearish'] = False
        
        for i in range(lookback, len(price)):
            # Look for regular bullish divergence (price makes lower low, indicator makes higher low)
            price_low = price_lows.iloc[i-lookback:i+1].dropna()
            ind_low = ind_lows.iloc[i-lookback:i+1].dropna()
            
            if len(price_low) >= 2 and len(ind_low) >= 2:
                if (price_low.iloc[-1] < price_low.iloc[-2] and 
                    ind_low.iloc[-1] > ind_low.iloc[-2]):
                    divergences.iloc[i]['regular_bullish'] = True
            
            # Look for regular bearish divergence (price makes higher high, indicator makes lower high)
            price_high = price_highs.iloc[i-lookback:i+1].dropna()
            ind_high = ind_highs.iloc[i-lookback:i+1].dropna()
            
            if len(price_high) >= 2 and len(ind_high) >= 2:
                if (price_high.iloc[-1] > price_high.iloc[-2] and 
                    ind_high.iloc[-1] < ind_high.iloc[-2]):
                    divergences.iloc[i]['regular_bearish'] = True
            
            # Hidden bullish divergence (price makes higher low, indicator makes lower low)
            if len(price_low) >= 2 and len(ind_low) >= 2:
                if (price_low.iloc[-1] > price_low.iloc[-2] and 
                    ind_low.iloc[-1] < ind_low.iloc[-2]):
                    divergences.iloc[i]['hidden_bullish'] = True
            
            # Hidden bearish divergence (price makes lower high, indicator makes higher high)
            if len(price_high) >= 2 and len(ind_high) >= 2:
                if (price_high.iloc[-1] < price_high.iloc[-2] and 
                    ind_high.iloc[-1] > ind_high.iloc[-2]):
                    divergences.iloc[i]['hidden_bearish'] = True
        
        return divergences
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect candlestick patterns
        
        Args:
            df: DataFrame with open, high, low, close columns
        
        Returns:
            DataFrame with pattern signals
        """
        patterns = pd.DataFrame(index=df.index)
        
        # Doji
        body = abs(df['close'] - df['open'])
        range_hl = df['high'] - df['low']
        patterns['doji'] = body < (range_hl * 0.1)
        
        # Hammer
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        body = abs(df['close'] - df['open'])
        
        patterns['hammer'] = (
            (lower_shadow > body * 2) &
            (upper_shadow < body * 0.5) &
            (df['close'] > df['open'])
        )
        
        # Shooting Star
        patterns['shooting_star'] = (
            (upper_shadow > body * 2) &
            (lower_shadow < body * 0.5) &
            (df['close'] < df['open'])
        )
        
        # Engulfing
        patterns['bullish_engulfing'] = (
            (df['close'].shift() < df['open'].shift()) &  # Previous red candle
            (df['close'] > df['open']) &  # Current green candle
            (df['open'] < df['close'].shift()) &  # Opens below previous close
            (df['close'] > df['open'].shift())  # Closes above previous open
        )
        
        patterns['bearish_engulfing'] = (
            (df['close'].shift() > df['open'].shift()) &  # Previous green candle
            (df['close'] < df['open']) &  # Current red candle
            (df['open'] > df['close'].shift()) &  # Opens above previous close
            (df['close'] < df['open'].shift())  # Closes below previous open
        )
        
        # Morning Star
        patterns['morning_star'] = (
            (df['close'].shift(2) < df['open'].shift(2)) &  # First red candle
            (abs(df['close'].shift() - df['open'].shift()) < body.shift() * 0.3) &  # Small body
            (df['close'] > df['open']) &  # Third green candle
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Closes above midpoint of first
        )
        
        # Evening Star
        patterns['evening_star'] = (
            (df['close'].shift(2) > df['open'].shift(2)) &  # First green candle
            (abs(df['close'].shift() - df['open'].shift()) < body.shift() * 0.3) &  # Small body
            (df['close'] < df['open']) &  # Third red candle
            (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Closes below midpoint of first
        )
        
        return patterns
    
    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            high: Highest point in the trend
            low: Lowest point in the trend
        
        Returns:
            Dictionary of Fibonacci levels
        """
        diff = high - low
        
        # Retracement levels
        retracement = {
            '0.0': high,
            '0.236': high - 0.236 * diff,
            '0.382': high - 0.382 * diff,
            '0.5': high - 0.5 * diff,
            '0.618': high - 0.618 * diff,
            '0.786': high - 0.786 * diff,
            '1.0': low
        }
        
        # Extension levels
        extension = {
            '1.272': high + 0.272 * diff,
            '1.382': high + 0.382 * diff,
            '1.5': high + 0.5 * diff,
            '1.618': high + 0.618 * diff,
            '2.0': high + 1.0 * diff,
            '2.618': high + 1.618 * diff
        }
        
        return {
            'retracement': retracement,
            'extension': extension
        }
    
    @staticmethod
    def calculate_fibonacci_retracement(df: pd.DataFrame, lookback_period: int = 20) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels for each bar based on recent swing points
        
        Args:
            df: DataFrame with high, low, close columns
            lookback_period: Number of bars to look back for swing points
        
        Returns:
            DataFrame with Fibonacci levels
        """
        fib_levels = pd.DataFrame(index=df.index)
        
        for i in range(lookback_period, len(df)):
            # Get recent swing high and low
            recent_high = df['high'].iloc[i-lookback_period:i+1].max()
            recent_low = df['low'].iloc[i-lookback_period:i+1].min()
            
            # Calculate Fibonacci levels
            fib = TechnicalIndicators.calculate_fibonacci_levels(recent_high, recent_low)
            
            # Store levels
            for level, price in fib['retracement'].items():
                fib_levels.loc[df.index[i], f'fib_ret_{level}'] = price
            
            # Distance to nearest level
            current_close = df['close'].iloc[i]
            levels = list(fib['retracement'].values())
            distances = [abs(current_close - l) for l in levels]
            nearest_idx = np.argmin(distances)
            
            fib_levels.loc[df.index[i], 'nearest_fib'] = levels[nearest_idx]
            fib_levels.loc[df.index[i], 'distance_to_fib'] = distances[nearest_idx]
        
        return fib_levels


# =============================================================================
# Legacy functions for backward compatibility
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Legacy function for RSI calculation"""
    return TechnicalIndicators.rsi(prices, period)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Legacy function for ATR calculation"""
    return TechnicalIndicators.atr(df['high'], df['low'], df['close'], period)


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Legacy function for MACD calculation"""
    result = TechnicalIndicators.macd(prices, fast, slow, signal)
    return result['MACD'], result['Signal'], result['Histogram']


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                             std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Legacy function for Bollinger Bands calculation"""
    result = TechnicalIndicators.bollinger_bands(prices, period, std_dev)
    return result['Middle'], result['Upper'], result['Lower']


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                        k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Legacy function for Stochastic calculation"""
    result = TechnicalIndicators.stochastic(high, low, close, k_period, d_period)
    return result['%K'], result['%D']


def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Legacy function for Momentum calculation"""
    return TechnicalIndicators.momentum(prices, period)


def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 24) -> pd.DataFrame:
    """Legacy function for Volume Profile calculation"""
    result = TechnicalIndicators.volume_profile(df, num_bins)
    return result.get('profile', pd.DataFrame())


def calculate_structure_features(df: pd.DataFrame, swing_length: int = 5) -> pd.DataFrame:
    """Legacy function for structure features"""
    # Simplified version for backward compatibility
    features = pd.DataFrame(index=df.index)
    
    features['dist_to_high'] = (df['close'] - df['high'].rolling(swing_length).max()) / df['close']
    features['dist_to_low'] = (df['close'] - df['low'].rolling(swing_length).min()) / df['close']
    
    recent_high = df['high'].rolling(swing_length).max().shift(1)
    recent_low = df['low'].rolling(swing_length).min().shift(1)
    
    features['break_high'] = (df['high'] > recent_high).astype(int)
    features['break_low'] = (df['low'] < recent_low).astype(int)
    
    return features


def calculate_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Legacy function for volatility calculation"""
    return TechnicalIndicators.volatility(df['high'], df['low'], df['close'], period=period)


def calculate_correlation(series1: pd.Series, series2: pd.Series, period: int = 20) -> pd.Series:
    """Legacy function for correlation calculation"""
    return TechnicalIndicators.correlation(series1, series2, period)


def calculate_fibonacci_levels(high: float, low: float) -> dict:
    """Legacy function for Fibonacci levels"""
    return TechnicalIndicators.calculate_fibonacci_levels(high, low)


def calculate_fibonacci_retracement(df: pd.DataFrame, lookback_period: int = 20) -> pd.DataFrame:
    """Legacy function for Fibonacci retracement"""
    return TechnicalIndicators.calculate_fibonacci_retracement(df, lookback_period)


def calculate_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                      tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    """Legacy function for Ichimoku Cloud calculation"""
    return TechnicalIndicators.ichimoku(high, low, close, tenkan, kijun, senkou)


def calculate_parabolic_sar(high: pd.Series, low: pd.Series, 
                           acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """Legacy function for Parabolic SAR calculation"""
    return TechnicalIndicators.parabolic_sar(high, low, acceleration, maximum)


# =============================================================================
# Export all functions
# =============================================================================

__all__ = [
    'TechnicalIndicators',
    'IndicatorError',
    'InsufficientDataError',
    'InvalidParameterError',
    'Trend',
    'Signal',
    'Divergence',
    'IndicatorResult',
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
    'calculate_fibonacci_levels',
    'calculate_fibonacci_retracement',
    'calculate_ichimoku',
    'calculate_parabolic_sar'
]