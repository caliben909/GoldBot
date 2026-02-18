"""
Technical Indicators - Comprehensive indicator calculations
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union
import warnings

warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Wrapper class for technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        return calculate_rsi(prices, period)
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return calculate_atr(df, period)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        return calculate_macd(prices, fast, slow, signal)
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        return calculate_bollinger_bands(prices, period, std_dev)
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        return calculate_stochastic(high, low, close, k_period, d_period)
    
    @staticmethod
    def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        return calculate_momentum(prices, period)
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 24) -> pd.DataFrame:
        return calculate_volume_profile(df, num_bins)
    
    @staticmethod
    def calculate_structure_features(df: pd.DataFrame, swing_length: int = 5) -> pd.DataFrame:
        return calculate_structure_features(df, swing_length)
    
    @staticmethod
    def calculate_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return calculate_volatility(df, period)
    
    @staticmethod
    def calculate_correlation(series1: pd.Series, series2: pd.Series, period: int = 20) -> pd.Series:
        return calculate_correlation(series1, series2, period)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: Series of prices
        period: RSI period
    
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period
    
    Returns:
        Series of ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Series of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Series of prices
        period: Moving average period
        std_dev: Number of standard deviations
    
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return middle, upper, lower


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        k_period: %K period
        d_period: %D period
    
    Returns:
        Tuple of (%K, %D)
    """
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    
    k = 100 * ((close - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    
    return k, d


def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Momentum
    
    Args:
        prices: Series of prices
        period: Momentum period
    
    Returns:
        Series of momentum values
    """
    return prices.diff(period)


def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 24) -> pd.DataFrame:
    """
    Calculate Volume Profile
    
    Args:
        df: DataFrame with high, low, volume columns
        num_bins: Number of price bins
    
    Returns:
        DataFrame with volume profile
    """
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    
    if price_range == 0:
        return pd.DataFrame()
    
    bin_size = price_range / num_bins
    
    volume_profile = []
    
    for i in range(num_bins):
        lower = price_min + (i * bin_size)
        upper = lower + bin_size
        
        # Find bars that intersect this price level
        mask = (df['high'] >= lower) & (df['low'] <= upper)
        
        if mask.any():
            # Calculate volume contribution
            volume_sum = 0
            for idx in df[mask].index:
                bar = df.loc[idx]
                overlap = min(bar['high'], upper) - max(bar['low'], lower)
                bar_range = bar['high'] - bar['low']
                
                if bar_range > 0:
                    contribution = (overlap / bar_range) * bar['volume']
                    volume_sum += contribution
            
            volume_profile.append({
                'price_level': (lower + upper) / 2,
                'volume': volume_sum,
                'lower': lower,
                'upper': upper
            })
    
    return pd.DataFrame(volume_profile)


def calculate_structure_features(df: pd.DataFrame, swing_length: int = 5) -> pd.DataFrame:
    """
    Calculate market structure features
    
    Args:
        df: DataFrame with high, low, close columns
        swing_length: Length for swing detection
    
    Returns:
        DataFrame with structure features
    """
    features = pd.DataFrame(index=df.index)
    
    # Swing highs
    swing_high = (
        (df['high'] > df['high'].shift(1)) &
        (df['high'] > df['high'].shift(2)) &
        (df['high'] > df['high'].shift(-1)) &
        (df['high'] > df['high'].shift(-2))
    )
    
    # Swing lows
    swing_low = (
        (df['low'] < df['low'].shift(1)) &
        (df['low'] < df['low'].shift(2)) &
        (df['low'] < df['low'].shift(-1)) &
        (df['low'] < df['low'].shift(-2))
    )
    
    # Distance to nearest swing
    last_high = df['high'].where(swing_high).ffill()
    last_low = df['low'].where(swing_low).ffill()
    
    features['dist_to_high'] = (df['close'] - last_high) / df['close']
    features['dist_to_low'] = (df['close'] - last_low) / df['close']
    
    # Structure break
    recent_high = df['high'].rolling(swing_length).max().shift(1)
    recent_low = df['low'].rolling(swing_length).min().shift(1)
    
    features['break_high'] = (df['high'] > recent_high).astype(int)
    features['break_low'] = (df['low'] < recent_low).astype(int)
    
    return features


def calculate_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate various volatility metrics
    
    Args:
        df: DataFrame with high, low, close columns
        period: Rolling period
    
    Returns:
        Series with volatility metrics
    """
    # Close-to-close volatility
    returns = df['close'].pct_change()
    vol_close = returns.rolling(period).std() * np.sqrt(252)
    
    # Parkinson volatility (high-low)
    log_hl = np.log(df['high'] / df['low'])
    vol_parkinson = log_hl.rolling(period).std() * np.sqrt(252 * 4 * np.log(2))
    
    # Garman-Klass volatility
    log_co = np.log(df['close'] / df['close'].shift())
    log_oc = np.log(df['open'] / df['close'].shift())
    
    vol_gk = np.sqrt(
        0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
    ).rolling(period).mean() * np.sqrt(252)
    
    return pd.DataFrame({
        'close_vol': vol_close,
        'parkinson_vol': vol_parkinson,
        'garman_klass_vol': vol_gk
    })


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
    return series1.rolling(period).corr(series2)