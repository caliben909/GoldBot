"""
Strategy Engine - Advanced SMC strategy with market regime detection and adaptive parameters
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
from scipy import stats
from sklearn.cluster import KMeans
import warnings

from core.liquidity_engine import LiquidityEngine
from core.session_engine import SessionEngine
from utils.indicators import TechnicalIndicators

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class SignalType(Enum):
    """Types of trading signals"""
    BOS_BREAKOUT = "bos_breakout"
    CHOCH_REVERSAL = "choch_reversal"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    FVG_ENTRY = "fvg_entry"
    ORDER_BLOCK_BOUNCE = "order_block_bounce"
    CONTRARIAN = "contrarian"
    KILL_ZONE = "kill_zone"
    MULTI_TIMEFRAME = "multi_timeframe"
    REGIME_SHIFT = "regime_shift"
    FIBONACCI = "fibonacci"


@dataclass
class MarketRegimeFeatures:
    """Features for regime detection"""
    volatility: float
    trend_strength: float
    volume_profile: float
    range_width: float
    adx: float
    rsi: float
    macd_histogram: float
    bollinger_width: float
    atr_percent: float
    hurst_exponent: float
    fractal_dimension: float
    regime: Optional[MarketRegime] = None
    confidence: float = 0.0


@dataclass
class TradingSignal:
    """Complete trading signal with regime context"""
    timestamp: datetime
    symbol: str
    direction: str
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: float
    regime: MarketRegime
    regime_confidence: float
    quantity: float = 0.0
    
    # SMC components
    order_block: Optional[Dict] = None
    fvg: Optional[Dict] = None
    liquidity_sweep: Optional[Dict] = None
    
    # Confluence
    confluences: List[str] = field(default_factory=list)
    
    # Metadata
    session_type: str = ""
    kill_zone: bool = False
    contrarian: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyEngine:
    """
    Advanced strategy engine with:
    - Market regime detection (HMM, Clustering, Volatility-based)
    - Adaptive parameters based on regime
    - Multiple SMC concepts integration
    - Dynamic position sizing based on regime
    - Regime transition detection
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate strategy parameters
        self._validate_parameters()
        
        # Initialize components
        self.liquidity_engine = LiquidityEngine(config)
        self.session_engine = SessionEngine(config)
        self.indicators = TechnicalIndicators()
        
        # Risk engine reference (will be set externally)
        self.risk_engine = None
        
        # Strategy parameters
        self.swing_length = config['strategy']['smc']['swing_length']
        self.fvg_min_size = config['strategy']['smc']['fvg_min_size']
        self.liquidity_lookback = config['strategy']['smc']['liquidity_lookback']
        
        # Regime detection
        self.regime_detection_enabled = config['strategy']['regime_detection']['enabled']
        self.regime_method = config['strategy']['regime_detection']['method']
        self.regime_lookback = config['strategy']['regime_detection']['lookback_periods']
        self.regime_update_frequency = config['strategy']['regime_detection']['update_frequency']
        
        # Regime-specific parameters
        self.regime_params = self._init_regime_params()
        
        # Signal tracking
        self.signals_history: List[TradingSignal] = []
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self.regime_transitions: List[Dict] = []
        
        # Performance tracking
        self.regime_performance: Dict[MarketRegime, Dict] = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'avg_r_multiple': 0.0
        })
        
        logger.info("StrategyEngine initialized with regime detection")
    
    def _validate_parameters(self):
        """Validate strategy configuration parameters"""
        # Validate SMC parameters
        smc_config = self.config['strategy']['smc']
        if not isinstance(smc_config['swing_length'], int) or smc_config['swing_length'] < 2 or smc_config['swing_length'] > 50:
            raise ValueError(f"Invalid swing_length: {smc_config['swing_length']} (must be between 2 and 50)")
        if not isinstance(smc_config['fvg_min_size'], (int, float)) or smc_config['fvg_min_size'] <= 0 or smc_config['fvg_min_size'] > 100:
            raise ValueError(f"Invalid fvg_min_size: {smc_config['fvg_min_size']} (must be positive and <= 100)")
        if not isinstance(smc_config['liquidity_lookback'], int) or smc_config['liquidity_lookback'] < 5 or smc_config['liquidity_lookback'] > 200:
            raise ValueError(f"Invalid liquidity_lookback: {smc_config['liquidity_lookback']} (must be between 5 and 200)")
        
        # Validate regime detection parameters
        regime_config = self.config['strategy']['regime_detection']
        if not isinstance(regime_config['enabled'], bool):
            raise ValueError(f"Invalid regime detection enabled: {regime_config['enabled']} (must be boolean)")
        if regime_config['method'] not in ['hmm', 'clustering', 'volatility_based', 'rule_based']:
            raise ValueError(f"Invalid regime detection method: {regime_config['method']} (must be hmm, clustering, volatility_based, or rule_based)")
        if not isinstance(regime_config['lookback_periods'], int) or regime_config['lookback_periods'] < 10 or regime_config['lookback_periods'] > 500:
            raise ValueError(f"Invalid regime lookback: {regime_config['lookback_periods']} (must be between 10 and 500)")
        if not isinstance(regime_config['update_frequency'], int) or regime_config['update_frequency'] < 1 or regime_config['update_frequency'] > 100:
            raise ValueError(f"Invalid regime update frequency: {regime_config['update_frequency']} (must be between 1 and 100)")
        
        logger.info("Strategy parameters validated successfully")
    
    def _init_regime_params(self) -> Dict[MarketRegime, Dict]:
        """Initialize regime-specific trading parameters"""
        return {
            MarketRegime.TRENDING_BULLISH: {
                'min_confidence': 0.70,
                'position_multiplier': 1.2,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.5,
                'max_trades': 5,
                'preferred_strategies': ['order_block_bounce', 'liquidity_sweep', 'fvg_entry']
            },
            MarketRegime.TRENDING_BEARISH: {
                'min_confidence': 0.70,
                'position_multiplier': 1.2,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.5,
                'max_trades': 5,
                'preferred_strategies': ['order_block_bounce', 'liquidity_sweep', 'fvg_entry']
            },
            MarketRegime.RANGING: {
                'min_confidence': 0.75,
                'position_multiplier': 0.8,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.2,
                'max_trades': 3,
                'preferred_strategies': ['liquidity_sweep', 'fvg_entry', 'order_block_bounce']
            },
            MarketRegime.VOLATILE: {
                'min_confidence': 0.80,
                'position_multiplier': 0.6,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 2.0,
                'max_trades': 2,
                'preferred_strategies': ['liquidity_sweep', 'order_block_bounce', 'fvg_entry']
            },
            MarketRegime.QUIET: {
                'min_confidence': 0.85,
                'position_multiplier': 0.5,
                'stop_loss_multiplier': 0.7,
                'take_profit_multiplier': 1.0,
                'max_trades': 1,
                'preferred_strategies': ['order_block_bounce', 'fvg_entry']
            },
            MarketRegime.BREAKOUT: {
                'min_confidence': 0.65,
                'position_multiplier': 1.5,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 2.0,
                'max_trades': 4,
                'preferred_strategies': ['liquidity_sweep', 'order_block_bounce', 'fvg_entry']
            },
            MarketRegime.REVERSAL: {
                'min_confidence': 0.75,
                'position_multiplier': 1.0,
                'stop_loss_multiplier': 1.1,
                'take_profit_multiplier': 1.8,
                'max_trades': 3,
                'preferred_strategies': ['order_block_bounce', 'liquidity_sweep', 'fvg_entry']
            }
        }
    
    async def detect_market_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime using multiple methods
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Detected regime and confidence score
        """
        # Extract features
        features = self._extract_regime_features(df)
        
        if self.regime_method == "hmm":
            regime, confidence = self._detect_regime_hmm(features)
        elif self.regime_method == "clustering":
            regime, confidence = self._detect_regime_clustering(features)
        elif self.regime_method == "volatility_based":
            regime, confidence = self._detect_regime_volatility_based(features)
        else:
            regime, confidence = self._detect_regime_rule_based(features)
        
        # Store in history
        self.regime_history.append((datetime.now(), regime))
        
        # Check for regime transition
        if len(self.regime_history) >= 2:
            prev_regime = self.regime_history[-2][1]
            if prev_regime != regime:
                self.regime_transitions.append({
                    'timestamp': datetime.now(),
                    'from_regime': prev_regime,
                    'to_regime': regime,
                    'confidence': confidence
                })
                logger.info(f"Regime transition detected: {prev_regime.value} -> {regime.value}")
        
        return regime, confidence
    
    def _extract_regime_features(self, df: pd.DataFrame) -> MarketRegimeFeatures:
        """Extract features for regime detection"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)
        
        # Volatility
        returns = np.diff(np.log(close))
        volatility = np.std(returns[-20:]) * np.sqrt(252 * 24 * 60)
        
        # Trend strength (using linear regression)
        x = np.arange(len(close[-20:]))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close[-20:])
        trend_strength = abs(r_value)
        
        # Volume profile
        volume_profile = np.mean(volume[-20:]) / np.mean(volume) if np.mean(volume) > 0 else 1.0
        
        # Range width
        range_width = (high[-20:].max() - low[-20:].min()) / close[-1]
        
        # Technical indicators
        adx = self._calculate_adx(df, 14)
        rsi = self._calculate_rsi(close, 14)[-1]
        macd = self._calculate_macd(close)
        
        # Bollinger width
        bb_middle = np.mean(close[-20:])
        bb_std = np.std(close[-20:])
        bb_width = (bb_std * 4) / bb_middle if bb_middle > 0 else 0
        
        # ATR percentage
        atr = self._calculate_atr(df, 14)
        atr_percent = atr[-1] / close[-1] if close[-1] > 0 else 0
        
        # Advanced metrics
        hurst = self._calculate_hurst_exponent(close)
        fractal = self._calculate_fractal_dimension(high, low)
        
        return MarketRegimeFeatures(
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            range_width=range_width,
            adx=adx[-1] if len(adx) > 0 else 0,
            rsi=rsi,
            macd_histogram=macd[2][-1] if len(macd[2]) > 0 else 0,
            bollinger_width=bb_width,
            atr_percent=atr_percent,
            hurst_exponent=hurst,
            fractal_dimension=fractal
        )
    
    def _detect_regime_rule_based(self, features: MarketRegimeFeatures) -> Tuple[MarketRegime, float]:
        """Rule-based regime detection"""
        confidence = 0.7  # Base confidence
        
        # Check for trending regimes
        if features.trend_strength > 0.7 and features.adx > 25:
            if features.rsi > 60:
                return MarketRegime.TRENDING_BULLISH, confidence
            elif features.rsi < 40:
                return MarketRegime.TRENDING_BEARISH, confidence
            else:
                return MarketRegime.TRENDING_BULLISH if features.macd_histogram > 0 else MarketRegime.TRENDING_BEARISH, confidence * 0.9
        
        # Check for ranging
        if features.trend_strength < 0.3 and features.adx < 20 and features.bollinger_width < 0.1:
            return MarketRegime.RANGING, confidence
        
        # Check for volatile
        if features.volatility > 0.3 and features.atr_percent > 0.02:
            return MarketRegime.VOLATILE, confidence * 0.8
        
        # Check for quiet
        if features.volatility < 0.1 and features.atr_percent < 0.005:
            return MarketRegime.QUIET, confidence
        
        # Check for breakout
        if features.bollinger_width > 0.2 and features.volume_profile > 1.5:
            direction = "bullish" if features.macd_histogram > 0 else "bearish"
            return MarketRegime.BREAKOUT, confidence * 0.85
        
        # Check for reversal
        if abs(features.rsi - 50) > 20 and features.trend_strength < 0.4:
            return MarketRegime.REVERSAL, confidence * 0.75
        
        return MarketRegime.RANGING, 0.5
    
    def _detect_regime_clustering(self, features: MarketRegimeFeatures) -> Tuple[MarketRegime, float]:
        """K-means clustering for regime detection"""
        # Simplified - in production, would use trained model
        feature_vector = np.array([
            features.volatility,
            features.trend_strength,
            features.adx / 100,
            features.rsi / 100,
            features.bollinger_width * 10,
            features.atr_percent * 100
        ]).reshape(1, -1)
        
        # Simple heuristic clustering
        if features.trend_strength > 0.6:
            if features.rsi > 60:
                return MarketRegime.TRENDING_BULLISH, 0.8
            elif features.rsi < 40:
                return MarketRegime.TRENDING_BEARISH, 0.8
        elif features.volatility > 0.25:
            return MarketRegime.VOLATILE, 0.7
        elif features.volatility < 0.1:
            return MarketRegime.QUIET, 0.7
        else:
            return MarketRegime.RANGING, 0.6
    
    def _detect_regime_hmm(self, features: MarketRegimeFeatures) -> Tuple[MarketRegime, float]:
        """Hidden Markov Model for regime detection"""
        # Placeholder for HMM implementation
        # In production, would use a trained HMM model
        return self._detect_regime_rule_based(features)
    
    def _detect_regime_volatility_based(self, features: MarketRegimeFeatures) -> Tuple[MarketRegime, float]:
        """Volatility-based regime classification"""
        vol_percentile = stats.percentileofscore(
            self.volatility_history if hasattr(self, 'volatility_history') else [0.2],
            features.volatility
        ) / 100
        
        if vol_percentile > 0.8:
            if features.trend_strength > 0.6:
                return MarketRegime.TRENDING_BULLISH if features.rsi > 50 else MarketRegime.TRENDING_BEARISH, 0.75
            else:
                return MarketRegime.VOLATILE, 0.7
        elif vol_percentile < 0.2:
            return MarketRegime.QUIET, 0.7
        else:
            if features.trend_strength > 0.6:
                return MarketRegime.TRENDING_BULLISH if features.rsi > 50 else MarketRegime.TRENDING_BEARISH, 0.65
            else:
                return MarketRegime.RANGING, 0.6
    
    def _calculate_hurst_exponent(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst exponent for mean reversion/trend detection"""
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Price difference with lag
            pp = np.subtract(prices[lag:], prices[:-lag])
            tau.append(np.std(pp))
        
        if len(tau) == 0 or np.std(np.log(lags)) == 0:
            return 0.5
        
        # Linear regression to estimate Hurst exponent
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = m[0] * 2
        
        return np.clip(hurst, 0, 1)
    
    def _calculate_fractal_dimension(self, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate fractal dimension of price series"""
        # Simplified box-counting dimension
        returns = np.diff(np.log((high + low) / 2))
        ranges = np.maximum(high[1:], high[:-1]) - np.minimum(low[1:], low[:-1])
        
        if np.std(ranges) == 0:
            return 1.0
        
        correlation = np.corrcoef(np.abs(returns), ranges)[0, 1]
        fractal = 2 - correlation if not np.isnan(correlation) else 1.5
        
        return np.clip(fractal, 1, 2)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate ADX indicator"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = self._ema(tr, period)
        plus_di = 100 * self._ema(plus_dm, period) / atr
        minus_di = 100 * self._ema(minus_dm, period) / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = self._ema(dx, period)
        
        return adx
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 100 - 100 / (1 + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100 - 100 / (1 + rs)
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate ATR"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr = self._ema(tr, period)
        
        return atr
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def set_risk_engine(self, risk_engine):
        """Set the risk engine reference for correlation filtering"""
        self.risk_engine = risk_engine
        logger.info("Risk engine reference set for DXY correlation filtering")
    
    def get_regime_parameters(self, regime: MarketRegime) -> Dict:
        """Get trading parameters for current regime"""
        return self.regime_params.get(regime, self.regime_params[MarketRegime.RANGING])
        
    def _check_multi_timeframe_confluence(self, signals: List[TradingSignal], tf_data: Dict) -> List[TradingSignal]:
        """
        Check if signals have multi-timeframe confluence
        
        Args:
            signals: List of trading signals to check
            tf_data: Dictionary of lower timeframe data
            
        Returns:
            List of signals with multi-timeframe confluence
        """
        if not tf_data:
            return signals
            
        confirmed_signals = []
        
        for signal in signals:
            has_confluence = True
            
            # Check lower timeframes for confirmation
            for tf, tf_df in tf_data.items():
                # Find the corresponding candle in lower timeframe
                tf_signal = self._find_corresponding_tf_signal(signal, tf_df, tf)
                if not tf_signal:
                    has_confluence = False
                    break
                    
                # Check if lower timeframe signal confirms
                if not self._confirm_signal_on_lower_tf(signal, tf_signal, tf_df, tf):
                    has_confluence = False
                    break
                    
            if has_confluence:
                # Increase confidence for multi-timeframe confirmation
                signal.confidence = min(signal.confidence + 0.15, 1.0)
                signal.confluences.append('multi_timeframe')
                confirmed_signals.append(signal)
                
        return confirmed_signals
        
    def _find_corresponding_tf_signal(self, signal: TradingSignal, tf_df: pd.DataFrame, timeframe: str) -> Optional[TradingSignal]:
        """
        Find corresponding signal in lower timeframe
        
        Args:
            signal: Primary timeframe signal
            tf_df: Lower timeframe DataFrame
            timeframe: Timeframe string
            
        Returns:
            Corresponding lower timeframe signal or None
        """
        try:
            # Find the candle in lower timeframe that contains the primary signal time
            tf_candle = tf_df[tf_df.index <= signal.timestamp].iloc[-1:]
            if tf_candle.empty:
                return None
                
            # Perform quick analysis on lower timeframe
            liquidity_analysis = self.liquidity_engine.analyze_market(tf_df)
            
            # Check for signals in lower timeframe
            tf_signals = []
            
            # Check for liquidity sweeps
            liquidity_signals = self._generate_liquidity_signals(tf_df, liquidity_analysis, signal.regime, signal.symbol)
            tf_signals.extend(liquidity_signals)
            
            # Check for FVG entries
            fvg_signals = self._generate_fvg_signals(tf_df, liquidity_analysis, signal.regime, signal.symbol)
            tf_signals.extend(fvg_signals)
            
            # Check for order block bounces
            ob_signals = self._generate_order_block_signals(tf_df, liquidity_analysis, signal.regime, signal.symbol)
            tf_signals.extend(ob_signals)
            
            return tf_signals[0] if tf_signals else None
            
        except Exception as e:
            logger.warning(f"Failed to find corresponding signal for {timeframe} timeframe: {e}")
            return None
            
    def _confirm_signal_on_lower_tf(self, primary_signal: TradingSignal, tf_signal: TradingSignal, 
                                   tf_df: pd.DataFrame, timeframe: str) -> bool:
        """
        Check if lower timeframe signal confirms primary signal
        
        Args:
            primary_signal: Primary timeframe signal
            tf_signal: Lower timeframe signal
            tf_df: Lower timeframe DataFrame
            timeframe: Timeframe string
            
        Returns:
            True if lower timeframe confirms, False otherwise
        """
        # Direction must match
        if primary_signal.direction != tf_signal.direction:
            return False
            
        # Check if lower timeframe has supporting structure
        liquidity_analysis = self.liquidity_engine.analyze_market(tf_df)
        
        # For bullish signals, check for bullish structure
        if primary_signal.direction == 'long':
            # Check if lower timeframe has bullish order blocks or FVG
            bullish_ob = any(ob.direction == 'bullish' and not ob.mitigated 
                          for ob in liquidity_analysis['order_blocks'])
            bullish_fvg = any(fvg.type == 'bullish' and not fvg.mitigated 
                           for fvg in liquidity_analysis['fvg_zones'])
            
            return bullish_ob or bullish_fvg
            
        # For bearish signals, check for bearish structure
        elif primary_signal.direction == 'short':
            # Check if lower timeframe has bearish order blocks or FVG
            bearish_ob = any(ob.direction == 'bearish' and not ob.mitigated 
                           for ob in liquidity_analysis['order_blocks'])
            bearish_fvg = any(fvg.type == 'bearish' and not fvg.mitigated 
                           for fvg in liquidity_analysis['fvg_zones'])
            
            return bearish_ob or bearish_fvg
            
        return False
    
    async def generate_trading_signals(self, df: pd.DataFrame, symbol: str = "default", tf_data: Optional[Dict] = None) -> List[TradingSignal]:
        """
        Generate trading signals with regime awareness and multi-timeframe confirmation
        
        Args:
            df: OHLCV DataFrame (primary timeframe, e.g., 4H)
            symbol: Trading symbol
            tf_data: Optional dictionary of lower timeframe data (e.g., {'15M': df_15m})
            
        Returns:
            List of trading signals
        """
        # Detect current regime
        regime, regime_confidence = await self.detect_market_regime(df)
        
        # Get regime-specific parameters
        regime_params = self.get_regime_parameters(regime)
        
        # Perform SMC analysis
        liquidity_analysis = self.liquidity_engine.analyze_market(df)
        
        # Get session info
        session_type, session_config = self.session_engine.get_current_session()
        
        # Determine current trend from liquidity analysis
        current_trend = liquidity_analysis['structure']['current_trend']
        
        signals = []
        
        # Generate signals from each strategy type
        for strategy in regime_params['preferred_strategies']:
            if strategy == 'bos_breakout':
                new_signals = self._generate_bos_signals(df, liquidity_analysis, regime, symbol)
            elif strategy == 'choch_reversal':
                new_signals = self._generate_choch_signals(df, liquidity_analysis, regime, symbol)
            elif strategy == 'liquidity_sweep':
                new_signals = self._generate_liquidity_signals(df, liquidity_analysis, regime, symbol)
            elif strategy == 'fvg_entry':
                new_signals = self._generate_fvg_signals(df, liquidity_analysis, regime, symbol)
            elif strategy == 'order_block_bounce':
                new_signals = self._generate_order_block_signals(df, liquidity_analysis, regime, symbol)
            elif strategy == 'contrarian':
                new_signals = self._generate_contrarian_signals(df, liquidity_analysis, regime, symbol)
            elif strategy == 'fibonacci':
                new_signals = self._generate_fibonacci_signals(df, liquidity_analysis, regime, symbol)
            else:
                continue
            
            # Filter signals to only trade with the trend
            filtered_signals = []
            for signal in new_signals:
                if current_trend == 'bullish' and signal.direction == 'long':
                    filtered_signals.append(signal)
                elif current_trend == 'bearish' and signal.direction == 'short':
                    filtered_signals.append(signal)
            
            signals.extend(filtered_signals)
        
        # Apply regime-based filters and adjustments
        filtered_signals = self._apply_regime_filters(signals, regime, regime_params)
        
        # Apply DXY correlation filter
        if self.risk_engine is not None and self.config['risk_management']['dxy_correlation']['enabled']:
            correlation_filtered_signals = []
            for signal in filtered_signals:
                signal_dict = {
                    'symbol': symbol,
                    'direction': signal.direction,
                    'timestamp': signal.timestamp,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit
                }
                
                should_filter, reason = await self.risk_engine.should_filter_signal_by_dxy_correlation(signal_dict)
                
                if should_filter:
                    logger.debug(f"Signal filtered by DXY correlation: {reason}")
                    continue
                
                # Adjust position size based on DXY correlation
                if self.config['risk_management']['dxy_correlation']['adjust_position_size_by_correlation']:
                    base_size = 1.0  # Will be calculated by risk engine
                    adjustment = await self.risk_engine.adjust_position_size_by_dxy_correlation(symbol, base_size)
                    signal.metadata['dxy_correlation_adjustment'] = adjustment
                
                correlation_filtered_signals.append(signal)
            
            filtered_signals = correlation_filtered_signals
            logger.debug(f"DXY correlation filter removed {len(signals) - len(filtered_signals)} signals")
        
        # Add regime info to signals
        for signal in filtered_signals:
            signal.regime = regime
            signal.regime_confidence = regime_confidence
            signal.session_type = session_type.value if session_type else 'unknown'
            
            # Check kill zone
            kill_zones = self.session_engine.get_active_kill_zones()
            signal.kill_zone = any(kz.session_type == session_type for kz in kill_zones)
        
        # Apply multi-timeframe confirmation if lower timeframe data available
        if tf_data:
            filtered_signals = self._check_multi_timeframe_confluence(filtered_signals, tf_data)
            logger.debug(f"Multi-timeframe filter removed {len(signals) - len(filtered_signals)} signals")
        
        logger.info(f"Generated {len(filtered_signals)} signals for {symbol} in {regime.value} regime")
        
        return filtered_signals
    
    def _apply_regime_filters(self, signals: List[TradingSignal], regime: MarketRegime,
                              regime_params: Dict) -> List[TradingSignal]:
        """Apply regime-specific filters to signals"""
        filtered = []
        
        # Get global confidence configuration
        confidence_config = self.config.get('strategy', {}).get('signal_confidence', {})
        enabled = confidence_config.get('enabled', True)
        min_confidence = confidence_config.get('min_confidence', 0.75)
        
        for signal in signals:
            # Apply global confidence threshold if enabled
            if enabled and signal.confidence < min_confidence:
                logger.debug(f"Signal filtered - low confidence: {signal.confidence:.2f} < {min_confidence:.2f}")
                continue
                
            # Apply regime-specific confidence threshold
            if signal.confidence < regime_params['min_confidence']:
                logger.debug(f"Signal filtered - low regime confidence: {signal.confidence:.2f} < {regime_params['min_confidence']:.2f}")
                continue
                
            # Apply regime-specific strategy preferences
            if signal.signal_type.value not in regime_params['preferred_strategies']:
                logger.debug(f"Signal filtered - strategy not preferred: {signal.signal_type.value}")
                continue
                
            # Adjust position size based on regime
            signal.metadata['position_multiplier'] = regime_params['position_multiplier']
            
            filtered.append(signal)
        
        logger.debug(f"Applied regime filters: {len(signals)} -> {len(filtered)} signals")
        return filtered
    
    def _generate_bos_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                              regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Break of Structure signals"""
        signals = []
        
        # Find swing points
        swing_highs = liquidity_analysis['structure']['swing_highs']
        swing_lows = liquidity_analysis['structure']['swing_lows']
        
        current_price = df['close'].iloc[-1]
        
        # Bullish BOS
        if swing_highs and current_price > swing_highs[-1]['price']:
            signal = TradingSignal(
                timestamp=df.index[-1],
                symbol=symbol,
                direction='long',
                signal_type=SignalType.BOS_BREAKOUT,
                entry_price=current_price,
                stop_loss=swing_highs[-1]['price'] * 0.995,
                take_profit=current_price + (current_price - swing_highs[-1]['price']) * 2,
                risk_reward=2.0,
                confidence=0.7 + 0.1 * (regime == MarketRegime.TRENDING_BULLISH),
                regime=regime,
                regime_confidence=0.8,
                confluences=['bos_breakout']
            )
            signals.append(signal)
        
        # Bearish BOS
        if swing_lows and current_price < swing_lows[-1]['price']:
            signal = TradingSignal(
                timestamp=df.index[-1],
                symbol=symbol,
                direction='short',
                signal_type=SignalType.BOS_BREAKOUT,
                entry_price=current_price,
                stop_loss=swing_lows[-1]['price'] * 1.005,
                take_profit=current_price - (swing_lows[-1]['price'] - current_price) * 2,
                risk_reward=2.0,
                confidence=0.7 + 0.1 * (regime == MarketRegime.TRENDING_BEARISH),
                regime=regime,
                regime_confidence=0.8,
                confluences=['bos_breakout']
            )
            signals.append(signal)
        
        return signals
    
    def _generate_choch_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                                regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Change of Character signals"""
        signals = []
        
        # Simplified CHOCH detection
        swing_highs = liquidity_analysis['structure']['swing_highs']
        swing_lows = liquidity_analysis['structure']['swing_lows']
        
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            # Bullish CHOCH (higher low after lower lows)
            if (swing_lows[-1]['price'] > swing_lows[-2]['price'] and
                swing_lows[-2]['price'] < swing_lows[-3]['price']):
                
                signal = TradingSignal(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    direction='long',
                    signal_type=SignalType.CHOCH_REVERSAL,
                    entry_price=df['close'].iloc[-1],
                    stop_loss=swing_lows[-2]['price'] * 0.998,
                    take_profit=swing_highs[-1]['price'],
                    risk_reward=self._calculate_risk_reward(swing_lows[-2]['price'], swing_highs[-1]['price']),
                    confidence=0.75,
                    regime=regime,
                    regime_confidence=0.8,
                    confluences=['choch_reversal']
                )
                signals.append(signal)
            
            # Bearish CHOCH (lower high after higher highs)
            if (swing_highs[-1]['price'] < swing_highs[-2]['price'] and
                swing_highs[-2]['price'] > swing_highs[-3]['price']):
                
                signal = TradingSignal(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    direction='short',
                    signal_type=SignalType.CHOCH_REVERSAL,
                    entry_price=df['close'].iloc[-1],
                    stop_loss=swing_highs[-2]['price'] * 1.002,
                    take_profit=swing_lows[-1]['price'],
                    risk_reward=self._calculate_risk_reward(swing_highs[-2]['price'], swing_lows[-1]['price']),
                    confidence=0.75,
                    regime=regime,
                    regime_confidence=0.8,
                    confluences=['choch_reversal']
                )
                signals.append(signal)
        
        return signals
    
    def _generate_liquidity_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                                    regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Liquidity Sweep signals"""
        signals = []
        
        liquidity = liquidity_analysis['liquidity']
        current_price = df['close'].iloc[-1]
        
        # Buy-side liquidity sweeps
        for zone in liquidity['buy_side']:
            if zone.is_swept and abs(current_price - zone.price_level) / zone.price_level < 0.001:
                signal = TradingSignal(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    direction='short',  # Sell after sweeping buy-side liquidity
                    signal_type=SignalType.LIQUIDITY_SWEEP,
                    entry_price=current_price,
                    stop_loss=zone.price_level * 1.005,
                    take_profit=current_price - (zone.price_level - current_price) * 2,
                    risk_reward=2.0,
                    confidence=0.8,
                    regime=regime,
                    regime_confidence=0.8,
                    confluences=['liquidity_sweep'],
                    liquidity_sweep={'level': zone.price_level, 'type': 'buy_side'}
                )
                signals.append(signal)
        
        # Sell-side liquidity sweeps
        for zone in liquidity['sell_side']:
            if zone.is_swept and abs(current_price - zone.price_level) / zone.price_level < 0.001:
                signal = TradingSignal(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    direction='long',  # Buy after sweeping sell-side liquidity
                    signal_type=SignalType.LIQUIDITY_SWEEP,
                    entry_price=current_price,
                    stop_loss=zone.price_level * 0.995,
                    take_profit=current_price + (current_price - zone.price_level) * 2,
                    risk_reward=2.0,
                    confidence=0.8,
                    regime=regime,
                    regime_confidence=0.8,
                    confluences=['liquidity_sweep'],
                    liquidity_sweep={'level': zone.price_level, 'type': 'sell_side'}
                )
                signals.append(signal)
        
        return signals
    
    def _generate_fvg_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                               regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Fair Value Gap signals"""
        signals = []
        
        fvg_zones = liquidity_analysis['fvg_zones']
        current_price = df['close'].iloc[-1]
        
        for fvg in fvg_zones:
            if not fvg.mitigated:
                # Check if price is in OTE zone
                if fvg.ote_zone_62 and fvg.ote_zone_79:
                    if fvg.type == 'bullish' and fvg.ote_zone_62 <= current_price <= fvg.ote_zone_79:
                        signal = TradingSignal(
                            timestamp=df.index[-1],
                             symbol=symbol,
                            direction='long',
                            signal_type=SignalType.FVG_ENTRY,
                            entry_price=current_price,
                            stop_loss=fvg.bottom * 0.998,
                            take_profit=fvg.top + (fvg.top - fvg.bottom),
                            risk_reward=2.0,
                            confidence=0.75,
                            regime=regime,
                            regime_confidence=0.8,
                            confluences=['fvg_entry', 'ote_zone'],
                            fvg={'top': fvg.top, 'bottom': fvg.bottom, 'type': fvg.type}
                        )
                        signals.append(signal)
                    
                    elif fvg.type == 'bearish' and fvg.ote_zone_79 <= current_price <= fvg.ote_zone_62:
                        signal = TradingSignal(
                            timestamp=df.index[-1],
                             symbol=symbol,
                            direction='short',
                            signal_type=SignalType.FVG_ENTRY,
                            entry_price=current_price,
                            stop_loss=fvg.top * 1.002,
                            take_profit=fvg.bottom - (fvg.top - fvg.bottom),
                            risk_reward=2.0,
                            confidence=0.75,
                            regime=regime,
                            regime_confidence=0.8,
                            confluences=['fvg_entry', 'ote_zone'],
                            fvg={'top': fvg.top, 'bottom': fvg.bottom, 'type': fvg.type}
                        )
                        signals.append(signal)
        
        return signals
    
    def _generate_fibonacci_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                                      regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Fibonacci-based trading signals"""
        signals = []
        
        # Find swing points for Fibonacci analysis
        swing_highs = liquidity_analysis['structure']['swing_highs']
        swing_lows = liquidity_analysis['structure']['swing_lows']
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return signals
            
        # Get recent swing points
        recent_high = swing_highs[-1]['price']
        recent_low = swing_lows[-1]['price']
        
        # Calculate Fibonacci levels based on recent swing
        fib_calc = self.indicators.calculate_fibonacci_levels(recent_high, recent_low)
        current_price = df['close'].iloc[-1]
        
        # Only generate signals if there's a significant swing (at least 1% move)
        swing_size = abs(recent_high - recent_low) / recent_low
        if swing_size < 0.01:
            return signals
            
        # Check for Fibonacci retracement bounce (bullish)
        if current_price > recent_low and current_price < recent_high:
            # Check 61.8% retracement (strong support)
            if abs(current_price - fib_calc['retracement']['0.618']) / current_price < 0.002:
                signal = TradingSignal(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    direction='long',
                    signal_type=SignalType.FIBONACCI,
                    entry_price=current_price,
                    stop_loss=recent_low * 0.998,
                    take_profit=fib_calc['extension']['1.272'],  # Use 1.272 extension instead of recent high
                    risk_reward=self._calculate_risk_reward(current_price, fib_calc['extension']['1.272'], recent_low * 0.998),
                    confidence=0.85,
                    regime=regime,
                    regime_confidence=0.8,
                    confluences=['fibonacci_retracement', '61.8_level'],
                    metadata={'fibonacci_level': '0.618', 'swing_high': recent_high, 'swing_low': recent_low}
                )
                signals.append(signal)
            
            # Check 50% retracement
            if abs(current_price - fib_calc['retracement']['0.5']) / current_price < 0.002:
                signal = TradingSignal(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    direction='long',
                    signal_type=SignalType.FIBONACCI,
                    entry_price=current_price,
                    stop_loss=recent_low * 0.998,
                    take_profit=fib_calc['extension']['1.272'],
                    risk_reward=self._calculate_risk_reward(current_price, fib_calc['extension']['1.272'], recent_low * 0.998),
                    confidence=0.75,
                    regime=regime,
                    regime_confidence=0.8,
                    confluences=['fibonacci_retracement', '50_level'],
                    metadata={'fibonacci_level': '0.5', 'swing_high': recent_high, 'swing_low': recent_low}
                )
                signals.append(signal)
        
        # Check for Fibonacci retracement resistance (bearish)
        if current_price < recent_high and current_price > recent_low:
            # Check 61.8% retracement (strong resistance)
            if abs(current_price - fib_calc['retracement']['0.382']) / current_price < 0.002:
                signal = TradingSignal(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    direction='short',
                    signal_type=SignalType.FIBONACCI,
                    entry_price=current_price,
                    stop_loss=recent_high * 1.002,
                    take_profit=fib_calc['extension']['1.272'],
                    risk_reward=self._calculate_risk_reward(current_price, fib_calc['extension']['1.272'], recent_high * 1.002),
                    confidence=0.85,
                    regime=regime,
                    regime_confidence=0.8,
                    confluences=['fibonacci_retracement', '38.2_level'],
                    metadata={'fibonacci_level': '0.382', 'swing_high': recent_high, 'swing_low': recent_low}
                )
                signals.append(signal)
        
        return signals
    
    def _generate_order_block_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                                      regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Order Block signals"""
        signals = []
        
        order_blocks = liquidity_analysis['order_blocks']
        current_price = df['close'].iloc[-1]
        
        for ob in order_blocks:
            if not ob.mitigated:
                # Check if price is near order block
                if ob.direction == 'bullish' and abs(current_price - ob.mitigation_price) / ob.mitigation_price < 0.001:
                    signal = TradingSignal(
                        timestamp=df.index[-1],
                             symbol=symbol,
                        direction='long',
                        signal_type=SignalType.ORDER_BLOCK_BOUNCE,
                        entry_price=current_price,
                        stop_loss=ob.price_range[0] * 0.998,
                        take_profit=ob.price_range[1] + (ob.price_range[1] - ob.price_range[0]) * 2,
                        risk_reward=2.0,
                        confidence=0.8,
                        regime=regime,
                        regime_confidence=0.8,
                        confluences=['order_block'],
                        order_block={'level': ob.mitigation_price, 'type': ob.direction}
                    )
                    signals.append(signal)
                
                elif ob.direction == 'bearish' and abs(current_price - ob.mitigation_price) / ob.mitigation_price < 0.001:
                    signal = TradingSignal(
                        timestamp=df.index[-1],
                             symbol=symbol,
                        direction='short',
                        signal_type=SignalType.ORDER_BLOCK_BOUNCE,
                        entry_price=current_price,
                        stop_loss=ob.price_range[1] * 1.002,
                        take_profit=ob.price_range[0] - (ob.price_range[1] - ob.price_range[0]) * 2,
                        risk_reward=2.0,
                        confidence=0.8,
                        regime=regime,
                        regime_confidence=0.8,
                        confluences=['order_block'],
                        order_block={'level': ob.mitigation_price, 'type': ob.direction}
                    )
                    signals.append(signal)
        
        return signals
    
    def _generate_enhanced_bearish_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                                           regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Enhanced Bearish Strategy signals based on market structure"""
        signals = []
        
        # Enhanced bearish strategy based on market conditions rather than hardcoded dates
        current_price = df['close'].iloc[-1]
        current_time = df.index[-1]
        
        # Check for specific market conditions that indicate bearish opportunities
        # (This replaces the hardcoded date-based approach)
        
        # Conditions for enhanced bearish strategy:
        # 1. Market in strong bearish regime
        # 2. Liquidity sweeps detected
        # 3. Order blocks identified
        # 4. Key levels broken
        if regime == MarketRegime.TRENDING_BEARISH and \
           liquidity_analysis.get('liquidity_sweeps', False) and \
           liquidity_analysis.get('order_blocks', False):
            
            # Calculate dynamic entry, stop loss, and take profit based on market structure
            recent_high = liquidity_analysis['recent_high']
            recent_low = liquidity_analysis['recent_low']
            
            # Entry: Near recent high
            if abs(current_price - recent_high) / recent_high < 0.005:
                stop_loss = recent_high * 1.002
                target_risk_reward = 3.0
                
                # Calculate risk and take profit
                risk = current_price - stop_loss
                take_profit = current_price + (risk * target_risk_reward)
                
                signal = TradingSignal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction='short',
                    signal_type=SignalType.CONTRARIAN,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=target_risk_reward,
                    confidence=0.85,
                    regime=regime,
                    regime_confidence=0.9,
                    confluences=['enhanced_bearish', '1:3_rr'],
                    contrarian=True,
                    metadata={
                        'trailing_stop': {
                            'activate_threshold': 0.03,  # Activate after 3% drop from entry
                            'trailing_percent': 0.02  # Trail by 2%
                        },
                        'scaling': {
                            'initial_position': 0.5,  # 50% initial position
                            'scale_in_price': recent_high * 1.001,  # Scale in if price rises further
                            'scale_in_position': 0.5  # 50% additional position
                        },
                        'multi_tier_exit': [
                            {'level': take_profit, 'percentage': 0.5, 'reason': '1:3_risk_reward'},
                            {'level': take_profit * 1.1, 'percentage': 0.5, 'reason': 'extended_target'}
                        ]
                    }
                )
                
                signals.append(signal)
        
        return signals

    def _generate_contrarian_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                                      regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Contrarian signals with strict risk management"""
        signals = []
        
        # First, try enhanced bearish strategy
        enhanced_signals = self._generate_enhanced_bearish_signals(df, liquidity_analysis, regime, symbol)
        if enhanced_signals:
            signals.extend(enhanced_signals)
        
        # Then try enhanced bullish strategy
        bullish_signals = self._generate_enhanced_bullish_signals(df, liquidity_analysis, regime, symbol)
        if bullish_signals:
            signals.extend(bullish_signals)
        
        # Finally, check for general contrarian opportunities based on overbought/oversold conditions
        if not signals:
            general_signals = self._generate_general_contrarian_signals(df, liquidity_analysis, regime, symbol)
            if general_signals:
                signals.extend(general_signals)
        
        return signals
    
    def _generate_enhanced_bullish_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                                           regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Enhanced Bullish Strategy signals based on market structure"""
        signals = []
        
        # Enhanced bullish strategy based on market conditions rather than hardcoded dates
        current_price = df['close'].iloc[-1]
        current_time = df.index[-1]
        
        # Conditions for enhanced bullish strategy:
        # 1. Market in strong bullish regime
        # 2. Liquidity sweeps detected
        # 3. Order blocks identified
        # 4. Key levels broken
        if regime == MarketRegime.TRENDING_BULLISH and \
           liquidity_analysis.get('liquidity_sweeps', False) and \
           liquidity_analysis.get('order_blocks', False):
            
            # Calculate dynamic entry, stop loss, and take profit based on market structure
            recent_high = liquidity_analysis['recent_high']
            recent_low = liquidity_analysis['recent_low']
            
            # Entry: Near recent low
            if abs(current_price - recent_low) / recent_low < 0.005:
                stop_loss = recent_low * 0.998
                target_risk_reward = 3.0
                
                # Calculate risk and take profit
                risk = stop_loss - current_price
                take_profit = current_price - (risk * target_risk_reward)
                
                signal = TradingSignal(
                    timestamp=current_time,
                    symbol=symbol,
                    direction='long',
                    signal_type=SignalType.CONTRARIAN,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=target_risk_reward,
                    confidence=0.85,
                    regime=regime,
                    regime_confidence=0.9,
                    confluences=['enhanced_bullish', '1:3_rr'],
                    contrarian=True,
                    metadata={
                        'trailing_stop': {
                            'activate_threshold': 0.03,  # Activate after 3% rise from entry
                            'trailing_percent': 0.02  # Trail by 2%
                        },
                        'scaling': {
                            'initial_position': 0.5,  # 50% initial position
                            'scale_in_price': recent_low * 0.999,  # Scale in if price drops further
                            'scale_in_position': 0.5  # 50% additional position
                        },
                        'multi_tier_exit': [
                            {'level': take_profit, 'percentage': 0.5, 'reason': '1:3_risk_reward'},
                            {'level': take_profit * 1.1, 'percentage': 0.5, 'reason': 'extended_target'}
                        ]
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def _generate_general_contrarian_signals(self, df: pd.DataFrame, liquidity_analysis: Dict,
                                             regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate general contrarian signals based on overbought/oversold conditions"""
        signals = []
        
        current_price = df['close'].iloc[-1]
        current_time = df.index[-1]
        
        # Calculate RSI for overbought/oversold conditions
        rsi = self._calculate_rsi(df['close'].values, 14)[-1]
        
        # Overbought condition - bearish signal
        if rsi > 70:
            # Find recent support as stop loss
            recent_low = df['low'].rolling(20).min().iloc[-1]
            stop_loss = recent_low * 0.995
            target_risk_reward = 2.0
            
            # Calculate take profit
            risk = current_price - stop_loss
            take_profit = current_price + (risk * target_risk_reward)
            
            signal = TradingSignal(
                timestamp=current_time,
                symbol=symbol,
                direction='bearish',
                signal_type=SignalType.CONTRARIAN,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=target_risk_reward,
                confidence=0.7,
                regime=regime,
                regime_confidence=0.8,
                confluences=['rsi_overbought'],
                contrarian=True
            )
            
            signals.append(signal)
        
        # Oversold condition - bullish signal
        if rsi < 30:
            # Find recent resistance as stop loss
            recent_high = df['high'].rolling(20).max().iloc[-1]
            stop_loss = recent_high * 1.005
            target_risk_reward = 2.0
            
            # Calculate take profit
            risk = stop_loss - current_price
            take_profit = current_price - (risk * target_risk_reward)
            
            signal = TradingSignal(
                timestamp=current_time,
                symbol=symbol,
                direction='bullish',
                signal_type=SignalType.CONTRARIAN,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=target_risk_reward,
                confidence=0.7,
                regime=regime,
                regime_confidence=0.8,
                confluences=['rsi_oversold'],
                contrarian=True
            )
            
            signals.append(signal)
        
        return signals
    
    def _calculate_risk_reward(self, entry: float, target: float, stop: Optional[float] = None) -> float:
        """Calculate risk-reward ratio"""
        if stop is None:
            return 2.0  # Default
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        return reward / risk if risk > 0 else 0
    
    def update_performance(self, trade_result: Dict):
        """Update performance tracking by regime"""
        regime = trade_result.get('regime')
        if regime and regime in MarketRegime:
            perf = self.regime_performance[regime]
            perf['trades'] += 1
            if trade_result['profit'] > 0:
                perf['wins'] += 1
            else:
                perf['losses'] += 1
            perf['total_pnl'] += trade_result['profit']
            perf['avg_r_multiple'] = (
                (perf['avg_r_multiple'] * (perf['trades'] - 1) + trade_result.get('r_multiple', 0)) /
                perf['trades']
            )
    
    def get_regime_performance(self) -> Dict:
        """Get performance metrics by regime"""
        return dict(self.regime_performance)
    
    def get_best_regime(self) -> Tuple[MarketRegime, float]:
        """Get best performing regime"""
        best_regime = None
        best_win_rate = 0
        
        for regime, perf in self.regime_performance.items():
            if perf['trades'] >= 10:
                win_rate = perf['wins'] / perf['trades'] * 100
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_regime = regime
        
        return best_regime, best_win_rate