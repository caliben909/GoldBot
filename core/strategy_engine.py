"""
Strategy Engine v2.0 - Production-Ready SMC Strategy with Regime Detection
Integrated with Risk Engine and Liquidity Engine
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
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class SignalType(Enum):
    BOS_BREAKOUT = "bos_breakout"
    CHOCH_REVERSAL = "choch_reversal"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    FVG_ENTRY = "fvg_entry"
    ORDER_BLOCK_BOUNCE = "order_block_bounce"
    CONTRARIAN = "contrarian"
    KILL_ZONE = "kill_zone"
    FIBONACCI = "fibonacci"
    TREND_CONTINUATION = "trend_continuation"


class SessionType(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    UNKNOWN = "unknown"


@dataclass
class MarketRegimeFeatures:
    """Features for regime detection"""
    volatility: float
    trend_strength: float
    adx: float
    rsi: float
    atr_percent: float
    volume_profile: float
    regime: Optional[MarketRegime] = None
    confidence: float = 0.0


@dataclass
class TradingSignal:
    """Complete trading signal with regime context"""
    timestamp: datetime
    symbol: str
    direction: str  # 'long' or 'short'
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


class SessionEngine:
    """Market session detection and kill zone identification"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def get_current_session(self, timestamp: Optional[datetime] = None) -> Tuple[SessionType, Dict]:
        """Determine current trading session"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        hour = timestamp.hour
        
        # Asian: 00:00 - 08:00 UTC
        if 0 <= hour < 8:
            return SessionType.ASIAN, {'liquidity': 'low', 'volatility': 'low'}
        
        # London: 08:00 - 17:00 UTC
        elif 8 <= hour < 13:
            return SessionType.LONDON, {'liquidity': 'increasing', 'volatility': 'increasing'}
        
        # London/NY Overlap: 13:00 - 17:00 UTC (highest volatility)
        elif 13 <= hour < 17:
            return SessionType.LONDON_NY_OVERLAP, {'liquidity': 'high', 'volatility': 'high'}
        
        # New York: 13:00 - 22:00 UTC
        elif 17 <= hour < 22:
            return SessionType.NEW_YORK, {'liquidity': 'moderate', 'volatility': 'moderate'}
        
        return SessionType.UNKNOWN, {'liquidity': 'low', 'volatility': 'low'}
    
    def is_kill_zone(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if current time is in a kill zone (high probability window)"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        hour = timestamp.hour
        minute = timestamp.minute
        
        # London Open Kill Zone: 08:00 - 10:00 UTC
        london_open = (hour == 8) or (hour == 9)
        
        # NY Open Kill Zone: 13:00 - 15:00 UTC
        ny_open = (hour == 13) or (hour == 14)
        
        # London Close Kill Zone: 16:00 - 17:00 UTC
        london_close = (hour == 16)
        
        return london_open or ny_open or london_close
    
    def get_session_high_low(self, df: pd.DataFrame, session: SessionType) -> Tuple[float, float]:
        """Get high/low for specific session"""
        df['hour'] = df.index.hour
        
        if session == SessionType.ASIAN:
            session_data = df[(df['hour'] >= 0) & (df['hour'] < 8)]
        elif session == SessionType.LONDON:
            session_data = df[(df['hour'] >= 8) & (df['hour'] < 17)]
        elif session == SessionType.NEW_YORK:
            session_data = df[(df['hour'] >= 13) & (df['hour'] < 22)]
        else:
            return df['high'].max(), df['low'].min()
        
        if len(session_data) == 0:
            return df['high'].iloc[-1], df['low'].iloc[-1]
        
        return session_data['high'].max(), session_data['low'].min()


class TechnicalIndicators:
    """Technical indicator calculations"""
    
    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, Dict[str, float]]:
        """Calculate Fibonacci retracement and extension levels"""
        diff = high - low
        
        return {
            'retracement': {
                '0.236': high - diff * 0.236,
                '0.382': high - diff * 0.382,
                '0.5': high - diff * 0.5,
                '0.618': high - diff * 0.618,
                '0.786': high - diff * 0.786
            },
            'extension': {
                '1.272': high + diff * 0.272,
                '1.618': high + diff * 0.618,
                '2.0': high + diff,
                '2.618': high + diff * 1.618
            }
        }
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram


class StrategyEngine:
    """
    Production-ready strategy engine with regime detection and adaptive parameters
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.liquidity_engine = None  # Will be injected
        self.session_engine = SessionEngine(config)
        self.indicators = TechnicalIndicators()
        self.risk_engine = None  # Will be injected
        
        # Strategy parameters
        strategy_config = config.get('strategy', {})
        self.smc_config = strategy_config.get('smc', {})
        self.swing_length = self.smc_config.get('swing_length', 5)
        self.fvg_min_size = self.smc_config.get('fvg_min_size', 0.0001)
        
        # Regime detection
        regime_config = strategy_config.get('regime_detection', {})
        self.regime_detection_enabled = regime_config.get('enabled', True)
        self.regime_method = regime_config.get('method', 'rule_based')
        
        # Regime-specific parameters
        self.regime_params = self._init_regime_params()
        
        # Tracking
        self.signals_history: List[TradingSignal] = []
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self.regime_performance: Dict[MarketRegime, Dict] = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0
        })
        
        logger.info("StrategyEngine initialized")
    
    def set_liquidity_engine(self, engine):
        """Inject liquidity engine"""
        self.liquidity_engine = engine
    
    def set_risk_engine(self, engine):
        """Inject risk engine"""
        self.risk_engine = engine
    
    def _init_regime_params(self) -> Dict[MarketRegime, Dict]:
        """Initialize regime-specific trading parameters"""
        return {
            MarketRegime.TRENDING_BULLISH: {
                'min_confidence': 0.70,
                'position_multiplier': 1.2,
                'preferred_strategies': ['order_block_bounce', 'fvg_entry', 'trend_continuation'],
                'trend_only': True,
                'direction': 'long'
            },
            MarketRegime.TRENDING_BEARISH: {
                'min_confidence': 0.70,
                'position_multiplier': 1.2,
                'preferred_strategies': ['order_block_bounce', 'fvg_entry', 'trend_continuation'],
                'trend_only': True,
                'direction': 'short'
            },
            MarketRegime.RANGING: {
                'min_confidence': 0.75,
                'position_multiplier': 0.8,
                'preferred_strategies': ['liquidity_sweep', 'fvg_entry', 'order_block_bounce'],
                'trend_only': False,
                'direction': 'both'
            },
            MarketRegime.VOLATILE: {
                'min_confidence': 0.80,
                'position_multiplier': 0.6,
                'preferred_strategies': ['liquidity_sweep', 'order_block_bounce'],
                'trend_only': False,
                'direction': 'both'
            },
            MarketRegime.QUIET: {
                'min_confidence': 0.85,
                'position_multiplier': 0.5,
                'preferred_strategies': ['order_block_bounce', 'fvg_entry'],
                'trend_only': False,
                'direction': 'both'
            },
            MarketRegime.BREAKOUT: {
                'min_confidence': 0.65,
                'position_multiplier': 1.5,
                'preferred_strategies': ['bos_breakout', 'liquidity_sweep'],
                'trend_only': False,
                'direction': 'both'
            },
            MarketRegime.REVERSAL: {
                'min_confidence': 0.75,
                'position_multiplier': 1.0,
                'preferred_strategies': ['choch_reversal', 'order_block_bounce', 'liquidity_sweep'],
                'trend_only': False,
                'direction': 'both'
            }
        }
    
    def detect_market_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime using rule-based approach
        """
        features = self._extract_regime_features(df)
        
        if self.regime_method == "rule_based":
            regime, confidence = self._detect_regime_rule_based(features)
        else:
            regime, confidence = self._detect_regime_rule_based(features)  # Fallback
        
        # Store in history
        self.regime_history.append((datetime.now(), regime))
        
        # Check for transition
        if len(self.regime_history) >= 2:
            prev_regime = self.regime_history[-2][1]
            if prev_regime != regime:
                logger.info(f"Regime transition: {prev_regime.value} -> {regime.value}")
        
        return regime, confidence
    
    def _extract_regime_features(self, df: pd.DataFrame) -> MarketRegimeFeatures:
        """Extract features for regime detection"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Returns and volatility
        returns = close.pct_change().dropna()
        volatility = returns.tail(20).std() * np.sqrt(252)
        
        # Trend strength (R² of linear regression)
        x = np.arange(len(close.tail(20)))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close.tail(20))
        trend_strength = abs(r_value)
        
        # Technical indicators
        adx = self.indicators.calculate_adx(df, 14).iloc[-1]
        rsi = self.indicators.calculate_rsi(close, 14).iloc[-1]
        atr = self.indicators.calculate_atr(df, 14).iloc[-1]
        atr_percent = atr / close.iloc[-1] if close.iloc[-1] > 0 else 0
        
        # Volume profile
        volume_profile = 1.0
        if 'volume' in df.columns:
            volume_profile = df['volume'].tail(20).mean() / df['volume'].mean() if df['volume'].mean() > 0 else 1.0
        
        return MarketRegimeFeatures(
            volatility=volatility,
            trend_strength=trend_strength,
            adx=adx if not np.isnan(adx) else 0,
            rsi=rsi if not np.isnan(rsi) else 50,
            atr_percent=atr_percent,
            volume_profile=volume_profile
        )
    
    def _detect_regime_rule_based(self, features: MarketRegimeFeatures) -> Tuple[MarketRegime, float]:
        """Rule-based regime detection"""
        
        # Trending check
        if features.trend_strength > 0.7 and features.adx > 25:
            if features.rsi > 60:
                return MarketRegime.TRENDING_BULLISH, 0.85
            elif features.rsi < 40:
                return MarketRegime.TRENDING_BEARISH, 0.85
        
        # Volatile check
        if features.volatility > 0.25 or features.atr_percent > 0.02:
            return MarketRegime.VOLATILE, 0.80
        
        # Quiet check
        if features.volatility < 0.10 and features.atr_percent < 0.005:
            return MarketRegime.QUIET, 0.75
        
        # Breakout check
        if features.volume_profile > 1.5 and features.atr_percent > 0.015:
            return MarketRegime.BREAKOUT, 0.75
        
        # Reversal check
        if abs(features.rsi - 50) > 25 and features.trend_strength < 0.4:
            return MarketRegime.REVERSAL, 0.70
        
        # Default to ranging
        return MarketRegime.RANGING, 0.60
    
    def generate_trading_signals(self, df: pd.DataFrame, symbol: str = "EURUSD",
                                  tf_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[TradingSignal]:
        """
        Generate trading signals with full integration
        """
        if self.liquidity_engine is None:
            raise ValueError("LiquidityEngine not set. Call set_liquidity_engine() first.")
        
        # Detect regime
        regime, regime_confidence = self.detect_market_regime(df)
        
        # Get regime parameters
        regime_params = self.regime_params.get(regime, self.regime_params[MarketRegime.RANGING])
        
        # Get SMC analysis
        liquidity_analysis = self.liquidity_engine.analyze_market(df)
        
        # Get session info
        session_type = self.session_engine.get_current_session()[0]
        is_kill_zone = self.session_engine.is_kill_zone()
        
        # Current trend from SMC
        current_trend = liquidity_analysis.get('structure', {}).get('current_trend', 'neutral')
        
        signals = []
        
        # Generate signals based on preferred strategies
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
            elif strategy == 'trend_continuation':
                new_signals = self._generate_trend_continuation_signals(df, liquidity_analysis, regime, symbol)
            elif strategy == 'fibonacci':
                new_signals = self._generate_fibonacci_signals(df, liquidity_analysis, regime, symbol)
            else:
                continue
            
            # Filter by trend alignment if required
            if regime_params.get('trend_only', False):
                filtered = []
                for sig in new_signals:
                    if current_trend == 'bullish' and sig.direction == 'long':
                        filtered.append(sig)
                    elif current_trend == 'bearish' and sig.direction == 'short':
                        filtered.append(sig)
                new_signals = filtered
            
            signals.extend(new_signals)
        
        # Apply regime filters
        signals = self._apply_regime_filters(signals, regime, regime_params)
        
        # Apply DXY correlation filter if risk engine available
        if self.risk_engine and hasattr(self.risk_engine, 'dxy_filter'):
            signals = self._apply_dxy_filter(signals, symbol)
        
        # Multi-timeframe confirmation
        if tf_data:
            signals = self._check_multi_timeframe_confluence(signals, tf_data)
        
        # Add metadata
        for signal in signals:
            signal.regime = regime
            signal.regime_confidence = regime_confidence
            signal.session_type = session_type.value
            signal.kill_zone = is_kill_zone
            signal.metadata['position_multiplier'] = regime_params['position_multiplier']
        
        logger.info(f"Generated {len(signals)} signals for {symbol} in {regime.value}")
        return signals
    
    def _apply_regime_filters(self, signals: List[TradingSignal], regime: MarketRegime,
                              regime_params: Dict) -> List[TradingSignal]:
        """Apply regime-specific filters"""
        filtered = []
        
        min_confidence = regime_params['min_confidence']
        
        for signal in signals:
            # Confidence filter
            if signal.confidence < min_confidence:
                continue
            
            # Kill zone boost
            if signal.kill_zone and signal.confidence < 1.0:
                signal.confidence = min(signal.confidence + 0.05, 1.0)
                signal.confluences.append('kill_zone')
            
            filtered.append(signal)
        
        return filtered
    
    def _apply_dxy_filter(self, signals: List[TradingSignal], symbol: str) -> List[TradingSignal]:
        """Filter signals based on DXY correlation"""
        filtered = []
        
        for signal in signals:
            # Check DXY correlation
            should_filter, reason = self.risk_engine.dxy_filter.should_filter_signal(
                symbol, signal.direction
            )
            
            if should_filter:
                logger.debug(f"Signal filtered by DXY: {reason}")
                continue
            
            # Adjust position size
            adjustment = self.risk_engine.dxy_filter.adjust_position_size(symbol, 1.0)
            signal.metadata['dxy_adjustment'] = adjustment
            
            filtered.append(signal)
        
        return filtered
    
    def _check_multi_timeframe_confluence(self, signals: List[TradingSignal],
                                          tf_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Confirm signals on lower timeframes"""
        confirmed = []
        
        for signal in signals:
            confirmation_count = 0
            
            for tf_name, tf_df in tf_data.items():
                # Quick check: is price action aligned?
                recent_close = tf_df['close'].iloc[-1]
                recent_trend = 'up' if tf_df['close'].iloc[-1] > tf_df['close'].iloc[-5] else 'down'
                
                if signal.direction == 'long' and recent_trend == 'up':
                    confirmation_count += 1
                elif signal.direction == 'short' and recent_trend == 'down':
                    confirmation_count += 1
            
            if confirmation_count >= len(tf_data) // 2:
                signal.confidence = min(signal.confidence + 0.1, 1.0)
                signal.confluences.append('multi_timeframe')
                confirmed.append(signal)
        
        return confirmed if confirmed else signals  # Return original if none confirmed
    
    # ========================================================================
    # SIGNAL GENERATION METHODS
    # ========================================================================
    
    def _generate_bos_signals(self, df: pd.DataFrame, analysis: Dict,
                              regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Break of Structure signals"""
        signals = []
        
        structure = analysis.get('structure', {})
        swing_highs = structure.get('swing_highs', [])
        swing_lows = structure.get('swing_lows', [])
        
        if not swing_highs or not swing_lows:
            return signals
        
        current_price = df['close'].iloc[-1]
        
        # Bullish BOS
        if swing_highs and current_price > swing_highs[-1]['price']:
            sl = swing_highs[-1]['price'] * 0.995
            tp = current_price + (current_price - swing_highs[-1]['price']) * 2
            
            signals.append(TradingSignal(
                timestamp=df.index[-1],
                symbol=symbol,
                direction='long',
                signal_type=SignalType.BOS_BREAKOUT,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                risk_reward=abs(tp - current_price) / abs(current_price - sl),
                confidence=0.75 if regime == MarketRegime.TRENDING_BULLISH else 0.65,
                regime=regime,
                regime_confidence=0.8,
                confluences=['bos_breakout']
            ))
        
        # Bearish BOS
        if swing_lows and current_price < swing_lows[-1]['price']:
            sl = swing_lows[-1]['price'] * 1.005
            tp = current_price - (swing_lows[-1]['price'] - current_price) * 2
            
            signals.append(TradingSignal(
                timestamp=df.index[-1],
                symbol=symbol,
                direction='short',
                signal_type=SignalType.BOS_BREAKOUT,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                risk_reward=abs(current_price - tp) / abs(sl - current_price),
                confidence=0.75 if regime == MarketRegime.TRENDING_BEARISH else 0.65,
                regime=regime,
                regime_confidence=0.8,
                confluences=['bos_breakout']
            ))
        
        return signals
    
    def _generate_choch_signals(self, df: pd.DataFrame, analysis: Dict,
                                regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Change of Character signals"""
        signals = []
        
        structure = analysis.get('structure', {})
        swing_highs = structure.get('swing_highs', [])
        swing_lows = structure.get('swing_lows', [])
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return signals
        
        current_price = df['close'].iloc[-1]
        
        # Bullish CHOCH (higher low)
        if (swing_lows[-1]['price'] > swing_lows[-2]['price'] and 
            swing_lows[-2]['price'] < swing_lows[-3]['price'] if len(swing_lows) >= 3 else True):
            
            sl = swing_lows[-2]['price'] * 0.998
            tp = swing_highs[-1]['price']
            
            signals.append(TradingSignal(
                timestamp=df.index[-1],
                symbol=symbol,
                direction='long',
                signal_type=SignalType.CHOCH_REVERSAL,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                risk_reward=abs(tp - current_price) / abs(current_price - sl),
                confidence=0.80,
                regime=regime,
                regime_confidence=0.8,
                confluences=['choch_reversal', 'higher_low']
            ))
        
        # Bearish CHOCH (lower high)
        if (swing_highs[-1]['price'] < swing_highs[-2]['price'] and 
            swing_highs[-2]['price'] > swing_highs[-3]['price'] if len(swing_highs) >= 3 else True):
            
            sl = swing_highs[-2]['price'] * 1.002
            tp = swing_lows[-1]['price']
            
            signals.append(TradingSignal(
                timestamp=df.index[-1],
                symbol=symbol,
                direction='short',
                signal_type=SignalType.CHOCH_REVERSAL,
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                risk_reward=abs(current_price - tp) / abs(sl - current_price),
                confidence=0.80,
                regime=regime,
                regime_confidence=0.8,
                confluences=['choch_reversal', 'lower_high']
            ))
        
        return signals
    
    def _generate_liquidity_signals(self, df: pd.DataFrame, analysis: Dict,
                                    regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Liquidity Sweep signals"""
        signals = []
        
        liquidity = analysis.get('liquidity', {})
        current_price = df['close'].iloc[-1]
        
        # Check buy-side liquidity (swept highs -> short)
        for zone in liquidity.get('buy_side', []):
            if zone.is_swept:
                sl = zone.price_level * 1.005
                tp = current_price - (zone.price_level - current_price) * 2
                
                signals.append(TradingSignal(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    direction='short',
                    signal_type=SignalType.LIQUIDITY_SWEEP,
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    risk_reward=2.0,
                    confidence=0.85,
                    regime=regime,
                    regime_confidence=0.8,
                    confluences=['liquidity_sweep', 'buy_side_swept'],
                    liquidity_sweep={'level': zone.price_level, 'type': 'buy_side'}
                ))
                break  # Only take the most recent
        
        # Check sell-side liquidity (swept lows -> long)
        for zone in liquidity.get('sell_side', []):
            if zone.is_swept:
                sl = zone.price_level * 0.995
                tp = current_price + (current_price - zone.price_level) * 2
                
                signals.append(TradingSignal(
                    timestamp=df.index[-1],
                    symbol=symbol,
                    direction='long',
                    signal_type=SignalType.LIQUIDITY_SWEEP,
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    risk_reward=2.0,
                    confidence=0.85,
                    regime=regime,
                    regime_confidence=0.8,
                    confluences=['liquidity_sweep', 'sell_side_swept'],
                    liquidity_sweep={'level': zone.price_level, 'type': 'sell_side'}
                ))
                break
        
        return signals
    
    def _generate_fvg_signals(self, df: pd.DataFrame, analysis: Dict,
                              regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Fair Value Gap signals"""
        signals = []
        
        fvg_zones = analysis.get('fvg_zones', [])
        current_price = df['close'].iloc[-1]
        
        for fvg in fvg_zones:
            if fvg.mitigated:
                continue
            
            # Bullish FVG
            if fvg.type == 'bullish' and fvg.ote_zone_62 and fvg.ote_zone_79:
                if fvg.ote_zone_62 <= current_price <= fvg.ote_zone_79:
                    signals.append(TradingSignal(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        direction='long',
                        signal_type=SignalType.FVG_ENTRY,
                        entry_price=current_price,
                        stop_loss=fvg.bottom * 0.998,
                        take_profit=fvg.top + (fvg.top - fvg.bottom),
                        risk_reward=2.0,
                        confidence=0.80,
                        regime=regime,
                        regime_confidence=0.8,
                        confluences=['fvg_entry', 'ote_zone'],
                        fvg={'top': fvg.top, 'bottom': fvg.bottom, 'type': fvg.type}
                    ))
            
            # Bearish FVG
            elif fvg.type == 'bearish' and fvg.ote_zone_62 and fvg.ote_zone_79:
                if fvg.ote_zone_79 <= current_price <= fvg.ote_zone_62:
                    signals.append(TradingSignal(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        direction='short',
                        signal_type=SignalType.FVG_ENTRY,
                        entry_price=current_price,
                        stop_loss=fvg.top * 1.002,
                        take_profit=fvg.bottom - (fvg.top - fvg.bottom),
                        risk_reward=2.0,
                        confidence=0.80,
                        regime=regime,
                        regime_confidence=0.8,
                        confluences=['fvg_entry', 'ote_zone'],
                        fvg={'top': fvg.top, 'bottom': fvg.bottom, 'type': fvg.type}
                    ))
        
        return signals
    
    def _generate_order_block_signals(self, df: pd.DataFrame, analysis: Dict,
                                      regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Order Block signals"""
        signals = []
        
        order_blocks = analysis.get('order_blocks', [])
        current_price = df['close'].iloc[-1]
        
        for ob in order_blocks:
            if ob.mitigated:
                continue
            
            # Bullish OB
            if ob.direction == 'bullish':
                if abs(current_price - ob.mitigation_price) / ob.mitigation_price < 0.002:
                    sl = ob.price_range[0] * 0.998
                    tp = ob.price_range[1] + (ob.price_range[1] - ob.price_range[0]) * 3
                    
                    signals.append(TradingSignal(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        direction='long',
                        signal_type=SignalType.ORDER_BLOCK_BOUNCE,
                        entry_price=current_price,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_reward=3.0,
                        confidence=0.85,
                        regime=regime,
                        regime_confidence=0.8,
                        confluences=['order_block', 'bullish_ob'],
                        order_block={'level': ob.mitigation_price, 'type': ob.direction}
                    ))
            
            # Bearish OB
            elif ob.direction == 'bearish':
                if abs(current_price - ob.mitigation_price) / ob.mitigation_price < 0.002:
                    sl = ob.price_range[1] * 1.002
                    tp = ob.price_range[0] - (ob.price_range[1] - ob.price_range[0]) * 3
                    
                    signals.append(TradingSignal(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        direction='short',
                        signal_type=SignalType.ORDER_BLOCK_BOUNCE,
                        entry_price=current_price,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_reward=3.0,
                        confidence=0.85,
                        regime=regime,
                        regime_confidence=0.8,
                        confluences=['order_block', 'bearish_ob'],
                        order_block={'level': ob.mitigation_price, 'type': ob.direction}
                    ))
        
        return signals
    
    def _generate_trend_continuation_signals(self, df: pd.DataFrame, analysis: Dict,
                                             regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate trend continuation signals"""
        signals = []
        
        # Only in strong trends
        if regime not in [MarketRegime.TRENDING_BULLISH, MarketRegime.TRENDING_BEARISH]:
            return signals
        
        structure = analysis.get('structure', {})
        current_trend = structure.get('current_trend', 'neutral')
        
        if current_trend == 'neutral':
            return signals
        
        order_blocks = analysis.get('order_blocks', [])
        current_price = df['close'].iloc[-1]
        
        for ob in order_blocks:
            if ob.mitigated or ob.times_tested > 0:
                continue
            
            # Bullish continuation
            if current_trend == 'bullish' and ob.direction == 'bullish':
                if current_price <= ob.price_range[1] * 1.005:
                    sl = ob.price_range[0] * 0.998
                    tp = current_price + (current_price - ob.price_range[0]) * 4
                    
                    signals.append(TradingSignal(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        direction='long',
                        signal_type=SignalType.TREND_CONTINUATION,
                        entry_price=current_price,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_reward=4.0,
                        confidence=0.90,
                        regime=regime,
                        regime_confidence=0.9,
                        confluences=['trend_continuation', 'fresh_ob'],
                        order_block={'level': ob.mitigation_price, 'type': ob.direction}
                    ))
            
            # Bearish continuation
            elif current_trend == 'bearish' and ob.direction == 'bearish':
                if current_price >= ob.price_range[0] * 0.995:
                    sl = ob.price_range[1] * 1.002
                    tp = current_price - (ob.price_range[1] - current_price) * 4
                    
                    signals.append(TradingSignal(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        direction='short',
                        signal_type=SignalType.TREND_CONTINUATION,
                        entry_price=current_price,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_reward=4.0,
                        confidence=0.90,
                        regime=regime,
                        regime_confidence=0.9,
                        confluences=['trend_continuation', 'fresh_ob'],
                        order_block={'level': ob.mitigation_price, 'type': ob.direction}
                    ))
        
        return signals
    
    def _generate_fibonacci_signals(self, df: pd.DataFrame, analysis: Dict,
                                    regime: MarketRegime, symbol: str) -> List[TradingSignal]:
        """Generate Fibonacci-based signals"""
        signals = []
        
        structure = analysis.get('structure', {})
        swing_highs = structure.get('swing_highs', [])
        swing_lows = structure.get('swing_lows', [])
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return signals
        
        recent_high = swing_highs[-1]['price']
        recent_low = swing_lows[-1]['price']
        
        # Calculate Fibonacci levels
        fib_levels = self.indicators.calculate_fibonacci_levels(recent_high, recent_low)
        current_price = df['close'].iloc[-1]
        
        # Check for retracement to key levels
        for level_name, level_price in fib_levels['retracement'].items():
            if abs(current_price - level_price) / current_price < 0.002:
                # Determine direction based on trend
                if current_price > level_price:  # Above support -> bullish
                    sl = recent_low * 0.998
                    tp = recent_high
                    
                    signals.append(TradingSignal(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        direction='long',
                        signal_type=SignalType.FIBONACCI,
                        entry_price=current_price,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_reward=abs(tp - current_price) / abs(current_price - sl),
                        confidence=0.80,
                        regime=regime,
                        regime_confidence=0.8,
                        confluences=[f'fib_{level_name}'],
                        metadata={'fib_level': level_name, 'level_price': level_price}
                    ))
                else:  # Below resistance -> bearish
                    sl = recent_high * 1.002
                    tp = recent_low
                    
                    signals.append(TradingSignal(
                        timestamp=df.index[-1],
                        symbol=symbol,
                        direction='short',
                        signal_type=SignalType.FIBONACCI,
                        entry_price=current_price,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_reward=abs(current_price - tp) / abs(sl - current_price),
                        confidence=0.80,
                        regime=regime,
                        regime_confidence=0.8,
                        confluences=[f'fib_{level_name}'],
                        metadata={'fib_level': level_name, 'level_price': level_price}
                    ))
        
        return signals
    
    def update_performance(self, signal: TradingSignal, pnl: float):
        """Update performance tracking"""
        regime = signal.regime
        perf = self.regime_performance[regime]
        
        perf['trades'] += 1
        if pnl > 0:
            perf['wins'] += 1
        else:
            perf['losses'] += 1
        perf['total_pnl'] += pnl
    
    def get_best_regime(self) -> Tuple[Optional[MarketRegime], float]:
        """Get best performing regime"""
        best_regime = None
        best_win_rate = 0.0
        
        for regime, perf in self.regime_performance.items():
            if perf['trades'] >= 5:
                win_rate = perf['wins'] / perf['trades']
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_regime = regime
        
        return best_regime, best_win_rate


# ============================================================================
# COMPLETE TRADING SYSTEM INTEGRATION
# ============================================================================

class CompleteTradingSystem:
    """
    Fully integrated trading system with all components
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Initialize all engines
        from core.liquidity_engine import LiquidityEngine  # Import your existing engine
        from core.risk_engine import RiskEngine  # Import the risk engine from previous code
        
        self.liquidity_engine = LiquidityEngine(config)
        self.risk_engine = RiskEngine(config)
        self.strategy_engine = StrategyEngine(config)
        
        # Wire up dependencies
        self.strategy_engine.set_liquidity_engine(self.liquidity_engine)
        self.strategy_engine.set_risk_engine(self.risk_engine)
        
        # Execution
        self.executor = None  # MT5Executor would go here
        
        logger.info("CompleteTradingSystem initialized")
    
    def process_market_data(self, df: pd.DataFrame, symbol: str = "EURUSD",
                           tf_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process market data through complete pipeline
        """
        # Step 1: Generate signals
        signals = self.strategy_engine.generate_trading_signals(df, symbol, tf_data)
        
        # Step 2: Calculate position sizes with risk management
        managed_signals = []
        for signal in signals:
            position_calc = self.risk_engine.calculate_position_size(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                confidence=signal.confidence,
                volatility=self._get_volatility(df)
            )
            
            if 'error' not in position_calc:
                signal.quantity = position_calc['position_size']
                signal.metadata['risk_amount'] = position_calc['risk_amount']
                signal.metadata['margin_required'] = position_calc['margin_required']
                managed_signals.append(signal)
        
        # Step 3: Filter for highest probability
        final_signals = sorted(managed_signals, key=lambda x: x.confidence, reverse=True)[:3]
        
        return {
            'signals': final_signals,
            'regime': self.strategy_engine.regime_history[-1][1] if self.strategy_engine.regime_history else None,
            'risk_metrics': self.risk_engine.get_risk_metrics(),
            'can_trade': self.risk_engine.can_trade()[0]
        }
    
    def _get_volatility(self, df: pd.DataFrame) -> float:
        """Calculate current volatility"""
        return df['close'].pct_change().tail(20).std()
    
    def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute trading signal"""
        if not self.executor:
            logger.error("No executor configured")
            return False
        
        # Final risk check
        can_trade, reason = self.risk_engine.can_trade()
        if not can_trade:
            logger.warning(f"Trade blocked: {reason}")
            return False
        
        # Execute
        result = self.executor.execute_signal(signal, signal.quantity)
        
        if result.get('success'):
            # Register with risk engine
            trade_risk = TradeRisk(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size=signal.quantity,
                risk_amount=signal.metadata.get('risk_amount', 0),
                risk_percentage=0,
                risk_reward=signal.risk_reward,
                margin_required=signal.metadata.get('margin_required', 0),
                correlation_risk=0,
                dxy_hedge_ratio=0
            )
            self.risk_engine.register_trade(trade_risk, result['ticket'])
            return True
        
        return False


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'symbol': 'EURUSD',
        'strategy': {
            'smc': {
                'swing_length': 5,
                'fvg_min_size': 0.0001
            },
            'regime_detection': {
                'enabled': True,
                'method': 'rule_based'
            }
        },
        'risk_management': {
            'initial_balance': 10000,
            'risk_per_trade': 0.015,
            'max_positions': 3,
            'dxy_correlation': {
                'enabled': True
            }
        }
    }
    
    # Initialize system
    system = CompleteTradingSystem(CONFIG)
    
    # Example: Process market data
    # df = pd.read_csv('market_data.csv', parse_dates=['time'], index_col='time')
    # result = system.process_market_data(df)
    # print(f"Generated {len(result['signals'])} signals")