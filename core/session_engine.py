"""
Session Engine - Institutional-grade session detection and parameter management
Features:
- Multi-session detection (Asia, London, NY, Overlap, Pre/Post)
- Dynamic volatility adjustment per session
- Session-specific strategy bias
- Kill zone management
- Holiday and weekend filtering
- News event integration
- Session transition handling
- Performance tracking per session
- Adaptive parameter tuning
"""

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import pytz
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import logging
import asyncio
from collections import defaultdict
import json
import calendar
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class SessionType(Enum):
    """Trading session types"""
    ASIA = "asia"
    LONDON = "london"
    NY = "ny"
    OVERLAP = "overlap"
    PRE_ASIA = "pre_asia"
    POST_NY = "post_ny"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    NEWS = "news"
    CLOSED = "closed"


class StrategyBias(Enum):
    """Strategy bias for different sessions"""
    TREND = "trend"
    RANGE = "range"
    MOMENTUM = "momentum"
    SCALP = "scalp"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    NEUTRAL = "neutral"


@dataclass
class SessionConfig:
    """Comprehensive session configuration"""
    type: SessionType
    name: str
    start: time
    end: time
    timezone: str = "UTC"
    
    # Volatility parameters
    volatility_multiplier: float = 1.0
    min_volatility: float = 10.0
    max_volatility: float = 50.0
    base_atr_period: int = 14
    
    # Strategy parameters
    strategy_bias: StrategyBias = StrategyBias.NEUTRAL
    preferred_pairs: List[str] = field(default_factory=list)
    excluded_pairs: List[str] = field(default_factory=list)
    
    # Risk parameters
    risk_multiplier: float = 1.0
    max_trades_per_session: int = 10
    min_confidence: float = 0.6
    
    # Execution parameters
    max_spread_multiplier: float = 1.0
    min_volume_multiplier: float = 1.0
    
    # Dynamic adjustment
    adaptive_params: bool = True
    lookback_days: int = 30
    
    def __post_init__(self):
        if isinstance(self.start, str):
            self.start = datetime.strptime(self.start, "%H:%M").time()
        if isinstance(self.end, str):
            self.end = datetime.strptime(self.end, "%H:%M").time()


@dataclass
class SessionStats:
    """Performance statistics per session"""
    session_type: SessionType
    date: datetime
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_volatility: float = 0.0
    max_volatility: float = 0.0
    min_volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_r_multiple: float = 0.0


@dataclass
class KillZone:
    """High-probability trading zones"""
    name: str
    start: time
    end: time
    session_type: SessionType
    probability_score: float
    avg_move_pips: float
    win_rate: float


class SessionEngine:
    """
    Professional session management engine with:
    - Real-time session detection
    - Dynamic parameter adjustment
    - Kill zone identification
    - Holiday/weekend filtering
    - News event integration
    - Performance tracking per session
    - Adaptive parameter tuning
    - Multi-timezone support
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Timezone configuration
        self.timezone = pytz.timezone(config.get('timezone', 'UTC'))
        self.gmt = pytz.UTC
        
        # Session configurations
        self.sessions: Dict[SessionType, SessionConfig] = self._init_sessions()
        self.kill_zones: List[KillZone] = self._init_kill_zones()
        
        # Current state
        self.current_session: Optional[Tuple[SessionType, SessionConfig]] = None
        self.next_session: Optional[Tuple[SessionType, SessionConfig]] = None
        self.session_start_time: Optional[datetime] = None
        
        # Performance tracking
        self.session_stats: Dict[SessionType, List[SessionStats]] = defaultdict(list)
        self.daily_stats: Dict[str, Dict] = {}
        self.weekly_stats: Dict[str, Dict] = {}
        
        # Dynamic parameters
        self.adaptive_parameters: Dict[SessionType, Dict] = defaultdict(dict)
        self.volatility_history: Dict[SessionType, List[float]] = defaultdict(list)
        
        # Market hours cache
        self.market_hours_cache: Dict[str, bool] = {}
        self.holiday_cache: Dict[str, List[str]] = {}
        
        # News calendar (would be populated from external source)
        self.news_events: List[Dict] = []
        
        logger.info("SessionEngine initialized")
    
    def _init_sessions(self) -> Dict[SessionType, SessionConfig]:
        """Initialize session configurations from config"""
        sessions = {}
        
        # Asia session
        asia_cfg = self.config['sessions']['asia']
        sessions[SessionType.ASIA] = SessionConfig(
            type=SessionType.ASIA,
            name="Asian Session",
            start=asia_cfg['start_time'],
            end=asia_cfg['end_time'],
            timezone="Asia/Tokyo",
            volatility_multiplier=asia_cfg['volatility_multiplier'],
            min_volatility=asia_cfg['min_volatility'],
            max_volatility=asia_cfg['max_volatility'],
            strategy_bias=StrategyBias(asia_cfg['strategy_bias']),
            preferred_pairs=asia_cfg['pairs'],
            max_trades_per_session=asia_cfg.get('max_trades', 5)
        )
        
        # London session
        london_cfg = self.config['sessions']['london']
        sessions[SessionType.LONDON] = SessionConfig(
            type=SessionType.LONDON,
            name="London Session",
            start=london_cfg['start_time'],
            end=london_cfg['end_time'],
            timezone="Europe/London",
            volatility_multiplier=london_cfg['volatility_multiplier'],
            min_volatility=london_cfg['min_volatility'],
            max_volatility=london_cfg['max_volatility'],
            strategy_bias=StrategyBias(london_cfg['strategy_bias']),
            preferred_pairs=london_cfg['pairs'],
            max_trades_per_session=london_cfg.get('max_trades', 8)
        )
        
        # NY session
        ny_cfg = self.config['sessions']['ny']
        sessions[SessionType.NY] = SessionConfig(
            type=SessionType.NY,
            name="New York Session",
            start=ny_cfg['start_time'],
            end=ny_cfg['end_time'],
            timezone="America/New_York",
            volatility_multiplier=ny_cfg['volatility_multiplier'],
            min_volatility=ny_cfg['min_volatility'],
            max_volatility=ny_cfg['max_volatility'],
            strategy_bias=StrategyBias(ny_cfg['strategy_bias']),
            preferred_pairs=ny_cfg['pairs'],
            max_trades_per_session=ny_cfg.get('max_trades', 8)
        )
        
        # Overlap session (London-NY overlap)
        overlap_cfg = self.config['sessions']['overlap']
        sessions[SessionType.OVERLAP] = SessionConfig(
            type=SessionType.OVERLAP,
            name="London-NY Overlap",
            start=overlap_cfg['start_time'],
            end=overlap_cfg['end_time'],
            timezone="America/New_York",
            volatility_multiplier=overlap_cfg['volatility_multiplier'],
            min_volatility=overlap_cfg['min_volatility'],
            max_volatility=overlap_cfg['max_volatility'],
            strategy_bias=StrategyBias(overlap_cfg['strategy_bias']),
            preferred_pairs=overlap_cfg['pairs'],
            max_trades_per_session=overlap_cfg.get('max_trades', 10)
        )
        
        # Pre-Asia session
        sessions[SessionType.PRE_ASIA] = SessionConfig(
            type=SessionType.PRE_ASIA,
            name="Pre-Asia Session",
            start="22:00",
            end="00:00",
            timezone="UTC",
            volatility_multiplier=0.6,
            min_volatility=5,
            max_volatility=20,
            strategy_bias=StrategyBias.RANGE,
            preferred_pairs=["AUDUSD", "NZDUSD"],
            max_trades_per_session=3
        )
        
        # Post-NY session
        sessions[SessionType.POST_NY] = SessionConfig(
            type=SessionType.POST_NY,
            name="Post-NY Session",
            start="22:00",
            end="23:59",
            timezone="America/New_York",
            volatility_multiplier=0.5,
            min_volatility=5,
            max_volatility=15,
            strategy_bias=StrategyBias.RANGE,
            preferred_pairs=["EURUSD", "GBPUSD"],
            max_trades_per_session=2
        )
        
        return sessions
    
    def _init_kill_zones(self) -> List[KillZone]:
        """Initialize high-probability kill zones"""
        return [
            KillZone(
                name="London Open",
                start=time(8, 0),
                end=time(9, 0),
                session_type=SessionType.LONDON,
                probability_score=0.75,
                avg_move_pips=25,
                win_rate=0.68
            ),
            KillZone(
                name="NY Open",
                start=time(13, 0),
                end=time(14, 0),
                session_type=SessionType.NY,
                probability_score=0.78,
                avg_move_pips=30,
                win_rate=0.72
            ),
            KillZone(
                name="London-NY Overlap",
                start=time(13, 0),
                end=time(16, 0),
                session_type=SessionType.OVERLAP,
                probability_score=0.82,
                avg_move_pips=35,
                win_rate=0.75
            ),
            KillZone(
                name="Tokyo Open",
                start=time(0, 0),
                end=time(1, 0),
                session_type=SessionType.ASIA,
                probability_score=0.65,
                avg_move_pips=15,
                win_rate=0.62
            ),
            KillZone(
                name="London Lunch",
                start=time(12, 0),
                end=time(13, 0),
                session_type=SessionType.LONDON,
                probability_score=0.55,
                avg_move_pips=10,
                win_rate=0.58
            )
        ]
    
    def get_current_session(self, dt: Optional[datetime] = None) -> Tuple[SessionType, Optional[SessionConfig]]:
        """
        Detect current trading session with proper timezone handling
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        else:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=self.timezone)
        
        # Convert to UTC for comparison
        dt_utc = dt.astimezone(self.gmt)
        current_time_utc = dt_utc.time()
        
        # Check if weekend
        if self._is_weekend(dt_utc):
            return SessionType.WEEKEND, None
        
        # Check if holiday
        if self._is_holiday(dt_utc):
            return SessionType.HOLIDAY, None
        
        # Check news events
        if self._is_news_time(dt_utc):
            return SessionType.NEWS, None
        
        # Check overlap first (highest priority)
        overlap = self.sessions[SessionType.OVERLAP]
        if self._time_in_range(overlap.start, overlap.end, current_time_utc):
            self.current_session = (SessionType.OVERLAP, overlap)
            return SessionType.OVERLAP, overlap
        
        # Check other sessions in order of importance
        session_order = [SessionType.LONDON, SessionType.NY, SessionType.ASIA,
                        SessionType.PRE_ASIA, SessionType.POST_NY]
        
        for session_type in session_order:
            config = self.sessions.get(session_type)
            if config and self._time_in_range(config.start, config.end, current_time_utc):
                self.current_session = (session_type, config)
                return session_type, config
        
        return SessionType.CLOSED, None
    
    def get_next_session(self, dt: Optional[datetime] = None) -> Tuple[SessionType, SessionConfig, timedelta]:
        """
        Get next upcoming trading session
        """
        if dt is None:
            dt = datetime.now(self.timezone)
        
        dt_utc = dt.astimezone(self.gmt)
        current_time = dt_utc.time()
        
        # Find the next session
        next_session = None
        min_time_diff = timedelta(days=1)
        
        for session_type, config in self.sessions.items():
            if session_type == SessionType.OVERLAP:
                continue
            
            # Calculate next start time
            session_start = datetime.combine(dt_utc.date(), config.start)
            if session_start <= dt_utc:
                session_start += timedelta(days=1)
            
            time_diff = session_start - dt_utc
            
            if time_diff < min_time_diff and time_diff > timedelta(0):
                min_time_diff = time_diff
                next_session = (session_type, config)
        
        if next_session:
            return (*next_session, min_time_diff)
        
        return SessionType.CLOSED, None, timedelta(0)
    
    def get_session_parameters(self, symbol: str, volatility: float, 
                              session_type: Optional[SessionType] = None) -> Dict[str, Any]:
        """
        Get comprehensive trading parameters adjusted for current session
        
        Args:
            symbol: Trading symbol
            volatility: Current volatility in pips
            session_type: Optional session type (uses current if not provided)
        
        Returns:
            Dictionary with adjusted parameters
        """
        if session_type is None:
            session_type, config = self.get_current_session()
        else:
            config = self.sessions.get(session_type)
        
        if session_type in [SessionType.CLOSED, SessionType.WEEKEND, 
                           SessionType.HOLIDAY, SessionType.NEWS]:
            return {'trading_allowed': False, 'reason': f'Session: {session_type.value}'}
        
        if config is None:
            return {'trading_allowed': False, 'reason': 'No session config'}
        
        # Apply adaptive parameters if enabled
        if config.adaptive_params:
            config = self._apply_adaptive_parameters(config, symbol)
        
        # Adjust volatility thresholds based on session
        adjusted_min_vol = config.min_volatility * config.volatility_multiplier
        adjusted_max_vol = config.max_volatility * config.volatility_multiplier
        
        # Check if volatility is suitable
        if volatility < adjusted_min_vol or volatility > adjusted_max_vol:
            return {
                'trading_allowed': False,
                'reason': f'Volatility {volatility:.1f} outside range {adjusted_min_vol:.1f}-{adjusted_max_vol:.1f}'
            }
        
        # Check if symbol is allowed
        is_preferred = symbol in config.preferred_pairs
        is_excluded = symbol in config.excluded_pairs
        
        if is_excluded:
            return {
                'trading_allowed': False,
                'reason': f'Symbol {symbol} excluded in {config.name}'
            }
        
        # Calculate dynamic risk multiplier
        risk_multiplier = self._calculate_risk_multiplier(config, volatility)
        
        # Calculate dynamic confidence threshold
        confidence_threshold = self._calculate_confidence_threshold(config, volatility, is_preferred)
        
        # Get active kill zones
        active_kill_zones = self.get_active_kill_zones()
        
        # Determine if in high-probability zone
        in_kill_zone = any(kz.session_type == session_type for kz in active_kill_zones)
        
        return {
            'trading_allowed': True,
            'session_type': session_type.value,
            'session_name': config.name,
            'strategy_bias': config.strategy_bias.value,
            'volatility_multiplier': config.volatility_multiplier,
            'is_preferred': is_preferred,
            'min_volatility': adjusted_min_vol,
            'max_volatility': adjusted_max_vol,
            'current_volatility': volatility,
            'risk_multiplier': risk_multiplier,
            'confidence_threshold': confidence_threshold,
            'max_trades': config.max_trades_per_session,
            'in_kill_zone': in_kill_zone,
            'time_remaining': self._get_time_remaining(config)
        }
    
    def _apply_adaptive_parameters(self, config: SessionConfig, symbol: str) -> SessionConfig:
        """
        Apply adaptive parameters based on historical performance with validation
        """
        # Validate input parameters
        if not isinstance(config, SessionConfig):
            raise TypeError("config must be an instance of SessionConfig")
        if not isinstance(symbol, str) or len(symbol) < 3:
            raise ValueError(f"Invalid symbol: {symbol} (must be at least 3 characters)")
        
        # Get recent stats for this session
        recent_stats = self.session_stats.get(config.type, [])[-config.lookback_days:]
        
        if not recent_stats:
            return config
        
        # Calculate average win rate with validation
        valid_stats = [s for s in recent_stats if s.trades_taken > 0]
        if valid_stats:
            avg_win_rate = np.mean([s.win_rate for s in valid_stats])
            # Validate win rate is within reasonable range
            avg_win_rate = np.clip(avg_win_rate, 0, 100)
            
            # Adjust confidence threshold based on win rate
            if avg_win_rate > 70:  # 70% win rate
                config.min_confidence = max(0.5, config.min_confidence - 0.05)
            elif avg_win_rate < 50:  # 50% win rate
                config.min_confidence = min(0.85, config.min_confidence + 0.05)
        
        # Adjust volatility thresholds based on recent volatility with validation
        recent_volatility = self.volatility_history.get(config.type, [])
        if recent_volatility and len(recent_volatility) >= 5:  # Need at least 5 data points
            # Remove outliers (3 standard deviations from mean)
            volatility_mean = np.mean(recent_volatility)
            volatility_std = np.std(recent_volatility)
            valid_volatility = [v for v in recent_volatility if abs(v - volatility_mean) <= 3 * volatility_std]
            
            if valid_volatility:
                avg_vol = np.mean(valid_volatility)
                std_vol = np.std(valid_volatility)
                
                # Calculate new volatility thresholds with bounds
                new_min_vol = max(5, avg_vol - std_vol)
                new_max_vol = min(200, avg_vol + std_vol)
                
                # Ensure min < max and thresholds are reasonable
                if new_min_vol < new_max_vol and new_max_vol - new_min_vol >= 5:
                    config.min_volatility = new_min_vol
                    config.max_volatility = new_max_vol
        
        # Validate all adaptive parameters are within valid ranges
        self._validate_adaptive_parameters(config)
        
        # Store adaptive parameters
        self.adaptive_parameters[config.type] = {
            'min_confidence': config.min_confidence,
            'min_volatility': config.min_volatility,
            'max_volatility': config.max_volatility
        }
        
        logger.debug(f"Adaptive parameters for {config.name} ({symbol}): "
                     f"confidence={config.min_confidence:.2f}, "
                     f"min_vol={config.min_volatility:.1f}, "
                     f"max_vol={config.max_volatility:.1f}")
        
        return config
    
    def _validate_adaptive_parameters(self, config: SessionConfig):
        """Validate adaptive parameters are within valid ranges"""
        # Validate confidence threshold
        if not isinstance(config.min_confidence, (int, float)):
            raise TypeError("min_confidence must be a number")
        if config.min_confidence < 0.5 or config.min_confidence > 0.95:
            raise ValueError(f"min_confidence must be between 0.5 and 0.95, got {config.min_confidence}")
        
        # Validate volatility thresholds
        if not isinstance(config.min_volatility, (int, float)) or not isinstance(config.max_volatility, (int, float)):
            raise TypeError("Volatility thresholds must be numbers")
        if config.min_volatility <= 0 or config.max_volatility <= 0:
            raise ValueError(f"Volatility thresholds must be positive, got min={config.min_volatility}, max={config.max_volatility}")
        if config.min_volatility >= config.max_volatility:
            raise ValueError(f"min_volatility ({config.min_volatility}) must be less than max_volatility ({config.max_volatility})")
        if config.max_volatility - config.min_volatility < 5:
            raise ValueError(f"Volatility range must be at least 5, got {config.max_volatility - config.min_volatility}")
        
        # Ensure volatility thresholds are within reasonable bounds
        config.min_volatility = max(5, min(config.min_volatility, 100))
        config.max_volatility = max(config.min_volatility + 5, min(config.max_volatility, 200))
    
    def _calculate_risk_multiplier(self, config: SessionConfig, volatility: float) -> float:
        """
        Calculate dynamic risk multiplier based on session and volatility
        """
        base_multiplier = config.risk_multiplier
        
        # Adjust based on volatility
        if volatility < config.min_volatility * 1.2:
            # Low volatility - reduce risk
            base_multiplier *= 0.8
        elif volatility > config.max_volatility * 0.8:
            # High volatility - reduce risk
            base_multiplier *= 0.7
        else:
            # Optimal volatility - full risk
            base_multiplier *= 1.0
        
        # Adjust based on session performance
        recent_stats = self.session_stats.get(config.type, [])[-10:]
        if recent_stats:
            win_rate = np.mean([s.win_rate for s in recent_stats if s.trades_taken > 0])
            if win_rate > 0.65:
                base_multiplier *= 1.2
            elif win_rate < 0.45:
                base_multiplier *= 0.6
        
        return max(0.3, min(2.0, base_multiplier))
    
    def _calculate_confidence_threshold(self, config: SessionConfig, 
                                       volatility: float, is_preferred: bool) -> float:
        """
        Calculate dynamic confidence threshold
        """
        base_threshold = config.min_confidence
        
        # Adjust for volatility
        if volatility < config.min_volatility * 1.2:
            base_threshold += 0.1  # Need more confidence in low volatility
        elif volatility > config.max_volatility * 0.8:
            base_threshold += 0.15  # Need more confidence in high volatility
        
        # Adjust for preferred pairs
        if is_preferred:
            base_threshold -= 0.05
        
        # Adjust for time of session
        time_remaining = self._get_time_remaining_minutes(config)
        if time_remaining < 30:
            base_threshold += 0.1  # Last 30 minutes - be more selective
        
        return max(0.5, min(0.95, base_threshold))
    
    def should_trade(self, symbol: str, volatility: float, 
                    confidence: float = 0.0) -> Tuple[bool, str]:
        """
        Comprehensive trade eligibility check
        """
        params = self.get_session_parameters(symbol, volatility)
        
        if not params['trading_allowed']:
            return False, params.get('reason', 'Trading not allowed')
        
        # Check confidence threshold
        if confidence < params['confidence_threshold']:
            return False, f"Confidence {confidence:.2f} below threshold {params['confidence_threshold']:.2f}"
        
        # Check session-specific logic
        session_type = params['session_type']
        strategy_bias = params['strategy_bias']
        
        if session_type == 'asia' and not params['is_preferred']:
            return False, 'Non-preferred pair for Asia session'
        
        if strategy_bias == 'scalp' and volatility > 40:
            return False, 'Volatility too high for scalp strategy'
        
        if strategy_bias == 'range' and volatility < 15:
            return False, 'Volatility too low for range strategy'
        
        # Check kill zone bonus
        if params['in_kill_zone'] and confidence < params['confidence_threshold'] - 0.1:
            # In kill zone, we can accept slightly lower confidence
            return True, 'OK (kill zone)'
        
        return True, 'OK'
    
    def get_active_kill_zones(self) -> List[KillZone]:
        """
        Get currently active kill zones
        """
        now_utc = datetime.now(self.gmt).time()
        active = []
        
        for kz in self.kill_zones:
            if self._time_in_range(kz.start, kz.end, now_utc):
                active.append(kz)
        
        return active
    
    def get_best_kill_zone(self) -> Optional[KillZone]:
        """
        Get the highest probability kill zone currently active
        """
        active = self.get_active_kill_zones()
        if not active:
            return None
        
        return max(active, key=lambda kz: kz.probability_score)
    
    def _is_weekend(self, dt: datetime) -> bool:
        """Check if given time is weekend"""
        # Weekend is Saturday (5) and Sunday (6)
        return dt.weekday() >= 5
    
    def _is_holiday(self, dt: datetime) -> bool:
        """Check if given time is a major holiday"""
        date_str = dt.strftime("%Y-%m-%d")
        
        # Common forex holidays
        holidays = {
            "01-01": "New Year's Day",
            "12-25": "Christmas Day",
            "12-26": "Boxing Day",
            "01-02": "New Year's (observed)"
        }
        
        month_day = dt.strftime("%m-%d")
        return month_day in holidays
    
    def _is_news_time(self, dt: datetime) -> bool:
        """Check if currently in news event"""
        # In production, this would check against economic calendar
        # Simplified version - avoid NFP, FOMC, etc.
        news_times = [
            (time(13, 30), "NFP"),  # First Friday of month
            (time(14, 0), "FOMC"),   # Various Wednesdays
            (time(8, 30), "CPI"),    # Monthly
        ]
        
        current_time = dt.time()
        
        for news_time, name in news_times:
            if abs((current_time.hour * 60 + current_time.minute) - 
                   (news_time.hour * 60 + news_time.minute)) < 30:
                return True
        
        return False
    
    def _time_in_range(self, start: time, end: time, x: time) -> bool:
        """Check if time x is within range [start, end]"""
        if start <= end:
            return start <= x <= end
        else:
            # Range crosses midnight
            return start <= x or x <= end
    
    def _get_time_remaining(self, config: SessionConfig) -> str:
        """Get time remaining in current session"""
        now = datetime.now(self.gmt).time()
        end = config.end
        
        now_minutes = now.hour * 60 + now.minute
        end_minutes = end.hour * 60 + end.minute
        
        remaining = end_minutes - now_minutes
        if remaining < 0:
            remaining += 24 * 60
        
        hours = remaining // 60
        minutes = remaining % 60
        
        return f"{hours}h {minutes}m"
    
    def _get_time_remaining_minutes(self, config: SessionConfig) -> int:
        """Get time remaining in minutes"""
        now = datetime.now(self.gmt).time()
        end = config.end
        
        now_minutes = now.hour * 60 + now.minute
        end_minutes = end.hour * 60 + end.minute
        
        remaining = end_minutes - now_minutes
        if remaining < 0:
            remaining += 24 * 60
        
        return remaining
    
    async def calculate_volatility(self, symbol: str, data_engine, 
                                  period: int = 20, timeframe: str = "H1") -> float:
        """
        Calculate current volatility using multiple methods
        """
        end = datetime.now(self.timezone)
        start = end - timedelta(days=7)  # Get enough data
        
        df = await data_engine.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end
        )
        
        if df is None or len(df) < period:
            return 0.0
        
        # Calculate multiple volatility metrics
        # 1. ATR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-period:])
        
        # 2. Standard deviation of returns
        returns = np.diff(np.log(close))
        std_vol = np.std(returns[-period:]) * np.sqrt(252 * 24 * 60)  # Annualized
        
        # 3. Garman-Klass volatility
        log_hl = np.log(high[-period:] / low[-period:])
        log_co = np.log(close[-period:] / close[-period-1:-1])
        gk_vol = np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2)
        
        # Combine metrics (weighted average)
        volatility = (atr * 0.5 + std_vol * 0.3 + np.mean(gk_vol) * 0.2)
        
        # Convert to pips
        if 'JPY' in symbol:
            volatility = volatility * 100
        else:
            volatility = volatility * 10000
        
        # Store for adaptive parameters
        current_session, _ = self.get_current_session()
        if current_session not in [SessionType.CLOSED, SessionType.WEEKEND]:
            self.volatility_history[current_session].append(volatility)
            # Keep last 100 values
            if len(self.volatility_history[current_session]) > 100:
                self.volatility_history[current_session].pop(0)
        
        return volatility
    
    def update_session_stats(self, trade_result: Dict):
        """
        Update session statistics with trade result
        """
        session_type = SessionType(trade_result.get('session_type', 'closed'))
        date = datetime.now().date()
        
        # Find or create stats for this session/date
        stats_list = self.session_stats[session_type]
        stats = next((s for s in stats_list if s.date.date() == date), None)
        
        if stats is None:
            stats = SessionStats(
                session_type=session_type,
                date=datetime.now()
            )
            stats_list.append(stats)
        
        # Update stats
        stats.trades_taken += 1
        if trade_result['result'] == 'win':
            stats.wins += 1
        else:
            stats.losses += 1
        
        stats.total_pnl += trade_result['profit']
        stats.avg_r_multiple = (stats.avg_r_multiple * (stats.trades_taken - 1) + 
                               trade_result['r_multiple']) / stats.trades_taken
        
        # Calculate derived metrics
        if stats.trades_taken > 0:
            stats.win_rate = (stats.wins / stats.trades_taken) * 100
            
            gross_profit = sum(t['profit'] for t in self.trade_history 
                              if t.get('session_type') == session_type.value and t['profit'] > 0)
            gross_loss = abs(sum(t['profit'] for t in self.trade_history 
                                if t.get('session_type') == session_type.value and t['profit'] < 0))
            
            stats.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def get_session_performance(self, session_type: SessionType, days: int = 30) -> Dict:
        """
        Get performance metrics for specific session
        """
        stats_list = self.session_stats.get(session_type, [])
        recent_stats = [s for s in stats_list if (datetime.now() - s.date).days <= days]
        
        if not recent_stats:
            return {}
        
        trades = sum(s.trades_taken for s in recent_stats)
        wins = sum(s.wins for s in recent_stats)
        pnl = sum(s.total_pnl for s in recent_stats)
        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        avg_r = np.mean([s.avg_r_multiple for s in recent_stats if s.trades_taken > 0])
        
        return {
            'trades': trades,
            'wins': wins,
            'losses': trades - wins,
            'win_rate': win_rate,
            'total_pnl': pnl,
            'avg_r_multiple': avg_r,
            'profit_factor': np.mean([s.profit_factor for s in recent_stats if s.profit_factor > 0])
        }
    
    def get_best_session(self) -> Tuple[SessionType, Dict]:
        """
        Get the best performing session based on historical data
        """
        best_session = None
        best_metrics = None
        best_score = -float('inf')
        
        for session_type in SessionType:
            if session_type in [SessionType.CLOSED, SessionType.WEEKEND, 
                               SessionType.HOLIDAY, SessionType.NEWS]:
                continue
            
            metrics = self.get_session_performance(session_type, days=30)
            if not metrics or metrics['trades'] < 5:
                continue
            
            # Calculate composite score
            score = (metrics['win_rate'] * 0.4 + 
                    metrics['profit_factor'] * 20 * 0.3 + 
                    metrics['avg_r_multiple'] * 10 * 0.3)
            
            if score > best_score:
                best_score = score
                best_session = session_type
                best_metrics = metrics
        
        return best_session, best_metrics
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of current session
        """
        session_type, config = self.get_current_session()
        
        if session_type in [SessionType.CLOSED, SessionType.WEEKEND, 
                           SessionType.HOLIDAY, SessionType.NEWS]:
            return {
                'status': session_type.value,
                'message': f'Market closed: {session_type.value}'
            }
        
        # Get performance for this session
        performance = self.get_session_performance(session_type, days=30)
        
        # Get active kill zones
        active_kill_zones = [kz.name for kz in self.get_active_kill_zones()]
        
        # Get next session
        next_session, next_config, time_until = self.get_next_session()
        
        return {
            'status': 'active',
            'current_session': {
                'name': config.name,
                'type': session_type.value,
                'start': config.start.strftime("%H:%M"),
                'end': config.end.strftime("%H:%M"),
                'time_remaining': self._get_time_remaining(config),
                'strategy_bias': config.strategy_bias.value,
                'volatility_multiplier': config.volatility_multiplier,
                'preferred_pairs': config.preferred_pairs[:5],  # Top 5
                'active_kill_zones': active_kill_zones
            },
            'performance': {
                'win_rate': f"{performance.get('win_rate', 0):.1f}%",
                'profit_factor': f"{performance.get('profit_factor', 0):.2f}",
                'avg_r': f"{performance.get('avg_r_multiple', 0):.2f}",
                'trades_30d': performance.get('trades', 0)
            },
            'next_session': {
                'name': next_config.name if next_config else 'Unknown',
                'type': next_session.value if next_session else 'unknown',
                'in': str(time_until).split('.')[0]  # Remove microseconds
            },
            'adaptive_parameters': self.adaptive_parameters.get(session_type, {})
        }
    
    def should_stop_trading(self, daily_losses: int, consecutive_losses: int) -> bool:
        """
        Determine if trading should stop based on session rules
        """
        session_type, config = self.get_current_session()
        
        if session_type in [SessionType.CLOSED, SessionType.WEEKEND, SessionType.HOLIDAY]:
            return True
        
        # Check max trades per session
        trades_today = len([t for t in self.trade_history 
                           if t.get('date', datetime.now()).date() == datetime.now().date()])
        
        if trades_today >= config.max_trades_per_session:
            self.logger.info(f"Max trades ({config.max_trades_per_session}) reached for {config.name}")
            return True
        
        # Check consecutive losses
        max_consecutive = self.config['risk_management'].get('max_consecutive_losses', 3)
        if consecutive_losses >= max_consecutive:
            self.logger.info(f"Max consecutive losses ({max_consecutive}) reached")
            return True
        
        # Check daily loss limit
        daily_loss_limit = self.config['risk_management'].get('max_daily_loss', 3)  # percent
        if daily_losses > 0:
            loss_percent = (abs(daily_losses) / 10000) * 100  # Simplified
            if loss_percent >= daily_loss_limit:
                self.logger.info(f"Daily loss limit ({daily_loss_limit}%) reached")
                return True
        
        return False
    
    async def shutdown(self):
        """Clean shutdown of session engine"""
        self.logger.info("Shutting down Session Engine...")
        
        # Log final session statistics
        summary = self.get_session_summary()
        self.logger.info(f"Final Session Summary: {json.dumps(summary, indent=2)}")
        
        self.logger.info("Session Engine shutdown complete")