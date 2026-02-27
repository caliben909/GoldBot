"""
Liquidity Engine - Complete SMC Implementation
Includes: Market Structure, Order Blocks, Liquidity Sweeps, FVG, Mitigation, Contrarian Flow
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)

class StructureType(Enum):
    BOS = "break_of_structure"
    CHOCH = "change_of_character"
    LIQUIDITY = "liquidity_sweep"
    FVG = "fair_value_gap"
    ORDER_BLOCK = "order_block"
    MITIGATION = "mitigation"
    IMBALANCE = "imbalance"
    DISPLACEMENT = "displacement"
    CONTRARIAN_BOS = "contrarian_bos"
    CONTRARIAN_CHOCH = "contrarian_choch"

class OrderBlockType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    MITIGATED = "mitigated"
    ACTIVE = "active"
    REJECTION = "rejection"
    ABSORPTION = "absorption"

class LiquidityType(Enum):
    BUY_SIDE = "buy_side"
    SELL_SIDE = "sell_side"
    ASIAN = "asian_range"
    PREVIOUS_DAY = "previous_day"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"

@dataclass
class OrderBlock:
    """Institutional Order Block"""
    type: OrderBlockType
    direction: str  # bullish, bearish
    price_range: Tuple[float, float]
    timestamp: pd.Timestamp
    strength: float
    volume_profile: float
    mitigation_price: Optional[float] = None
    mitigated: bool = False
    mitigation_time: Optional[pd.Timestamp] = None
    order_flow_imbalance: float = 0.0
    absorption_ratio: float = 0.0
    rejection_wicks: float = 0.0
    
@dataclass
class FairValueGap:
    """Fair Value Gap (Imbalance)"""
    type: str  # bullish, bearish
    top: float
    bottom: float
    midpoint: float
    timestamp: pd.Timestamp
    size_pips: float
    volume_confirmation: bool
    mitigated: bool = False
    mitigation_time: Optional[pd.Timestamp] = None
    ote_zone_62: Optional[float] = None  # Optimal Trade Entry 62%
    ote_zone_79: Optional[float] = None  # Optimal Trade Entry 79%

@dataclass
class LiquidityZone:
    """Liquidity Zone"""
    type: LiquidityType
    price_level: float
    zone_range: Tuple[float, float]
    strength: float
    touches: int
    sweeps: int
    is_swept: bool = False
    sweep_time: Optional[pd.Timestamp] = None
    volume_at_sweep: float = 0.0
    
@dataclass
class StructurePoint:
    """Market Structure Point"""
    type: StructureType
    price: float
    timestamp: pd.Timestamp
    strength: float
    direction: str  # bullish, bearish
    volume: float
    confirmation: bool
    mitigated: bool = False
    mitigation_price: Optional[float] = None

@dataclass
class ContrarianSignal:
    """Contrarian Trading Signal"""
    type: str  # liquidity_grab, fvg_rejection, order_block_mitigation
    direction: str  # counter to trend
    confidence: float
    entry_zone: Tuple[float, float]
    stop_loss: float
    take_profit: float
    risk_reward: float
    reason: str
    structure_confluence: List[str]

class LiquidityEngine:
    """
    Complete SMC Engine with Contrarian and Order Block Flow
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.swing_length = config['strategy']['smc']['swing_length']
        self.fvg_min_size = config['strategy']['smc']['fvg_min_size']
        self.liquidity_lookback = config['strategy']['smc']['liquidity_lookback']
        self.order_block_lookback = config['strategy']['smc']['order_block_lookback']
        
        # Contrarian settings
        self.contrarian_enabled = True
        self.min_contrarian_confluence = 3
        self.max_contrarian_risk = 0.5  # Half of normal risk
        
        # Order flow tracking
        self.active_order_blocks = deque(maxlen=100)
        self.mitigated_order_blocks = deque(maxlen=100)
        self.fvg_zones = deque(maxlen=100)
        self.liquidity_zones = deque(maxlen=100)
        
    def analyze_market(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete market analysis including all SMC concepts
        """
        # Reset internal state for each analysis to prevent carryover from previous bars
        self.active_order_blocks.clear()
        self.mitigated_order_blocks.clear()
        self.fvg_zones.clear()
        self.liquidity_zones.clear()
        
        result = {
            'structure': self._analyze_structure(df),
            'order_blocks': self._analyze_order_blocks(df),
            'fvg_zones': self._detect_fvg(df),
            'liquidity': self._analyze_liquidity(df),
            'contrarian': self._analyze_contrarian(df),
            'order_flow': self._analyze_order_flow(df),
            'mitigation': self._check_mitigation(df),
            'confluence_zones': []
        }
        
        # Find confluence between different concepts
        result['confluence_zones'] = self._find_confluence_zones(result)
        
        # Update active zones
        self._update_active_zones(result)
        
        return result
        
    def _analyze_structure(self, df: pd.DataFrame) -> Dict[str, List]:
        """Analyze market structure including BOS and CHOCH"""
        swings = self._find_swing_points(df)
        
        return {
            'swing_highs': swings['highs'],
            'swing_lows': swings['lows'],
            'bos': self._detect_bos(df, swings),
            'choch': self._detect_choch(df, swings),
            'current_trend': self._determine_trend(swings),
            'structure_strength': self._calculate_structure_strength(swings)
        }
        
    def _analyze_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Advanced Order Block Detection with Volume Profile
        """
        order_blocks = []
        
        for i in range(self.order_block_lookback, len(df) - 1):
            # Bullish Order Block (last bearish candle before rally)
            if self._is_bullish_order_block(df, i):
                ob = self._create_order_block(df, i, 'bullish')
                order_blocks.append(ob)
                self.active_order_blocks.append(ob)
                
            # Bearish Order Block (last bullish candle before dump)
            if self._is_bearish_order_block(df, i):
                ob = self._create_order_block(df, i, 'bearish')
                order_blocks.append(ob)
                self.active_order_blocks.append(ob)
                
        return order_blocks
        
    def _is_bullish_order_block(self, df: pd.DataFrame, index: int) -> bool:
        """Detect bullish order block"""
        if index < 3 or index >= len(df) - 3:
            return False
            
        # Criteria for bullish OB:
        # 1. Current candle is bearish (down)
        # 2. Next 2-3 candles are bullish (up)
        # 3. Volume increases on the move
        # 4. Price breaks structure
        
        current = df.iloc[index]
        next1 = df.iloc[index + 1]
        next2 = df.iloc[index + 2]
        next3 = df.iloc[index + 3] if index + 3 < len(df) else None
        
        # Current bearish
        if current['close'] >= current['open']:
            return False
            
        # Next candles bullish
        bullish_count = 0
        for candle in [next1, next2, next3]:
            if candle is not None and candle['close'] > candle['open']:
                bullish_count += 1
                
        if bullish_count < 2:
            return False
            
        # Volume confirmation
        avg_volume = df['volume'].iloc[max(0, index-10):index].mean()
        volume_surge = next1['volume'] > avg_volume * 1.5
        
        # Structure break
        structure_break = next1['high'] > df['high'].iloc[max(0, index-5):index].max()
        
        return volume_surge or structure_break
        
    def _is_bearish_order_block(self, df: pd.DataFrame, index: int) -> bool:
        """Detect bearish order block"""
        if index < 3 or index >= len(df) - 3:
            return False
            
        current = df.iloc[index]
        next1 = df.iloc[index + 1]
        next2 = df.iloc[index + 2]
        next3 = df.iloc[index + 3] if index + 3 < len(df) else None
        
        # Current bullish
        if current['close'] <= current['open']:
            return False
            
        # Next candles bearish
        bearish_count = 0
        for candle in [next1, next2, next3]:
            if candle is not None and candle['close'] < candle['open']:
                bearish_count += 1
                
        if bearish_count < 2:
            return False
            
        # Volume confirmation
        avg_volume = df['volume'].iloc[max(0, index-10):index].mean()
        volume_surge = next1['volume'] > avg_volume * 1.5
        
        # Structure break
        structure_break = next1['low'] < df['low'].iloc[max(0, index-5):index].min()
        
        return volume_surge or structure_break
        
    def _create_order_block(self, df: pd.DataFrame, index: int, direction: str) -> OrderBlock:
        """Create Order Block object with detailed metrics"""
        candle = df.iloc[index]
        
        # Calculate order block range
        if direction == 'bullish':
            price_range = (candle['low'], candle['high'])
            mitigation_price = candle['high']  # Price to mitigate
        else:
            price_range = (candle['low'], candle['high'])
            mitigation_price = candle['low']  # Price to mitigate
            
        # Calculate order flow imbalance
        volume_profile = candle['volume']
        prev_volume = df['volume'].iloc[max(0, index-5):index].mean()
        absorption_ratio = volume_profile / prev_volume if prev_volume > 0 else 1
        
        # Calculate rejection wicks
        if direction == 'bullish':
            rejection_wicks = (candle['high'] - candle['close']) / (candle['high'] - candle['low']) if candle['high'] > candle['low'] else 0
        else:
            rejection_wicks = (candle['open'] - candle['low']) / (candle['high'] - candle['low']) if candle['high'] > candle['low'] else 0
            
        return OrderBlock(
            type=OrderBlockType.ACTIVE,
            direction=direction,
            price_range=price_range,
            timestamp=candle.name,
            strength=absorption_ratio,
            volume_profile=volume_profile,
            mitigation_price=mitigation_price,
            mitigated=False,
            order_flow_imbalance=absorption_ratio - 1,
            absorption_ratio=absorption_ratio,
            rejection_wicks=rejection_wicks
        )
        
    def _detect_fvg(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps (Imbalances) with OTE zones
        """
        fvg_zones = []
        
        for i in range(2, len(df) - 2):
            # Bullish FVG: current low > previous high
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                
                if gap_size >= self.fvg_min_size:
                    fvg = FairValueGap(
                        type='bullish',
                        top=df['low'].iloc[i],
                        bottom=df['high'].iloc[i-2],
                        midpoint=(df['low'].iloc[i] + df['high'].iloc[i-2]) / 2,
                        timestamp=df.index[i],
                        size_pips=gap_size * 10000 if 'JPY' not in str(df.index) else gap_size * 100,
                        volume_confirmation=df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean(),
                        mitigated=False
                    )
                    
                    # Calculate OTE zones
                    fvg.ote_zone_62 = fvg.bottom + (gap_size * 0.62)
                    fvg.ote_zone_79 = fvg.bottom + (gap_size * 0.79)
                    
                    fvg_zones.append(fvg)
                    self.fvg_zones.append(fvg)
                    
            # Bearish FVG: current high < previous low
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                
                if gap_size >= self.fvg_min_size:
                    fvg = FairValueGap(
                        type='bearish',
                        top=df['low'].iloc[i-2],
                        bottom=df['high'].iloc[i],
                        midpoint=(df['low'].iloc[i-2] + df['high'].iloc[i]) / 2,
                        timestamp=df.index[i],
                        size_pips=gap_size * 10000 if 'JPY' not in str(df.index) else gap_size * 100,
                        volume_confirmation=df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean(),
                        mitigated=False
                    )
                    
                    # Calculate OTE zones
                    fvg.ote_zone_62 = fvg.top - (gap_size * 0.62)
                    fvg.ote_zone_79 = fvg.top - (gap_size * 0.79)
                    
                    fvg_zones.append(fvg)
                    self.fvg_zones.append(fvg)
                    
        return fvg_zones
        
    def _analyze_liquidity(self, df: pd.DataFrame) -> Dict[str, List[LiquidityZone]]:
        """
        Comprehensive liquidity analysis
        """
        liquidity = {
            'buy_side': [],  # Above price (liquidity to take out)
            'sell_side': [],  # Below price (liquidity to take out)
            'asian_range': None,
            'previous_day': None,
            'equal_highs': [],
            'equal_lows': []
        }
        
        # Find swing points
        swings = self._find_swing_points(df)
        
        # Buy-side liquidity (swing highs)
        for high in swings['highs'][-10:]:
            zone = LiquidityZone(
                type=LiquidityType.BUY_SIDE,
                price_level=high['price'],
                zone_range=(high['price'] * 0.999, high['price'] * 1.001),
                strength=high['strength'],
                touches=self._count_touches(df, high['price']),
                sweeps=0
            )
            liquidity['buy_side'].append(zone)
            self.liquidity_zones.append(zone)
            
        # Sell-side liquidity (swing lows)
        for low in swings['lows'][-10:]:
            zone = LiquidityZone(
                type=LiquidityType.SELL_SIDE,
                price_level=low['price'],
                zone_range=(low['price'] * 0.999, low['price'] * 1.001),
                strength=low['strength'],
                touches=self._count_touches(df, low['price']),
                sweeps=0
            )
            liquidity['sell_side'].append(zone)
            self.liquidity_zones.append(zone)
            
        # Detect equal highs (double tops)
        liquidity['equal_highs'] = self._detect_equal_levels(swings['highs'], 'high')
        
        # Detect equal lows (double bottoms)
        liquidity['equal_lows'] = self._detect_equal_levels(swings['lows'], 'low')
        
        return liquidity
        
    def _analyze_contrarian(self, df: pd.DataFrame) -> List[ContrarianSignal]:
        """
        Contrarian Strategy Detection
        Looks for:
        1. Liquidity grabs with rejection
        2. FVG fills with reversal patterns
        3. Order block mitigations with structure break
        """
        contrarian_signals = []
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # 1. Check for liquidity grabs that reversed
        for zone in self.liquidity_zones:
            if not zone.is_swept:
                continue
                
            # Check if price swept and reversed
            swept_candle = df[df.index == zone.sweep_time]
            if len(swept_candle) == 0:
                continue
                
            # Look for reversal pattern after sweep
            if self._is_reversal_after_sweep(df, zone, current_price):
                signal = ContrarianSignal(
                    type='liquidity_grab',
                    direction='bullish' if zone.type == LiquidityType.SELL_SIDE else 'bearish',
                    confidence=0.85,
                    entry_zone=self._calculate_entry_zone(df, zone),
                    stop_loss=self._calculate_contrarian_sl(df, zone),
                    take_profit=self._calculate_contrarian_tp(df, zone),
                    risk_reward=2.0,
                    reason=f"Liquidity grab reversal at {zone.type.value}",
                    structure_confluence=['liquidity_sweep', 'rejection']
                )
                contrarian_signals.append(signal)
                
        # 2. Check for FVG fills with reversal
        for fvg in self.fvg_zones:
            if fvg.mitigated:
                continue
                
            # Check if price entered FVG
            if fvg.bottom <= current_price <= fvg.top:
                # Look for reversal within FVG
                if self._is_reversal_in_fvg(df, fvg):
                    signal = ContrarianSignal(
                        type='fvg_rejection',
                        direction='bearish' if fvg.type == 'bullish' else 'bullish',
                        confidence=0.80,
                        entry_zone=(fvg.ote_zone_62, fvg.ote_zone_79),
                        stop_loss=self._calculate_fvg_sl(df, fvg),
                        take_profit=self._calculate_fvg_tp(df, fvg),
                        risk_reward=2.5,
                        reason=f"FVG rejection at {fvg.type} gap",
                        structure_confluence=['fvg', 'ote_zone']
                    )
                    contrarian_signals.append(signal)
                    
        # 3. Check for order block mitigations
        for ob in self.active_order_blocks:
            if ob.mitigated:
                continue
                
            # Check if price mitigated order block
            if ob.direction == 'bullish' and current_price <= ob.mitigation_price:
                if self._is_bounce_from_ob(df, ob):
                    signal = ContrarianSignal(
                        type='order_block_mitigation',
                        direction='bullish',
                        confidence=0.90,
                        entry_zone=ob.price_range,
                        stop_loss=ob.price_range[0] * 0.998,
                        take_profit=self._calculate_ob_tp(df, ob),
                        risk_reward=3.0,
                        reason="Order block mitigation bounce",
                        structure_confluence=['order_block', 'support']
                    )
                    contrarian_signals.append(signal)
                    
            elif ob.direction == 'bearish' and current_price >= ob.mitigation_price:
                if self._is_rejection_from_ob(df, ob):
                    signal = ContrarianSignal(
                        type='order_block_mitigation',
                        direction='bearish',
                        confidence=0.90,
                        entry_zone=ob.price_range,
                        stop_loss=ob.price_range[1] * 1.002,
                        take_profit=self._calculate_ob_tp(df, ob),
                        risk_reward=3.0,
                        reason="Order block mitigation rejection",
                        structure_confluence=['order_block', 'resistance']
                    )
                    contrarian_signals.append(signal)
                    
        return contrarian_signals
        
    def _analyze_order_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Order Flow Analysis
        Detects absorption, exhaustion, and institutional activity
        """
        result = {
            'buying_pressure': [],
            'selling_pressure': [],
            'absorption_zones': [],
            'exhaustion_signals': [],
            'institutional_footprints': []
        }
        
        # Calculate delta (buying vs selling pressure)
        for i in range(10, len(df)):
            delta = self._calculate_delta(df, i)
            
            if delta > 1.5:  # Strong buying pressure
                result['buying_pressure'].append({
                    'time': df.index[i],
                    'price': df['close'].iloc[i],
                    'delta': delta
                })
            elif delta < 0.67:  # Strong selling pressure
                result['selling_pressure'].append({
                    'time': df.index[i],
                    'price': df['close'].iloc[i],
                    'delta': delta
                })
                
            # Detect absorption (price not moving despite volume)
            if self._is_absorption(df, i):
                result['absorption_zones'].append({
                    'time': df.index[i],
                    'price': df['close'].iloc[i],
                    'volume': df['volume'].iloc[i]
                })
                
        return result
        
    def _check_mitigation(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        Check for mitigation of all SMC concepts
        """
        mitigated = {
            'order_blocks': [],
            'fvg_zones': [],
            'liquidity_zones': []
        }
        
        current_price = df['close'].iloc[-1]
        
        # Check order block mitigation
        for ob in self.active_order_blocks:
            if not ob.mitigated:
                if ob.direction == 'bullish' and current_price <= ob.mitigation_price:
                    ob.mitigated = True
                    ob.mitigation_time = df.index[-1]
                    mitigated['order_blocks'].append(ob)
                elif ob.direction == 'bearish' and current_price >= ob.mitigation_price:
                    ob.mitigated = True
                    ob.mitigation_time = df.index[-1]
                    mitigated['order_blocks'].append(ob)
                    
        # Check FVG mitigation
        for fvg in self.fvg_zones:
            if not fvg.mitigated:
                if fvg.bottom <= current_price <= fvg.top:
                    fvg.mitigated = True
                    fvg.mitigation_time = df.index[-1]
                    mitigated['fvg_zones'].append(fvg)
                    
        # Check liquidity sweeps
        for zone in self.liquidity_zones:
            if not zone.is_swept:
                if zone.zone_range[0] <= current_price <= zone.zone_range[1]:
                    # Check if price swept through
                    recent_df = df.tail(5)
                    if zone.type in [LiquidityType.BUY_SIDE, LiquidityType.EQUAL_HIGHS]:
                        if recent_df['high'].max() >= zone.price_level:
                            zone.is_swept = True
                            zone.sweep_time = recent_df[recent_df['high'] >= zone.price_level].index[0]
                            zone.volume_at_sweep = recent_df['volume'].iloc[-1]
                            mitigated['liquidity_zones'].append(zone)
                    else:
                        if recent_df['low'].min() <= zone.price_level:
                            zone.is_swept = True
                            zone.sweep_time = recent_df[recent_df['low'] <= zone.price_level].index[0]
                            zone.volume_at_sweep = recent_df['volume'].iloc[-1]
                            mitigated['liquidity_zones'].append(zone)
                            
        return mitigated
        
    def _find_confluence_zones(self, analysis: Dict) -> List[Dict]:
        """
        Find areas where multiple SMC concepts align
        """
        confluence_zones = []
        
        # Collect all significant levels
        levels = []
        
        # Add order blocks
        for ob in analysis['order_blocks']:
            if not ob.mitigated:
                levels.append({
                    'price': sum(ob.price_range) / 2,
                    'type': f"OB_{ob.direction}",
                    'strength': ob.strength,
                    'range': ob.price_range
                })
                
        # Add FVG zones
        for fvg in analysis['fvg_zones']:
            if not fvg.mitigated:
                levels.append({
                    'price': fvg.midpoint,
                    'type': f"FVG_{fvg.type}",
                    'strength': fvg.size_pips / 10,
                    'range': (fvg.bottom, fvg.top)
                })
                
        # Add liquidity zones
        for liq in analysis['liquidity']['buy_side'] + analysis['liquidity']['sell_side']:
            if not liq.is_swept:
                levels.append({
                    'price': liq.price_level,
                    'type': f"LIQ_{liq.type.value}",
                    'strength': liq.strength,
                    'range': liq.zone_range
                })
                
        # Find overlapping zones
        for i in range(len(levels)):
            for j in range(i+1, len(levels)):
                # Check if ranges overlap
                range1 = levels[i]['range']
                range2 = levels[j]['range']
                
                if max(range1[0], range2[0]) <= min(range1[1], range2[1]):
                    # Overlap found
                    confluence_zones.append({
                        'zone': (max(range1[0], range2[0]), min(range1[1], range2[1])),
                        'midpoint': (max(range1[0], range2[0]) + min(range1[1], range2[1])) / 2,
                        'confluence_count': 2,
                        'types': [levels[i]['type'], levels[j]['type']],
                        'strength': (levels[i]['strength'] + levels[j]['strength']) / 2
                    })
                    
        return confluence_zones
        
    def _update_active_zones(self, analysis: Dict):
        """Update internal tracking of active zones"""
        # Keep only unmitigated zones
        self.active_order_blocks = deque(
            [ob for ob in self.active_order_blocks if not ob.mitigated],
            maxlen=100
        )
        self.fvg_zones = deque(
            [fvg for fvg in self.fvg_zones if not fvg.mitigated],
            maxlen=100
        )
        
    def _find_swing_points(self, df: pd.DataFrame) -> Dict[str, List]:
        """Find swing highs and lows"""
        highs = []
        lows = []
        
        high_values = df['high'].values
        low_values = df['low'].values
        index = df.index.values
        
        length = self.swing_length
        
        for i in range(length, len(df) - length):
            # Swing high
            if all(high_values[i] > high_values[i-j] for j in range(1, length+1)) and \
               all(high_values[i] > high_values[i+j] for j in range(1, length+1)):
                highs.append({
                    'price': high_values[i],
                    'index': i,
                    'time': index[i],
                    'strength': self._calculate_swing_strength(df, i, 'high')
                })
                
            # Swing low
            if all(low_values[i] < low_values[i-j] for j in range(1, length+1)) and \
               all(low_values[i] < low_values[i+j] for j in range(1, length+1)):
                lows.append({
                    'price': low_values[i],
                    'index': i,
                    'time': index[i],
                    'strength': self._calculate_swing_strength(df, i, 'low')
                })
                
        return {'highs': highs, 'lows': lows}
        
    def _calculate_swing_strength(self, df: pd.DataFrame, index: int, swing_type: str) -> float:
        """Calculate strength of swing point"""
        if swing_type == 'high':
            price = df['high'].iloc[index]
            surrounding = df['high'].iloc[max(0, index-5):min(len(df), index+6)]
            avg = surrounding.mean()
            return (price / avg - 1) * 100
        else:
            price = df['low'].iloc[index]
            surrounding = df['low'].iloc[max(0, index-5):min(len(df), index+6)]
            avg = surrounding.mean()
            return (1 - price / avg) * 100
            
    def _detect_bos(self, df: pd.DataFrame, swings: Dict) -> List[Dict]:
        """Detect Break of Structure"""
        bos_points = []
        
        highs = swings['highs']
        lows = swings['lows']
        
        # Bullish BOS
        for i in range(1, len(highs)):
            prev_high = highs[i-1]
            curr_high = highs[i]
            
            mask = (df.index > prev_high['time']) & (df.index < curr_high['time'])
            prices_between = df.loc[mask]
            
            if len(prices_between) > 0 and prices_between['high'].max() > prev_high['price']:
                bos_points.append({
                    'type': 'bullish',
                    'price': prev_high['price'],
                    'time': prices_between[prices_between['high'] > prev_high['price']].index[0],
                    'strength': (curr_high['price'] - prev_high['price']) / prev_high['price'] * 100
                })
                
        # Bearish BOS
        for i in range(1, len(lows)):
            prev_low = lows[i-1]
            curr_low = lows[i]
            
            mask = (df.index > prev_low['time']) & (df.index < curr_low['time'])
            prices_between = df.loc[mask]
            
            if len(prices_between) > 0 and prices_between['low'].min() < prev_low['price']:
                bos_points.append({
                    'type': 'bearish',
                    'price': prev_low['price'],
                    'time': prices_between[prices_between['low'] < prev_low['price']].index[0],
                    'strength': (prev_low['price'] - curr_low['price']) / prev_low['price'] * 100
                })
                
        return bos_points
        
    def _detect_choch(self, df: pd.DataFrame, swings: Dict) -> List[Dict]:
        """Detect Change of Character"""
        choch_points = []
        
        # Look for trend change
        recent_highs = swings['highs'][-5:]
        recent_lows = swings['lows'][-5:]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Check for bearish CHOCH (lower highs, lower lows)
            if recent_highs[-1]['price'] < recent_highs[-2]['price'] and \
               recent_lows[-1]['price'] < recent_lows[-2]['price']:
                choch_points.append({
                    'type': 'bearish',
                    'price': recent_highs[-1]['price'],
                    'time': recent_highs[-1]['time'],
                    'strength': (recent_highs[-2]['price'] - recent_highs[-1]['price']) / recent_highs[-2]['price'] * 100
                })
                
            # Check for bullish CHOCH (higher highs, higher lows)
            if recent_highs[-1]['price'] > recent_highs[-2]['price'] and \
               recent_lows[-1]['price'] > recent_lows[-2]['price']:
                choch_points.append({
                    'type': 'bullish',
                    'price': recent_lows[-1]['price'],
                    'time': recent_lows[-1]['time'],
                    'strength': (recent_lows[-1]['price'] - recent_lows[-2]['price']) / recent_lows[-2]['price'] * 100
                })
                
        return choch_points
        
    def _determine_trend(self, swings: Dict) -> str:
        """Determine current trend from swings"""
        if len(swings['highs']) < 2 or len(swings['lows']) < 2:
            return 'neutral'
            
        recent_highs = [h['price'] for h in swings['highs'][-3:]]
        recent_lows = [l['price'] for l in swings['lows'][-3:]]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
                return 'bullish'
            elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
                return 'bearish'
                
        return 'neutral'
        
    def _calculate_structure_strength(self, swings: Dict) -> float:
        """Calculate overall structure strength"""
        if len(swings['highs']) < 3 or len(swings['lows']) < 3:
            return 0.5
            
        # Calculate average swing strength
        high_strength = np.mean([h['strength'] for h in swings['highs'][-5:]])
        low_strength = np.mean([l['strength'] for l in swings['lows'][-5:]])
        
        return (high_strength + low_strength) / 200  # Normalize to 0-1
        
    def _count_touches(self, df: pd.DataFrame, level: float) -> int:
        """Count how many times price touched a level"""
        touches = 0
        tolerance = level * 0.001  # 0.1% tolerance
        
        for i in range(len(df)):
            if abs(df['high'].iloc[i] - level) <= tolerance or \
               abs(df['low'].iloc[i] - level) <= tolerance:
                touches += 1
                
        return touches
        
    def _detect_equal_levels(self, swings: List, swing_type: str) -> List[Dict]:
        """Detect equal highs or lows"""
        equal_levels = []
        
        for i in range(len(swings)):
            for j in range(i+1, len(swings)):
                price_diff = abs(swings[i]['price'] - swings[j]['price']) / swings[i]['price']
                
                if price_diff < 0.001:  # Within 0.1%
                    equal_levels.append({
                        'price': swings[i]['price'],
                        'first_time': swings[i]['time'],
                        'second_time': swings[j]['time'],
                        'strength': (swings[i]['strength'] + swings[j]['strength']) / 2,
                        'type': swing_type
                    })
                    
        return equal_levels
        
    def _calculate_delta(self, df: pd.DataFrame, index: int) -> float:
        """Calculate buying vs selling pressure delta"""
        if index < 5:
            return 1.0
            
        recent = df.iloc[index-5:index]
        up_volume = recent[recent['close'] > recent['open']]['volume'].sum()
        down_volume = recent[recent['close'] < recent['open']]['volume'].sum()
        
        if down_volume == 0:
            return 2.0
            
        return up_volume / down_volume
        
    def _is_absorption(self, df: pd.DataFrame, index: int) -> bool:
        """Detect absorption (high volume, small price movement)"""
        if index < 10:
            return False
            
        current = df.iloc[index]
        prev_10 = df.iloc[index-10:index]
        
        # High volume
        if current['volume'] < prev_10['volume'].mean() * 1.5:
            return False
            
        # Small price movement
        price_range = (current['high'] - current['low']) / current['low']
        avg_range = (prev_10['high'] - prev_10['low']).mean() / prev_10['close'].mean()
        
        return price_range < avg_range * 0.5
        
    def _is_reversal_after_sweep(self, df: pd.DataFrame, zone: LiquidityZone, current_price: float) -> bool:
        """Check if price reversed after sweeping liquidity"""
        if zone.sweep_time is None:
            return False
            
        # Get candles after sweep
        after_sweep = df[df.index > zone.sweep_time]
        if len(after_sweep) < 3:
            return False
            
        # Check for reversal pattern
        if zone.type in [LiquidityType.SELL_SIDE, LiquidityType.EQUAL_LOWS]:
            # Bullish reversal after sweeping lows
            return after_sweep['close'].iloc[0] < after_sweep['close'].iloc[-1] and \
                   after_sweep['low'].iloc[1:].min() > after_sweep['low'].iloc[0]
        else:
            # Bearish reversal after sweeping highs
            return after_sweep['close'].iloc[0] > after_sweep['close'].iloc[-1] and \
                   after_sweep['high'].iloc[1:].max() < after_sweep['high'].iloc[0]
                   
    def _is_reversal_in_fvg(self, df: pd.DataFrame, fvg: FairValueGap) -> bool:
        """Check for reversal pattern within FVG"""
        # Get candles inside FVG
        in_fvg = df[(df.index >= fvg.timestamp) & 
                    (df['low'] <= fvg.top) & 
                    (df['high'] >= fvg.bottom)]
                    
        if len(in_fvg) < 2:
            return False
            
        # Look for rejection candle
        last_candle = in_fvg.iloc[-1]
        
        if fvg.type == 'bullish':
            # Bullish FVG - look for bearish rejection
            return last_candle['high'] > last_candle['open'] and \
                   last_candle['close'] < last_candle['open'] and \
                   (last_candle['high'] - last_candle['close']) > (last_candle['close'] - last_candle['low']) * 2
        else:
            # Bearish FVG - look for bullish rejection
            return last_candle['low'] < last_candle['open'] and \
                   last_candle['close'] > last_candle['open'] and \
                   (last_candle['close'] - last_candle['low']) > (last_candle['high'] - last_candle['close']) * 2
                   
    def _is_bounce_from_ob(self, df: pd.DataFrame, ob: OrderBlock) -> bool:
        """Check if price bounced from order block"""
        recent = df.tail(3)
        
        if len(recent) < 2:
            return False
            
        # Check for bounce candle
        if ob.direction == 'bullish':
            return recent['low'].iloc[-1] > recent['low'].iloc[-2] and \
                   recent['close'].iloc[-1] > recent['open'].iloc[-1]
        else:
            return recent['high'].iloc[-1] < recent['high'].iloc[-2] and \
                   recent['close'].iloc[-1] < recent['open'].iloc[-1]
                   
    def _is_rejection_from_ob(self, df: pd.DataFrame, ob: OrderBlock) -> bool:
        """Check if price rejected from order block"""
        return self._is_bounce_from_ob(df, ob)  # Same logic
        
    def _calculate_entry_zone(self, df: pd.DataFrame, zone: LiquidityZone) -> Tuple[float, float]:
        """Calculate contrarian entry zone"""
        if zone.type in [LiquidityType.SELL_SIDE, LiquidityType.EQUAL_LOWS]:
            # Buy after sweep
            return (zone.price_level * 0.998, zone.price_level)
        else:
            # Sell after sweep
            return (zone.price_level, zone.price_level * 1.002)
            
    def _calculate_contrarian_sl(self, df: pd.DataFrame, zone: LiquidityZone) -> float:
        """Calculate stop loss for contrarian trade"""
        atr = (df['high'] - df['low']).tail(14).mean()
        
        if zone.type in [LiquidityType.SELL_SIDE, LiquidityType.EQUAL_LOWS]:
            return zone.price_level - atr
        else:
            return zone.price_level + atr
            
    def _calculate_contrarian_tp(self, df: pd.DataFrame, zone: LiquidityZone) -> float:
        """Calculate take profit for contrarian trade"""
        atr = (df['high'] - df['low']).tail(14).mean()
        
        if zone.type in [LiquidityType.SELL_SIDE, LiquidityType.EQUAL_LOWS]:
            return zone.price_level + (atr * 2)
        else:
            return zone.price_level - (atr * 2)
            
    def _calculate_fvg_sl(self, df: pd.DataFrame, fvg: FairValueGap) -> float:
        """Calculate stop loss for FVG trade"""
        if fvg.type == 'bullish':
            return fvg.bottom - (fvg.size_pips / 10000)
        else:
            return fvg.top + (fvg.size_pips / 10000)
            
    def _calculate_fvg_tp(self, df: pd.DataFrame, fvg: FairValueGap) -> float:
        """Calculate take profit for FVG trade"""
        if fvg.type == 'bullish':
            return fvg.top + (fvg.size_pips / 10000 * 1.5)
        else:
            return fvg.bottom - (fvg.size_pips / 10000 * 1.5)
            
    def _calculate_ob_tp(self, df: pd.DataFrame, ob: OrderBlock) -> float:
        """Calculate take profit for order block trade"""
        atr = (df['high'] - df['low']).tail(14).mean()
        
        if ob.direction == 'bullish':
            return ob.price_range[1] + (atr * 2)
        else:
            return ob.price_range[0] - (atr * 2)