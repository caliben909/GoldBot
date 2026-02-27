"""
Position Tracker - Production-Grade Position & Risk Management
Real-time P&L, exposure monitoring, and risk analytics
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Set, Callable
from enum import Enum, auto
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPEN = auto()
    PARTIALLY_CLOSED = auto()
    CLOSING = auto()
    CLOSED = auto()
    ERROR = auto()


@dataclass
class Position:
    """Immutable position state for thread safety"""
    order_id: str
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    quantity: float
    filled_quantity: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    open_time: datetime
    status: PositionStatus
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    swap_costs: float = 0.0
    breakeven_set: bool = False
    partial_closed: bool = False
    trailing_active: bool = False
    highest_price: float = 0.0
    lowest_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_long(self) -> bool:
        return self.direction == 'long'
    
    @property
    def market_value(self) -> float:
        """Position market value (unsigned)"""
        return self.current_price * self.filled_quantity
    
    @property
    def notional_exposure(self) -> float:
        """Signed notional exposure (+ for long, - for short)"""
        return self.market_value if self.is_long else -self.market_value
    
    @property
    def initial_risk(self) -> float:
        """Initial risk amount in account currency"""
        if self.stop_loss is None or self.stop_loss == 0:
            return 0.0
        risk_per_unit = abs(self.entry_price - self.stop_loss)
        return risk_per_unit * self.filled_quantity
    
    @property
    def r_multiple(self) -> float:
        """Current R-multiple (profit relative to initial risk)"""
        if self.initial_risk == 0:
            return 0.0
        return self.unrealized_pnl / self.initial_risk
    
    def update_price(self, new_price: float, timestamp: Optional[datetime] = None) -> 'Position':
        """Create new Position with updated price (immutable)"""
        # Update high/low water marks
        new_high = max(self.highest_price, new_price) if self.highest_price > 0 else new_price
        new_low = min(self.lowest_price, new_price) if self.lowest_price > 0 else new_price
        
        # Calculate new unrealized P&L
        if self.is_long:
            new_pnl = (new_price - self.entry_price) * self.filled_quantity
        else:
            new_pnl = (self.entry_price - new_price) * self.filled_quantity
        
        return Position(
            order_id=self.order_id,
            symbol=self.symbol,
            direction=self.direction,
            entry_price=self.entry_price,
            current_price=new_price,
            quantity=self.quantity,
            filled_quantity=self.filled_quantity,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            open_time=self.open_time,
            status=self.status,
            unrealized_pnl=new_pnl,
            realized_pnl=self.realized_pnl,
            total_commission=self.total_commission,
            swap_costs=self.swap_costs,
            breakeven_set=self.breakeven_set,
            partial_closed=self.partial_partial_closed,
            trailing_active=self.trailing_active,
            highest_price=new_high,
            lowest_price=new_low,
            last_update=timestamp or datetime.now(),
            metadata=self.metadata
        )
    
    def close(self, exit_price: float, exit_time: datetime, 
              closed_quantity: Optional[float] = None) -> 'Position':
        """Close or partially close position"""
        close_qty = closed_quantity or self.filled_quantity
        
        if self.is_long:
            close_pnl = (exit_price - self.entry_price) * close_qty
        else:
            close_pnl = (self.entry_price - exit_price) * close_qty
        
        new_realized = self.realized_pnl + close_pnl
        remaining = self.filled_quantity - close_qty
        
        new_status = PositionStatus.CLOSED if remaining <= 0 else PositionStatus.PARTIALLY_CLOSED
        
        return Position(
            order_id=self.order_id,
            symbol=self.symbol,
            direction=self.direction,
            entry_price=self.entry_price,
            current_price=exit_price if remaining <= 0 else self.current_price,
            quantity=self.quantity,
            filled_quantity=remaining,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            open_time=self.open_time,
            status=new_status,
            unrealized_pnl=0.0 if remaining <= 0 else self.unrealized_pnl * (remaining / self.filled_quantity),
            realized_pnl=new_realized,
            total_commission=self.total_commission,
            swap_costs=self.swap_costs,
            breakeven_set=self.breakeven_set,
            partial_closed=remaining > 0,
            trailing_active=self.trailing_active and remaining > 0,
            highest_price=self.highest_price,
            lowest_price=self.lowest_price,
            last_update=exit_time,
            metadata={**self.metadata, 'exit_price': exit_price, 'exit_time': exit_time}
        )


@dataclass
class PortfolioSnapshot:
    """Real-time portfolio state"""
    timestamp: datetime
    total_exposure_long: float
    total_exposure_short: float
    net_exposure: float
    gross_exposure: float
    total_unrealized_pnl: float
    total_realized_pnl_day: float
    open_positions_count: int
    margin_used: float
    margin_available: float
    portfolio_heat: float  # Max loss as % of equity
    concentration_risk: float  # Largest position % of portfolio


class TimePeriodPnL:
    """Track P&L for specific time period"""
    def __init__(self):
        self.realized_pnl: float = 0.0
        self.commissions: float = 0.0
        self.swap_costs: float = 0.0
        self.trade_count: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0
        self.last_reset: datetime = datetime.now()
    
    def add_trade(self, realized_pnl: float, commission: float = 0.0, swap: float = 0.0):
        """Record a closed trade"""
        self.realized_pnl += realized_pnl
        self.commissions += commission
        self.swap_costs += swap
        self.trade_count += 1
        
        if realized_pnl > 0:
            self.win_count += 1
        elif realized_pnl < 0:
            self.loss_count += 1
    
    def reset(self):
        """Reset period stats"""
        self.__init__()
    
    @property
    def net_pnl(self) -> float:
        return self.realized_pnl - self.commissions - self.swap_costs
    
    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count > 0 else 0.0


class PositionTrackerConfig:
    """Configuration for position tracking"""
    def __init__(self):
        self.max_position_history = 10000  # Limit memory usage
        self.price_stale_seconds = 30  # Max age for price updates
        self.heat_warning_threshold = 0.05  # 5% portfolio heat warning
        self.heat_critical_threshold = 0.10  # 10% critical
        self.max_concentration_pct = 0.20  # 20% max single position
        self.enable_hedging = True  # Allow long/short same symbol
        self.margin_requirement_pct = 0.02  # 2% margin per position


class PositionTracker:
    """
    Production-grade position tracker with:
    - Thread-safe concurrent access
    - Accurate time-based P&L tracking
    - Risk monitoring and alerts
    - Immutable position state
    """
    
    def __init__(self, config: Optional[PositionTrackerConfig] = None):
        self.config = config or PositionTrackerConfig()
        
        # Thread-safe state
        self._positions: Dict[str, Position] = {}  # order_id -> Position
        self._symbol_positions: Dict[str, Set[str]] = {}  # symbol -> order_ids
        self._lock = asyncio.Lock()
        
        # P&L tracking by time period
        self._daily_pnl = TimePeriodPnL()
        self._weekly_pnl = TimePeriodPnL()
        self._monthly_pnl = TimePeriodPnL()
        self._last_date_check = date.today()
        
        # Position history (limited size)
        self._position_history: List[Dict] = []
        
        # Risk monitoring
        self._equity: float = 0.0
        self._margin_used: float = 0.0
        
        # Callbacks
        self.on_heat_warning: Optional[Callable[[float], None]] = None
        self.on_heat_critical: Optional[Callable[[float], None]] = None
        self.on_concentration_warning: Optional[Callable[[str, float], None]] = None
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("PositionTracker initialized")
    
    async def start(self):
        """Start background monitoring"""
        self._monitor_task = asyncio.create_task(self._periodic_checks())
        logger.info("PositionTracker monitoring started")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down PositionTracker...")
        self._shutdown_event.set()
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("PositionTracker shutdown complete")
    
    # ==================== POSITION LIFECYCLE ====================
    
    async def add_position(self, order: Dict) -> Optional[Position]:
        """
        Add new position from filled order
        
        Args:
            order: Dict with keys: id, symbol, direction, fill_price, 
                   quantity, stop_loss, take_profit, commission
        """
        try:
            # Validate
            required = ['id', 'symbol', 'direction', 'fill_price', 'quantity']
            if not all(k in order for k in required):
                logger.error(f"Missing required fields in order: {order}")
                return None
            
            # Check for existing
            async with self._lock:
                if order['id'] in self._positions:
                    logger.warning(f"Position {order['id']} already exists")
                    return self._positions[order['id']]
                
                # Check hedging policy
                if not self.config.enable_hedging:
                    existing = self._get_symbol_position(order['symbol'])
                    if existing and existing.direction != order['direction']:
                        logger.error(f"Hedging not enabled, conflicting position exists")
                        return None
            
            # Create position
            position = Position(
                order_id=order['id'],
                symbol=order['symbol'],
                direction=order['direction'],
                entry_price=order['fill_price'],
                current_price=order['fill_price'],
                quantity=order['quantity'],
                filled_quantity=order['quantity'],  # Assume fully filled for now
                stop_loss=order.get('stop_loss'),
                take_profit=order.get('take_profit'),
                open_time=datetime.now(),
                status=PositionStatus.OPEN,
                highest_price=order['fill_price'],
                lowest_price=order['fill_price'],
                total_commission=order.get('commission', 0.0)
            )
            
            # Store
            async with self._lock:
                self._positions[order['id']] = position
                self._symbol_positions.setdefault(order['symbol'], set()).add(order['id'])
                self._update_margin(position)
            
            logger.info(f"Position added: {position.symbol} {position.direction} "
                       f"@ {position.entry_price} x {position.filled_quantity}")
            
            # Check concentration risk immediately
            await self._check_concentration_risk(position.symbol)
            
            return position
            
        except Exception as e:
            logger.error(f"Error adding position: {e}", exc_info=True)
            return None
    
    async def update_price(self, order_id: str, current_price: float, 
                          timestamp: Optional[datetime] = None) -> Optional[Position]:
        """
        Update position with current market price
        
        Returns updated position or None if not found
        """
        try:
            # Validate price recency
            if timestamp and (datetime.now() - timestamp).seconds > self.config.price_stale_seconds:
                logger.warning(f"Stale price for {order_id}: {timestamp}")
            
            async with self._lock:
                if order_id not in self._positions:
                    return None
                
                old_position = self._positions[order_id]
                
                # Create updated position (immutable)
                new_position = old_position.update_price(current_price, timestamp)
                self._positions[order_id] = new_position
                
                # Check risk thresholds
                await self._check_portfolio_heat()
                
                return new_position
                
        except Exception as e:
            logger.error(f"Error updating price for {order_id}: {e}")
            return None
    
    async def close_position(self, order_id: str, exit_price: float, 
                            exit_reason: str, 
                            closed_quantity: Optional[float] = None,
                            commission: float = 0.0,
                            swap_costs: float = 0.0) -> Optional[Position]:
        """
        Close or partially close position
        
        Args:
            closed_quantity: None for full close, or specific quantity for partial
        """
        try:
            async with self._lock:
                if order_id not in self._positions:
                    logger.error(f"Position {order_id} not found for closing")
                    return None
                
                old_position = self._positions[order_id]
                
                # Calculate realized P&L for this close
                close_qty = closed_quantity or old_position.filled_quantity
                if old_position.is_long:
                    trade_pnl = (exit_price - old_position.entry_price) * close_qty
                else:
                    trade_pnl = (old_position.entry_price - exit_price) * close_qty
                
                # Update position
                new_position = old_position.close(
                    exit_price, datetime.now(), closed_quantity
                )
                
                # Update P&L tracking (REALIZED only)
                self._daily_pnl.add_trade(trade_pnl, commission, swap_costs)
                self._weekly_pnl.add_trade(trade_pnl, commission, swap_costs)
                self._monthly_pnl.add_trade(trade_pnl, commission, swap_costs)
                
                # Update state
                if new_position.status == PositionStatus.CLOSED:
                    # Move to history
                    history_entry = {
                        'position': new_position,
                        'exit_reason': exit_reason,
                        'closed_at': datetime.now()
                    }
                    self._position_history.append(history_entry)
                    
                    # Limit history size
                    if len(self._position_history) > self.config.max_position_history:
                        self._position_history.pop(0)
                    
                    # Remove from active
                    del self._positions[order_id]
                    self._symbol_positions.get(new_position.symbol, set()).discard(order_id)
                    self._release_margin(old_position)
                    
                    logger.info(f"Position {order_id} closed: {exit_reason} P&L: ${trade_pnl:.2f}")
                else:
                    # Partial close - update position
                    self._positions[order_id] = new_position
                    logger.info(f"Position {order_id} partially closed: {close_qty} lots "
                               f"P&L: ${trade_pnl:.2f}")
                
                return new_position
                
        except Exception as e:
            logger.error(f"Error closing position {order_id}: {e}", exc_info=True)
            return None
    
    async def modify_position(self, order_id: str, 
                             stop_loss: Optional[float] = None,
                             take_profit: Optional[float] = None,
                             breakeven_set: Optional[bool] = None,
                             trailing_active: Optional[bool] = None) -> bool:
        """Modify position parameters"""
        try:
            async with self._lock:
                if order_id not in self._positions:
                    return False
                
                position = self._positions[order_id]
                
                # Create modified position
                updates = {}
                if stop_loss is not None:
                    updates['stop_loss'] = stop_loss
                if take_profit is not None:
                    updates['take_profit'] = take_profit
                if breakeven_set is not None:
                    updates['breakeven_set'] = breakeven_set
                if trailing_active is not None:
                    updates['trailing_active'] = trailing_active
                
                # Apply updates (create new instance)
                new_position = Position(
                    **{**position.__dict__, **updates, 'last_update': datetime.now()}
                )
                
                self._positions[order_id] = new_position
                logger.info(f"Position {order_id} modified: {updates}")
                return True
                
        except Exception as e:
            logger.error(f"Error modifying position {order_id}: {e}")
            return False
    
    # ==================== QUERIES ====================
    
    async def get_position(self, order_id: str) -> Optional[Position]:
        """Get position by order ID"""
        async with self._lock:
            return self._positions.get(order_id)
    
    async def get_symbol_position(self, symbol: str, direction: Optional[str] = None) -> Optional[Position]:
        """Get position for symbol (optionally filtered by direction)"""
        async with self._lock:
            ids = self._symbol_positions.get(symbol, set())
            for oid in ids:
                pos = self._positions.get(oid)
                if pos and pos.status == PositionStatus.OPEN:
                    if direction is None or pos.direction == direction:
                        return pos
            return None
    
    async def get_all_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all open positions"""
        async with self._lock:
            if symbol:
                ids = self._symbol_positions.get(symbol, set())
                return [self._positions[oid] for oid in ids 
                       if oid in self._positions]
            else:
                return list(self._positions.values())
    
    async def has_position(self, symbol: str, direction: Optional[str] = None) -> bool:
        """Check if position exists"""
        pos = await self.get_symbol_position(symbol, direction)
        return pos is not None
    
    async def get_position_count(self) -> int:
        """Get number of open positions"""
        async with self._lock:
            return len(self._positions)
    
    # ==================== RISK & EXPOSURE ====================
    
    async def get_portfolio_snapshot(self, current_equity: float) -> PortfolioSnapshot:
        """Get comprehensive portfolio risk snapshot"""
        async with self._lock:
            positions = list(self._positions.values())
            
            if not positions:
                return PortfolioSnapshot(
                    timestamp=datetime.now(),
                    total_exposure_long=0.0,
                    total_exposure_short=0.0,
                    net_exposure=0.0,
                    gross_exposure=0.0,
                    total_unrealized_pnl=0.0,
                    total_realized_pnl_day=self._daily_pnl.net_pnl,
                    open_positions_count=0,
                    margin_used=self._margin_used,
                    margin_available=current_equity - self._margin_used,
                    portfolio_heat=0.0,
                    concentration_risk=0.0
                )
            
            # Calculate exposures
            long_exposure = sum(p.market_value for p in positions if p.is_long)
            short_exposure = sum(p.market_value for p in positions if not p.is_long)
            net_exposure = long_exposure - short_exposure
            gross_exposure = long_exposure + short_exposure
            
            # Calculate heat (worst case loss based on stops)
            total_heat = 0.0
            for p in positions:
                if p.stop_loss:
                    risk = abs(p.current_price - p.stop_loss) * p.filled_quantity
                    total_heat += risk
            
            portfolio_heat = total_heat / current_equity if current_equity > 0 else 0.0
            
            # Concentration risk (largest position %)
            max_position = max(p.market_value for p in positions)
            concentration = max_position / gross_exposure if gross_exposure > 0 else 0.0
            
            total_unrealized = sum(p.unrealized_pnl for p in positions)
            
            return PortfolioSnapshot(
                timestamp=datetime.now(),
                total_exposure_long=long_exposure,
                total_exposure_short=short_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                total_unrealized_pnl=total_unrealized,
                total_realized_pnl_day=self._daily_pnl.net_pnl,
                open_positions_count=len(positions),
                margin_used=self._margin_used,
                margin_available=current_equity - self._margin_used,
                portfolio_heat=portfolio_heat,
                concentration_risk=concentration
            )
    
    async def get_position_risk_metrics(self, order_id: str) -> Optional[Dict]:
        """Get detailed risk metrics for position"""
        async with self._lock:
            if order_id not in self._positions:
                return None
            
            p = self._positions[order_id]
            
            # Calculate distance to stop/target
            if p.is_long:
                distance_to_sl = ((p.current_price - p.stop_loss) / p.current_price * 100) if p.stop_loss else None
                distance_to_tp = ((p.take_profit - p.current_price) / p.current_price * 100) if p.take_profit else None
            else:
                distance_to_sl = ((p.stop_loss - p.current_price) / p.current_price * 100) if p.stop_loss else None
                distance_to_tp = ((p.current_price - p.take_profit) / p.current_price * 100) if p.take_profit else None
            
            # Max favorable/adverse excursion
            mfe = (p.highest_price - p.entry_price) / p.entry_price * 100 if p.is_long else (p.entry_price - p.lowest_price) / p.entry_price * 100
            mae = (p.entry_price - p.lowest_price) / p.entry_price * 100 if p.is_long else (p.highest_price - p.entry_price) / p.entry_price * 100
            
            return {
                'r_multiple': p.r_multiple,
                'distance_to_stop_pct': distance_to_sl,
                'distance_to_target_pct': distance_to_tp,
                'max_favorable_excursion_pct': mfe,
                'max_adverse_excursion_pct': mae,
                'time_in_trade_minutes': (datetime.now() - p.open_time).total_seconds() / 60,
                'current_drawdown_from_peak_pct': ((p.highest_price - p.current_price) / p.highest_price * 100) if p.is_long else ((p.current_price - p.lowest_price) / p.lowest_price * 100)
            }
    
    # ==================== P&L REPORTING ====================
    
    async def get_pnl_summary(self) -> Dict:
        """Get P&L summary by time period"""
        await self._check_time_resets()
        
        return {
            'daily': {
                'realized_pnl': self._daily_pnl.realized_pnl,
                'net_pnl': self._daily_pnl.net_pnl,
                'trades': self._daily_pnl.trade_count,
                'win_rate': self._daily_pnl.win_rate,
                'wins': self._daily_pnl.win_count,
                'losses': self._daily_pnl.loss_count
            },
            'weekly': {
                'realized_pnl': self._weekly_pnl.realized_pnl,
                'net_pnl': self._weekly_pnl.net_pnl,
                'trades': self._weekly_pnl.trade_count,
                'win_rate': self._weekly_pnl.win_rate
            },
            'monthly': {
                'realized_pnl': self._monthly_pnl.realized_pnl,
                'net_pnl': self._monthly_pnl.net_pnl,
                'trades': self._monthly_pnl.trade_count,
                'win_rate': self._monthly_pnl.win_rate
            },
            'unrealized': sum(p.unrealized_pnl for p in self._positions.values()),
            'total_exposure': await self.get_total_exposure()
        }
    
    async def reset_daily_pnl(self):
        """Manually reset daily P&L"""
        async with self._lock:
            self._daily_pnl.reset()
            logger.info("Daily P&L reset")
    
    # ==================== INTERNAL METHODS ====================
    
    def _update_margin(self, position: Position):
        """Calculate margin used by position"""
        margin = position.market_value * self.config.margin_requirement_pct
        self._margin_used += margin
    
    def _release_margin(self, position: Position):
        """Release margin when position closes"""
        margin = position.market_value * self.config.margin_requirement_pct
        self._margin_used = max(0, self._margin_used - margin)
    
    async def _check_portfolio_heat(self):
        """Check if portfolio heat exceeds thresholds"""
        if self._equity <= 0:
            return
        
        total_heat = 0.0
        for p in self._positions.values():
            if p.stop_loss:
                total_heat += abs(p.current_price - p.stop_loss) * p.filled_quantity
        
        heat_pct = total_heat / self._equity
        
        if heat_pct > self.config.heat_critical_threshold:
            logger.critical(f"CRITICAL PORTFOLIO HEAT: {heat_pct:.2%}")
            if self.on_heat_critical:
                asyncio.create_task(self._safe_callback(self.on_heat_critical, heat_pct))
        elif heat_pct > self.config.heat_warning_threshold:
            logger.warning(f"High portfolio heat: {heat_pct:.2%}")
            if self.on_heat_warning:
                asyncio.create_task(self._safe_callback(self.on_heat_warning, heat_pct))
    
    async def _check_concentration_risk(self, symbol: str):
        """Check if symbol concentration exceeds limit"""
        async with self._lock:
            ids = self._symbol_positions.get(symbol, set())
            symbol_exposure = sum(
                self._positions[oid].market_value 
                for oid in ids if oid in self._positions
            )
            
            total_exposure = sum(p.market_value for p in self._positions.values())
            
            if total_exposure > 0:
                concentration = symbol_exposure / total_exposure
                if concentration > self.config.max_concentration_pct:
                    logger.warning(f"Concentration risk for {symbol}: {concentration:.1%}")
                    if self.on_concentration_warning:
                        asyncio.create_task(
                            self._safe_callback(self.on_concentration_warning, symbol, concentration)
                        )
    
    async def _safe_callback(self, callback, *args):
        """Safely execute callback without blocking"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    async def _check_time_resets(self):
        """Auto-reset P&L periods on time boundaries"""
        today = date.today()
        
        if today != self._last_date_check:
            # New day - reset daily
            self._daily_pnl.reset()
            
            # Check for new week (Monday)
            if today.weekday() == 0:  # Monday
                self._weekly_pnl.reset()
            
            # Check for new month
            if today.day == 1:
                self._monthly_pnl.reset()
            
            self._last_date_check = today
            logger.info(f"PnL periods reset for {today}")
    
    async def _periodic_checks(self):
        """Background task for periodic maintenance"""
        while not self._shutdown_event.is_set():
            try:
                await self._check_time_resets()
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=60)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Periodic check error: {e}")
                await asyncio.sleep(60)
    
    async def get_total_exposure(self) -> Dict[str, float]:
        """Get detailed exposure breakdown"""
        async with self._lock:
            positions = list(self._positions.values())
            
            long_exposure = sum(p.market_value for p in positions if p.is_long)
            short_exposure = sum(p.market_value for p in positions if not p.is_long)
            
            return {
                'long': long_exposure,
                'short': short_exposure,
                'net': long_exposure - short_exposure,
                'gross': long_exposure + short_exposure,
                'count': len(positions)
            }


# ==================== USAGE EXAMPLE ====================

async def example_usage():
    """Example of production usage"""
    config = PositionTrackerConfig()
    config.heat_warning_threshold = 0.03
    config.max_concentration_pct = 0.15
    
    tracker = PositionTracker(config)
    
    # Set callbacks
    def on_heat_warning(heat):
        print(f"⚠️  Portfolio heat warning: {heat:.2%}")
    
    def on_heat_critical(heat):
        print(f"🚨 CRITICAL HEAT: {heat:.2%} - Consider reducing positions!")
    
    tracker.on_heat_warning = on_heat_warning
    tracker.on_heat_critical = on_heat_critical
    
    await tracker.start()
    
    # Add positions
    order1 = {
        'id': 'ord_001',
        'symbol': 'XAUUSD',
        'direction': 'long',
        'fill_price': 2000.0,
        'quantity': 1.0,
        'stop_loss': 1990.0,
        'take_profit': 2030.0,
        'commission': 7.0
    }
    
    pos1 = await tracker.add_position(order1)
    print(f"Added position: {pos1}")
    
    # Update prices
    for price in [2005.0, 2010.0, 2008.0]:
        updated = await tracker.update_price('ord_001', price)
        print(f"Price {price}: Unrealized P&L: ${updated.unrealized_pnl:.2f}, "
              f"R-Multiple: {updated.r_multiple:.2f}R")
        
        # Get risk metrics
        risk = await tracker.get_position_risk_metrics('ord_001')
        print(f"  Distance to SL: {risk['distance_to_stop_pct']:.2f}%")
    
    # Get portfolio snapshot
    snapshot = await tracker.get_portfolio_snapshot(current_equity=10000.0)
    print(f"\nPortfolio Heat: {snapshot.portfolio_heat:.2%}")
    print(f"Concentration Risk: {snapshot.concentration_risk:.2%}")
    
    # Close position
    closed = await tracker.close_position('ord_001', exit_price=2015.0, 
                                         exit_reason='take_profit',
                                         commission=7.0)
    print(f"\nClosed with realized P&L: ${closed.realized_pnl:.2f}")
    
    # Check P&L summary
    pnl = await tracker.get_pnl_summary()
    print(f"\nDaily P&L: ${pnl['daily']['net_pnl']:.2f} "
          f"(Win Rate: {pnl['daily']['win_rate']:.0%})")
    
    await tracker.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())