"""
Order Manager - Production-Ready Order Lifecycle Management
Handles placement, modification, cancellation with institutional-grade safety
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable
from contextlib import asynccontextmanager
import copy

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = auto()
    SUBMITTING = auto()
    OPEN = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLING = auto()
    CANCELLED = auto()
    REJECTED = auto()
    ERROR = auto()
    EXPIRED = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


@dataclass
class Order:
    """Immutable order state"""
    id: str
    symbol: str
    direction: str  # 'long' or 'short'
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    external_id: Optional[str] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    remaining_quantity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    
    @property
    def is_active(self) -> bool:
        return self.status in [
            OrderStatus.PENDING, OrderStatus.SUBMITTING, 
            OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED
        ]
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    def with_status(self, status: OrderStatus, **kwargs) -> 'Order':
        """Create new Order with updated status (immutable)"""
        data = {
            'status': status,
            'updated_at': datetime.now(),
            **kwargs
        }
        return Order(**{**self.__dict__, **data})


@dataclass 
class Position:
    """Track filled position state separately from orders"""
    order_id: str
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    breakeven_set: bool = False
    partial_closed: bool = False
    trailing_active: bool = False
    highest_price: float = 0.0
    lowest_price: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManagerConfig:
    """Configuration for order management behavior"""
    def __init__(self):
        self.max_retries = 3
        self.retry_delay_base = 1.0  # seconds
        self.order_timeout = 30.0  # seconds
        self.max_slippage_pct = 0.001  # 0.1% max slippage
        self.partial_fill_threshold = 0.95  # 95% filled = considered complete
        self.breakeven_trigger_rr = 1.0
        self.partial_close_trigger_rr = 2.0
        self.trailing_trigger_rr = 3.0
        self.trailing_distance_atr_multiplier = 1.0
        self.enable_breakeven = True
        self.enable_partial_close = True
        self.enable_trailing_stop = True


class OrderManager:
    """
    Production-ready order manager with:
    - Async safety with locks
    - State machine enforcement
    - Automatic retry logic
    - Position lifecycle management
    - Comprehensive logging
    """
    
    def __init__(self, config: Optional[OrderManagerConfig] = None, 
                 execution_engine=None):
        self.config = config or OrderManagerConfig()
        self.execution_engine = execution_engine
        
        # Thread-safe state management
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}  # order_id -> Position
        self._symbol_orders: Dict[str, Set[str]] = {}  # symbol -> order_ids
        self._lock = asyncio.Lock()
        
        # Background tasks
        self._monitor_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Callbacks
        self.on_order_fill: Optional[Callable[[Order], None]] = None
        self.on_position_update: Optional[Callable[[Position], None]] = None
        
        logger.info("OrderManager initialized")
    
    async def shutdown(self):
        """Graceful shutdown - cancel all pending orders"""
        logger.info("Shutting down OrderManager...")
        self._shutdown_event.set()
        
        # Cancel all active orders
        async with self._lock:
            active = [oid for oid, o in self._orders.items() if o.is_active]
        
        tasks = [self.cancel_order(oid) for oid in active]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cancel monitor tasks
        for task in self._monitor_tasks:
            task.cancel()
        
        if self._monitor_tasks:
            await asyncio.gather(*self._monitor_tasks, return_exceptions=True)
        
        logger.info("OrderManager shutdown complete")
    
    # ==================== ORDER PLACEMENT ====================
    
    async def place_order(self, signal: Dict) -> Optional[Order]:
        """
        Place new order with validation, retry logic, and state tracking
        
        Args:
            signal: Dict with keys: symbol, direction, position_size, 
                   entry_price, stop_loss, take_profit, order_type
        
        Returns:
            Order object or None if failed
        """
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return None
            
            # Check for existing identical order (idempotency)
            async with self._lock:
                existing = self._find_similar_order(signal)
                if existing:
                    logger.warning(f"Similar order already exists: {existing.id}")
                    return existing
            
            # Create order object
            order = Order(
                id=str(uuid.uuid4())[:12],
                symbol=signal['symbol'],
                direction=signal['direction'],
                order_type=OrderType[signal.get('order_type', 'MARKET').upper()],
                quantity=signal['position_size'],
                price=signal.get('entry_price'),
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit'),
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                remaining_quantity=signal['position_size']
            )
            
            # Store order
            async with self._lock:
                self._orders[order.id] = order
                self._symbol_orders.setdefault(order.symbol, set()).add(order.id)
            
            # Execute with retry logic
            filled_order = await self._execute_with_retry(order)
            
            if filled_order and filled_order.is_filled:
                # Create position tracking
                await self._initialize_position(filled_order)
                
                if self.on_order_fill:
                    try:
                        self.on_order_fill(filled_order)
                    except Exception as e:
                        logger.error(f"Order fill callback error: {e}")
            
            return filled_order
            
        except Exception as e:
            logger.error(f"Critical error placing order: {e}", exc_info=True)
            return None
    
    def _validate_signal(self, signal: Dict) -> bool:
        """Validate order signal parameters"""
        required = ['symbol', 'direction', 'position_size']
        for field in required:
            if field not in signal or signal[field] is None:
                logger.error(f"Missing required field: {field}")
                return False
        
        if signal['direction'] not in ['long', 'short']:
            logger.error(f"Invalid direction: {signal['direction']}")
            return False
        
        if signal['position_size'] <= 0:
            logger.error(f"Invalid position size: {signal['position_size']}")
            return False
        
        # Validate price for limit orders
        if signal.get('order_type', 'MARKET').upper() == 'LIMIT':
            if 'entry_price' not in signal or signal['entry_price'] <= 0:
                logger.error("Limit order requires valid entry_price")
                return False
        
        return True
    
    def _find_similar_order(self, signal: Dict) -> Optional[Order]:
        """Find existing similar order (idempotency check)"""
        symbol = signal['symbol']
        direction = signal['direction']
        
        for oid in self._symbol_orders.get(symbol, set()):
            order = self._orders.get(oid)
            if not order or not order.is_active:
                continue
            
            # Check if similar (same direction, within 1% price, within 60s)
            if order.direction == direction:
                price_similar = True
                if order.price and signal.get('entry_price'):
                    price_diff = abs(order.price - signal['entry_price']) / order.price
                    price_similar = price_diff < 0.01
                
                time_similar = (datetime.now() - order.created_at) < timedelta(seconds=60)
                
                if price_similar and time_similar:
                    return order
        
        return None
    
    async def _execute_with_retry(self, order: Order) -> Optional[Order]:
        """Execute order with exponential backoff retry"""
        for attempt in range(self.config.max_retries):
            try:
                # Update status
                async with self._lock:
                    order = order.with_status(OrderStatus.SUBMITTING, retry_count=attempt)
                    self._orders[order.id] = order
                
                # Attempt execution
                result = await self._submit_to_exchange(order)
                
                if result.get('success'):
                    # Validate fill
                    fill_price = result.get('fill_price', order.price)
                    if not self._validate_fill(order, fill_price):
                        raise ValueError(f"Fill price {fill_price} exceeds slippage tolerance")
                    
                    # Update to filled
                    filled_order = order.with_status(
                        OrderStatus.FILLED,
                        external_id=result.get('order_id'),
                        filled_quantity=order.quantity,
                        avg_fill_price=fill_price,
                        remaining_quantity=0.0
                    )
                    
                    async with self._lock:
                        self._orders[order.id] = filled_order
                    
                    logger.info(f"Order {order.id} filled at {fill_price}")
                    return filled_order
                
                elif result.get('retryable', False):
                    raise asyncio.TimeoutError("Exchange timeout")
                else:
                    # Non-retryable error
                    error_order = order.with_status(
                        OrderStatus.REJECTED,
                        error_message=result.get('message', 'Unknown rejection')
                    )
                    async with self._lock:
                        self._orders[order.id] = error_order
                    logger.error(f"Order {order.id} rejected: {error_order.error_message}")
                    return error_order
                    
            except Exception as e:
                logger.warning(f"Order {order.id} attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_base * (2 ** attempt)
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    # Final failure
                    failed_order = order.with_status(
                        OrderStatus.ERROR,
                        error_message=str(e)
                    )
                    async with self._lock:
                        self._orders[order.id] = failed_order
                    return failed_order
        
        return None
    
    async def _submit_to_exchange(self, order: Order) -> Dict:
        """Submit order to appropriate exchange"""
        if not self.execution_engine:
            raise RuntimeError("No execution engine configured")
        
        # Route to correct exchange
        if order.symbol.endswith(('USDT', 'BTC', 'ETH')):
            return await self.execution_engine.execute_binance_trade(
                symbol=order.symbol,
                direction=order.direction,
                position_size=order.quantity,
                entry_price=order.price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                order_type=order.order_type.name
            )
        else:
            return await self.execution_engine.execute_mt5_trade(
                symbol=order.symbol,
                direction=order.direction,
                position_size=order.quantity,
                entry_price=order.price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                order_type=order.order_type.name
            )
    
    def _validate_fill(self, order: Order, fill_price: float) -> bool:
        """Validate fill price against slippage tolerance"""
        if not order.price or order.order_type == OrderType.MARKET:
            return True  # Market orders accept any fill
        
        slippage = abs(fill_price - order.price) / order.price
        return slippage <= self.config.max_slippage_pct
    
    async def _initialize_position(self, order: Order):
        """Initialize position tracking for filled order"""
        position = Position(
            order_id=order.id,
            symbol=order.symbol,
            direction=order.direction,
            entry_price=order.avg_fill_price,
            quantity=order.filled_quantity,
            stop_loss=order.stop_loss or 0.0,
            take_profit=order.take_profit or 0.0,
            opened_at=datetime.now(),
            current_price=order.avg_fill_price,
            highest_price=order.avg_fill_price,
            lowest_price=order.avg_fill_price
        )
        
        async with self._lock:
            self._positions[order.id] = position
        
        logger.info(f"Position initialized for order {order.id}")
    
    # ==================== ORDER MODIFICATION ====================
    
    async def modify_order(self, order_id: str, updates: Dict) -> bool:
        """
        Modify existing order with state validation
        
        Supports: stop_loss, take_profit, price (for pending limits)
        """
        async with self._lock:
            if order_id not in self._orders:
                logger.error(f"Order {order_id} not found")
                return False
            
            order = self._orders[order_id]
            
            # Validate state
            if not order.is_active:
                logger.error(f"Cannot modify order {order_id} with status {order.status}")
                return False
            
            # For filled orders, modify position instead
            if order.is_filled:
                return await self._modify_position(order_id, updates)
            
            # For pending orders, amend the order
            return await self._amend_pending_order(order, updates)
    
    async def _amend_pending_order(self, order: Order, updates: Dict) -> bool:
        """Amend pending limit/stop order"""
        try:
            # Validate updates
            new_price = updates.get('price', order.price)
            new_sl = updates.get('stop_loss', order.stop_loss)
            new_tp = updates.get('take_profit', order.take_profit)
            
            # Call exchange
            result = await self.execution_engine.amend_order(
                symbol=order.symbol,
                order_id=order.external_id,
                price=new_price,
                stop_loss=new_sl,
                take_profit=new_tp
            )
            
            if result.get('success'):
                updated_order = order.with_status(
                    OrderStatus.OPEN,  # Remains open
                    price=new_price,
                    stop_loss=new_sl,
                    take_profit=new_tp
                )
                
                async with self._lock:
                    self._orders[order.id] = updated_order
                
                logger.info(f"Order {order.id} amended")
                return True
            else:
                logger.error(f"Failed to amend order {order.id}: {result.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"Error amending order {order.id}: {e}")
            return False
    
    async def _modify_position(self, order_id: str, updates: Dict) -> bool:
        """Modify filled position (stop loss / take profit)"""
        try:
            async with self._lock:
                if order_id not in self._positions:
                    return False
                position = self._positions[order_id]
            
            # Update stop loss
            if 'stop_loss' in updates:
                result = await self.execution_engine.modify_position(
                    symbol=position.symbol,
                    position_id=position.order_id,  # Or external position ID
                    stop_loss=updates['stop_loss']
                )
                
                if result.get('success'):
                    position.stop_loss = updates['stop_loss']
                    if 'breakeven_set' in updates:
                        position.breakeven_set = updates['breakeven_set']
                    logger.info(f"Position {order_id} SL updated to {updates['stop_loss']}")
                else:
                    return False
            
            # Update take profit
            if 'take_profit' in updates:
                result = await self.execution_engine.modify_position(
                    symbol=position.symbol,
                    position_id=position.order_id,
                    take_profit=updates['take_profit']
                )
                
                if result.get('success'):
                    position.take_profit = updates['take_profit']
                    logger.info(f"Position {order_id} TP updated to {updates['take_profit']}")
                else:
                    return False
            
            async with self._lock:
                self._positions[order_id] = position
            
            return True
            
        except Exception as e:
            logger.error(f"Error modifying position {order_id}: {e}")
            return False
    
    # ==================== ORDER CANCELLATION ====================
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        async with self._lock:
            if order_id not in self._orders:
                return False
            
            order = self._orders[order_id]
            
            if not order.is_active:
                logger.info(f"Order {order_id} already inactive ({order.status})")
                return True  # Already done
            
            if order.is_filled:
                logger.error(f"Cannot cancel filled order {order_id}")
                return False
            
            # Update status to prevent race conditions
            cancelling_order = order.with_status(OrderStatus.CANCELLING)
            self._orders[order_id] = cancelling_order
        
        try:
            result = await self.execution_engine.cancel_order(
                symbol=order.symbol,
                order_id=order.external_id
            )
            
            async with self._lock:
                if result.get('success'):
                    cancelled_order = cancelling_order.with_status(
                        OrderStatus.CANCELLED,
                        cancelled_at=datetime.now()
                    )
                    self._orders[order_id] = cancelled_order
                    
                    # Cleanup
                    self._symbol_orders.get(order.symbol, set()).discard(order_id)
                    
                    logger.info(f"Order {order_id} cancelled")
                    return True
                else:
                    # Revert status
                    self._orders[order_id] = order
                    logger.error(f"Failed to cancel order {order_id}: {result.get('message')}")
                    return False
                    
        except Exception as e:
            async with self._lock:
                self._orders[order_id] = order
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    # ==================== POSITION MANAGEMENT ====================
    
    async def update_position_price(self, order_id: str, current_price: float):
        """Update position with current market price and check exits"""
        async with self._lock:
            if order_id not in self._positions:
                return
            
            position = self._positions[order_id]
            position.current_price = current_price
            
            # Update high/low water marks
            if current_price > position.highest_price:
                position.highest_price = current_price
            if current_price < position.lowest_price:
                position.lowest_price = current_price
            
            # Calculate unrealized PnL
            if position.direction == 'long':
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
        
        # Check exit conditions (outside lock to avoid blocking)
        await self._check_exit_conditions(order_id, position)
        
        # Callback
        if self.on_position_update:
            try:
                self.on_position_update(position)
            except Exception as e:
                logger.error(f"Position update callback error: {e}")
    
    async def _check_exit_conditions(self, order_id: str, position: Position):
        """Check and execute exit conditions"""
        try:
            risk = abs(position.entry_price - position.stop_loss)
            if risk == 0:
                return
            
            profit = position.unrealized_pnl
            rr = profit / (risk * position.quantity) if risk > 0 else 0
            
            # Breakeven at 1R
            if (self.config.enable_breakeven and 
                not position.breakeven_set and 
                rr >= self.config.breakeven_trigger_rr):
                
                success = await self._modify_position(order_id, {
                    'stop_loss': position.entry_price,
                    'breakeven_set': True
                })
                
                if success:
                    async with self._lock:
                        self._positions[order_id].breakeven_set = True
                    logger.info(f"Position {order_id} moved to breakeven at 1R")
            
            # Partial close at 2R
            if (self.config.enable_partial_close and 
                not position.partial_closed and 
                rr >= self.config.partial_close_trigger_rr):
                
                await self._execute_partial_close(order_id, position, 50)
            
            # Trailing stop at 3R
            if (self.config.enable_trailing_stop and 
                not position.trailing_active and 
                rr >= self.config.trailing_trigger_rr):
                
                new_stop = self._calculate_trailing_stop(position)
                success = await self._modify_position(order_id, {
                    'stop_loss': new_stop,
                    'trailing_active': True
                })
                
                if success:
                    async with self._lock:
                        self._positions[order_id].trailing_active = True
                    logger.info(f"Position {order_id} trailing stop activated at 3R")
            
            # Update trailing stop if active
            if position.trailing_active:
                await self._update_trailing_stop(order_id, position)
                
        except Exception as e:
            logger.error(f"Error checking exits for {order_id}: {e}")
    
    async def _execute_partial_close(self, order_id: str, position: Position, percent: int):
        """Execute partial position close"""
        try:
            close_quantity = position.quantity * (percent / 100)
            
            result = await self.execution_engine.close_position_partial(
                symbol=position.symbol,
                position_id=order_id,
                quantity=close_quantity
            )
            
            if result.get('success'):
                async with self._lock:
                    position.quantity -= close_quantity
                    position.partial_closed = True
                    self._positions[order_id] = position
                
                logger.info(f"Position {order_id} partially closed ({percent}%)")
            else:
                logger.error(f"Partial close failed for {order_id}: {result.get('message')}")
                
        except Exception as e:
            logger.error(f"Error partial closing {order_id}: {e}")
    
    def _calculate_trailing_stop(self, position: Position) -> float:
        """Calculate trailing stop price based on ATR or percentage"""
        # Use 1x original risk distance for trailing
        trail_distance = abs(position.entry_price - position.stop_loss)
        
        if position.direction == 'long':
            return position.highest_price - trail_distance
        else:
            return position.lowest_price + trail_distance
    
    async def _update_trailing_stop(self, order_id: str, position: Position):
        """Update trailing stop if price moved favorably"""
        new_stop = self._calculate_trailing_stop(position)
        
        # Only move stop in favorable direction
        if position.direction == 'long' and new_stop > position.stop_loss:
            await self._modify_position(order_id, {'stop_loss': new_stop})
        elif position.direction == 'short' and new_stop < position.stop_loss:
            await self._modify_position(order_id, {'stop_loss': new_stop})
    
    async def close_position(self, order_id: str, reason: str = "manual") -> bool:
        """Fully close position"""
        async with self._lock:
            if order_id not in self._positions:
                return False
            position = self._positions[order_id]
        
        try:
            result = await self.execution_engine.close_position(
                symbol=position.symbol,
                position_id=order_id
            )
            
            if result.get('success'):
                async with self._lock:
                    del self._positions[order_id]
                
                # Update order status
                if order_id in self._orders:
                    closed_order = self._orders[order_id].with_status(
                        OrderStatus.CANCELLED,
                        metadata={'close_reason': reason, 'closed_at': datetime.now()}
                    )
                    self._orders[order_id] = closed_order
                
                logger.info(f"Position {order_id} closed: {reason}")
                return True
            else:
                logger.error(f"Failed to close position {order_id}: {result.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position {order_id}: {e}")
            return False
    
    # ==================== QUERIES ====================
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        async with self._lock:
            return self._orders.get(order_id)
    
    async def get_position(self, order_id: str) -> Optional[Position]:
        """Get position by order ID"""
        async with self._lock:
            return self._positions.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active orders, optionally filtered by symbol"""
        orders = []
        async with self._lock:
            if symbol:
                ids = self._symbol_orders.get(symbol, set())
                orders = [self._orders[oid] for oid in ids 
                         if oid in self._orders and self._orders[oid].is_active]
            else:
                orders = [o for o in self._orders.values() if o.is_active]
        return orders
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all positions"""
        async with self._lock:
            positions = list(self._positions.values())
            if symbol:
                positions = [p for p in positions if p.symbol == symbol]
        return positions
    
    def get_order_history(self, symbol: Optional[str] = None, 
                         limit: int = 100) -> List[Order]:
        """Get historical orders"""
        async with self._lock:
            orders = list(self._orders.values())
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            orders.sort(key=lambda x: x.created_at, reverse=True)
        return orders[:limit]
    
    # ==================== MONITORING ====================
    
    async def start_position_monitor(self, interval: float = 1.0):
        """Start background task to monitor positions"""
        task = asyncio.create_task(self._position_monitor_loop(interval))
        self._monitor_tasks.add(task)
        task.add_done_callback(self._monitor_tasks.discard)
    
    async def _position_monitor_loop(self, interval: float):
        """Background loop to update positions"""
        while not self._shutdown_event.is_set():
            try:
                positions = self.get_positions()
                for position in positions:
                    # Get current price from execution engine
                    price = await self.execution_engine.get_current_price(
                        position.symbol
                    )
                    if price:
                        await self.update_position_price(position.order_id, price)
                
                await asyncio.wait_for(
                    self._shutdown_event.wait(), 
                    timeout=interval
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(interval)
    
    def _monitor_tasks(self) -> Set[asyncio.Task]:
        return self._monitor_tasks


# ==================== USAGE EXAMPLE ====================

async def example_usage():
    """Example of using OrderManager"""
    
    # Mock execution engine
    class MockExecutionEngine:
        async def execute_binance_trade(self, **kwargs):
            await asyncio.sleep(0.1)
            return {
                'success': True,
                'order_id': 'binance_123',
                'fill_price': kwargs.get('entry_price', 50000)
            }
        
        async def execute_mt5_trade(self, **kwargs):
            return {'success': True, 'order_id': 'mt5_456', 'fill_price': 2000}
        
        async def modify_position(self, **kwargs):
            return {'success': True}
        
        async def cancel_order(self, **kwargs):
            return {'success': True}
        
        async def get_current_price(self, symbol):
            return 2000.0
    
    # Initialize
    config = OrderManagerConfig()
    config.enable_partial_close = True
    config.enable_trailing_stop = True
    
    engine = MockExecutionEngine()
    manager = OrderManager(config, engine)
    
    # Set callbacks
    def on_fill(order):
        print(f"Order filled: {order.id} at {order.avg_fill_price}")
    
    def on_position_update(pos):
        print(f"Position update: {pos.symbol} PnL: {pos.unrealized_pnl:.2f}")
    
    manager.on_order_fill = on_fill
    manager.on_position_update = on_position_update
    
    # Start monitoring
    await manager.start_position_monitor(interval=5.0)
    
    # Place order
    signal = {
        'symbol': 'XAUUSD',
        'direction': 'long',
        'position_size': 0.5,
        'entry_price': 2000.0,
        'stop_loss': 1990.0,
        'take_profit': 2030.0,
        'order_type': 'MARKET'
    }
    
    order = await manager.place_order(signal)
    print(f"Placed order: {order}")
    
    # Simulate price updates
    for price in [2005, 2010, 2020, 2035]:
        await asyncio.sleep(1)
        if order:
            await manager.update_position_price(order.id, float(price))
            pos = await manager.get_position(order.id)
            if pos:
                print(f"Price: {price}, RR: {(price - pos.entry_price) / (pos.entry_price - pos.stop_loss):.2f}")
    
    # Cleanup
    await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())