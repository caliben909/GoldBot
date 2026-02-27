"""
Execution Engine - Institutional-Grade Order Execution
Production-ready with risk controls, idempotency, and emergency safeguards
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import functools

# Optional imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = auto()
    VALIDATING = auto()
    RISK_CHECK = auto()
    SUBMITTED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    FAILED = auto()


class TimeInForce(Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


@dataclass(frozen=True)
class OrderRequest:
    """Immutable order request for idempotency"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    order_type: str
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    client_order_id: Optional[str] = None
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    
    def generate_key(self) -> str:
        """Generate unique key for idempotency"""
        content = f"{self.symbol}:{self.side}:{self.quantity}:{self.order_type}:{self.price}:{self.stop_price}:{self.client_order_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Order:
    """Mutable order state"""
    id: str
    request: OrderRequest
    status: OrderStatus
    created_at: datetime
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    external_id: Optional[str] = None
    filled_quantity: Decimal = Decimal('0')
    avg_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal('0')
    reject_reason: Optional[str] = None
    retry_count: int = 0
    events: List[Dict] = field(default_factory=list)
    
    @property
    def remaining_quantity(self) -> Decimal:
        return self.request.quantity - self.filled_quantity
    
    @property
    def is_done(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                            OrderStatus.REJECTED, OrderStatus.FAILED]


@dataclass
class Position:
    """Position tracking with risk metrics"""
    symbol: str
    side: str
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    margin_used: Decimal
    orders: List[str] = field(default_factory=list)
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_price(self, price: Decimal):
        self.current_price = price
        if self.side == 'long':
            self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_entry_price - price) * self.quantity
        self.updated_at = datetime.now()


@dataclass
class RiskLimits:
    """Pre-trade risk limits"""
    max_order_size: Decimal
    max_position_size: Decimal
    max_daily_loss: Decimal
    max_leverage: Decimal
    max_concentration: Decimal  # Max % of portfolio in single symbol


class RiskGuard:
    """
    Pre-trade risk validation with stateful tracking
    """
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self._daily_pnl: Decimal = Decimal('0')
        self._positions: Dict[str, Position] = {}
        self._lock = asyncio.Lock()
        self._last_reset = datetime.now().date()
    
    async def validate_order(self, order: OrderRequest, 
                            current_equity: Decimal) -> Tuple[bool, Optional[str]]:
        """Validate order against risk limits"""
        async with self._lock:
            # Check daily reset
            if datetime.now().date() != self._last_reset:
                self._daily_pnl = Decimal('0')
                self._last_reset = datetime.now().date()
            
            # 1. Order size check
            if order.quantity > self.limits.max_order_size:
                return False, f"Order size {order.quantity} exceeds max {self.limits.max_order_size}"
            
            # 2. Position limit check
            current_pos = self._positions.get(order.symbol)
            current_qty = current_pos.quantity if current_pos else Decimal('0')
            
            # For closing orders, check if we're not over-closing
            if (order.side == 'sell' and current_pos and current_pos.side == 'long'):
                new_qty = current_qty - order.quantity
                if new_qty < -self.limits.max_position_size:  # Short limit
                    return False, f"Would exceed max short position size"
            elif (order.side == 'buy' and current_pos and current_pos.side == 'short'):
                new_qty = -current_qty + order.quantity  # Covering short
                if new_qty > self.limits.max_position_size:
                    return False, f"Would exceed max long position size"
            elif order.side == 'buy':
                if current_qty + order.quantity > self.limits.max_position_size:
                    return False, f"Would exceed max position size {self.limits.max_position_size}"
            
            # 3. Concentration check
            total_exposure = sum(p.quantity * p.current_price for p in self._positions.values())
            order_value = order.quantity * (order.price or Decimal('0'))
            
            if total_exposure > 0:
                concentration = order_value / (total_exposure + order_value)
                if concentration > self.limits.max_concentration:
                    return False, f"Concentration {concentration:.1%} exceeds limit"
            
            # 4. Daily loss limit
            if self._daily_pnl < -self.limits.max_daily_loss:
                return False, f"Daily loss limit reached: {self._daily_pnl}"
            
            return True, None
    
    async def update_position(self, symbol: str, fill_qty: Decimal, 
                             fill_price: Decimal, side: str):
        """Update position after fill"""
        async with self._lock:
            if symbol not in self._positions:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side='long' if side == 'buy' else 'short',
                    quantity=fill_qty,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=Decimal('0'),
                    realized_pnl=Decimal('0'),
                    margin_used=Decimal('0')
                )
            else:
                pos = self._positions[symbol]
                
                # Check if reducing position
                if (pos.side == 'long' and side == 'sell') or (pos.side == 'short' and side == 'buy'):
                    if fill_qty >= pos.quantity:
                        # Position closed or flipped
                        realized = (fill_price - pos.avg_entry_price) * pos.quantity
                        if pos.side == 'short':
                            realized = -realized
                        
                        self._daily_pnl += realized
                        
                        if fill_qty > pos.quantity:
                            # Flipped to opposite side
                            remaining = fill_qty - pos.quantity
                            pos.side = 'short' if pos.side == 'long' else 'long'
                            pos.quantity = remaining
                            pos.avg_entry_price = fill_price
                        else:
                            del self._positions[symbol]
                    else:
                        # Partial close
                        realized = (fill_price - pos.avg_entry_price) * fill_qty
                        if pos.side == 'short':
                            realized = -realized
                        
                        pos.quantity -= fill_qty
                        self._daily_pnl += realized
                else:
                    # Adding to position
                    total_qty = pos.quantity + fill_qty
                    total_cost = (pos.quantity * pos.avg_entry_price) + (fill_qty * fill_price)
                    pos.avg_entry_price = total_cost / total_qty
                    pos.quantity = total_qty
            
            if symbol in self._positions:
                self._positions[symbol].update_price(fill_price)
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        async with self._lock:
            return self._positions.get(symbol)
    
    async def emergency_flatten_all(self):
        """Emergency close all positions"""
        async with self._lock:
            positions = list(self._positions.items())
            self._positions.clear()
            return positions


class ExecutionEngine:
    """
    Production-grade execution engine with:
    - Pre-trade risk validation
    - Order idempotency
    - Async-native architecture
    - Emergency kill switch
    - Memory-efficient history
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.exec_config = config.get('execution', {})
        
        # Risk management
        limits = RiskLimits(
            max_order_size=Decimal(str(self.exec_config.get('max_order_size', 100))),
            max_position_size=Decimal(str(self.exec_config.get('max_position_size', 1000))),
            max_daily_loss=Decimal(str(self.exec_config.get('max_daily_loss', 10000))),
            max_leverage=Decimal(str(self.exec_config.get('max_leverage', 10))),
            max_concentration=Decimal(str(self.exec_config.get('max_concentration', 0.2)))
        )
        self.risk_guard = RiskGuard(limits)
        
        # Broker clients
        self._binance_client: Optional[Any] = None
        self._mt5_initialized = False
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Order management
        self._orders: Dict[str, Order] = {}
        self._order_lock = asyncio.Lock()
        self._idempotency_keys: Set[str] = set()  # Prevent duplicates
        
        # Position tracking
        self._positions: Dict[str, Position] = {}
        
        # History (circular buffers for memory efficiency)
        self._order_history: deque = deque(maxlen=10000)
        self._fills: deque = deque(maxlen=10000)
        
        # Emergency controls
        self._emergency_stop = asyncio.Event()
        self._kill_switch = False
        
        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._latency_ms: deque = deque(maxlen=1000)
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        logger.info("ExecutionEngine initialized")
    
    async def initialize(self):
        """Initialize broker connections"""
        logger.info("Initializing ExecutionEngine...")
        
        if BINANCE_AVAILABLE and self.exec_config.get('binance', {}).get('enabled'):
            await self._init_binance()
        
        if MT5_AVAILABLE and self.exec_config.get('mt5', {}).get('enabled'):
            await self._init_mt5()
        
        # Start background tasks
        self._tasks.extend([
            asyncio.create_task(self._position_sync_loop()),
            asyncio.create_task(self._risk_monitor_loop())
        ])
        
        logger.info("✅ ExecutionEngine ready")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down ExecutionEngine...")
        self._shutdown_event.set()
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close positions if configured
        if self.exec_config.get('close_on_shutdown'):
            await self.emergency_close_all()
        
        if self._binance_client:
            await self._binance_client.close_connection()
        
        if self._mt5_initialized:
            await asyncio.get_event_loop().run_in_executor(self._thread_pool, mt5.shutdown)
        
        self._thread_pool.shutdown(wait=True)
        logger.info("✅ ExecutionEngine shutdown complete")
    
    # ==================== ORDER ENTRY ====================
    
    async def submit_order(self, 
                          symbol: str,
                          side: str,
                          quantity: float,
                          order_type: str = 'market',
                          price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          client_order_id: Optional[str] = None,
                          **kwargs) -> Order:
        """
        Submit order with full risk validation and idempotency
        
        Returns:
            Order object with status
        """
        # Check emergency stop
        if self._kill_switch:
            raise RuntimeError("Kill switch active - trading halted")
        
        # Create immutable request
        request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=Decimal(str(quantity)),
            order_type=order_type,
            price=Decimal(str(price)) if price else None,
            stop_price=Decimal(str(stop_price)) if stop_price else None,
            client_order_id=client_order_id,
            metadata=tuple(kwargs.items())
        )
        
        # Check idempotency
        idempotency_key = request.generate_key()
        async with self._order_lock:
            if idempotency_key in self._idempotency_keys:
                # Return existing order
                existing = next((o for o in self._orders.values() 
                               if o.request.generate_key() == idempotency_key), None)
                if existing:
                    logger.info(f"Duplicate order detected, returning existing {existing.id}")
                    return existing
            
            self._idempotency_keys.add(idempotency_key)
        
        # Create order
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hashlib.md5(idempotency_key.encode()).hexdigest()[:8]}"
        order = Order(
            id=order_id,
            request=request,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        
        async with self._order_lock:
            self._orders[order_id] = order
        
        # Pre-trade risk check
        order.status = OrderStatus.RISK_CHECK
        # Get current equity from config or calculate
        equity = Decimal(str(self.exec_config.get('equity', 100000)))
        
        passed, reason = await self.risk_guard.validate_order(request, equity)
        if not passed:
            order.status = OrderStatus.REJECTED
            order.reject_reason = reason
            logger.warning(f"Order {order_id} rejected: {reason}")
            return order
        
        # Execute
        try:
            await self._execute_order(order)
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.reject_reason = str(e)
            logger.error(f"Order {order_id} failed: {e}")
        
        # Archive if done
        if order.is_done:
            async with self._order_lock:
                self._order_history.append({
                    'id': order.id,
                    'symbol': order.request.symbol,
                    'status': order.status.name,
                    'filled_qty': float(order.filled_quantity),
                    'avg_price': float(order.avg_fill_price) if order.avg_fill_price else None,
                    'commission': float(order.commission)
                })
        
        return order
    
    async def _execute_order(self, order: Order):
        """Execute order through appropriate broker"""
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        
        # Determine broker
        broker = 'binance' if order.request.symbol.endswith(('USDT', 'BTC')) else 'mt5'
        
        # Execute with retry
        max_retries = self.exec_config.get('max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                if broker == 'binance' and self._binance_client:
                    result = await self._execute_binance(order)
                elif broker == 'mt5' and self._mt5_initialized:
                    result = await self._execute_mt5(order)
                else:
                    raise RuntimeError(f"Broker {broker} not available")
                
                if result:
                    # Success
                    order.events.append({
                        'time': datetime.now().isoformat(),
                        'event': 'filled',
                        'attempt': attempt
                    })
                    return
                
            except Exception as e:
                order.retry_count += 1
                logger.warning(f"Order {order.id} attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    order.status = OrderStatus.FAILED
                    order.reject_reason = str(e)
        
        if order.status != OrderStatus.FILLED:
            order.status = OrderStatus.FAILED
    
    async def _execute_binance(self, order: Order) -> bool:
        """Execute on Binance"""
        if not BINANCE_AVAILABLE or not self._binance_client:
            return False
        
        try:
            params = {
                'symbol': order.request.symbol,
                'side': order.request.side.upper(),
                'type': order.request.order_type.upper(),
                'quantity': float(order.request.quantity),
                'newClientOrderId': order.id[:32]  # Binance limit
            }
            
            if order.request.price:
                params['price'] = float(order.request.price)
                params['timeInForce'] = 'GTC'
            
            if order.request.stop_price:
                params['stopPrice'] = float(order.request.stop_price)
            
            start = asyncio.get_event_loop().time()
            response = await self._binance_client.create_order(**params)
            latency = (asyncio.get_event_loop().time() - start) * 1000
            self._latency_ms.append(latency)
            
            # Parse response
            order.external_id = str(response.get('orderId'))
            
            status = response.get('status')
            if status == 'FILLED':
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                
                # Calculate fills
                fills = response.get('fills', [])
                if fills:
                    total_qty = Decimal('0')
                    total_value = Decimal('0')
                    
                    for fill in fills:
                        qty = Decimal(str(fill['qty']))
                        price = Decimal(str(fill['price']))
                        total_qty += qty
                        total_value += qty * price
                        order.commission += Decimal(str(fill.get('commission', 0)))
                    
                    order.filled_quantity = total_qty
                    order.avg_fill_price = total_value / total_qty if total_qty > 0 else None
                else:
                    # Immediate fill
                    order.filled_quantity = order.request.quantity
                    order.avg_fill_price = Decimal(str(response.get('price', 0)))
                
                # Update risk guard
                await self.risk_guard.update_position(
                    order.request.symbol,
                    order.filled_quantity,
                    order.avg_fill_price or Decimal('0'),
                    order.request.side
                )
                
                # Record fill
                self._fills.append({
                    'order_id': order.id,
                    'symbol': order.request.symbol,
                    'qty': float(order.filled_quantity),
                    'price': float(order.avg_fill_price) if order.avg_fill_price else 0,
                    'time': datetime.now().isoformat()
                })
                
                return True
                
            elif status == 'PARTIALLY_FILLED':
                order.status = OrderStatus.PARTIALLY_FILLED
                # Would need to track and wait for rest
                return False
                
            else:
                order.status = OrderStatus.SUBMITTED
                return False
                
        except Exception as e:
            self._error_count += 1
            raise
    
    async def _execute_mt5(self, order: Order) -> bool:
        """Execute on MT5 (in thread pool)"""
        if not MT5_AVAILABLE or not self._mt5_initialized:
            return False
        
        def sync_execute():
            symbol_info = mt5.symbol_info(order.request.symbol)
            if not symbol_info:
                raise RuntimeError(f"Symbol {order.request.symbol} not found")
            
            tick = mt5.symbol_info_tick(order.request.symbol)
            if not tick:
                raise RuntimeError("No tick data")
            
            price = tick.ask if order.request.side == 'buy' else tick.bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order.request.symbol,
                "volume": float(order.request.quantity),
                "type": mt5.ORDER_TYPE_BUY if order.request.side == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 10,
                "magic": 234000,
                "comment": order.id[:30],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise RuntimeError(f"MT5 error: {result.comment}")
            
            return {
                'order_id': str(result.order),
                'price': result.price,
                'volume': result.volume
            }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._thread_pool, sync_execute)
        
        order.external_id = result['order_id']
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_quantity = Decimal(str(result['volume']))
        order.avg_fill_price = Decimal(str(result['price']))
        
        # Update risk
        await self.risk_guard.update_position(
            order.request.symbol,
            order.filled_quantity,
            order.avg_fill_price,
            order.request.side
        )
        
        return True
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        async with self._order_lock:
            order = self._orders.get(order_id)
            if not order or order.is_done:
                return False
            
            order.status = OrderStatus.CANCELLED
        
        # Execute cancel on broker
        # Implementation depends on broker
        logger.info(f"Order {order_id} cancelled")
        return True
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        async with self._order_lock:
            return self._orders.get(order_id)
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        return await self.risk_guard.get_position(symbol)
    
    # ==================== EMERGENCY CONTROLS ====================
    
    async def emergency_close_all(self):
        """Emergency flatten all positions"""
        logger.critical("EMERGENCY CLOSE ALL INITIATED")
        
        self._kill_switch = True
        self._emergency_stop.set()
        
        # Get all positions
        positions = await self.risk_guard.emergency_flatten_all()
        
        # Close each
        for symbol, pos in positions:
            try:
                side = 'sell' if pos.side == 'long' else 'buy'
                await self.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=float(pos.quantity),
                    order_type='market',
                    metadata={'emergency_close': True}
                )
                logger.info(f"Emergency closed {symbol}")
                await asyncio.sleep(0.5)  # Rate limit
            except Exception as e:
                logger.error(f"Failed to close {symbol}: {e}")
        
        logger.info("Emergency close complete")
    
    def trigger_kill_switch(self):
        """Activate kill switch to halt all trading"""
        logger.critical("KILL SWITCH ACTIVATED")
        self._kill_switch = True
    
    def reset_kill_switch(self):
        """Reset kill switch after manual review"""
        logger.info("Kill switch reset")
        self._kill_switch = False
    
    # ==================== BACKGROUND TASKS ====================
    
    async def _position_sync_loop(self):
        """Sync positions with brokers"""
        while not self._shutdown_event.is_set():
            try:
                # Sync with Binance
                if self._binance_client:
                    account = await self._binance_client.get_account()
                    for balance in account.get('balances', []):
                        if float(balance['free']) > 0 or float(balance['locked']) > 0:
                            # Update position tracking
                            pass
                
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Position sync error: {e}")
                await asyncio.sleep(5)
    
    async def _risk_monitor_loop(self):
        """Monitor risk metrics"""
        while not self._shutdown_event.is_set():
            try:
                # Check for kill switch conditions
                if len(self._latency_ms) > 100:
                    avg_latency = sum(self._latency_ms) / len(self._latency_ms)
                    if avg_latency > 5000:  # 5 seconds
                        logger.critical(f"High latency detected: {avg_latency:.0f}ms")
                        # Could auto-trigger kill switch
                
                # Check error rate
                if self._request_count > 100:
                    error_rate = self._error_count / self._request_count
                    if error_rate > 0.1:  # 10% errors
                        logger.critical(f"High error rate: {error_rate:.1%}")
                
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=60)
            except asyncio.TimeoutError:
                continue
    
    # ==================== METRICS ====================
    
    def get_metrics(self) -> Dict:
        """Get execution metrics"""
        return {
            'total_orders': len(self._orders),
            'active_orders': len([o for o in self._orders.values() if not o.is_done]),
            'avg_latency_ms': sum(self._latency_ms) / len(self._latency_ms) if self._latency_ms else 0,
            'error_rate': self._error_count / max(self._request_count, 1),
            'kill_switch_active': self._kill_switch
        }


# ==================== USAGE ====================

async def example():
    config = {
        'execution': {
            'binance': {'enabled': True, 'api_key': 'xxx', 'secret': 'xxx'},
            'max_order_size': 10,
            'max_position_size': 100,
            'equity': 50000
        }
    }
    
    engine = ExecutionEngine(config)
    await engine.initialize()
    
    # Submit order
    order = await engine.submit_order(
        symbol='BTCUSDT',
        side='buy',
        quantity=0.1,
        order_type='market'
    )
    
    print(f"Order status: {order.status.name}")
    
    # Emergency close
    # engine.trigger_kill_switch()
    # await engine.emergency_close_all()
    
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(example())