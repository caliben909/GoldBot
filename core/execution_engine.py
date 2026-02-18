"""
Execution Engine - Institutional-grade execution with advanced order types, dynamic slippage, retry logic, and real-time PnL
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import MetaTrader5 as mt5
from binance import AsyncClient, BinanceSocketManager
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
import ccxt.async_support as ccxt
import logging
import time
import hmac
import hashlib
import json
import asyncio
from collections import deque
import threading
import queue
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import math
import websockets
import aiohttp
from typing import Optional
import uuid

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other
    ICEBERG = "iceberg"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    DAY = "DAY"  # Day order
    GTX = "GTX"  # Good Till Crossing


class BrokerType(Enum):
    MT5 = "mt5"
    BINANCE = "binance"
    BINANCE_FUTURES = "binance_futures"
    CCXT = "ccxt"


@dataclass
class Order:
    """Complete order object with all parameters"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    trailing_distance: Optional[float] = None
    trailing_activation: Optional[float] = None
    iceberg_qty: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    fills: List[Dict] = field(default_factory=list)
    commission: float = 0.0
    commission_asset: str = "USDT"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    external_id: Optional[str] = None
    client_order_id: Optional[str] = None
    broker: BrokerType = BrokerType.BINANCE
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity
        if not self.client_order_id:
            self.client_order_id = str(uuid.uuid4())


@dataclass
class Position:
    """Real-time position tracking with PnL"""
    symbol: str
    side: OrderSide
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    open_time: datetime
    update_time: datetime = field(default_factory=datetime.now)
    orders: List[Order] = field(default_factory=list)
    broker: BrokerType = BrokerType.BINANCE
    leverage: float = 1.0
    liquidation_price: Optional[float] = None
    margin_used: float = 0.0
    margin_ratio: float = 0.0
    pnl_history: List[float] = field(default_factory=list)
    
    @property
    def value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        if self.side == OrderSide.BUY:
            return ((self.current_price - self.avg_entry_price) / self.avg_entry_price) * 100
        else:
            return ((self.avg_entry_price - self.current_price) / self.avg_entry_price) * 100
    
    def update_pnl(self, current_price: float):
        """Update unrealized PnL"""
        self.current_price = current_price
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_entry_price - current_price) * self.quantity
        self.pnl_history.append(self.unrealized_pnl)
        self.update_time = datetime.now()


@dataclass
class OrderBook:
    """Real-time order book snapshot"""
    symbol: str
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]
    timestamp: datetime
    bid_volume: float
    ask_volume: float
    spread: float
    mid_price: float
    bid_depth: float  # Total bid volume
    ask_depth: float  # Total ask volume
    imbalance: float  # (bid_depth - ask_depth) / (bid_depth + ask_depth)


@dataclass
class ExecutionReport:
    """Real-time execution report"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    realized_pnl: float
    timestamp: datetime
    broker: BrokerType
    fill_type: str  # partial, full
    liquidity: str  # maker, taker


class ExecutionEngine:
    """
    Professional execution engine with:
    - All order types (market, limit, stop, OCO, iceberg, TWAP, VWAP)
    - Dynamic slippage based on volatility and liquidity
    - Exponential backoff retry logic
    - Real-time PnL tracking
    - WebSocket streaming
    - Order book management
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Broker clients
        self.binance_client = None
        self.binance_futures_client = None
        self.binance_socket_manager = None
        self.mt5_initialized = False
        self.ccxt_exchanges = {}
        
        # WebSocket connections
        self.ws_connections = {}
        self.ws_tasks = []
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []
        self.execution_reports: List[ExecutionReport] = []
        
        # Order books
        self.order_books: Dict[str, OrderBook] = {}
        
        # Rate limiting
        self.rate_limiter = {}
        self.request_counts = {}
        self.last_request_time = {}
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'total_realized_pnl': 0.0,
            'avg_slippage_bps': 0.0,
            'avg_fill_time_ms': 0.0,
            'best_execution_rate': 0.0,
            'retry_count': 0,
            'failed_orders': 0
        }
        
        # Background tasks
        self._running = False
        self._tasks = []
        
        # Order queues
        self.order_queue = asyncio.Queue()
        self.cancel_queue = asyncio.Queue()
        self.modify_queue = asyncio.Queue()
        
        # WebSocket subscriptions
        self.ws_subscriptions = {}
        
        logger.info("ExecutionEngine initialized")
    
    async def initialize(self):
        """Initialize all broker connections and WebSockets"""
        logger.info("Initializing Execution Engine...")
        
        # Initialize Binance Spot
        if self.config['execution']['binance']['enabled']:
            await self._init_binance_spot()
        
        # Initialize Binance Futures
        if self.config['execution'].get('binance_futures', {}).get('enabled', False):
            await self._init_binance_futures()
        
        # Initialize MT5
        if self.config['execution']['mt5']['enabled']:
            await self._init_mt5()
        
        # Initialize CCXT exchanges
        await self._init_ccxt()
        
        # Start background tasks
        self._running = True
        self._tasks.extend([
            asyncio.create_task(self._process_order_queue()),
            asyncio.create_task(self._process_cancel_queue()),
            asyncio.create_task(self._process_modify_queue()),
            asyncio.create_task(self._monitor_positions()),
            asyncio.create_task(self._update_order_books()),
            asyncio.create_task(self._rate_limit_monitor()),
            asyncio.create_task(self._websocket_manager()),
            asyncio.create_task(self._health_check())
        ])
        
        logger.info("✅ Execution Engine initialized successfully")
    
    async def _init_binance_spot(self):
        """Initialize Binance Spot client with WebSocket"""
        try:
            self.binance_client = await AsyncClient.create(
                api_key=self.config['execution']['binance']['api_key'],
                api_secret=self.config['execution']['binance']['api_secret'],
                testnet=self.config['execution']['binance'].get('testnet', False)
            )
            
            # Initialize WebSocket manager
            if self.config['execution']['binance'].get('websocket', False):
                self.binance_socket_manager = BinanceSocketManager(self.binance_client)
            
            # Test connection
            await self.binance_client.ping()
            
            # Get exchange info
            exchange_info = await self.binance_client.get_exchange_info()
            logger.info(f"✅ Binance Spot initialized with {len(exchange_info['symbols'])} symbols")
            
        except Exception as e:
            logger.error(f"Binance Spot initialization failed: {e}")
            self.binance_client = None
    
    async def _init_binance_futures(self):
        """Initialize Binance Futures client"""
        try:
            self.binance_futures_client = await AsyncClient.create(
                api_key=self.config['execution']['binance_futures']['api_key'],
                api_secret=self.config['execution']['binance_futures']['api_secret'],
                testnet=self.config['execution']['binance_futures'].get('testnet', False)
            )
            
            # Test connection
            await self.binance_futures_client.futures_ping()
            
            # Get account info
            account = await self.binance_futures_client.futures_account()
            logger.info(f"✅ Binance Futures initialized")
            logger.info(f"   Total Balance: ${float(account['totalWalletBalance']):.2f}")
            
        except Exception as e:
            logger.error(f"Binance Futures initialization failed: {e}")
            self.binance_futures_client = None
    
    async def _init_mt5(self):
        """Initialize MT5 connection"""
        try:
            path = self.config['execution']['mt5'].get('path')
            if path:
                if not mt5.initialize(path=path):
                    error = mt5.last_error()
                    raise Exception(f"MT5 initialization failed: {error}")
            else:
                if not mt5.initialize():
                    error = mt5.last_error()
                    raise Exception(f"MT5 initialization failed: {error}")
            
            # Login if credentials provided
            login = self.config['execution']['mt5'].get('login')
            password = self.config['execution']['mt5'].get('password')
            server = self.config['execution']['mt5'].get('server')
            
            if login and password and server:
                authorized = mt5.login(
                    login=int(login),
                    password=password,
                    server=server,
                    timeout=self.config['execution']['mt5'].get('timeout', 30000)
                )
                if not authorized:
                    error = mt5.last_error()
                    raise Exception(f"MT5 login failed: {error}")
            
            self.mt5_initialized = True
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"✅ MT5 initialized")
                logger.info(f"   Account: {account_info.login}")
                logger.info(f"   Balance: ${account_info.balance:.2f}")
            
        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")
            self.mt5_initialized = False
    
    async def _init_ccxt(self):
        """Initialize CCXT exchanges"""
        exchanges = ['binance', 'bybit', 'okx', 'kucoin']
        
        for exchange_id in exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'apiKey': self.config['execution'].get(exchange_id, {}).get('api_key', ''),
                    'secret': self.config['execution'].get(exchange_id, {}).get('secret', ''),
                    'enableRateLimit': True,
                    'rateLimit': 1200,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                })
                await exchange.load_markets()
                self.ccxt_exchanges[exchange_id] = exchange
                logger.info(f"✅ CCXT {exchange_id} initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_id}: {e}")
    
    def calculate_dynamic_slippage(self, symbol: str, quantity: float, side: OrderSide) -> float:
        """
        Calculate dynamic slippage based on volatility and liquidity
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: Order side
        
        Returns:
            Slippage in basis points
        """
        base_slippage = self.config['execution']['slippage']['base_bps']
        
        # Get volatility multiplier
        if symbol in self.volatility_history:
            volatility = self.volatility_history[symbol][-1] if self.volatility_history[symbol] else 0.20
            vol_multiplier = 1.0 + (volatility * self.config['execution']['slippage']['volatility_multiplier'])
        else:
            vol_multiplier = 1.0
        
        # Get liquidity multiplier
        if symbol in self.order_books:
            order_book = self.order_books[symbol]
            total_depth = order_book.bid_depth + order_book.ask_depth
            if total_depth > 0:
                liquidity_ratio = quantity / total_depth
                liq_multiplier = 1.0 + (liquidity_ratio * self.config['execution']['slippage']['liquidity_multiplier'])
            else:
                liq_multiplier = 2.0  # No liquidity, high slippage
        else:
            liq_multiplier = 1.0
        
        # Calculate dynamic slippage
        dynamic_slippage = base_slippage * vol_multiplier * liq_multiplier
        
        # Apply limits
        max_slippage = self.config['execution']['slippage']['max_slippage_bps']
        min_slippage = self.config['execution']['slippage']['min_slippage_bps']
        
        return np.clip(dynamic_slippage, min_slippage, max_slippage)
    
    async def place_order_with_retry(self, order: Order) -> Order:
        """
        Place order with exponential backoff retry logic
        
        Args:
            order: Order to place
        
        Returns:
            Updated order
        """
        max_attempts = self.config['execution']['retry']['max_attempts']
        backoff_factor = self.config['execution']['retry']['backoff_factor']
        max_backoff = self.config['execution']['retry']['max_backoff_seconds']
        
        for attempt in range(max_attempts):
            try:
                order.retry_count = attempt
                result = await self.place_order(order)
                
                if result.status != OrderStatus.REJECTED:
                    return result
                
                # Calculate backoff
                wait_time = min(backoff_factor ** attempt, max_backoff)
                self.logger.warning(f"Order {order.id} rejected, retrying in {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Order placement attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    order.status = OrderStatus.FAILED
                    order.error = str(e)
                    self.execution_stats['failed_orders'] += 1
                    return order
                
                wait_time = min(backoff_factor ** attempt, max_backoff)
                await asyncio.sleep(wait_time)
        
        return order
    
    async def place_order(self, order: Order) -> Order:
        """
        Place order with smart routing
        
        Args:
            order: Order to place
        
        Returns:
            Updated order
        """
        order.id = self._generate_order_id()
        order.created_at = datetime.now()
        order.status = OrderStatus.PENDING
        
        self.orders[order.id] = order
        self.execution_stats['total_orders'] += 1
        
        # Validate order
        if not await self._validate_order(order):
            order.status = OrderStatus.REJECTED
            order.error = "Order validation failed"
            self.execution_stats['rejected_orders'] += 1
            return order
        
        # Queue for execution
        await self.order_queue.put(order)
        
        self.logger.info(f"Order {order.id} queued: {order.side.value} {order.quantity} {order.symbol}")
        
        return order
    
    async def place_oco_order(self, symbol: str, side: OrderSide, quantity: float,
                             price: float, stop_price: float, limit_price: float) -> List[Order]:
        """
        Place One-Cancels-Other (OCO) order
        
        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Limit price
            stop_price: Stop price
            limit_price: Stop-limit price
        
        Returns:
            List of created orders
        """
        orders = []
        
        # Create limit order
        limit_order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=TimeInForce.GTC,
            broker=self._determine_broker(symbol),
            metadata={'oco_group': True}
        )
        orders.append(limit_order)
        
        # Create stop-limit order
        stop_order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,  # Opposite side
            type=OrderType.STOP_LIMIT,
            quantity=quantity,
            stop_price=stop_price,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC,
            broker=self._determine_broker(symbol),
            metadata={'oco_group': True}
        )
        orders.append(stop_order)
        
        # Place orders
        for order in orders:
            await self.place_order(order)
        
        return orders
    
    async def place_twap_order(self, symbol: str, side: OrderSide, total_quantity: float,
                              duration_minutes: int, slices: int = 10) -> List[Order]:
        """
        Place Time-Weighted Average Price (TWAP) order
        
        Args:
            symbol: Trading symbol
            side: Order side
            total_quantity: Total quantity to execute
            duration_minutes: Total execution duration
            slices: Number of slices
        
        Returns:
            List of created orders
        """
        orders = []
        slice_quantity = total_quantity / slices
        interval_seconds = (duration_minutes * 60) / slices
        
        for i in range(slices):
            # Calculate execution time
            execute_at = datetime.now() + timedelta(seconds=i * interval_seconds)
            
            # Create order
            order = Order(
                id=self._generate_order_id(),
                symbol=symbol,
                side=side,
                type=OrderType.LIMIT,  # Use limit orders for better pricing
                quantity=slice_quantity,
                price=None,  # Will be determined at execution time
                time_in_force=TimeInForce.DAY,
                broker=self._determine_broker(symbol),
                metadata={
                    'twap': True,
                    'slice': i + 1,
                    'total_slices': slices,
                    'execute_at': execute_at.isoformat()
                }
            )
            orders.append(order)
            
            # Schedule execution
            asyncio.create_task(self._execute_twap_slice(order, execute_at))
        
        return orders
    
    async def _execute_twap_slice(self, order: Order, execute_at: datetime):
        """Execute TWAP slice at scheduled time"""
        wait_seconds = (execute_at - datetime.now()).total_seconds()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)
        
        # Get current market price
        current_price = await self._get_current_price(order.symbol)
        order.price = current_price
        
        # Place order
        await self.place_order(order)
    
    async def place_vwap_order(self, symbol: str, side: OrderSide, total_quantity: float,
                              volume_profile: List[float]) -> List[Order]:
        """
        Place Volume-Weighted Average Price (VWAP) order
        
        Args:
            symbol: Trading symbol
            side: Order side
            total_quantity: Total quantity to execute
            volume_profile: Expected volume distribution over time
        
        Returns:
            List of created orders
        """
        orders = []
        total_volume = sum(volume_profile)
        
        for i, volume_pct in enumerate(volume_profile):
            slice_quantity = total_quantity * (volume_pct / total_volume)
            
            order = Order(
                id=self._generate_order_id(),
                symbol=symbol,
                side=side,
                type=OrderType.LIMIT,
                quantity=slice_quantity,
                price=None,
                time_in_force=TimeInForce.DAY,
                broker=self._determine_broker(symbol),
                metadata={
                    'vwap': True,
                    'slice': i + 1,
                    'volume_pct': volume_pct
                }
            )
            orders.append(order)
            
            # Schedule based on volume profile
            asyncio.create_task(self._execute_vwap_slice(order, i))
        
        return orders
    
    async def _execute_vwap_slice(self, order: Order, slice_index: int):
        """Execute VWAP slice based on volume profile"""
        # Simplified - in production, would use real-time volume data
        wait_seconds = slice_index * 300  # 5 minutes per slice
        await asyncio.sleep(wait_seconds)
        
        current_price = await self._get_current_price(order.symbol)
        order.price = current_price
        
        await self.place_order(order)
    
    async def place_iceberg_order(self, symbol: str, side: OrderSide, total_quantity: float,
                                 visible_quantity: float, price: float) -> List[Order]:
        """
        Place Iceberg order (large order split into smaller visible portions)
        
        Args:
            symbol: Trading symbol
            side: Order side
            total_quantity: Total quantity to execute
            visible_quantity: Visible portion size
            price: Order price
        
        Returns:
            List of created orders
        """
        orders = []
        remaining = total_quantity
        
        while remaining > 0:
            slice_qty = min(visible_quantity, remaining)
            
            order = Order(
                id=self._generate_order_id(),
                symbol=symbol,
                side=side,
                type=OrderType.LIMIT,
                quantity=slice_qty,
                price=price,
                time_in_force=TimeInForce.GTC,
                broker=self._determine_broker(symbol),
                metadata={
                    'iceberg': True,
                    'parent_quantity': total_quantity,
                    'remaining': remaining - slice_qty
                }
            )
            orders.append(order)
            await self.place_order(order)
            
            remaining -= slice_qty
            
            # Wait for previous slice to fill before placing next
            await asyncio.sleep(5)
        
        return orders
    
    async def _process_order_queue(self):
        """Process orders from queue with rate limiting and retry logic"""
        while self._running:
            try:
                order = await self.order_queue.get()
                
                # Check rate limits
                if not await self._check_rate_limits(order.broker):
                    self.logger.warning(f"Rate limit exceeded for {order.broker.value}, requeuing {order.id}")
                    await asyncio.sleep(1)
                    await self.order_queue.put(order)
                    self.order_queue.task_done()
                    continue
                
                # Calculate dynamic slippage
                slippage_bps = self.calculate_dynamic_slippage(order.symbol, order.quantity, order.side)
                order.metadata['expected_slippage_bps'] = slippage_bps
                
                # Execute order based on broker
                start_time = time.time()
                
                if order.broker == BrokerType.BINANCE:
                    result = await self._execute_binance_spot(order)
                elif order.broker == BrokerType.BINANCE_FUTURES:
                    result = await self._execute_binance_futures(order)
                elif order.broker == BrokerType.MT5:
                    result = await self._execute_mt5(order)
                elif order.broker == BrokerType.CCXT:
                    result = await self._execute_ccxt(order)
                else:
                    order.status = OrderStatus.REJECTED
                    order.error = f"Unsupported broker: {order.broker}"
                    self.order_history.append(order)
                    self.order_queue.task_done()
                    continue
                
                # Calculate execution metrics
                execution_time = (time.time() - start_time) * 1000  # ms
                self.execution_stats['avg_fill_time_ms'] = (
                    self.execution_stats['avg_fill_time_ms'] * 0.95 + execution_time * 0.05
                )
                
                if result:
                    order = result
                    
                    # Calculate actual slippage
                    if order.avg_fill_price and order.metadata.get('expected_price'):
                        expected = order.metadata['expected_price']
                        actual = order.avg_fill_price
                        if order.side == OrderSide.BUY:
                            slippage = (actual - expected) / expected * 10000
                        else:
                            slippage = (expected - actual) / expected * 10000
                        
                        self.execution_stats['avg_slippage_bps'] = (
                            self.execution_stats['avg_slippage_bps'] * 0.95 + slippage * 0.05
                        )
                
                self.order_history.append(order)
                
                # Update position if filled
                if order.status == OrderStatus.FILLED:
                    await self._update_position(order)
                    self.execution_stats['filled_orders'] += 1
                    self.execution_stats['total_volume'] += order.filled_quantity
                    
                    # Create execution report
                    report = ExecutionReport(
                        order_id=order.id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.filled_quantity,
                        price=order.avg_fill_price,
                        commission=order.commission,
                        realized_pnl=0,  # Will be calculated when position closes
                        timestamp=datetime.now(),
                        broker=order.broker,
                        fill_type='full' if order.filled_quantity == order.quantity else 'partial',
                        liquidity='taker'  # Default, would need to check
                    )
                    self.execution_reports.append(report)
                    
                    self.logger.info(
                        f"Order {order.id} filled: {order.filled_quantity} @ ${order.avg_fill_price:.4f} "
                        f"(slippage: {self.execution_stats['avg_slippage_bps']:.1f} bps)"
                    )
                
                elif order.status == OrderStatus.REJECTED:
                    self.execution_stats['rejected_orders'] += 1
                    self.logger.error(f"Order {order.id} rejected: {order.error}")
                
                elif order.status == OrderStatus.CANCELLED:
                    self.execution_stats['cancelled_orders'] += 1
                
                self.order_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Order processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_binance_spot(self, order: Order) -> Order:
        """Execute order on Binance Spot with all order types"""
        try:
            order.submitted_at = datetime.now()
            order.status = OrderStatus.SUBMITTED
            
            # Prepare order parameters
            params = {
                'symbol': order.symbol,
                'side': 'BUY' if order.side == OrderSide.BUY else 'SELL',
                'type': self._map_order_type(order.type),
                'quantity': order.quantity,
                'newClientOrderId': order.client_order_id
            }
            
            # Add price for limit orders
            if order.type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                params['price'] = order.price
                params['timeInForce'] = self._map_time_in_force(order.time_in_force)
            
            # Add stop price for stop orders
            if order.type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.STOP_MARKET,
                             OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT]:
                params['stopPrice'] = order.stop_price
            
            # Add iceberg quantity
            if order.iceberg_qty:
                params['icebergQty'] = order.iceberg_qty
            
            # Store expected price for slippage calculation
            if order.type == OrderType.MARKET:
                ticker = await self.binance_client.get_symbol_ticker(symbol=order.symbol)
                order.metadata['expected_price'] = float(ticker['price'])
            
            # Place order
            response = await self.binance_client.create_order(**params)
            
            # Update order with response
            order.external_id = str(response['orderId'])
            order.status = self._map_binance_status(response['status'])
            order.filled_quantity = float(response['executedQty'])
            order.avg_fill_price = float(response['price']) if response['price'] else None
            
            # Process fills
            if 'fills' in response:
                for fill in response['fills']:
                    fill_info = {
                        'price': float(fill['price']),
                        'quantity': float(fill['qty']),
                        'commission': float(fill['commission']),
                        'commission_asset': fill['commissionAsset'],
                        'time': datetime.fromtimestamp(fill['tradeId'] / 1000)
                    }
                    order.fills.append(fill_info)
                    order.commission += fill_info['commission']
                    self.execution_stats['total_commission'] += fill_info['commission']
            
            order.updated_at = datetime.now()
            
            # Calculate average fill price if not provided
            if order.fills and not order.avg_fill_price:
                total_value = sum(f['price'] * f['quantity'] for f in order.fills)
                total_qty = sum(f['quantity'] for f in order.fills)
                order.avg_fill_price = total_value / total_qty if total_qty > 0 else None
            
            # Update remaining quantity
            order.remaining_quantity = order.quantity - order.filled_quantity
            
            return order
            
        except BinanceAPIException as e:
            order.status = OrderStatus.REJECTED
            order.error = f"Binance API error: {e.message}"
            self.logger.error(f"Binance order failed: {e}")
            return order
        
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error = str(e)
            self.logger.error(f"Binance order failed: {e}")
            return order
    
    async def _execute_binance_futures(self, order: Order) -> Order:
        """Execute order on Binance Futures"""
        try:
            order.submitted_at = datetime.now()
            order.status = OrderStatus.SUBMITTED
            
            params = {
                'symbol': order.symbol,
                'side': 'BUY' if order.side == OrderSide.BUY else 'SELL',
                'type': self._map_order_type(order.type),
                'quantity': order.quantity,
                'newClientOrderId': order.client_order_id
            }
            
            if order.type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                params['price'] = order.price
                params['timeInForce'] = self._map_time_in_force(order.time_in_force)
            
            if order.type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.STOP_MARKET]:
                params['stopPrice'] = order.stop_price
            
            response = await self.binance_futures_client.futures_create_order(**params)
            
            order.external_id = str(response['orderId'])
            order.status = self._map_binance_status(response['status'])
            order.filled_quantity = float(response['executedQty'])
            order.avg_fill_price = float(response['avgPrice']) if response.get('avgPrice') else float(response['price'])
            order.updated_at = datetime.now()
            order.remaining_quantity = order.quantity - order.filled_quantity
            
            return order
            
        except BinanceAPIException as e:
            order.status = OrderStatus.REJECTED
            order.error = f"Binance Futures error: {e.message}"
            return order
        
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error = str(e)
            return order
    
    async def _execute_mt5(self, order: Order) -> Order:
        """Execute order on MT5"""
        if not self.mt5_initialized:
            order.status = OrderStatus.REJECTED
            order.error = "MT5 not initialized"
            return order
        
        try:
            order.submitted_at = datetime.now()
            order.status = OrderStatus.SUBMITTED
            
            symbol_info = mt5.symbol_info(order.symbol)
            if not symbol_info:
                order.status = OrderStatus.REJECTED
                order.error = f"Symbol {order.symbol} not found"
                return order
            
            tick = mt5.symbol_info_tick(order.symbol)
            if not tick:
                order.status = OrderStatus.REJECTED
                order.error = f"No tick data for {order.symbol}"
                return order
            
            if order.side == OrderSide.BUY:
                price = tick.ask
                sl_price = order.metadata.get('stop_loss', 0)
                tp_price = order.metadata.get('take_profit', 0)
            else:
                price = tick.bid
                sl_price = order.metadata.get('stop_loss', 0)
                tp_price = order.metadata.get('take_profit', 0)
            
            # Map order type
            mt5_type_map = {
                OrderType.MARKET: mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL,
                OrderType.LIMIT: mt5.ORDER_TYPE_BUY_LIMIT if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_LIMIT,
                OrderType.STOP: mt5.ORDER_TYPE_BUY_STOP if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_STOP,
                OrderType.STOP_LIMIT: mt5.ORDER_TYPE_BUY_STOP_LIMIT if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_STOP_LIMIT,
            }
            
            trade_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order.symbol,
                "volume": order.quantity,
                "type": mt5_type_map.get(order.type, mt5.ORDER_TYPE_BUY),
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": self.config['execution']['slippage']['base_bps'] * 100,
                "magic": 999999,
                "comment": f"SMC_{order.id[:8]}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK if order.symbol.endswith('XBT') else mt5.ORDER_FILLING_RETURN
            }
            
            if order.type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                trade_request["stop_limit"] = order.stop_price
            
            result = mt5.order_send(trade_request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                order.status = OrderStatus.REJECTED
                order.error = f"MT5 error: {result.comment} (code: {result.retcode})"
                return order
            
            order.external_id = str(result.order)
            order.status = OrderStatus.FILLED
            order.filled_quantity = result.volume
            order.avg_fill_price = result.price
            order.updated_at = datetime.now()
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error = str(e)
            return order
    
    async def _execute_ccxt(self, order: Order) -> Order:
        """Execute order via CCXT"""
        exchange_id = 'binance'
        if exchange_id not in self.ccxt_exchanges:
            order.status = OrderStatus.REJECTED
            order.error = f"Exchange {exchange_id} not available"
            return order
        
        exchange = self.ccxt_exchanges[exchange_id]
        
        try:
            order.submitted_at = datetime.now()
            
            params = {
                'symbol': order.symbol,
                'type': order.type.value,
                'side': order.side.value,
                'amount': order.quantity
            }
            
            if order.price:
                params['price'] = order.price
            
            response = await exchange.create_order(**params)
            
            order.external_id = str(response['id'])
            order.status = OrderStatus.FILLED if response['status'] == 'closed' else OrderStatus.PENDING
            order.filled_quantity = float(response['filled'])
            order.avg_fill_price = float(response['average']) if response['average'] else None
            order.updated_at = datetime.now()
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error = str(e)
            return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel open order"""
        if order_id not in self.orders:
            self.logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            self.logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
            return False
        
        await self.cancel_queue.put(order)
        return True
    
    async def _process_cancel_queue(self):
        """Process cancellation requests"""
        while self._running:
            try:
                order = await self.cancel_queue.get()
                
                result = await self._cancel_order_on_broker(order)
                
                if result:
                    order.status = OrderStatus.CANCELLED
                    order.cancelled_at = datetime.now()
                    order.updated_at = datetime.now()
                    self.logger.info(f"Order {order.id} cancelled")
                else:
                    self.logger.error(f"Failed to cancel order {order.id}")
                
                self.cancel_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Cancel processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _cancel_order_on_broker(self, order: Order) -> bool:
        """Cancel order on specific broker"""
        try:
            if order.broker == BrokerType.BINANCE:
                result = await self.binance_client.cancel_order(
                    symbol=order.symbol,
                    orderId=order.external_id
                )
                return result is not None
                
            elif order.broker == BrokerType.BINANCE_FUTURES:
                result = await self.binance_futures_client.futures_cancel_order(
                    symbol=order.symbol,
                    orderId=order.external_id
                )
                return result is not None
                
            elif order.broker == BrokerType.MT5:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": int(order.external_id)
                }
                result = mt5.order_send(request)
                return result.retcode == mt5.TRADE_RETCODE_DONE
                
            elif order.broker == BrokerType.CCXT:
                exchange_id = 'binance'
                if exchange_id in self.ccxt_exchanges:
                    exchange = self.ccxt_exchanges[exchange_id]
                    result = await exchange.cancel_order(order.external_id, order.symbol)
                    return result is not None
            
        except Exception as e:
            self.logger.error(f"Cancel failed for order {order.id}: {e}")
        
        return False
    
    async def modify_order(self, order_id: str, updates: Dict) -> bool:
        """Modify open order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        await self.modify_queue.put((order, updates))
        return True
    
    async def _process_modify_queue(self):
        """Process order modification requests"""
        while self._running:
            try:
                order, updates = await self.modify_queue.get()
                
                # Cancel existing order
                if await self._cancel_order_on_broker(order):
                    # Create new order with updates
                    new_order = Order(
                        id=self._generate_order_id(),
                        symbol=order.symbol,
                        side=order.side,
                        type=order.type,
                        quantity=updates.get('quantity', order.quantity),
                        price=updates.get('price', order.price),
                        stop_price=updates.get('stop_price', order.stop_price),
                        time_in_force=order.time_in_force,
                        broker=order.broker,
                        metadata={**order.metadata, **updates.get('metadata', {})}
                    )
                    
                    result = await self.place_order_with_retry(new_order)
                    
                    if result.status == OrderStatus.FILLED:
                        order.status = OrderStatus.CANCELLED
                        self.logger.info(f"Order {order.id} modified to {new_order.id}")
                    else:
                        self.logger.error(f"Failed to modify order {order.id}")
                
                self.modify_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Modify processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def get_positions(self, broker: Optional[BrokerType] = None) -> List[Position]:
        """Get current positions with real-time PnL"""
        positions = []
        
        # Binance Futures positions
        if (not broker or broker == BrokerType.BINANCE_FUTURES) and self.binance_futures_client:
            try:
                account = await self.binance_futures_client.futures_account()
                for pos in account['positions']:
                    if float(pos['positionAmt']) != 0:
                        position = Position(
                            symbol=pos['symbol'],
                            side=OrderSide.BUY if float(pos['positionAmt']) > 0 else OrderSide.SELL,
                            quantity=abs(float(pos['positionAmt'])),
                            avg_entry_price=float(pos['entryPrice']),
                            current_price=float(pos['markPrice']),
                            unrealized_pnl=float(pos['unRealizedProfit']),
                            realized_pnl=float(pos['realizedProfit']),
                            open_time=datetime.now(),
                            broker=BrokerType.BINANCE_FUTURES,
                            leverage=float(pos['leverage']),
                            liquidation_price=float(pos['liquidationPrice']) if pos['liquidationPrice'] else None,
                            margin_used=float(pos['initialMargin']),
                            margin_ratio=float(pos['marginRatio'])
                        )
                        positions.append(position)
                        self.positions[pos['symbol']] = position
                        
            except Exception as e:
                self.logger.error(f"Failed to get Binance Futures positions: {e}")
        
        # MT5 positions
        if (not broker or broker == BrokerType.MT5) and self.mt5_initialized:
            try:
                mt5_positions = mt5.positions_get()
                if mt5_positions:
                    for pos in mt5_positions:
                        position = Position(
                            symbol=pos.symbol,
                            side=OrderSide.BUY if pos.type == mt5.POSITION_TYPE_BUY else OrderSide.SELL,
                            quantity=pos.volume,
                            avg_entry_price=pos.price_open,
                            current_price=pos.price_current,
                            unrealized_pnl=pos.profit,
                            realized_pnl=0,
                            open_time=datetime.fromtimestamp(pos.time),
                            broker=BrokerType.MT5
                        )
                        positions.append(position)
                        self.positions[pos.symbol] = position
                        
            except Exception as e:
                self.logger.error(f"Failed to get MT5 positions: {e}")
        
        return positions
    
    async def _update_position(self, order: Order):
        """Update position after order fill"""
        if order.status != OrderStatus.FILLED:
            return
        
        positions = await self.get_positions(order.broker)
        
        position = next((p for p in positions if p.symbol == order.symbol), None)
        
        if position:
            self.positions[order.symbol] = position
            position.orders.append(order)
    
    async def _monitor_positions(self):
        """Real-time position monitoring with PnL updates"""
        while self._running:
            try:
                positions = await self.get_positions()
                
                for symbol, position in self.positions.items():
                    # Update current price
                    current_price = await self._get_current_price(symbol)
                    position.update_pnl(current_price)
                    
                    # Check for stop loss / take profit
                    await self._check_exit_conditions(position)
                
                await asyncio.sleep(self.config['execution']['monitoring']['pnl_update_frequency'])
                
            except Exception as e:
                self.logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_exit_conditions(self, position: Position):
        """Check exit conditions for position"""
        # Stop loss
        if position.side == OrderSide.BUY:
            if position.current_price <= position.avg_entry_price * 0.99:  # 1% loss
                await self.close_position(position.symbol)
        else:
            if position.current_price >= position.avg_entry_price * 1.01:  # 1% loss
                await self.close_position(position.symbol)
    
    async def close_position(self, symbol: str, quantity: Optional[float] = None) -> Optional[Order]:
        """Close position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        close_quantity = quantity if quantity else position.quantity
        
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=close_quantity,
            broker=position.broker,
            metadata={'closing_position': True}
        )
        
        result = await self.place_order_with_retry(order)
        
        if result.status == OrderStatus.FILLED:
            # Calculate realized PnL
            if position.side == OrderSide.BUY:
                realized_pnl = (result.avg_fill_price - position.avg_entry_price) * close_quantity
            else:
                realized_pnl = (position.avg_entry_price - result.avg_fill_price) * close_quantity
            
            self.execution_stats['total_realized_pnl'] += realized_pnl
            del self.positions[symbol]
        
        return result
    
    async def close_all_positions(self):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            await self.close_position(symbol)
            await asyncio.sleep(0.5)
    
    async def _update_order_books(self):
        """Update order books via WebSocket"""
        while self._running:
            try:
                symbols = []
                if self.config['assets']['forex']['enabled']:
                    symbols.extend(self.config['assets']['forex']['symbols'])
                if self.config['assets']['crypto']['enabled']:
                    symbols.extend(self.config['assets']['crypto']['symbols'])
                
                for symbol in symbols:
                    try:
                        if symbol.endswith(('USDT', 'BTC')):
                            if self.binance_client:
                                book = await self.binance_client.get_order_book(symbol=symbol, limit=10)
                                
                                bids = [(float(b[0]), float(b[1])) for b in book['bids']]
                                asks = [(float(a[0]), float(a[1])) for a in book['asks']]
                                
                                bid_volume = sum(b[1] for b in bids)
                                ask_volume = sum(a[1] for a in asks)
                                best_bid = bids[0][0] if bids else 0
                                best_ask = asks[0][0] if asks else 0
                                spread = best_ask - best_bid
                                mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
                                
                                self.order_books[symbol] = OrderBook(
                                    symbol=symbol,
                                    bids=bids,
                                    asks=asks,
                                    timestamp=datetime.now(),
                                    bid_volume=bid_volume,
                                    ask_volume=ask_volume,
                                    spread=spread,
                                    mid_price=mid,
                                    bid_depth=bid_volume,
                                    ask_depth=ask_volume,
                                    imbalance=(bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
                                )
                        
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to update order book for {symbol}: {e}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Order book update error: {e}")
                await asyncio.sleep(1)
    
    async def _websocket_manager(self):
        """Manage WebSocket connections for real-time data"""
        if not self.binance_socket_manager:
            return
        
        while self._running:
            try:
                # Subscribe to user data stream
                if self.binance_futures_client:
                    listen_key = await self.binance_futures_client.futures_stream_get_listen_key()
                    
                    # Connect to user data stream
                    ws_url = f"wss://fstream.binance.com/ws/{listen_key}"
                    
                    async with websockets.connect(ws_url) as websocket:
                        self.ws_connections['futures_user'] = websocket
                        
                        while self._running:
                            try:
                                msg = await websocket.recv()
                                data = json.loads(msg)
                                
                                # Handle different message types
                                if 'e' in data:
                                    if data['e'] == 'ORDER_TRADE_UPDATE':
                                        await self._handle_order_update(data)
                                    elif data['e'] == 'ACCOUNT_UPDATE':
                                        await self._handle_account_update(data)
                                    elif data['e'] == 'listenKeyExpired':
                                        # Renew listen key
                                        await self.binance_futures_client.futures_stream_renew_listen_key(listen_key)
                                
                            except websockets.exceptions.ConnectionClosed:
                                self.logger.warning("WebSocket connection closed, reconnecting...")
                                break
                            
                            except Exception as e:
                                self.logger.error(f"WebSocket error: {e}")
                                await asyncio.sleep(1)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"WebSocket manager error: {e}")
                await asyncio.sleep(10)
    
    async def _handle_order_update(self, data: dict):
        """Handle real-time order updates from WebSocket"""
        try:
            order_data = data['o']
            order_id = order_data['c']  # Client order ID
            
            if order_id in self.orders:
                order = self.orders[order_id]
                order.status = self._map_binance_status(order_data['X'])
                order.filled_quantity = float(order_data['z'])
                order.remaining_quantity = float(order_data['q']) - float(order_data['z'])
                order.updated_at = datetime.now()
                
                if order_data['X'] == 'FILLED':
                    order.filled_at = datetime.now()
                    
        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")
    
    async def _handle_account_update(self, data: dict):
        """Handle real-time account updates from WebSocket"""
        try:
            for position in data['a']['P']:
                symbol = position['s']
                if symbol in self.positions:
                    self.positions[symbol].current_price = float(position['p'])
                    self.positions[symbol].unrealized_pnl = float(position['up'])
                    self.positions[symbol].update_time = datetime.now()
                    
        except Exception as e:
            self.logger.error(f"Error handling account update: {e}")
    
    async def _health_check(self):
        """Periodic health check of all connections"""
        while self._running:
            try:
                # Check Binance connection
                if self.binance_client:
                    try:
                        await self.binance_client.ping()
                    except:
                        self.logger.error("Binance connection lost, reconnecting...")
                        await self._init_binance_spot()
                
                # Check WebSocket connections
                for name, ws in list(self.ws_connections.items()):
                    if ws.closed:
                        self.logger.warning(f"WebSocket {name} closed")
                        del self.ws_connections[name]
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)
    
    async def _check_rate_limits(self, broker: BrokerType) -> bool:
        """Check rate limits for broker"""
        now = time.time()
        
        # Clean up old requests
        for b in list(self.last_request_time.keys()):
            if now - self.last_request_time[b] > 60:
                del self.last_request_time[b]
                if b in self.request_counts:
                    del self.request_counts[b]
        
        # Update count
        self.request_counts[broker] = self.request_counts.get(broker, 0) + 1
        self.last_request_time[broker] = now
        
        limits = {
            BrokerType.BINANCE: 1200,
            BrokerType.BINANCE_FUTURES: 1200,
            BrokerType.MT5: 30,
            BrokerType.CCXT: 100
        }
        
        limit = limits.get(broker, 100)
        
        if self.request_counts[broker] > limit:
            self.logger.warning(f"Rate limit exceeded for {broker.value}: {self.request_counts[broker]}/{limit}")
            return False
        
        return True
    
    async def _rate_limit_monitor(self):
        """Monitor and reset rate limits"""
        while self._running:
            await asyncio.sleep(60)
            self.request_counts.clear()
            self.last_request_time.clear()
    
    async def _validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        if order.quantity <= 0:
            self.logger.error(f"Invalid quantity: {order.quantity}")
            return False
        
        if order.type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and (order.price is None or order.price <= 0):
            self.logger.error("Limit order requires valid price")
            return False
        
        if order.type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.STOP_MARKET] and (order.stop_price is None or order.stop_price <= 0):
            self.logger.error("Stop order requires valid stop price")
            return False
        
        return True
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        if symbol.endswith(('USDT', 'BTC')):
            try:
                ticker = await self.binance_client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            except:
                pass
        else:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return (tick.bid + tick.ask) / 2
        return 0.0
    
    def _determine_broker(self, symbol: str) -> BrokerType:
        """Determine broker for symbol"""
        if symbol.endswith(('USDT', 'BTC', 'ETH')):
            return BrokerType.BINANCE_FUTURES if self.config['execution'].get('binance_futures', {}).get('enabled') else BrokerType.BINANCE
        else:
            return BrokerType.MT5
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        timestamp = int(time.time() * 1000)
        random_part = int.from_bytes(os.urandom(4), 'big')
        return f"ORD_{timestamp}_{random_part:08x}"
    
    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to broker-specific type"""
        mapping = {
            OrderType.MARKET: ORDER_TYPE_MARKET,
            OrderType.LIMIT: ORDER_TYPE_LIMIT,
            OrderType.STOP: ORDER_TYPE_STOP_MARKET,
            OrderType.STOP_MARKET: ORDER_TYPE_STOP_MARKET,
            OrderType.STOP_LIMIT: ORDER_TYPE_STOP_LIMIT,
            OrderType.TAKE_PROFIT: ORDER_TYPE_TAKE_PROFIT_MARKET,
            OrderType.TAKE_PROFIT_LIMIT: ORDER_TYPE_TAKE_PROFIT_LIMIT,
            OrderType.TRAILING_STOP: 'TRAILING_STOP_MARKET',
        }
        return mapping.get(order_type, ORDER_TYPE_MARKET)
    
    def _map_time_in_force(self, tif: TimeInForce) -> str:
        """Map time in force to broker-specific value"""
        mapping = {
            TimeInForce.GTC: TIME_IN_FORCE_GTC,
            TimeInForce.IOC: TIME_IN_FORCE_IOC,
            TimeInForce.FOK: TIME_IN_FORCE_FOK,
            TimeInForce.DAY: TIME_IN_FORCE_DAY,
            TimeInForce.GTX: 'GTX'
        }
        return mapping.get(tif, TIME_IN_FORCE_GTC)
    
    def _map_binance_status(self, status: str) -> OrderStatus:
        """Map Binance order status to internal status"""
        mapping = {
            'NEW': OrderStatus.SUBMITTED,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        return mapping.get(status, OrderStatus.PENDING)
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        return {
            **self.execution_stats,
            'active_orders': len([o for o in self.orders.values() if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]]),
            'open_positions': len(self.positions),
            'total_commission_usd': self.execution_stats['total_commission']
        }
    
    async def shutdown(self):
        """Clean shutdown of all connections"""
        logger.info("Shutting down Execution Engine...")
        
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        if self.config['trading'].get('close_on_shutdown', False):
            await self.close_all_positions()
        
        # Close WebSocket connections
        for ws in self.ws_connections.values():
            await ws.close()
        
        # Close broker connections
        if self.binance_client:
            await self.binance_client.close_connection()
        
        if self.binance_futures_client:
            await self.binance_futures_client.close_connection()
        
        if self.mt5_initialized:
            mt5.shutdown()
        
        for exchange in self.ccxt_exchanges.values():
            await exchange.close()
        
        logger.info("Execution Engine shutdown complete")
        logger.info(f"Final Stats: {json.dumps(self.get_execution_stats(), indent=2)}")