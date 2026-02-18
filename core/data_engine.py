"""
Data Engine - Institutional-grade data management with advanced quality features
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import MetaTrader5 as mt5
from binance import AsyncClient, BinanceSocketManager
from binance.enums import *
from binance.exceptions import BinanceAPIException
import ccxt.async_support as ccxt
import redis.asyncio as redis
import json
import logging
from contextlib import asynccontextmanager
from collections import deque
import aiofiles
import pickle
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics for a symbol"""
    symbol: str
    completeness: float  # Percentage of expected data points
    consistency: float   # Cross-source consistency score
    accuracy: float      # Accuracy score (based on outlier detection)
    timeliness: float    # Data freshness score
    gap_count: int       # Number of data gaps
    max_gap_minutes: int # Maximum gap duration
    outlier_count: int   # Number of outliers detected
    anomaly_score: float # Overall anomaly score
    cross_source_error: float  # Error between different sources
    updated_at: datetime


@dataclass
class MarketData:
    """Complete market data container with quality metadata"""
    symbol: str
    timeframe: str
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    timestamp: np.ndarray
    spread: Optional[np.ndarray] = None
    tick_volume: Optional[np.ndarray] = None
    bid: Optional[np.ndarray] = None
    ask: Optional[np.ndarray] = None
    vwap: Optional[np.ndarray] = None
    quality_score: float = 1.0
    data_source: str = "unknown"
    interpolation_method: Optional[str] = None
    gaps_filled: int = 0
    outliers_removed: int = 0


@dataclass
class TickData:
    """High-frequency tick data"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float
    bid_size: float
    ask_size: float
    spread: float
    exchange: str
    sequence: int
    trade_id: Optional[str] = None


class DataEngine:
    """
    Production-grade data engine with advanced quality features:
    - Multi-source validation and cross-checking
    - Advanced interpolation methods
    - Outlier detection (Isolation Forest, Z-score, IQR)
    - Gap detection and filling
    - Data quality scoring
    - Real-time WebSocket streaming
    - Tick data support
    - Source failover
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data sources
        self.redis_client = None
        self.mt5_initialized = False
        self.binance_client = None
        self.binance_socket_manager = None
        self.binance_futures_client = None
        self.ccxt_exchanges = {}
        self.polygon_client = None
        self.alphavantage_client = None
        
        # Data caches
        self.data_cache = {}
        self.tick_buffer = deque(maxlen=100000)  # Store last 100k ticks
        self.order_book_cache = {}
        
        # Quality tracking
        self.quality_metrics: Dict[str, DataQualityMetrics] = {}
        self.data_sources_status: Dict[str, bool] = {}
        self.source_priority = self._init_source_priority()
        
        # Subscribers
        self.subscribers = []
        self.ws_connections = {}
        
        # Symbol info
        self.symbol_info = {}
        self.market_hours = {}
        
        # Background tasks
        self._running = False
        self._tasks = []
        
        logger.info("DataEngine initialized with advanced quality features")
    
    def _init_source_priority(self) -> List[str]:
        """Initialize data source priority order"""
        sources = []
        for source, config in self.config['data_quality']['sources'].items():
            if config.get('enabled', False):
                sources.append((source, config.get('priority', 999)))
        
        # Sort by priority (lower number = higher priority)
        sources.sort(key=lambda x: x[1])
        return [s[0] for s in sources]
    
    async def initialize(self):
        """Initialize all data connections"""
        logger.info("Initializing Data Engine...")
        
        # Initialize Redis cache
        await self._init_redis()
        
        # Initialize MT5
        if self.config['execution']['mt5']['enabled']:
            await self._init_mt5()
        
        # Initialize Binance
        if self.config['execution']['binance']['enabled']:
            await self._init_binance()
        
        # Initialize CCXT exchanges
        await self._init_ccxt()
        
        # Initialize additional sources
        await self._init_additional_sources()
        
        # Start background tasks
        self._running = True
        self._tasks.extend([
            asyncio.create_task(self._monitor_data_quality()),
            asyncio.create_task(self._cleanup_cache()),
            asyncio.create_task(self._websocket_manager()),
            asyncio.create_task(self._failover_monitor()),
            asyncio.create_task(self._tick_data_collector())
        ])
        
        logger.info("✅ Data Engine initialized successfully")
    
    async def _init_redis(self):
        """Initialize Redis with connection pool"""
        try:
            self.redis_client = await redis.from_url(
                "redis://localhost:6379",
                decode_responses=False,
                encoding=None,
                retry_on_timeout=True,
                health_check_interval=30,
                max_connections=20
            )
            await self.redis_client.ping()
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
    
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
            self.data_sources_status['mt5'] = True
            logger.info("✅ MT5 connected")
            
        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")
            self.mt5_initialized = False
            self.data_sources_status['mt5'] = False
    
    async def _init_binance(self):
        """Initialize Binance with WebSocket support"""
        try:
            self.binance_client = await AsyncClient.create(
                api_key=self.config['execution']['binance']['api_key'],
                api_secret=self.config['execution']['binance']['api_secret'],
                testnet=self.config['execution']['binance'].get('testnet', False)
            )
            
            if self.config['execution']['binance'].get('websocket', False):
                self.binance_socket_manager = BinanceSocketManager(self.binance_client)
            
            # Initialize futures client if enabled
            if self.config['execution'].get('binance_futures', {}).get('enabled', False):
                self.binance_futures_client = await AsyncClient.create(
                    api_key=self.config['execution']['binance_futures']['api_key'],
                    api_secret=self.config['execution']['binance_futures']['api_secret'],
                    testnet=self.config['execution']['binance_futures'].get('testnet', False)
                )
            
            await self.binance_client.ping()
            self.data_sources_status['binance'] = True
            logger.info("✅ Binance client initialized")
            
        except Exception as e:
            logger.error(f"Binance initialization failed: {e}")
            self.binance_client = None
            self.data_sources_status['binance'] = False
    
    async def _init_ccxt(self):
        """Initialize CCXT exchanges"""
        exchanges = ['binance', 'bybit', 'okx', 'kucoin', 'bitget']
        
        for exchange_id in exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'rateLimit': 1200,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                })
                await exchange.load_markets()
                self.ccxt_exchanges[exchange_id] = exchange
                self.data_sources_status[exchange_id] = True
                logger.info(f"✅ CCXT {exchange_id} initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_id}: {e}")
                self.data_sources_status[exchange_id] = False
    
    async def _init_additional_sources(self):
        """Initialize additional data sources"""
        # Polygon.io
        if self.config['data_quality']['sources'].get('polygon', {}).get('enabled'):
            try:
                import polygon
                self.polygon_client = polygon.RESTClient(
                    api_key=self.config['data_quality']['sources']['polygon']['api_key']
                )
                self.data_sources_status['polygon'] = True
                logger.info("✅ Polygon.io client initialized")
            except Exception as e:
                logger.warning(f"Polygon.io initialization failed: {e}")
                self.data_sources_status['polygon'] = False
        
        # Alpha Vantage
        if self.config['data_quality']['sources'].get('alphavantage', {}).get('enabled'):
            try:
                from alpha_vantage import AlphaVantage
                self.alphavantage_client = AlphaVantage(
                    api_key=self.config['data_quality']['sources']['alphavantage']['api_key']
                )
                self.data_sources_status['alphavantage'] = True
                logger.info("✅ Alpha Vantage client initialized")
            except Exception as e:
                logger.warning(f"Alpha Vantage initialization failed: {e}")
                self.data_sources_status['alphavantage'] = False
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        source: str = "auto",
        quality_check: bool = True,
        fill_gaps: bool = True,
        detect_outliers: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data with advanced quality features
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start: Start datetime
            end: End datetime
            source: Data source (auto, mt5, binance, ccxt, polygon)
            quality_check: Perform quality checks
            fill_gaps: Fill missing data gaps
            detect_outliers: Detect and handle outliers
        
        Returns:
            DataFrame with quality-enhanced data
        """
        # Try multiple sources with failover
        if source == "auto":
            data = await self._get_data_with_failover(symbol, timeframe, start, end)
        else:
            data = await self._get_data_from_source(symbol, timeframe, start, end, source)
        
        if data is None or data.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Perform quality checks
        if quality_check:
            quality_metrics = await self._assess_data_quality(data, symbol)
            self.quality_metrics[symbol] = quality_metrics
            
            if quality_metrics.completeness < 0.5:
                logger.warning(f"Poor data quality for {symbol}: {quality_metrics.completeness:.1%} complete")
        
        # Detect and handle outliers
        if detect_outliers:
            data = await self._detect_and_handle_outliers(data, symbol)
        
        # Fill gaps
        if fill_gaps:
            data = await self._fill_data_gaps(data, timeframe, symbol)
        
        # Cross-validate with other sources
        if quality_check and len(self.source_priority) > 1:
            data = await self._cross_validate_data(data, symbol, timeframe, start, end)
        
        # Cache the data
        await self._cache_data(symbol, timeframe, start, end, data)
        
        logger.info(f"Retrieved {len(data)} bars for {symbol} with quality score: {data.attrs.get('quality_score', 1.0):.2f}")
        
        return data
    
    async def _get_data_with_failover(self, symbol: str, timeframe: str,
                                      start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Try multiple data sources with failover"""
        for source in self.source_priority:
            if self.data_sources_status.get(source, False):
                try:
                    data = await self._get_data_from_source(symbol, timeframe, start, end, source)
                    if data is not None and not data.empty:
                        data.attrs['data_source'] = source
                        logger.info(f"Retrieved data from {source} for {symbol}")
                        return data
                except Exception as e:
                    logger.warning(f"Failed to get data from {source}: {e}")
                    continue
        
        logger.error(f"All data sources failed for {symbol}")
        return None
    
    async def _get_data_from_source(self, symbol: str, timeframe: str,
                                    start: datetime, end: datetime,
                                    source: str) -> Optional[pd.DataFrame]:
        """Get data from specific source"""
        if source == "mt5" and self.mt5_initialized:
            return await self._fetch_mt5_data(symbol, timeframe, start, end)
        elif source == "binance" and self.binance_client:
            return await self._fetch_binance_data(symbol, timeframe, start, end)
        elif source == "binance_futures" and self.binance_futures_client:
            return await self._fetch_binance_futures_data(symbol, timeframe, start, end)
        elif source in self.ccxt_exchanges:
            return await self._fetch_ccxt_data(symbol, timeframe, start, end, self.ccxt_exchanges[source])
        elif source == "polygon" and self.polygon_client:
            return await self._fetch_polygon_data(symbol, timeframe, start, end)
        elif source == "alphavantage" and self.alphavantage_client:
            return await self._fetch_alphavantage_data(symbol, timeframe, start, end)
        
        return None
    
    async def _fetch_mt5_data(self, symbol: str, timeframe: str,
                              start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from MT5"""
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        
        rates = mt5.copy_rates_range(
            symbol,
            tf_map.get(timeframe, mt5.TIMEFRAME_M1),
            start,
            end
        )
        
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        return df
    
    async def _fetch_binance_data(self, symbol: str, timeframe: str,
                                  start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Binance Spot"""
        tf_map = {
            "M1": KLINE_INTERVAL_1MINUTE,
            "M5": KLINE_INTERVAL_5MINUTE,
            "M15": KLINE_INTERVAL_15MINUTE,
            "M30": KLINE_INTERVAL_30MINUTE,
            "H1": KLINE_INTERVAL_1HOUR,
            "H4": KLINE_INTERVAL_4HOUR,
            "D1": KLINE_INTERVAL_1DAY,
        }
        
        klines = await self.binance_client.get_historical_klines(
            symbol,
            tf_map.get(timeframe, KLINE_INTERVAL_1HOUR),
            start.strftime("%d %b %Y %H:%M:%S"),
            end.strftime("%d %b %Y %H:%M:%S") if end else None
        )
        
        if not klines:
            return None
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    async def _fetch_binance_futures_data(self, symbol: str, timeframe: str,
                                         start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Binance Futures"""
        tf_map = {
            "M1": KLINE_INTERVAL_1MINUTE,
            "M5": KLINE_INTERVAL_5MINUTE,
            "M15": KLINE_INTERVAL_15MINUTE,
            "M30": KLINE_INTERVAL_30MINUTE,
            "H1": KLINE_INTERVAL_1HOUR,
            "H4": KLINE_INTERVAL_4HOUR,
            "D1": KLINE_INTERVAL_1DAY,
        }
        
        klines = await self.binance_futures_client.futures_historical_klines(
            symbol,
            tf_map.get(timeframe, KLINE_INTERVAL_1HOUR),
            start.strftime("%d %b %Y %H:%M:%S"),
            limit=1000
        )
        
        if not klines:
            return None
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    async def _fetch_ccxt_data(self, symbol: str, timeframe: str,
                               start: datetime, end: datetime,
                               exchange) -> Optional[pd.DataFrame]:
        """Fetch data from CCXT exchange"""
        since = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000) if end else None
        
        all_ohlcv = []
        current_since = since
        
        while True:
            ohlcv = await exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=1000
            )
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            last_ts = ohlcv[-1][0]
            if end_ts and last_ts >= end_ts:
                all_ohlcv = [c for c in all_ohlcv if c[0] <= end_ts]
                break
            
            if len(ohlcv) < 1000:
                break
            
            current_since = last_ts + 1
            await asyncio.sleep(exchange.rateLimit / 1000)
        
        if not all_ohlcv:
            return None
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    async def _fetch_polygon_data(self, symbol: str, timeframe: str,
                                  start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io"""
        try:
            # Map timeframe to Polygon format
            multiplier, timespan = self._map_polygon_timeframe(timeframe)
            
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                limit=50000
            )
            
            if not aggs:
                return None
            
            df = pd.DataFrame(aggs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }, inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Polygon fetch failed: {e}")
            return None
    
    async def _fetch_alphavantage_data(self, symbol: str, timeframe: str,
                                       start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage"""
        try:
            # Map timeframe
            if timeframe == "D1":
                data = self.alphavantage_client.get_daily(symbol, outputsize='full')
                df = pd.DataFrame(data['Time Series (Daily)']).T
            elif timeframe == "H1":
                data = self.alphavantage_client.get_intraday(symbol, interval='60min', outputsize='full')
                df = pd.DataFrame(data['Time Series (60min)']).T
            else:
                return None
            
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[(df.index >= start) & (df.index <= end)]
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }, inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed: {e}")
            return None
    
    def _map_polygon_timeframe(self, timeframe: str) -> Tuple[int, str]:
        """Map timeframe to Polygon multiplier and timespan"""
        mapping = {
            "M1": (1, "minute"),
            "M5": (5, "minute"),
            "M15": (15, "minute"),
            "M30": (30, "minute"),
            "H1": (1, "hour"),
            "H4": (4, "hour"),
            "D1": (1, "day"),
        }
        return mapping.get(timeframe, (1, "minute"))
    
    async def _assess_data_quality(self, df: pd.DataFrame, symbol: str) -> DataQualityMetrics:
        """Assess comprehensive data quality metrics"""
        # Completeness
        expected_periods = self._calculate_expected_periods(df)
        completeness = len(df) / expected_periods if expected_periods > 0 else 1.0
        
        # Consistency (check for monotonic index)
        index_diff = df.index.to_series().diff().dt.total_seconds()
        expected_diff = self._timeframe_to_seconds(self._infer_timeframe(df))
        consistency = (abs(index_diff - expected_diff) < expected_diff * 0.1).mean()
        
        # Accuracy (outlier detection)
        outlier_count = 0
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                outliers = self._detect_outliers_zscore(df[col])
                outlier_count += outliers.sum()
        accuracy = max(0, 1 - (outlier_count / (len(df) * 4)))
        
        # Timeliness
        last_time = df.index[-1]
        now = datetime.now(last_time.tzinfo) if last_time.tzinfo else datetime.now()
        time_diff = (now - last_time).total_seconds()
        expected_diff = self._timeframe_to_seconds(self._infer_timeframe(df))
        timeliness = max(0, 1 - (time_diff / (expected_diff * 10)))
        
        # Gaps
        gaps = self._detect_gaps(df)
        gap_count = len(gaps)
        max_gap = max([g['duration'] for g in gaps]) if gaps else 0
        
        return DataQualityMetrics(
            symbol=symbol,
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            timeliness=timeliness,
            gap_count=gap_count,
            max_gap_minutes=max_gap / 60,
            outlier_count=outlier_count,
            anomaly_score=1 - (completeness * consistency * accuracy * timeliness),
            cross_source_error=0.0,
            updated_at=datetime.now()
        )
    
    async def _detect_and_handle_outliers(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Detect and handle outliers using multiple methods"""
        df_clean = df.copy()
        outliers_removed = 0
        
        method = self.config['data_quality']['outliers']['method']
        
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                continue
            
            if method == "zscore":
                outliers = self._detect_outliers_zscore(df[col])
            elif method == "iqr":
                outliers = self._detect_outliers_iqr(df[col])
            elif method == "isolation_forest":
                outliers = await self._detect_outliers_isolation_forest(df, col)
            else:
                outliers = pd.Series(False, index=df.index)
            
            if outliers.any():
                # Replace outliers with interpolated values
                df_clean.loc[outliers, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
                outliers_removed += outliers.sum()
        
        df_clean.attrs['outliers_removed'] = outliers_removed
        return df_clean
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method"""
        zscore