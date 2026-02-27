"""
Logger - Comprehensive structured logging with multiple outputs, contextual logging,
log aggregation support, and performance optimizations.
"""

import logging
import logging.config
import logging.handlers
import json
import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import traceback
import uuid
import socket
import os
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import atexit
import time
import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Optional imports for advanced features
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

try:
    import elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False


# =============================================================================
# Enums and Constants
# =============================================================================

class LogLevel(Enum):
    """Log levels with numeric values"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    TRADE = 15  # Custom level between INFO and DEBUG
    ORDER = 16
    POSITION = 17


class LogComponent(Enum):
    """System components for targeted logging"""
    SYSTEM = 'system'
    TRADING = 'trading'
    RISK = 'risk'
    DATA = 'data'
    STRATEGY = 'strategy'
    BACKTEST = 'backtest'
    API = 'api'
    DATABASE = 'database'
    CACHE = 'cache'
    NETWORK = 'network'


class LogOutput(Enum):
    """Log output destinations"""
    CONSOLE = 'console'
    FILE = 'file'
    SYSLOG = 'syslog'
    ELASTICSEARCH = 'elasticsearch'
    SENTRY = 'sentry'
    SLACK = 'slack'
    DATADOG = 'datadog'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LogContext:
    """Context information for logging"""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[LogComponent] = None
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    hostname: str = field(default_factory=socket.gethostname)
    pid: int = field(default_factory=os.getpid)
    thread_id: int = field(default_factory=threading.get_ident)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'component': self.component.value if self.component else None,
            'environment': self.environment,
            'hostname': self.hostname,
            'pid': self.pid,
            'thread_id': self.thread_id
        }


@dataclass
class LogEvent:
    """Structured log event"""
    timestamp: datetime
    level: str
    message: str
    logger: str
    module: str
    function: str
    line: int
    context: LogContext
    data: Optional[Dict[str, Any]] = None
    exception: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            '@timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'message': self.message,
            'logger': self.logger,
            'module': self.module,
            'function': self.function,
            'line': self.line,
            **self.context.to_dict()
        }
        
        if self.data:
            result['data'] = self.data
        if self.exception:
            result['exception'] = self.exception
        if self.duration_ms is not None:
            result['duration_ms'] = self.duration_ms
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


# =============================================================================
# Custom Formatters
# =============================================================================

class CustomJsonFormatter(logging.Formatter):
    """Enhanced JSON formatter with context and metadata"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_local = threading.local()
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_record = {
            '@timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
            'hostname': socket.gethostname()
        }
        
        # Add custom fields from record
        if hasattr(record, 'data'):
            log_record['data'] = record.data
        if hasattr(record, 'component'):
            log_record['component'] = record.component
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id
        
        # Add exception info
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add context from thread local
        if hasattr(self._context_local, 'context'):
            log_record.update(self._context_local.context)
        
        return json.dumps(log_record, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with colors"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors"""
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        record.levelname = f"{level_color}{record.levelname}{reset}"
        
        # Add timestamp in gray
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        timestamp = f"\033[90m{timestamp}\033[0m"
        
        # Add component if present
        component = ''
        if hasattr(record, 'component'):
            component = f" \033[33m[{record.component}]\033[0m"
        
        # Add correlation ID if present
        corr_id = ''
        if hasattr(record, 'correlation_id'):
            corr_id = f" \033[90m[{record.correlation_id[:8]}]\033[0m"
        
        return f"{timestamp}{component}{corr_id} - {record.levelname} - {record.getMessage()}"


# =============================================================================
# Custom Handlers
# =============================================================================

class AsyncRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Asynchronous file handler with buffering"""
    
    def __init__(self, *args, buffer_size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = Queue(maxsize=buffer_size)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_buffer)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        atexit.register(self.close)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record asynchronously"""
        try:
            # Try to put in buffer without blocking
            self.buffer.put_nowait(record)
        except:
            # Buffer full, write directly
            self._write_record(record)
    
    def _process_buffer(self):
        """Process buffer in background thread"""
        while not self._stop_event.is_set():
            try:
                record = self.buffer.get(timeout=0.1)
                self._write_record(record)
            except:
                continue
    
    def _write_record(self, record):
        """Write record to file"""
        try:
            super().emit(record)
        except Exception as e:
            print(f"Error writing log: {e}", file=sys.stderr)
    
    def close(self):
        """Close handler and flush buffer"""
        self._stop_event.set()
        
        # Write remaining records
        while not self.buffer.empty():
            try:
                record = self.buffer.get_nowait()
                self._write_record(record)
            except:
                break
        
        self.executor.shutdown(wait=True)
        super().close()


class ElasticsearchHandler(logging.Handler):
    """Handler for sending logs to Elasticsearch"""
    
    def __init__(self, hosts: List[str], index_prefix: str = 'logs',
                 buffer_size: int = 100, flush_interval: int = 5):
        super().__init__()
        
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("elasticsearch module is required")
        
        self.es = elasticsearch.Elasticsearch(hosts)
        self.index_prefix = index_prefix
        self.buffer = []
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._last_flush = time.time()
        self._lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to Elasticsearch"""
        try:
            log_entry = self.format(record)
            if isinstance(log_entry, str):
                log_entry = json.loads(log_entry)
            
            with self._lock:
                self.buffer.append(log_entry)
                
                # Flush if buffer full or interval elapsed
                if (len(self.buffer) >= self.buffer_size or 
                    time.time() - self._last_flush >= self.flush_interval):
                    self.flush()
        except Exception as e:
            self.handleError(record)
    
    def flush(self):
        """Flush buffer to Elasticsearch"""
        if not self.buffer:
            return
        
        try:
            index_name = f"{self.index_prefix}-{datetime.now():%Y.%m.%d}"
            
            # Prepare bulk operations
            actions = []
            for doc in self.buffer:
                actions.append({'index': {'_index': index_name}})
                actions.append(doc)
            
            if actions:
                self.es.bulk(body=actions, refresh=False)
            
            self.buffer.clear()
            self._last_flush = time.time()
        except Exception as e:
            print(f"Error flushing to Elasticsearch: {e}", file=sys.stderr)


class SlackHandler(logging.Handler):
    """Handler for sending critical logs to Slack"""
    
    def __init__(self, webhook_url: str, min_level: str = 'ERROR'):
        super().__init__()
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests module is required")
        
        self.webhook_url = webhook_url
        self.setLevel(getattr(logging, min_level))
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to Slack"""
        try:
            if record.levelno < self.level:
                return
            
            # Format message
            message = self.format(record)
            
            # Prepare Slack payload
            color = {
                'ERROR': 'danger',
                'CRITICAL': 'danger',
                'WARNING': 'warning'
            }.get(record.levelname, 'good')
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': f"Log Alert: {record.levelname}",
                    'text': message[:4000],  # Slack limit
                    'fields': [
                        {'title': 'Logger', 'value': record.name, 'short': True},
                        {'title': 'Module', 'value': record.module, 'short': True}
                    ],
                    'ts': int(record.created)
                }]
            }
            
            # Send asynchronously
            threading.Thread(
                target=self._send_slack,
                args=(payload,),
                daemon=True
            ).start()
            
        except Exception as e:
            self.handleError(record)
    
    def _send_slack(self, payload: Dict):
        """Send message to Slack"""
        try:
            requests.post(self.webhook_url, json=payload, timeout=2)
        except:
            pass


# =============================================================================
# Context Manager for Logging
# =============================================================================

class LogContextManager:
    """Context manager for adding context to logs"""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self._old_context = {}
    
    def __enter__(self):
        # Store existing context
        if hasattr(self.logger, '_context'):
            self._old_context = self.logger._context.copy()
            self.logger._context.update(self.context)
        else:
            self.logger._context = self.context.copy()
        
        # Update handlers with context
        for handler in self.logger.handlers:
            if hasattr(handler, 'formatter') and hasattr(handler.formatter, '_context_local'):
                handler.formatter._context_local.context = self.logger._context
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old context
        if hasattr(self.logger, '_context'):
            if self._old_context:
                self.logger._context = self._old_context
            else:
                delattr(self.logger, '_context')
            
            # Update handlers
            for handler in self.logger.handlers:
                if hasattr(handler, 'formatter') and hasattr(handler.formatter, '_context_local'):
                    if hasattr(self.logger, '_context'):
                        handler.formatter._context_local.context = self.logger._context
                    else:
                        handler.formatter._context_local.context = {}


# =============================================================================
# Performance Monitoring
# =============================================================================

class LogTimer:
    """Context manager for timing operations"""
    
    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.perf_counter() - self.start_time) * 1000  # ms
        
        extra = {
            'operation': self.operation,
            'duration_ms': duration,
            **self.kwargs
        }
        
        if exc_type:
            extra['error'] = exc_type.__name__
            self.logger.error(f"Operation {self.operation} failed", extra=extra)
        else:
            self.logger.info(f"Operation {self.operation} completed", extra=extra)


# =============================================================================
# Main Logger Configuration
# =============================================================================

class LoggerConfig:
    """Configuration for logger setup"""
    
    DEFAULT_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': 'utils.logger.CustomJsonFormatter'
            },
            'colored': {
                '()': 'utils.logger.ColoredConsoleFormatter'
            },
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'colored',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'utils.logger.AsyncRotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'json',
                'filename': 'logs/trading_bot.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 20,
                'buffer_size': 100
            },
            'error_file': {
                'class': 'utils.logger.AsyncRotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json',
                'filename': 'logs/errors.log',
                'maxBytes': 10485760,
                'backupCount': 10,
                'buffer_size': 50
            },
            'trade_file': {
                'class': 'utils.logger.AsyncRotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': 'logs/trades.log',
                'maxBytes': 10485760,
                'backupCount': 30,
                'buffer_size': 200
            },
            'performance_file': {
                'class': 'utils.logger.AsyncRotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': 'logs/performance.log',
                'maxBytes': 10485760,
                'backupCount': 10,
                'buffer_size': 100
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file', 'error_file'],
                'level': 'INFO',
                'propagate': True
            },
            'trading': {
                'handlers': ['console', 'file', 'trade_file'],
                'level': 'INFO',
                'propagate': False
            },
            'risk': {
                'handlers': ['console', 'file', 'error_file'],
                'level': 'INFO',
                'propagate': False
            },
            'data': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'performance': {
                'handlers': ['performance_file'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and config_path.exists():
            self.load_config(config_path)
    
    def load_config(self, config_path: Path):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    user_config = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    user_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
                
                self._merge_config(user_config)
        except Exception as e:
            print(f"Error loading logging config: {e}", file=sys.stderr)
    
    def _merge_config(self, user_config: Dict):
        """Merge user configuration with defaults"""
        for key, value in user_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def add_handler(self, name: str, handler_config: Dict):
        """Add custom handler"""
        self.config['handlers'][name] = handler_config
    
    def add_logger(self, name: str, logger_config: Dict):
        """Add custom logger"""
        self.config['loggers'][name] = logger_config
    
    def apply(self):
        """Apply logging configuration"""
        logging.config.dictConfig(self.config)
        
        # Add custom log levels
        logging.addLevelName(LogLevel.TRADE.value, 'TRADE')
        logging.addLevelName(LogLevel.ORDER.value, 'ORDER')
        logging.addLevelName(LogLevel.POSITION.value, 'POSITION')
        
        # Set levels for noisy libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('websockets').setLevel(logging.WARNING)
        logging.getLogger('elasticsearch').setLevel(logging.WARNING)


# =============================================================================
# Main Logger Setup Function
# =============================================================================

def setup_logger(name: str, config: Optional[Dict[str, Any]] = None,
                 config_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup structured logger with multiple outputs
    
    Args:
        name: Logger name
        config: Optional configuration dictionary
        config_file: Optional configuration file path
    
    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Setup configuration
    logger_config = LoggerConfig(config_file)
    
    if config:
        logger_config._merge_config(config)
    
    # Apply configuration
    logger_config.apply()
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Add custom methods
    logger = _add_custom_methods(logger)
    
    logger.info(f"Logger configured for {name} in {logger_config.config['loggers']['']['level']} mode")
    
    return logger


def _add_custom_methods(logger: logging.Logger) -> logging.Logger:
    """Add custom logging methods"""
    
    def log_with_context(level, msg, *args, **kwargs):
        """Log with context"""
        extra = kwargs.pop('extra', {})
        
        # Add component if specified
        if 'component' in kwargs:
            extra['component'] = kwargs.pop('component')
        
        # Add correlation ID if specified
        if 'correlation_id' in kwargs:
            extra['correlation_id'] = kwargs.pop('correlation_id')
        
        # Add data
        if 'data' in kwargs:
            extra['data'] = kwargs.pop('data')
        
        logger.log(level, msg, *args, extra=extra, **kwargs)
    
    logger.debug_with_context = lambda msg, **kwargs: log_with_context(logging.DEBUG, msg, **kwargs)
    logger.info_with_context = lambda msg, **kwargs: log_with_context(logging.INFO, msg, **kwargs)
    logger.warning_with_context = lambda msg, **kwargs: log_with_context(logging.WARNING, msg, **kwargs)
    logger.error_with_context = lambda msg, **kwargs: log_with_context(logging.ERROR, msg, **kwargs)
    
    # Add component-specific methods
    def log_trade(self, msg, **kwargs):
        """Log trade event"""
        if 'data' not in kwargs:
            kwargs['data'] = {}
        kwargs['data']['type'] = 'trade'
        self.info_with_context(msg, component='trading', **kwargs)
    
    def log_order(self, msg, **kwargs):
        """Log order event"""
        if 'data' not in kwargs:
            kwargs['data'] = {}
        kwargs['data']['type'] = 'order'
        self.info_with_context(msg, component='trading', **kwargs)
    
    def log_position(self, msg, **kwargs):
        """Log position event"""
        if 'data' not in kwargs:
            kwargs['data'] = {}
        kwargs['data']['type'] = 'position'
        self.info_with_context(msg, component='trading', **kwargs)
    
    logger.log_trade = log_trade.__get__(logger)
    logger.log_order = log_order.__get__(logger)
    logger.log_position = log_position.__get__(logger)
    
    # Add performance monitoring
    logger.timer = lambda operation, **kwargs: LogTimer(logger, operation, **kwargs)
    
    # Add context manager
    logger.context = lambda **kwargs: LogContextManager(logger, **kwargs)
    
    return logger


# =============================================================================
# Specialized Loggers
# =============================================================================

class TradeLogger:
    """Specialized logger for trade events with analytics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._trade_count = 0
        self._win_count = 0
        self._loss_count = 0
        self._total_pnl = 0.0
        self._lock = threading.Lock()
    
    def log_trade_open(self, trade_data: Dict[str, Any]):
        """Log trade open event"""
        with self._lock:
            self._trade_count += 1
        
        self.logger.log_trade("Trade opened", data={
            'action': 'open',
            'trade_id': trade_data.get('id'),
            'symbol': trade_data.get('symbol'),
            'direction': trade_data.get('direction'),
            'entry_price': trade_data.get('price'),
            'size': trade_data.get('size'),
            'stop_loss': trade_data.get('stop_loss'),
            'take_profit': trade_data.get('take_profit'),
            **trade_data
        })
    
    def log_trade_close(self, trade_data: Dict[str, Any]):
        """Log trade close event"""
        pnl = trade_data.get('pnl', 0)
        
        with self._lock:
            self._total_pnl += pnl
            if pnl > 0:
                self._win_count += 1
            elif pnl < 0:
                self._loss_count += 1
        
        self.logger.log_trade("Trade closed", data={
            'action': 'close',
            'trade_id': trade_data.get('id'),
            'symbol': trade_data.get('symbol'),
            'exit_price': trade_data.get('price'),
            'pnl': pnl,
            'pnl_percent': trade_data.get('pnl_percent'),
            'exit_reason': trade_data.get('exit_reason'),
            'duration': trade_data.get('duration'),
            **trade_data
        })
    
    def log_order(self, order_data: Dict[str, Any]):
        """Log order event"""
        self.logger.log_order("Order placed", data=order_data)
    
    def log_position_update(self, position_data: Dict[str, Any]):
        """Log position update"""
        self.logger.log_position("Position updated", data=position_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trade statistics"""
        with self._lock:
            total_trades = self._trade_count
            win_rate = (self._win_count / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'wins': self._win_count,
                'losses': self._loss_count,
                'win_rate': win_rate,
                'total_pnl': self._total_pnl,
                'avg_pnl': self._total_pnl / total_trades if total_trades > 0 else 0
            }


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logging.getLogger('performance')
        self._metrics = {}
        self._lock = threading.Lock()
    
    def log_metric(self, name: str, value: float, tags: Optional[Dict] = None):
        """Log a performance metric"""
        with self._lock:
            self._metrics[name] = {
                'value': value,
                'timestamp': datetime.utcnow().isoformat(),
                'tags': tags or {}
            }
        
        self.logger.info(f"Metric: {name}", extra={
            'data': {
                'metric': name,
                'value': value,
                'tags': tags
            }
        })
    
    def log_timing(self, name: str, duration_ms: float, tags: Optional[Dict] = None):
        """Log timing metric"""
        self.log_metric(f"{name}_ms", duration_ms, tags)
    
    def log_count(self, name: str, increment: int = 1, tags: Optional[Dict] = None):
        """Log count metric"""
        self.log_metric(name, increment, tags)
    
    def log_gauge(self, name: str, value: float, tags: Optional[Dict] = None):
        """Log gauge metric"""
        self.log_metric(name, value, tags)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            return self._metrics.copy()


# =============================================================================
# Log Aggregator
# =============================================================================

class LogAggregator:
    """Aggregate logs from multiple sources"""
    
    def __init__(self):
        self.handlers = []
        self._buffer = []
        self._lock = threading.Lock()
    
    def add_handler(self, handler: logging.Handler):
        """Add log handler"""
        self.handlers.append(handler)
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to all handlers"""
        with self._lock:
            for handler in self.handlers:
                try:
                    handler.emit(record)
                except Exception as e:
                    print(f"Error in log handler: {e}", file=sys.stderr)
    
    def flush(self):
        """Flush all handlers"""
        for handler in self.handlers:
            try:
                handler.flush()
            except Exception as e:
                print(f"Error flushing handler: {e}", file=sys.stderr)


# =============================================================================
# Sensitive Data Masking
# =============================================================================

class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in logs"""
    
    SENSITIVE_PATTERNS = [
        (r'password["\s]*[:=][\s]*["\']?([^"\'}\s]+)', 'password=[MASKED]'),
        (r'token["\s]*[:=][\s]*["\']?([^"\'}\s]+)', 'token=[MASKED]'),
        (r'api[_-]key["\s]*[:=][\s]*["\']?([^"\'}\s]+)', 'api_key=[MASKED]'),
        (r'secret["\s]*[:=][\s]*["\']?([^"\'}\s]+)', 'secret=[MASKED]'),
        (r'authorization["\s]*[:=][\s]*["\']?([^"\'}\s]+)', 'authorization=[MASKED]'),
        (r'credit[_-]card[^0-9]*([0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4})', 'credit_card=[MASKED]'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and mask sensitive data"""
        if isinstance(record.msg, str):
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                record.msg = re.sub(pattern, replacement, record.msg, flags=re.IGNORECASE)
        
        # Mask in args
        if record.args:
            masked_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.SENSITIVE_PATTERNS:
                        arg = re.sub(pattern, replacement, arg, flags=re.IGNORECASE)
                masked_args.append(arg)
            record.args = tuple(masked_args)
        
        # Mask in extra data
        if hasattr(record, 'data') and isinstance(record.data, dict):
            self._mask_dict(record.data)
        
        return True
    
    def _mask_dict(self, data: Dict):
        """Recursively mask sensitive data in dictionary"""
        sensitive_keys = ['password', 'token', 'api_key', 'secret', 'authorization', 'credit_card']
        
        for key, value in data.items():
            if any(sk in key.lower() for sk in sensitive_keys):
                data[key] = '[MASKED]'
            elif isinstance(value, dict):
                self._mask_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._mask_dict(item)


# =============================================================================
# Log Sampling
# =============================================================================

class SampledLogFilter(logging.Filter):
    """Filter for sampling high-frequency logs"""
    
    def __init__(self, sample_rate: float = 0.1, loggers: List[str] = None):
        self.sample_rate = sample_rate
        self.loggers = loggers or []
        self._counters = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if log should be sampled"""
        if self.loggers and record.name not in self.loggers:
            return True
        
        # Use deterministic sampling based on hash
        key = f"{record.name}:{record.module}:{record.funcName}"
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        return (hash_val % 100) < (self.sample_rate * 100)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get or create a logger"""
    logger = logging.getLogger(name)
    return _add_custom_methods(logger)


def get_trade_logger(name: str = 'trading') -> TradeLogger:
    """Get trade logger instance"""
    logger = get_logger(name)
    return TradeLogger(logger)


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance"""
    logger = logging.getLogger('performance')
    return PerformanceLogger(logger)


# =============================================================================
# Cleanup
# =============================================================================

def cleanup_loggers():
    """Cleanup loggers on exit"""
    for handler in logging.root.handlers[:]:
        try:
            handler.close()
            logging.root.removeHandler(handler)
        except:
            pass


atexit.register(cleanup_loggers)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Main setup
    'setup_logger',
    'get_logger',
    'get_trade_logger',
    'get_performance_logger',
    
    # Enums
    'LogLevel',
    'LogComponent',
    'LogOutput',
    
    # Data classes
    'LogContext',
    'LogEvent',
    
    # Context managers
    'LogContextManager',
    'LogTimer',
    
    # Specialized loggers
    'TradeLogger',
    'PerformanceLogger',
    
    # Configuration
    'LoggerConfig',
    
    # Filters and handlers
    'SensitiveDataFilter',
    'SampledLogFilter',
    'ElasticsearchHandler',
    'SlackHandler',
    'AsyncRotatingFileHandler',
    'CustomJsonFormatter',
    'ColoredConsoleFormatter',
    
    # Aggregator
    'LogAggregator',
    
    # Cleanup
    'cleanup_loggers'
]