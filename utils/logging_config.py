"""
Logging Configuration - Structured logging setup for production with rotation,
contextual logging, and performance optimizations.
"""

import logging
import logging.config
import logging.handlers
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
import traceback
import socket
import threading
from queue import Queue
import atexit
import time
import re
from dataclasses import dataclass, field
from enum import Enum

# Optional imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


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


class LogOutput(Enum):
    """Log output destinations"""
    CONSOLE = 'console'
    FILE = 'file'
    JSON_FILE = 'json_file'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LogContext:
    """Context information for logging"""
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    hostname: str = field(default_factory=socket.gethostname)
    pid: int = field(default_factory=os.getpid)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'component': self.component,
            'environment': self.environment,
            'hostname': self.hostname,
            'pid': self.pid
        }


# =============================================================================
# Custom Formatters
# =============================================================================

class JsonFormatter(logging.Formatter):
    """Enhanced JSON formatter with context and metadata"""
    
    def __init__(self, *args, include_context: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_context = include_context
        self._context_local = threading.local()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_record = {
            '@timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.threadName,
            'hostname': socket.gethostname()
        }
        
        # Add custom fields
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id
        if hasattr(record, 'component'):
            log_record['component'] = record.component
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        if hasattr(record, 'extra_data'):
            log_record['extra'] = record.extra_data
        
        # Add exception info
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add performance metrics
        if hasattr(record, 'duration_ms'):
            log_record['duration_ms'] = record.duration_ms
        
        # Add context
        if self.include_context and hasattr(self._context_local, 'context'):
            log_record['context'] = self._context_local.context
        
        return json.dumps(log_record, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with colors for better readability"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'TRADE': '\033[34m',     # Blue
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors"""
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        original_levelname = record.levelname
        record.levelname = f"{level_color}{record.levelname}{reset}"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        timestamp = f"\033[90m{timestamp}\033[0m"
        
        # Format message
        message = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        return f"{timestamp} {record.levelname} {message}"


# =============================================================================
# Custom Handlers
# =============================================================================

class AsyncRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Asynchronous file handler with buffering for better performance"""
    
    def __init__(self, *args, buffer_size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_buffer, daemon=True)
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
        except Exception:
            # Fallback to stderr
            print(f"Log write failed: {record.getMessage()}", file=sys.stderr)
    
    def close(self):
        """Close handler and flush buffer"""
        self._stop_event.set()
        
        # Write remaining records
        remaining = []
        while not self.buffer.empty():
            try:
                remaining.append(self.buffer.get_nowait())
            except:
                break
        
        for record in remaining:
            self._write_record(record)
        
        super().close()


# =============================================================================
# Context Manager for Logging
# =============================================================================

class LogContext:
    """Context manager for adding context to logs"""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self._previous_context = {}
        
        # Find JsonFormatter instances
        self._json_formatters = []
        self._find_json_formatters(logger)
    
    def _find_json_formatters(self, logger):
        """Find all JsonFormatter instances in logger handlers"""
        for handler in logger.handlers:
            if isinstance(handler.formatter, JsonFormatter):
                self._json_formatters.append(handler.formatter)
        
        # Check parent loggers
        if logger.parent:
            self._find_json_formatters(logger.parent)
    
    def __enter__(self):
        # Set context in all JsonFormatter instances
        for formatter in self._json_formatters:
            if hasattr(formatter, '_context_local'):
                if hasattr(formatter._context_local, 'context'):
                    self._previous_context[formatter] = formatter._context_local.context.copy()
                    formatter._context_local.context.update(self.context)
                else:
                    formatter._context_local.context = self.context.copy()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        for formatter in self._json_formatters:
            if hasattr(formatter, '_context_local'):
                if formatter in self._previous_context:
                    formatter._context_local.context = self._previous_context[formatter]
                else:
                    if hasattr(formatter._context_local, 'context'):
                        delattr(formatter._context_local, 'context')


# =============================================================================
# Sensitive Data Filter
# =============================================================================

class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in logs"""
    
    SENSITIVE_PATTERNS = [
        (r'password["\s]*[:=][\s]*["\']?([^"\'}\s]+)', '[PASSWORD]'),
        (r'token["\s]*[:=][\s]*["\']?([^"\'}\s]+)', '[TOKEN]'),
        (r'api[_-]key["\s]*[:=][\s]*["\']?([^"\'}\s]+)', '[API_KEY]'),
        (r'secret["\s]*[:=][\s]*["\']?([^"\'}\s]+)', '[SECRET]'),
        (r'authorization["\s]*[:=][\s]*["\']?([^"\'}\s]+)', '[AUTH]'),
        (r'credit[_-]card[^0-9]*([0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4})', '[CREDIT_CARD]'),
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
        if hasattr(record, 'extra_data') and isinstance(record.extra_data, dict):
            self._mask_dict(record.extra_data)
        
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
# Log Sampling Filter
# =============================================================================

class SampledLogFilter(logging.Filter):
    """Filter for sampling high-frequency logs"""
    
    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = min(1.0, max(0.0, sample_rate))
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if log should be sampled"""
        # Always keep errors and above
        if record.levelno >= logging.ERROR:
            return True
        
        # Use hash-based sampling for consistency
        import hashlib
        key = f"{record.name}:{record.module}:{record.funcName}"
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        
        return (hash_val % 100) < (self.sample_rate * 100)


# =============================================================================
# Main Setup Function
# =============================================================================

def setup_logging(config: Optional[Dict[str, Any]] = None, 
                  config_file: Optional[Path] = None,
                  app_name: str = 'trading_bot') -> logging.Logger:
    """
    Setup structured logging with console and rotating file handlers
    
    Args:
        config: Logging configuration dictionary
        config_file: Path to configuration file (YAML or JSON)
        app_name: Application name for log file naming
    
    Returns:
        Root logger instance
    """
    # Load configuration
    if config_file and config_file.exists():
        config = _load_config_file(config_file)
    
    # Default configuration
    if config is None:
        config = {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': os.getenv('LOG_FORMAT', 'json'),
            'outputs': {
                'console': os.getenv('LOG_CONSOLE', 'true').lower() == 'true',
                'file': os.getenv('LOG_FILE', 'true').lower() == 'true'
            },
            'file': {
                'max_bytes': int(os.getenv('LOG_MAX_BYTES', 10485760)),  # 10MB
                'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 10)),
                'buffer_size': int(os.getenv('LOG_BUFFER_SIZE', 100))
            },
            'filters': {
                'mask_sensitive': os.getenv('LOG_MASK_SENSITIVE', 'true').lower() == 'true',
                'sample_rate': float(os.getenv('LOG_SAMPLE_RATE', 1.0))
            }
        }
    
    log_level = getattr(logging, config.get('level', 'INFO'))
    log_format = config.get('format', 'json')
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add custom log level
    logging.addLevelName(LogLevel.TRADE.value, 'TRADE')
    
    # Create formatters
    json_formatter = JsonFormatter()
    console_formatter = (
        JsonFormatter() if log_format == 'json' 
        else ColoredConsoleFormatter('%(message)s')
    )
    
    # Console handler
    if config.get('outputs', {}).get('console', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        
        # Add filters
        if config.get('filters', {}).get('mask_sensitive', True):
            console_handler.addFilter(SensitiveDataFilter())
        
        if config.get('filters', {}).get('sample_rate', 1.0) < 1.0:
            console_handler.addFilter(SampledLogFilter(
                config['filters']['sample_rate']
            ))
        
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.get('outputs', {}).get('file', True):
        log_file = log_dir / f'{app_name}.log'
        file_config = config.get('file', {})
        
        file_handler = AsyncRotatingFileHandler(
            filename=log_file,
            maxBytes=file_config.get('max_bytes', 10485760),
            backupCount=file_config.get('backup_count', 10),
            buffer_size=file_config.get('buffer_size', 100)
        )
        file_handler.setFormatter(json_formatter)
        
        # Add filters
        if config.get('filters', {}).get('mask_sensitive', True):
            file_handler.addFilter(SensitiveDataFilter())
        
        root_logger.addHandler(file_handler)
    
    # Separate error log
    error_file = log_dir / f'{app_name}_error.log'
    error_handler = AsyncRotatingFileHandler(
        filename=error_file,
        maxBytes=10485760,
        backupCount=5,
        buffer_size=50
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    root_logger.addHandler(error_handler)
    
    # Set level for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info(
        f"Logging configured", 
        extra={
            'extra_data': {
                'level': logging.getLevelName(log_level),
                'format': log_format,
                'outputs': list(config.get('outputs', {}).keys()),
                'filters': config.get('filters', {})
            }
        }
    )
    
    # Add context manager to root logger
    root_logger.context = lambda **ctx: LogContext(root_logger, **ctx)
    
    return root_logger


def _load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml'] and YAML_AVAILABLE:
                return yaml.safe_load(f)
            elif config_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    except Exception as e:
        print(f"Error loading logging config: {e}", file=sys.stderr)
        return {}


# =============================================================================
# Convenience Functions
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)


def log_trade(logger: logging.Logger, trade_data: Dict[str, Any]):
    """Log a trade event"""
    logger.log(
        LogLevel.TRADE.value,
        f"Trade: {trade_data.get('symbol', 'Unknown')} {trade_data.get('action', '')}",
        extra={'extra_data': trade_data, 'component': 'trading'}
    )


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, **kwargs):
    """Log performance metrics"""
    logger.info(
        f"Performance: {operation}",
        extra={
            'extra_data': {
                'operation': operation,
                'duration_ms': duration_ms,
                **kwargs
            },
            'duration_ms': duration_ms,
            'component': 'performance'
        }
    )


# =============================================================================
# Cleanup
# =============================================================================

def cleanup():
    """Cleanup logging handlers on exit"""
    for handler in logging.root.handlers[:]:
        try:
            handler.close()
            logging.root.removeHandler(handler)
        except:
            pass


atexit.register(cleanup)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'setup_logging',
    'get_logger',
    'log_trade',
    'log_performance',
    'LogLevel',
    'LogOutput',
    'LogContext',
    'JsonFormatter',
    'ColoredConsoleFormatter',
    'AsyncRotatingFileHandler',
    'SensitiveDataFilter',
    'SampledLogFilter'
]