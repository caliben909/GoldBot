"""
Logging Configuration - Structured logging setup for production
"""
import logging
import logging.config
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_record['extra'] = record.extra
        
        return json.dumps(log_record)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup structured logging with console and file handlers
    
    Args:
        config: Logging configuration dictionary
    
    Returns:
        Root logger instance
    """
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
    
    # Console handler
    if config.get('outputs', {}).get('console', True):
        console_handler = logging.StreamHandler(sys.stdout)
        if log_format == 'json':
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        root_logger.addHandler(console_handler)
    
    # File handler
    if config.get('outputs', {}).get('file', True):
        log_file = log_dir / 'trading_bot.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)
    
    # Set level for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, format={log_format}")
    
    return root_logger