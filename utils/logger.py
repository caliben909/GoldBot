"""
Logger - Structured logging with multiple outputs
"""
import logging
import logging.config
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }


def setup_logger(name: str, config: Dict[str, Any]) -> logging.Logger:
    """
    Setup structured logger with multiple outputs
    
    Args:
        name: Logger name
        config: Logging configuration
    
    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': CustomJsonFormatter,
                'format': '%(timestamp)s %(level)s %(name)s %(message)s'
            },
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': config.get('level', 'INFO'),
                'formatter': 'standard',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': config.get('level', 'INFO'),
                'formatter': 'json',
                'filename': log_dir / 'trading_bot.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json',
                'filename': log_dir / 'errors.log',
                'maxBytes': 10485760,
                'backupCount': 10
            },
            'trade_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': log_dir / 'trades.json',
                'maxBytes': 10485760,
                'backupCount': 10
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file', 'error_file'],
                'level': config.get('level', 'INFO'),
                'propagate': True
            },
            'trades': {
                'handlers': ['trade_file'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(log_config)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    
    logger.info(f"Logger configured for {name}")
    
    return logger


class TradeLogger:
    """Specialized logger for trade events"""
    
    def __init__(self):
        self.logger = logging.getLogger('trades')
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade event"""
        self.logger.info(json.dumps({
            'type': 'trade',
            'data': trade_data,
            'timestamp': datetime.utcnow().isoformat()
        }))
    
    def log_order(self, order_data: Dict[str, Any]):
        """Log order event"""
        self.logger.info(json.dumps({
            'type': 'order',
            'data': order_data,
            'timestamp': datetime.utcnow().isoformat()
        }))
    
    def log_position(self, position_data: Dict[str, Any]):
        """Log position update"""
        self.logger.info(json.dumps({
            'type': 'position',
            'data': position_data,
            'timestamp': datetime.utcnow().isoformat()
        }))