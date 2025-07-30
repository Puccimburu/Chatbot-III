"""
Logging configuration for Conversational Analytics
Setup structured logging with different handlers and formatters
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path

from config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for production logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process', 
                          'message', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return str(log_entry)


def setup_logging():
    """Setup logging configuration for the application"""
    
    # Create logs directory if it doesn't exist
    if settings.ENABLE_FILE_LOGGING:
        log_dir = Path(settings.LOG_FILE_PATH).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logging level
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    if settings.DEBUG:
        # Colored formatter for development
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Simple formatter for production console
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for persistent logging
    if settings.ENABLE_FILE_LOGGING:
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE_PATH,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        if settings.DEBUG:
            # Detailed formatter for development
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
        else:
            # Structured JSON formatter for production
            file_formatter = StructuredFormatter()
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Error file handler for errors and above
    if settings.ENABLE_FILE_LOGGING:
        error_file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE_PATH.replace('.log', '.error.log'),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(module)s:%(funcName)s:%(lineno)d - %(message)s\n'
                '%(exc_info)s'
            )
        )
        root_logger.addHandler(error_file_handler)
    
    # Configure specific loggers
    configure_specific_loggers()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("🚀 Logging system initialized")
    logger.info(f"📊 Log level: {settings.LOG_LEVEL}")
    if settings.ENABLE_FILE_LOGGING:
        logger.info(f"📁 File logging: {settings.LOG_FILE_PATH}")


def configure_specific_loggers():
    """Configure logging levels for specific modules"""
    
    # Reduce noise from external libraries
    logging.getLogger('motor').setLevel(logging.WARNING)
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Set appropriate levels for our modules
    if not settings.DEBUG:
        logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
        logging.getLogger('fastapi').setLevel(logging.WARNING)
    
    # Enable detailed logging for our core modules in debug mode
    if settings.ENABLE_DETAILED_LOGGING:
        logging.getLogger('core').setLevel(logging.DEBUG)
        logging.getLogger('services').setLevel(logging.DEBUG)
        logging.getLogger('api').setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


def log_performance(func_name: str, duration_ms: float, extra_info: dict = None):
    """Log performance information"""
    logger = logging.getLogger('performance')
    
    message = f"⚡ {func_name} completed in {duration_ms:.2f}ms"
    
    if extra_info:
        message += f" - {extra_info}"
    
    if duration_ms > 1000:  # Log as warning if > 1 second
        logger.warning(message)
    elif duration_ms > 500:  # Log as info if > 500ms
        logger.info(message)
    else:
        logger.debug(message)


def log_schema_detection(collection_name: str, field_count: int, sample_size: int, duration_ms: float):
    """Log schema detection activity"""
    logger = logging.getLogger('schema_detection')
    logger.info(
        f"🔍 Schema detected for '{collection_name}': "
        f"{field_count} fields, {sample_size} samples, {duration_ms:.2f}ms"
    )


def log_query_execution(collection: str, query_type: str, result_count: int, duration_ms: float):
    """Log query execution"""
    logger = logging.getLogger('query_execution')
    logger.info(
        f"📊 Query executed on '{collection}': "
        f"{query_type}, {result_count} results, {duration_ms:.2f}ms"
    )


def log_gemini_interaction(operation: str, tokens_used: int, duration_ms: float, success: bool):
    """Log Gemini AI interactions"""
    logger = logging.getLogger('gemini')
    
    status = "✅" if success else "❌"
    message = f"{status} Gemini {operation}: {tokens_used} tokens, {duration_ms:.2f}ms"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)


def log_cache_operation(operation: str, key: str, hit: bool = None, size: int = None):
    """Log cache operations"""
    logger = logging.getLogger('cache')
    
    if hit is not None:
        status = "🎯 HIT" if hit else "❌ MISS"
        logger.debug(f"{status} Cache {operation}: {key}")
    else:
        size_info = f" ({size} bytes)" if size else ""
        logger.debug(f"💾 Cache {operation}: {key}{size_info}")


def log_user_activity(user_id: str, action: str, details: dict = None):
    """Log user activity for analytics"""
    logger = logging.getLogger('user_activity')
    
    message = f"👤 User {user_id}: {action}"
    
    if details:
        message += f" - {details}"
    
    logger.info(message)


# Performance monitoring decorator
def monitor_performance(operation_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                log_performance(
                    operation_name or func.__name__, 
                    duration_ms,
                    {"args_count": len(args), "kwargs_count": len(kwargs)}
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger = logging.getLogger('performance')
                logger.error(
                    f"❌ {operation_name or func.__name__} failed after {duration_ms:.2f}ms: {str(e)}"
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                log_performance(
                    operation_name or func.__name__, 
                    duration_ms,
                    {"args_count": len(args), "kwargs_count": len(kwargs)}
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger = logging.getLogger('performance')
                logger.error(
                    f"❌ {operation_name or func.__name__} failed after {duration_ms:.2f}ms: {str(e)}"
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator