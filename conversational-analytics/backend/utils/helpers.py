# utils/helpers.py

import re
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from functools import wraps
import time

logger = logging.getLogger(__name__)

# =============================================================================
# STRING UTILITIES
# =============================================================================

def clean_field_name(field_name: str) -> str:
    """
    Clean and normalize field names for consistent usage
    """
    if not field_name:
        return "unknown_field"
    
    # Remove special characters and normalize
    cleaned = re.sub(r'[^\w\s-]', '', str(field_name))
    cleaned = re.sub(r'[-\s]+', '_', cleaned)
    cleaned = cleaned.lower().strip('_')
    
    # Ensure it doesn't start with a number
    if cleaned and cleaned[0].isdigit():
        cleaned = f"field_{cleaned}"
    
    return cleaned or "unknown_field"

def format_number(value: Union[int, float, str], precision: int = 2) -> str:
    """
    Format numbers for human-readable display
    """
    try:
        num = float(value)
        
        if abs(num) >= 1_000_000_000:
            return f"{num/1_000_000_000:.{precision}f}B"
        elif abs(num) >= 1_000_000:
            return f"{num/1_000_000:.{precision}f}M"
        elif abs(num) >= 1_000:
            return f"{num/1_000:.{precision}f}K"
        elif num == int(num):
            return str(int(num))
        else:
            return f"{num:.{precision}f}"
    
    except (ValueError, TypeError):
        return str(value)

def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate string to specified length with suffix
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from text for analysis
    """
    if not text:
        return []
    
    # Remove common words and extract meaningful keywords
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'can', 'this', 'that', 'these', 'those', 'what',
        'show', 'me', 'get', 'find', 'give'
    }
    
    # Extract words and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [word for word in words if word not in common_words]
    
    return list(set(keywords))  # Remove duplicates

# =============================================================================
# DATA STRUCTURE UTILITIES
# =============================================================================

def safe_get(data: Dict, path: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary values using dot notation
    """
    try:
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    except Exception:
        return default

def flatten_dict(data: Dict, prefix: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary into dot-notation keys
    """
    flattened = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # Handle array of objects
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flattened.update(flatten_dict(item, f"{new_key}[{i}]", sep))
                else:
                    flattened[f"{new_key}[{i}]"] = item
        else:
            flattened[new_key] = value
    
    return flattened

def merge_dicts(*dicts: Dict) -> Dict:
    """
    Deep merge multiple dictionaries
    """
    result = {}
    
    for d in dicts:
        if not isinstance(d, dict):
            continue
            
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
    
    return result

def remove_none_values(data: Union[Dict, List], recursive: bool = True) -> Union[Dict, List]:
    """
    Remove None values from dictionary or list
    """
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if value is not None:
                if recursive and isinstance(value, (dict, list)):
                    cleaned[key] = remove_none_values(value, recursive)
                else:
                    cleaned[key] = value
        return cleaned
    
    elif isinstance(data, list):
        cleaned = []
        for item in data:
            if item is not None:
                if recursive and isinstance(item, (dict, list)):
                    cleaned.append(remove_none_values(item, recursive))
                else:
                    cleaned.append(item)
        return cleaned
    
    return data

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def is_valid_email(email: str) -> bool:
    """
    Validate email address format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def is_valid_url(url: str) -> bool:
    """
    Validate URL format
    """
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))

def is_valid_mongodb_id(obj_id: str) -> bool:
    """
    Validate MongoDB ObjectId format
    """
    pattern = r'^[a-fA-F0-9]{24}$'
    return bool(re.match(pattern, obj_id))

def validate_query_params(params: Dict, required: List[str] = None, 
                         optional: List[str] = None) -> Dict[str, Any]:
    """
    Validate query parameters
    """
    result = {"valid": True, "errors": [], "params": {}}
    
    required = required or []
    optional = optional or []
    allowed = set(required + optional)
    
    # Check required parameters
    for param in required:
        if param not in params:
            result["valid"] = False
            result["errors"].append(f"Missing required parameter: {param}")
        else:
            result["params"][param] = params[param]
    
    # Check optional parameters
    for param in optional:
        if param in params:
            result["params"][param] = params[param]
    
    # Check for unexpected parameters
    unexpected = set(params.keys()) - allowed
    if unexpected:
        result["errors"].extend([f"Unexpected parameter: {param}" for param in unexpected])
    
    return result

# =============================================================================
# ASYNC UTILITIES
# =============================================================================

def async_retry(max_attempts: int = 3, delay: float = 1.0, 
                backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Decorator for async functions with retry logic
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:  # Last attempt
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    wait_time = delay * (backoff_factor ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

async def run_with_timeout(coro, timeout: float, default=None):
    """
    Run coroutine with timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default

async def gather_with_limit(tasks: List, limit: int = 10):
    """
    Run tasks concurrently with concurrency limit
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def limited_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[limited_task(task) for task in tasks])

# =============================================================================
# CACHING UTILITIES
# =============================================================================

def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate consistent cache key from arguments
    """
    # Create deterministic string from arguments
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True, default=str))
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (dict, list)):
            key_parts.append(f"{key}:{json.dumps(value, sort_keys=True, default=str)}")
        else:
            key_parts.append(f"{key}:{value}")
    
    # Create hash
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def cache_with_ttl(ttl_seconds: int = 3600):
    """
    Simple in-memory cache decorator with TTL
    """
    cache = {}
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            # Check cache
            if cache_key in cache:
                data, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    return data
                else:
                    del cache[cache_key]
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator

# =============================================================================
# TIME UTILITIES
# =============================================================================

def parse_time_period(period_str: str) -> timedelta:
    """
    Parse time period string into timedelta
    """
    patterns = {
        r'(\d+)\s*s(?:ec|econds?)?': lambda x: timedelta(seconds=int(x)),
        r'(\d+)\s*m(?:in|inutes?)?': lambda x: timedelta(minutes=int(x)),
        r'(\d+)\s*h(?:our|ours?)?': lambda x: timedelta(hours=int(x)),
        r'(\d+)\s*d(?:ay|ays?)?': lambda x: timedelta(days=int(x)),
        r'(\d+)\s*w(?:eek|eeks?)?': lambda x: timedelta(weeks=int(x))
    }
    
    period_str = period_str.lower().strip()
    
    for pattern, converter in patterns.items():
        match = re.match(pattern, period_str)
        if match:
            return converter(match.group(1))
    
    raise ValueError(f"Invalid time period format: {period_str}")

def format_duration(duration: timedelta) -> str:
    """
    Format timedelta into human-readable string
    """
    total_seconds = int(duration.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s" if seconds else f"{minutes}m"
    elif total_seconds < 86400:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    else:
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        return f"{days}d {hours}h" if hours else f"{days}d"

def get_time_bucket(dt: datetime, bucket_size: str = "hour") -> datetime:
    """
    Round datetime to time bucket for aggregation
    """
    if bucket_size == "minute":
        return dt.replace(second=0, microsecond=0)
    elif bucket_size == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    elif bucket_size == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif bucket_size == "week":
        days_since_monday = dt.weekday()
        start_of_week = dt - timedelta(days=days_since_monday)
        return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    elif bucket_size == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        return dt

# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

class SafeDict(dict):
    """
    Dictionary that doesn't raise KeyError for missing keys
    """
    def __missing__(self, key):
        return None
    
    def get_nested(self, path: str, default=None):
        """Get nested value using dot notation"""
        return safe_get(self, path, default)

def safe_cast(value: Any, target_type: type, default=None):
    """
    Safely cast value to target type
    """
    try:
        if target_type == bool and isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return target_type(value)
    except (ValueError, TypeError):
        return default

def handle_exceptions(default_return=None, log_errors=True):
    """
    Decorator to handle exceptions gracefully
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================

def measure_time(func: Callable):
    """
    Decorator to measure function execution time
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

class PerformanceMonitor:
    """
    Context manager for monitoring performance
    """
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        logger.info(f"⏱️ {self.operation_name}: {execution_time:.3f}s")

# =============================================================================
# BUSINESS LOGIC UTILITIES
# =============================================================================

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    """
    if old_value == 0:
        return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0
    
    return ((new_value - old_value) / old_value) * 100

def detect_data_patterns(values: List[Union[int, float]]) -> Dict[str, Any]:
    """
    Detect patterns in numeric data
    """
    if not values:
        return {"pattern": "no_data"}
    
    if len(values) < 2:
        return {"pattern": "insufficient_data"}
    
    # Calculate statistics
    mean_val = sum(values) / len(values)
    
    # Detect trends
    increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
    decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
    
    # Detect volatility
    differences = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
    avg_difference = sum(differences) / len(differences) if differences else 0
    volatility = avg_difference / mean_val if mean_val != 0 else 0
    
    pattern = "stable"
    if increasing:
        pattern = "increasing"
    elif decreasing:
        pattern = "decreasing"
    elif volatility > 0.3:
        pattern = "volatile"
    
    return {
        "pattern": pattern,
        "trend": "up" if increasing else "down" if decreasing else "stable",
        "volatility": volatility,
        "mean": mean_val,
        "range": max(values) - min(values) if values else 0
    }