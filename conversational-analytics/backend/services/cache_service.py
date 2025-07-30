"""
Cache Service for intelligent caching with Redis and memory fallback
Supports TTL, pattern-based operations, and performance monitoring
"""

import asyncio
import logging
import json
import time
from typing import Any, Optional, Dict, List, Pattern
from datetime import datetime, timedelta
from collections import OrderedDict
import re

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import settings
from utils.logging_config import log_cache_operation

logger = logging.getLogger(__name__)


class MemoryCache:
    """In-memory cache with TTL support for fallback"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.ttl_map: Dict[str, float] = {}
        self.max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            if key not in self.cache:
                log_cache_operation("get", key, hit=False)
                return None
            
            # Check TTL
            if key in self.ttl_map:
                if time.time() > self.ttl_map[key]:
                    # Expired
                    del self.cache[key]
                    del self.ttl_map[key]
                    log_cache_operation("get", key, hit=False)
                    return None
            
            # Move to end (LRU)
            value = self.cache.pop(key)
            self.cache[key] = value
            
            log_cache_operation("get", key, hit=True)
            return value
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None):
        async with self._lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.ttl_map:
                    del self.ttl_map[oldest_key]
            
            self.cache[key] = value
            
            if ttl:
                self.ttl_map[key] = time.time() + ttl
            elif key in self.ttl_map:
                del self.ttl_map[key]
            
            log_cache_operation("set", key, size=len(value))
    
    async def delete(self, key: str):
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.ttl_map:
                    del self.ttl_map[key]
                log_cache_operation("delete", key)
    
    async def delete_pattern(self, pattern: str):
        """Delete keys matching pattern (simplified regex)"""
        async with self._lock:
            regex = re.compile(pattern.replace('*', '.*'))
            keys_to_delete = [key for key in self.cache.keys() if regex.match(key)]
            
            for key in keys_to_delete:
                del self.cache[key]
                if key in self.ttl_map:
                    del self.ttl_map[key]
            
            log_cache_operation("delete_pattern", pattern)
    
    async def clear(self):
        async with self._lock:
            self.cache.clear()
            self.ttl_map.clear()
            log_cache_operation("clear", "all")
    
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        async with self._lock:
            if pattern:
                regex = re.compile(pattern.replace('*', '.*'))
                return [key for key in self.cache.keys() if regex.match(key)]
            return list(self.cache.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "memory",
            "total_keys": len(self.cache),
            "max_size": self.max_size,
            "keys_with_ttl": len(self.ttl_map)
        }


class CacheService:
    """Intelligent cache service with Redis primary and memory fallback"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache = MemoryCache(max_size=settings.MAX_QUERY_RESULTS)
        self.use_redis = False
        self.connection_retries = 0
        self.max_retries = 3
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.error_count = 0
    
    async def initialize(self):
        """Initialize cache service with Redis connection attempt"""
        
        if not settings.ENABLE_CACHING:
            logger.info("📋 Caching disabled by configuration")
            return
        
        logger.info("🔧 Initializing Cache Service...")
        
        # Try to connect to Redis if available and configured
        if REDIS_AVAILABLE and settings.CACHE_TYPE == "redis" and settings.REDIS_URL:
            await self._connect_redis()
        
        if not self.use_redis:
            logger.info("💾 Using memory cache (Redis not available)")
        
        logger.info("✅ Cache Service initialized")
    
    async def _connect_redis(self):
        """Attempt Redis connection with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔗 Connecting to Redis... (attempt {attempt + 1})")
                
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                
                # Test connection
                await self.redis_client.ping()
                
                self.use_redis = True
                self.connection_retries = 0
                
                logger.info("✅ Redis connection established")
                return
                
            except Exception as e:
                logger.warning(f"⚠️ Redis connection attempt {attempt + 1} failed: {str(e)}")
                
                if self.redis_client:
                    try:
                        await self.redis_client.close()
                    except:
                        pass
                    self.redis_client = None
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.warning("❌ Redis connection failed after all retries, using memory cache")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache with Redis primary, memory fallback"""
        
        if not settings.ENABLE_CACHING:
            return None
        
        try:
            # Try Redis first
            if self.use_redis and self.redis_client:
                try:
                    value = await self.redis_client.get(key)
                    if value is not None:
                        self.hit_count += 1
                        log_cache_operation("get", key, hit=True)
                        return value
                except Exception as e:
                    logger.warning(f"⚠️ Redis get error: {str(e)}")
                    await self._handle_redis_error()
            
            # Fallback to memory cache
            value = await self.memory_cache.get(key)
            if value is not None:
                self.hit_count += 1
                return value
            
            self.miss_count += 1
            return None
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Cache get error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Set value in cache with TTL"""
        
        if not settings.ENABLE_CACHING:
            return
        
        if ttl is None:
            ttl = settings.CACHE_DEFAULT_TTL
        
        try:
            # Set in Redis
            if self.use_redis and self.redis_client:
                try:
                    await self.redis_client.setex(key, ttl, value)
                    log_cache_operation("set", key, size=len(value))
                except Exception as e:
                    logger.warning(f"⚠️ Redis set error: {str(e)}")
                    await self._handle_redis_error()
            
            # Always set in memory cache as backup
            await self.memory_cache.set(key, value, ttl)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Cache set error for key {key}: {str(e)}")
    
    async def delete(self, key: str):
        """Delete key from cache"""
        
        if not settings.ENABLE_CACHING:
            return
        
        try:
            # Delete from Redis
            if self.use_redis and self.redis_client:
                try:
                    await self.redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"⚠️ Redis delete error: {str(e)}")
                    await self._handle_redis_error()
            
            # Delete from memory cache
            await self.memory_cache.delete(key)
            
            log_cache_operation("delete", key)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Cache delete error for key {key}: {str(e)}")
    
    async def delete_pattern(self, pattern: str):
        """Delete keys matching pattern"""
        
        if not settings.ENABLE_CACHING:
            return
        
        try:
            # Delete from Redis
            if self.use_redis and self.redis_client:
                try:
                    # Redis SCAN with pattern
                    cursor = 0
                    while True:
                        cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                        if keys:
                            await self.redis_client.delete(*keys)
                        if cursor == 0:
                            break
                except Exception as e:
                    logger.warning(f"⚠️ Redis delete pattern error: {str(e)}")
                    await self._handle_redis_error()
            
            # Delete from memory cache
            await self.memory_cache.delete_pattern(pattern)
            
            log_cache_operation("delete_pattern", pattern)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Cache delete pattern error for {pattern}: {str(e)}")
    
    async def clear(self):
        """Clear all cache entries"""
        
        if not settings.ENABLE_CACHING:
            return
        
        try:
            # Clear Redis
            if self.use_redis and self.redis_client:
                try:
                    await self.redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"⚠️ Redis clear error: {str(e)}")
                    await self._handle_redis_error()
            
            # Clear memory cache
            await self.memory_cache.clear()
            
            log_cache_operation("clear", "all")
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Cache clear error: {str(e)}")
    
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get keys matching pattern"""
        
        if not settings.ENABLE_CACHING:
            return []
        
        try:
            # Get from Redis
            if self.use_redis and self.redis_client:
                try:
                    if pattern:
                        keys = []
                        cursor = 0
                        while True:
                            cursor, batch = await self.redis_client.scan(cursor, match=pattern, count=100)
                            keys.extend(batch)
                            if cursor == 0:
                                break
                        return keys
                    else:
                        return await self.redis_client.keys()
                except Exception as e:
                    logger.warning(f"⚠️ Redis keys error: {str(e)}")
                    await self._handle_redis_error()
            
            # Fallback to memory cache
            return await self.memory_cache.keys(pattern)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Cache keys error: {str(e)}")
            return []
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        
        if not settings.ENABLE_CACHING:
            return False
        
        try:
            # Check Redis first
            if self.use_redis and self.redis_client:
                try:
                    return bool(await self.redis_client.exists(key))
                except Exception as e:
                    logger.warning(f"⚠️ Redis exists error: {str(e)}")
                    await self._handle_redis_error()
            
            # Check memory cache
            value = await self.memory_cache.get(key)
            return value is not None
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Cache exists error for key {key}: {str(e)}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get TTL for key (-1 if no expiry, -2 if doesn't exist)"""
        
        if not settings.ENABLE_CACHING:
            return -2
        
        try:
            # Check Redis first
            if self.use_redis and self.redis_client:
                try:
                    return await self.redis_client.ttl(key)
                except Exception as e:
                    logger.warning(f"⚠️ Redis TTL error: {str(e)}")
                    await self._handle_redis_error()
            
            # Memory cache doesn't easily support TTL queries
            exists = await self.memory_cache.get(key) is not None
            return -1 if exists else -2
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Cache TTL error for key {key}: {str(e)}")
            return -2
    
    async def _handle_redis_error(self):
        """Handle Redis connection errors"""
        
        self.connection_retries += 1
        
        if self.connection_retries >= self.max_retries:
            logger.warning("🔄 Too many Redis errors, switching to memory cache")
            self.use_redis = False
            
            if self.redis_client:
                try:
                    await self.redis_client.close()
                except:
                    pass
                self.redis_client = None
        else:
            # Try to reconnect
            logger.info("🔄 Attempting Redis reconnection...")
            await self._connect_redis()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            "enabled": settings.ENABLE_CACHING,
            "backend": "redis" if self.use_redis else "memory",
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "error_count": self.error_count,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }
        
        # Add backend-specific stats
        if self.use_redis:
            stats["redis_connected"] = self.redis_client is not None
            stats["connection_retries"] = self.connection_retries
        else:
            stats.update(self.memory_cache.get_stats())
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform cache health check"""
        
        if not settings.ENABLE_CACHING:
            return {"status": "disabled"}
        
        try:
            # Test cache operations
            test_key = "health_check_test"
            test_value = "test_value"
            
            # Test set
            await self.set(test_key, test_value, ttl=60)
            
            # Test get
            retrieved_value = await self.get(test_key)
            
            # Test delete
            await self.delete(test_key)
            
            if retrieved_value == test_value:
                return {
                    "status": "healthy",
                    "backend": "redis" if self.use_redis else "memory",
                    "operations_tested": ["set", "get", "delete"],
                    **self.get_stats()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Cache operations failed",
                    **self.get_stats()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                **self.get_stats()
            }
    
    async def close(self):
        """Close cache connections"""
        
        logger.info("🛑 Closing cache connections...")
        
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("✅ Redis connection closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing Redis connection: {str(e)}")
        
        # Clear memory cache
        await self.memory_cache.clear()
        
        logger.info("✅ Cache Service closed")