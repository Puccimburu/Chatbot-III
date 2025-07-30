# api/middleware/rate_limiting.py

import time
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import asyncio

logger = logging.getLogger(__name__)

class RateLimitRule:
    """
    Defines a rate limiting rule
    """
    
    def __init__(self, max_requests: int, time_window: int, burst_limit: Optional[int] = None):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.burst_limit = burst_limit or max_requests * 2
        
    def __str__(self):
        return f"{self.max_requests} requests per {self.time_window}s"

class TokenBucket:
    """
    Token bucket algorithm for rate limiting
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket
        """
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_refill
            
            # Refill tokens based on time passed
            new_tokens = time_passed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get time to wait before tokens are available
        """
        if self.tokens >= tokens:
            return 0.0
        
        needed_tokens = tokens - self.tokens
        return needed_tokens / self.refill_rate

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter implementation
    """
    
    def __init__(self, rule: RateLimitRule):
        self.rule = rule
        self.requests = defaultdict(deque)  # client_id -> deque of timestamps
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        """
        Check if request is allowed for client
        """
        async with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests outside the time window
            cutoff_time = now - self.rule.time_window
            while client_requests and client_requests[0] <= cutoff_time:
                client_requests.popleft()
            
            # Check if we're within limits
            current_count = len(client_requests)
            
            if current_count >= self.rule.max_requests:
                # Calculate reset time
                oldest_request = client_requests[0] if client_requests else now
                reset_time = oldest_request + self.rule.time_window
                
                return False, {
                    "limit": self.rule.max_requests,
                    "remaining": 0,
                    "reset": reset_time,
                    "retry_after": reset_time - now
                }
            
            # Add current request
            client_requests.append(now)
            
            return True, {
                "limit": self.rule.max_requests,
                "remaining": self.rule.max_requests - current_count - 1,
                "reset": now + self.rule.time_window,
                "retry_after": 0
            }

class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on system load
    """
    
    def __init__(self, base_rule: RateLimitRule):
        self.base_rule = base_rule
        self.current_multiplier = 1.0
        self.system_load_samples = deque(maxlen=60)  # Last 60 samples
        self.last_adjustment = time.time()
        
    def update_system_load(self, cpu_percent: float, memory_percent: float, 
                          response_time: float) -> None:
        """
        Update system load metrics
        """
        # Simple load score (0-100)
        load_score = (cpu_percent * 0.4) + (memory_percent * 0.3) + (min(response_time, 10) * 10 * 0.3)
        self.system_load_samples.append(load_score)
        
        # Adjust rate limits every 30 seconds
        now = time.time()
        if now - self.last_adjustment > 30:
            self._adjust_rate_limits()
            self.last_adjustment = now
    
    def _adjust_rate_limits(self) -> None:
        """
        Adjust rate limits based on system load
        """
        if not self.system_load_samples:
            return
        
        avg_load = sum(self.system_load_samples) / len(self.system_load_samples)
        
        # Adjust multiplier based on load
        if avg_load > 80:  # High load
            self.current_multiplier = max(0.5, self.current_multiplier - 0.1)
            logger.warning(f"🔥 High system load ({avg_load:.1f}%), reducing rate limits to {self.current_multiplier:.1f}x")
        elif avg_load < 40:  # Low load
            self.current_multiplier = min(2.0, self.current_multiplier + 0.1)
            logger.info(f"⚡ Low system load ({avg_load:.1f}%), increasing rate limits to {self.current_multiplier:.1f}x")
    
    def get_effective_rule(self) -> RateLimitRule:
        """
        Get the current effective rate limiting rule
        """
        effective_max = int(self.base_rule.max_requests * self.current_multiplier)
        return RateLimitRule(
            max_requests=effective_max,
            time_window=self.base_rule.time_window,
            burst_limit=int(self.base_rule.burst_limit * self.current_multiplier)
        )

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Main rate limiting middleware
    """
    
    def __init__(self, app, rules: Dict[str, RateLimitRule] = None, 
                 key_func: Callable[[Request], str] = None,
                 adaptive: bool = False):
        super().__init__(app)
        
        # Default rules
        if rules is None:
            rules = {
                "default": RateLimitRule(max_requests=100, time_window=60),  # 100 per minute
                "/api/analytics/query": RateLimitRule(max_requests=20, time_window=60),  # 20 per minute for queries
                "/api/health": RateLimitRule(max_requests=200, time_window=60)  # 200 per minute for health checks
            }
        
        self.rules = rules
        self.key_func = key_func or self._default_key_func
        self.limiters = {}
        self.adaptive = adaptive
        
        # Initialize limiters
        for endpoint, rule in rules.items():
            if adaptive:
                self.limiters[endpoint] = AdaptiveRateLimiter(rule)
            else:
                self.limiters[endpoint] = SlidingWindowRateLimiter(rule)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "clients_blocked": set(),
            "last_reset": datetime.now()
        }
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting
        """
        start_time = time.time()
        
        # Get client identifier
        client_id = self.key_func(request)
        
        # Get appropriate limiter
        limiter = self._get_limiter_for_request(request)
        
        # Check rate limit
        if limiter:
            allowed, limit_info = await limiter.is_allowed(client_id)
            
            if not allowed:
                self._update_blocked_stats(client_id)
                return self._create_rate_limit_response(limit_info)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Update adaptive limiter if enabled
            if self.adaptive:
                processing_time = time.time() - start_time
                await self._update_system_metrics(processing_time)
            
            # Add rate limit headers
            if limiter and 'limit_info' in locals():
                self._add_rate_limit_headers(response, limit_info)
            
            self.stats["total_requests"] += 1
            return response
            
        except Exception as e:
            # Still count failed requests
            self.stats["total_requests"] += 1
            raise
    
    def _default_key_func(self, request: Request) -> str:
        """
        Default function to generate client identifier
        """
        # Try different identification methods
        
        # 1. API Key (if present)
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key:
            return f"api_key:{api_key[:10]}..."
        
        # 2. Client IP
        client_ip = self._get_client_ip(request)
        
        # 3. User agent for additional uniqueness
        user_agent = request.headers.get("User-Agent", "unknown")[:50]
        
        return f"ip:{client_ip}:ua:{hash(user_agent) % 10000}"
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request
        """
        # Check for forwarded headers first (load balancer/proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client
        return request.client.host if request.client else "unknown"
    
    def _get_limiter_for_request(self, request: Request) -> Optional[SlidingWindowRateLimiter]:
        """
        Get appropriate rate limiter for the request
        """
        path = request.url.path
        
        # Check for exact path match
        if path in self.limiters:
            return self.limiters[path]
        
        # Check for prefix matches
        for endpoint_pattern, limiter in self.limiters.items():
            if endpoint_pattern.startswith("/") and path.startswith(endpoint_pattern):
                return limiter
        
        # Use default limiter
        return self.limiters.get("default")
    
    def _create_rate_limit_response(self, limit_info: dict) -> Response:
        """
        Create rate limit exceeded response
        """
        from fastapi.responses import JSONResponse
        
        response_data = {
            "success": False,
            "error": {
                "type": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Maximum {limit_info['limit']} requests per window.",
                "limit": limit_info["limit"],
                "remaining": limit_info["remaining"],
                "reset": limit_info["reset"],
                "retry_after": int(limit_info["retry_after"]) + 1
            }
        }
        
        headers = {
            "X-RateLimit-Limit": str(limit_info["limit"]),
            "X-RateLimit-Remaining": str(limit_info["remaining"]),
            "X-RateLimit-Reset": str(int(limit_info["reset"])),
            "Retry-After": str(int(limit_info["retry_after"]) + 1)
        }
        
        return JSONResponse(
            status_code=429,
            content=response_data,
            headers=headers
        )
    
    def _add_rate_limit_headers(self, response: Response, limit_info: dict) -> None:
        """
        Add rate limiting headers to successful responses
        """
        response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(int(limit_info["reset"]))
    
    def _update_blocked_stats(self, client_id: str) -> None:
        """
        Update statistics for blocked requests
        """
        self.stats["blocked_requests"] += 1
        self.stats["clients_blocked"].add(client_id)
        
        logger.warning(f"🚫 Rate limit exceeded for client: {client_id}")
    
    async def _update_system_metrics(self, processing_time: float) -> None:
        """
        Update system metrics for adaptive rate limiting
        """
        if not self.adaptive:
            return
        
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Update all adaptive limiters
            for limiter in self.limiters.values():
                if isinstance(limiter, AdaptiveRateLimiter):
                    limiter.update_system_load(cpu_percent, memory_percent, processing_time)
                    
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            logger.warning(f"Error updating system metrics: {e}")
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """
        Get current rate limiting statistics
        """
        uptime = datetime.now() - self.stats["last_reset"]
        
        stats = {
            "total_requests": self.stats["total_requests"],
            "blocked_requests": self.stats["blocked_requests"],
            "block_rate": (self.stats["blocked_requests"] / max(1, self.stats["total_requests"])) * 100,
            "unique_clients_blocked": len(self.stats["clients_blocked"]),
            "uptime_minutes": uptime.total_seconds() / 60,
            "requests_per_minute": self.stats["total_requests"] / max(1, uptime.total_seconds() / 60)
        }
        
        # Add adaptive limiter info
        if self.adaptive:
            adaptive_info = {}
            for endpoint, limiter in self.limiters.items():
                if isinstance(limiter, AdaptiveRateLimiter):
                    effective_rule = limiter.get_effective_rule()
                    adaptive_info[endpoint] = {
                        "current_multiplier": limiter.current_multiplier,
                        "effective_limit": effective_rule.max_requests,
                        "base_limit": limiter.base_rule.max_requests
                    }
            stats["adaptive_limits"] = adaptive_info
        
        return stats
    
    def reset_stats(self) -> None:
        """
        Reset rate limiting statistics
        """
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "clients_blocked": set(),
            "last_reset": datetime.now()
        }
        logger.info("📊 Rate limiting statistics reset")

def create_api_rate_limiter(environment: str = "development") -> RateLimitingMiddleware:
    """
    Create rate limiter with environment-specific rules
    """
    if environment == "production":
        rules = {
            "default": RateLimitRule(max_requests=60, time_window=60),  # 60 per minute
            "/api/analytics/query": RateLimitRule(max_requests=10, time_window=60),  # 10 per minute
            "/api/health": RateLimitRule(max_requests=120, time_window=60),  # 120 per minute
            "/api/schema": RateLimitRule(max_requests=30, time_window=60)  # 30 per minute
        }
        adaptive = True
    
    elif environment == "development":
        rules = {
            "default": RateLimitRule(max_requests=200, time_window=60),  # 200 per minute
            "/api/analytics/query": RateLimitRule(max_requests=50, time_window=60),  # 50 per minute
            "/api/health": RateLimitRule(max_requests=500, time_window=60)  # 500 per minute
        }
        adaptive = False
    
    else:  # testing
        rules = {
            "default": RateLimitRule(max_requests=1000, time_window=60)  # Very permissive
        }
        adaptive = False
    
    return RateLimitingMiddleware(rules=rules, adaptive=adaptive)

def custom_key_function(request: Request) -> str:
    """
    Custom key function for specific rate limiting needs
    """
    # Example: Rate limit by API key if present, otherwise by IP
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api:{api_key}"
    
    # For analytics queries, be more restrictive on anonymous users
    if "/analytics" in request.url.path:
        client_ip = request.client.host if request.client else "unknown"
        return f"anon_analytics:{client_ip}"
    
    return f"general:{request.client.host if request.client else 'unknown'}"