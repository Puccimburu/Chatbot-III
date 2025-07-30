# api/middleware/cors.py

import logging
from typing import List, Optional, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

class CORSConfig:
    """
    Configuration class for CORS settings
    """
    
    def __init__(self, 
                 allow_origins: Union[List[str], str] = None,
                 allow_credentials: bool = True,
                 allow_methods: List[str] = None,
                 allow_headers: List[str] = None,
                 expose_headers: List[str] = None,
                 max_age: int = 600):
        
        # Set default allowed origins
        if allow_origins is None:
            allow_origins = [
                "http://localhost:3000",  # React development
                "http://localhost:5173",  # Vite development
                "http://localhost:8080",  # Alternative dev port
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:8080"
            ]
        
        # Set default allowed methods
        if allow_methods is None:
            allow_methods = [
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "OPTIONS",
                "HEAD",
                "PATCH"
            ]
        
        # Set default allowed headers
        if allow_headers is None:
            allow_headers = [
                "Accept",
                "Accept-Language",
                "Content-Language",
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "X-Request-ID",
                "X-API-Key",
                "Cache-Control",
                "Pragma"
            ]
        
        # Set default exposed headers
        if expose_headers is None:
            expose_headers = [
                "X-Request-ID",
                "X-Processing-Time",
                "X-Rate-Limit-Remaining",
                "X-Rate-Limit-Reset"
            ]
        
        self.allow_origins = allow_origins
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods
        self.allow_headers = allow_headers
        self.expose_headers = expose_headers
        self.max_age = max_age


def setup_cors(app: FastAPI, config: Optional[CORSConfig] = None, 
               environment: str = "development") -> None:
    """
    Setup CORS middleware with environment-specific configurations
    """
    if config is None:
        config = CORSConfig()
    
    # Environment-specific adjustments
    if environment == "production":
        # In production, be more restrictive
        production_origins = []
        
        # Add your production domains here
        production_domains = [
            "https://yourdomain.com",
            "https://www.yourdomain.com",
            "https://app.yourdomain.com"
        ]
        
        production_origins.extend(production_domains)
        
        # If no production origins specified, use config origins but log warning
        if not production_domains:
            logger.warning("⚠️ No production origins specified for CORS. Using development origins.")
            production_origins = config.allow_origins
        
        config.allow_origins = production_origins
        config.allow_credentials = True  # Usually needed for auth
        
    elif environment == "development":
        # In development, be more permissive
        config.allow_origins.extend([
            "http://localhost:*",  # Allow any localhost port
            "http://127.0.0.1:*"   # Allow any 127.0.0.1 port
        ])
        
    elif environment == "testing":
        # For testing, allow all origins
        config.allow_origins = ["*"]
        config.allow_credentials = False  # Can't use credentials with wildcard
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_credentials=config.allow_credentials,
        allow_methods=config.allow_methods,
        allow_headers=config.allow_headers,
        expose_headers=config.expose_headers,
        max_age=config.max_age
    )
    
    logger.info(f"✅ CORS configured for {environment} environment")
    logger.info(f"   Allowed origins: {config.allow_origins}")
    logger.info(f"   Allow credentials: {config.allow_credentials}")


def create_development_cors_config() -> CORSConfig:
    """
    Create a permissive CORS configuration for development
    """
    return CORSConfig(
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:8000"
        ],
        allow_credentials=True,
        max_age=3600  # 1 hour cache for preflight requests
    )


def create_production_cors_config(allowed_domains: List[str]) -> CORSConfig:
    """
    Create a secure CORS configuration for production
    """
    return CORSConfig(
        allow_origins=allowed_domains,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID"
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Processing-Time"
        ],
        max_age=86400  # 24 hours cache for preflight requests
    )


def create_api_only_cors_config() -> CORSConfig:
    """
    Create CORS configuration for API-only access (no browser credentials)
    """
    return CORSConfig(
        allow_origins=["*"],
        allow_credentials=False,  # No credentials with wildcard
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Accept",
            "Content-Type",
            "X-API-Key",
            "X-Requested-With"
        ],
        max_age=3600
    )


class DynamicCORSConfig:
    """
    Dynamic CORS configuration that can be updated at runtime
    """
    
    def __init__(self):
        self.allowed_origins = set()
        self.blocked_origins = set()
        self.default_config = CORSConfig()
    
    def add_allowed_origin(self, origin: str) -> None:
        """Add an origin to the allowed list"""
        self.allowed_origins.add(origin)
        logger.info(f"➕ Added allowed origin: {origin}")
    
    def remove_allowed_origin(self, origin: str) -> None:
        """Remove an origin from the allowed list"""
        self.allowed_origins.discard(origin)
        logger.info(f"➖ Removed allowed origin: {origin}")
    
    def block_origin(self, origin: str) -> None:
        """Block a specific origin"""
        self.blocked_origins.add(origin)
        self.allowed_origins.discard(origin)
        logger.warning(f"🚫 Blocked origin: {origin}")
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is allowed"""
        if origin in self.blocked_origins:
            return False
        
        if origin in self.allowed_origins:
            return True
        
        # Check against default patterns
        return origin in self.default_config.allow_origins
    
    def get_current_origins(self) -> List[str]:
        """Get current list of allowed origins"""
        return list(self.allowed_origins) + self.default_config.allow_origins


def validate_cors_origin(origin: str) -> bool:
    """
    Validate if an origin is properly formatted
    """
    import re
    
    # Basic URL pattern validation
    pattern = r'^https?://[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*(:[0-9]+)?$'
    
    if origin == "*":
        return True
    
    if re.match(pattern, origin):
        return True
    
    # Allow localhost patterns
    localhost_pattern = r'^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$'
    if re.match(localhost_pattern, origin):
        return True
    
    return False


def log_cors_request(origin: str, method: str, allowed: bool) -> None:
    """
    Log CORS request details for debugging
    """
    status = "✅ ALLOWED" if allowed else "❌ BLOCKED"
    logger.info(f"🌐 CORS {method} from {origin}: {status}")


class SecurityEnhancedCORS:
    """
    Enhanced CORS handler with security features
    """
    
    def __init__(self, config: CORSConfig):
        self.config = config
        self.request_counts = {}
        self.suspicious_origins = set()
    
    def is_suspicious_origin(self, origin: str) -> bool:
        """
        Check if an origin shows suspicious patterns
        """
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\.tk$',      # Suspicious TLD
            r'\.ml$',      # Suspicious TLD
            r'ngrok',      # Tunneling service
            r'localhost\.localdomain'  # Potentially malicious
        ]
        
        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, origin):
                return True
        
        return False
    
    def track_origin_requests(self, origin: str) -> bool:
        """
        Track requests from origins and flag excessive usage
        """
        current_count = self.request_counts.get(origin, 0) + 1
        self.request_counts[origin] = current_count
        
        # Flag if too many requests (simple rate limiting)
        if current_count > 1000:  # Adjust threshold as needed
            self.suspicious_origins.add(origin)
            logger.warning(f"🚨 Origin {origin} flagged for excessive requests: {current_count}")
            return False
        
        return True
    
    def should_allow_origin(self, origin: str) -> bool:
        """
        Comprehensive origin validation
        """
        # Check if origin is in blocked list
        if origin in self.suspicious_origins:
            return False
        
        # Check if origin is suspicious
        if self.is_suspicious_origin(origin):
            self.suspicious_origins.add(origin)
            return False
        
        # Track request count
        if not self.track_origin_requests(origin):
            return False
        
        # Check against allowed origins
        return origin in self.config.allow_origins or "*" in self.config.allow_origins
    
    def get_security_report(self) -> dict:
        """
        Get security report of CORS activity
        """
        return {
            "total_origins_seen": len(self.request_counts),
            "suspicious_origins": list(self.suspicious_origins),
            "top_requesting_origins": sorted(
                self.request_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }