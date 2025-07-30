"""
Configuration settings for Conversational Analytics
Environment variables and application configuration
"""

import os
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Settings:
    """Application settings with environment variable support"""
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database Configuration
    MONGODB_URL: str = os.getenv(
        "MONGODB_URL", 
        "mongodb://localhost:27017"
    )
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "analytics_db")
    
    # MongoDB Connection Settings
    MAX_POOL_SIZE: int = int(os.getenv("MAX_POOL_SIZE", "10"))
    MIN_POOL_SIZE: int = int(os.getenv("MIN_POOL_SIZE", "1"))
    CONNECTION_TIMEOUT: int = int(os.getenv("CONNECTION_TIMEOUT", "30"))
    
    # AI/Gemini Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "30"))
    GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    
    # Schema Detection Configuration
    ENABLE_AUTO_SCHEMA: bool = os.getenv("ENABLE_AUTO_SCHEMA", "True").lower() == "true"
    SCHEMA_CACHE_TTL: int = int(os.getenv("SCHEMA_CACHE_TTL", "21600"))  # 6 hours
    SCHEMA_SAMPLE_SIZE: int = int(os.getenv("SCHEMA_SAMPLE_SIZE", "1000"))
    SCHEMA_MAX_COLLECTIONS: int = int(os.getenv("SCHEMA_MAX_COLLECTIONS", "50"))
    
    # Caching Configuration
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "True").lower() == "true"
    CACHE_TYPE: str = os.getenv("CACHE_TYPE", "memory")  # memory, redis
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_DEFAULT_TTL: int = int(os.getenv("CACHE_DEFAULT_TTL", "600"))  # 10 minutes
    
    # Query Processing Configuration
    MAX_QUERY_RESULTS: int = int(os.getenv("MAX_QUERY_RESULTS", "1000"))
    QUERY_TIMEOUT: int = int(os.getenv("QUERY_TIMEOUT", "60"))
    ENABLE_QUERY_CACHE: bool = os.getenv("ENABLE_QUERY_CACHE", "True").lower() == "true"
    
    # Chart Generation Configuration
    CHART_CACHE_TTL: int = int(os.getenv("CHART_CACHE_TTL", "3600"))  # 1 hour
    MAX_CHART_DATA_POINTS: int = int(os.getenv("MAX_CHART_DATA_POINTS", "100"))
    ENABLE_CHART_SUGGESTIONS: bool = os.getenv("ENABLE_CHART_SUGGESTIONS", "True").lower() == "true"
    
    # Security Configuration
    ALLOWED_ORIGINS: List[str] = [
        origin.strip() 
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    ]
    ENABLE_RATE_LIMITING: bool = os.getenv("ENABLE_RATE_LIMITING", "True").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ENABLE_FILE_LOGGING: bool = os.getenv("ENABLE_FILE_LOGGING", "False").lower() == "true"
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "logs/app.log")
    
    # Performance Configuration
    ENABLE_ASYNC_SCHEMA_DETECTION: bool = os.getenv("ENABLE_ASYNC_SCHEMA_DETECTION", "True").lower() == "true"
    BACKGROUND_TASK_INTERVAL: int = int(os.getenv("BACKGROUND_TASK_INTERVAL", "3600"))  # 1 hour
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))
    
    # Feature Flags
    ENABLE_EXPERIMENTAL_FEATURES: bool = os.getenv("ENABLE_EXPERIMENTAL_FEATURES", "False").lower() == "true"
    ENABLE_METRICS_COLLECTION: bool = os.getenv("ENABLE_METRICS_COLLECTION", "True").lower() == "true"
    ENABLE_DETAILED_LOGGING: bool = os.getenv("ENABLE_DETAILED_LOGGING", "False").lower() == "true"
    
    def __init__(self):
        """Initialize settings and validate configuration"""
        self._validate_settings()
        self._log_configuration()
    
    def _validate_settings(self):
        """Validate critical configuration settings"""
        
        # Validate MongoDB URL
        if not self.MONGODB_URL:
            raise ValueError("MONGODB_URL is required")
        
        # Validate database name
        if not self.DATABASE_NAME:
            raise ValueError("DATABASE_NAME is required")
        
        # Validate Gemini API key if AI features are enabled
        if self.ENABLE_AUTO_SCHEMA and not self.GEMINI_API_KEY:
            logger.warning("⚠️ GEMINI_API_KEY not provided - AI features will be limited")
        
        # Validate cache configuration
        if self.ENABLE_CACHING and self.CACHE_TYPE == "redis" and not self.REDIS_URL:
            logger.warning("⚠️ Redis caching enabled but REDIS_URL not provided - falling back to memory cache")
            self.CACHE_TYPE = "memory"
        
        # Validate port range
        if not (1 <= self.PORT <= 65535):
            raise ValueError(f"Invalid port number: {self.PORT}")
        
        # Validate sample sizes
        if self.SCHEMA_SAMPLE_SIZE < 10:
            logger.warning("⚠️ SCHEMA_SAMPLE_SIZE is very low - may affect schema detection quality")
        
        if self.MAX_CHART_DATA_POINTS < 5:
            logger.warning("⚠️ MAX_CHART_DATA_POINTS is very low - may affect chart quality")
    
    def _log_configuration(self):
        """Log current configuration for debugging"""
        if self.DEBUG:
            logger.info("📋 Current Configuration:")
            logger.info(f"   Database: {self.DATABASE_NAME} @ {self.MONGODB_URL}")
            logger.info(f"   Auto Schema Detection: {self.ENABLE_AUTO_SCHEMA}")
            logger.info(f"   AI Integration: {self.GEMINI_API_KEY is not None}")
            logger.info(f"   Caching: {self.ENABLE_CACHING} ({self.CACHE_TYPE})")
            logger.info(f"   Rate Limiting: {self.ENABLE_RATE_LIMITING}")
            logger.info(f"   Debug Mode: {self.DEBUG}")
    
    @staticmethod
    def get_current_timestamp() -> str:
        """Get current timestamp in ISO format"""
        return datetime.utcnow().isoformat() + "Z"
    
    def get_gemini_config(self) -> dict:
        """Get Gemini API configuration"""
        return {
            "api_key": self.GEMINI_API_KEY,
            "model": self.GEMINI_MODEL,
            "timeout": self.GEMINI_TIMEOUT,
            "max_retries": self.GEMINI_MAX_RETRIES
        }
    
    def get_cache_config(self) -> dict:
        """Get cache configuration"""
        return {
            "enabled": self.ENABLE_CACHING,
            "type": self.CACHE_TYPE,
            "redis_url": self.REDIS_URL,
            "default_ttl": self.CACHE_DEFAULT_TTL,
            "schema_ttl": self.SCHEMA_CACHE_TTL,
            "chart_ttl": self.CHART_CACHE_TTL
        }
    
    def get_database_config(self) -> dict:
        """Get database configuration"""
        return {
            "url": self.MONGODB_URL,
            "database_name": self.DATABASE_NAME,
            "max_pool_size": self.MAX_POOL_SIZE,
            "min_pool_size": self.MIN_POOL_SIZE,
            "connection_timeout": self.CONNECTION_TIMEOUT
        }
    
    def get_schema_detection_config(self) -> dict:
        """Get schema detection configuration"""
        return {
            "enabled": self.ENABLE_AUTO_SCHEMA,
            "sample_size": self.SCHEMA_SAMPLE_SIZE,
            "max_collections": self.SCHEMA_MAX_COLLECTIONS,
            "cache_ttl": self.SCHEMA_CACHE_TTL,
            "async_detection": self.ENABLE_ASYNC_SCHEMA_DETECTION
        }


# Global settings instance
settings = Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    ENABLE_DETAILED_LOGGING = True


class ProductionSettings(Settings):
    """Production environment settings"""
    DEBUG = False
    LOG_LEVEL = "INFO"
    ENABLE_FILE_LOGGING = True
    ENABLE_RATE_LIMITING = True


class TestingSettings(Settings):
    """Testing environment settings"""
    DATABASE_NAME = "test_analytics_db"
    ENABLE_CACHING = False
    GEMINI_MAX_RETRIES = 1
    SCHEMA_SAMPLE_SIZE = 10


def get_settings():
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()