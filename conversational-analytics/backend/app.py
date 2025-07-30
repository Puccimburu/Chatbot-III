# app.py - Complete Main Application

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware

# Configuration and Database
from config.settings import get_settings
from config.database import get_database_manager, DatabaseManager

# Services
from services.analytics_service import AnalyticsService
from services.schema_service import SchemaService
from services.cache_service import CacheService
from services.database_service import DatabaseService

# Core Components
from core.query_generation.gemini_client import GeminiClient
from core.schema_detection.detector import SchemaDetector

# API Routes
from api.routes.analytics import router as analytics_router
from api.routes.health import router as health_router
from api.routes.schema import router as schema_router
from api.routes.charts import router as charts_router

# Middleware
from api.middleware.error_handling import ErrorHandlingMiddleware, setup_error_handlers
from api.middleware.cors import setup_cors, create_development_cors_config
from api.middleware.rate_limiting import create_api_rate_limiter

# Utilities
from utils.logging_config import setup_logging

# Global settings
settings = get_settings()
logger = logging.getLogger(__name__)

# Global service instances
_services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown
    """
    # Startup
    logger.info("🚀 Starting Conversational Analytics API")
    
    try:
        # Initialize services
        await initialize_services()
        
        # Perform startup tasks
        await startup_tasks()
        
        logger.info("✅ Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"💥 Startup failed: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Conversational Analytics API")
        await shutdown_services()
        logger.info("✅ Application shutdown completed")

async def initialize_services():
    """
    Initialize all application services
    """
    global _services
    
    logger.info("🔧 Initializing services...")
    
    # 1. Database Manager
    db_manager = await get_database_manager()
    _services['db_manager'] = db_manager
    
    # Test database connection
    try:
        await db_manager.test_connection()
        logger.info("✅ Database connection established")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise
    
    # 2. Cache Service
    cache_service = CacheService(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB
    )
    await cache_service.initialize()
    _services['cache_service'] = cache_service
    
    # 3. Database Service
    database_service = DatabaseService(db_manager)
    _services['database_service'] = database_service
    
    # 4. Gemini Client
    if settings.GOOGLE_API_KEY:
        gemini_client = GeminiClient(
            api_key=settings.GOOGLE_API_KEY,
            model_name=settings.GEMINI_MODEL_NAME
        )
        _services['gemini_client'] = gemini_client
        logger.info("✅ Gemini AI client initialized")
    else:
        logger.warning("⚠️ No Google API key provided - AI features will be disabled")
        _services['gemini_client'] = None
    
    # 5. Schema Service
    schema_service = SchemaService(
        database_service=database_service,
        cache_service=cache_service
    )
    _services['schema_service'] = schema_service
    
    # 6. Analytics Service
    analytics_service = AnalyticsService(
        database_service=database_service,
        gemini_client=_services['gemini_client'],
        schema_service=schema_service,
        cache_service=cache_service
    )
    _services['analytics_service'] = analytics_service
    
    logger.info("✅ All services initialized successfully")

async def startup_tasks():
    """
    Perform application startup tasks
    """
    logger.info("🔄 Performing startup tasks...")
    
    # Initialize schema detection in background
    if settings.AUTO_DETECT_SCHEMA_ON_STARTUP:
        asyncio.create_task(background_schema_detection())
    
    # Warm up cache
    if settings.WARMUP_CACHE_ON_STARTUP:
        asyncio.create_task(warmup_cache())
    
    # Log application info
    logger.info(f"📊 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🔗 Database: {settings.MONGODB_URL[:20]}...")
    logger.info(f"🤖 AI Enabled: {'Yes' if _services.get('gemini_client') else 'No'}")
    logger.info(f"💾 Cache Enabled: {'Yes' if _services.get('cache_service') else 'No'}")

async def background_schema_detection():
    """
    Perform initial schema detection in background
    """
    try:
        logger.info("🔍 Starting background schema detection...")
        schema_service = _services.get('schema_service')
        
        if schema_service:
            await schema_service.detect_and_cache_schema()
            logger.info("✅ Background schema detection completed")
        
    except Exception as e:
        logger.error(f"❌ Background schema detection failed: {e}")

async def warmup_cache():
    """
    Warm up cache with frequently accessed data
    """
    try:
        logger.info("🔥 Warming up cache...")
        
        # Add any cache warming logic here
        # For example, pre-load schema information
        
        logger.info("✅ Cache warmup completed")
        
    except Exception as e:
        logger.error(f"❌ Cache warmup failed: {e}")

async def shutdown_services():
    """
    Clean shutdown of all services
    """
    global _services
    
    logger.info("🧹 Cleaning up services...")
    
    # Close cache connections
    cache_service = _services.get('cache_service')
    if cache_service:
        await cache_service.close()
        logger.info("✅ Cache service closed")
    
    # Close database connections
    db_manager = _services.get('db_manager')
    if db_manager:
        await db_manager.close()
        logger.info("✅ Database connections closed")
    
    _services.clear()

# Create FastAPI application
def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application
    """
    # Setup logging first
    setup_logging()
    
    # Create FastAPI app with lifespan manager
    app = FastAPI(
        title="Conversational Analytics API",
        description="AI-powered natural language data analytics platform",
        version="1.0.0",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan
    )
    
    # Add middleware (order matters!)
    
    # 1. Error handling (should be first)
    app.add_middleware(
        ErrorHandlingMiddleware,
        debug=settings.DEBUG
    )
    
    # 2. CORS
    setup_cors(
        app, 
        config=create_development_cors_config() if settings.ENVIRONMENT == "development" else None,
        environment=settings.ENVIRONMENT
    )
    
    # 3. Rate limiting
    if settings.ENABLE_RATE_LIMITING:
        rate_limiter = create_api_rate_limiter(settings.ENVIRONMENT)
        app.add_middleware(type(rate_limiter), **rate_limiter.__dict__)
    
    # 4. GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Setup error handlers
    setup_error_handlers(app)
    
    # Include routers
    app.include_router(analytics_router)
    app.include_router(health_router)
    app.include_router(schema_router)
    app.include_router(charts_router)
    
    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Conversational Analytics API",
            "version": "1.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "docs": "/docs" if settings.ENVIRONMENT != "production" else "Documentation disabled in production",
            "endpoints": {
                "analytics": "/api/analytics",
                "health": "/api/health", 
                "schema": "/api/schema",
                "charts": "/api/charts"
            }
        }
    
    # Add startup message endpoint
    @app.get("/api/info")
    async def api_info():
        """Get API information and status"""
        return {
            "api_name": "Conversational Analytics",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "features": {
                "ai_powered": bool(_services.get('gemini_client')),
                "auto_schema_detection": True,
                "intelligent_caching": bool(_services.get('cache_service')),
                "chart_generation": True,
                "rate_limiting": settings.ENABLE_RATE_LIMITING
            },
            "status": "healthy",
            "uptime": "See /api/health/status for detailed uptime",
            "timestamp": datetime.now().isoformat()
        }
    
    logger.info("🎯 FastAPI application configured successfully")
    
    return app

# Dependency injection functions
async def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance"""
    service = _services.get('analytics_service')
    if not service:
        raise HTTPException(
            status_code=503,
            detail="Analytics service not available"
        )
    return service

async def get_schema_service() -> SchemaService:
    """Get schema service instance"""
    service = _services.get('schema_service')
    if not service:
        raise HTTPException(
            status_code=503,
            detail="Schema service not available"
        )
    return service

async def get_database_service() -> DatabaseService:
    """Get database service instance"""
    service = _services.get('database_service')
    if not service:
        raise HTTPException(
            status_code=503,
            detail="Database service not available"
        )
    return service

async def get_cache_service() -> CacheService:
    """Get cache service instance"""
    return _services.get('cache_service')

async def get_gemini_client() -> GeminiClient:
    """Get Gemini client instance"""
    return _services.get('gemini_client')

# Create the application instance
app = create_application()

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response

# Health check for load balancers
@app.get("/health")
async def simple_health_check():
    """Simple health check endpoint for load balancers"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Development and debugging helpers
if settings.ENVIRONMENT == "development":
    
    @app.get("/api/debug/services")
    async def debug_services():
        """Debug endpoint to check service status"""
        return {
            "services": {
                name: "available" if service else "unavailable"
                for name, service in _services.items()
            },
            "settings": {
                "environment": settings.ENVIRONMENT,
                "debug": settings.DEBUG,
                "database_configured": bool(settings.MONGODB_URL),
                "ai_configured": bool(settings.GOOGLE_API_KEY),
                "cache_configured": bool(settings.REDIS_HOST)
            }
        }
    
    @app.get("/api/debug/config")
    async def debug_config():
        """Debug endpoint to check configuration"""
        return {
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "features": {
                "auto_schema_detection": settings.AUTO_DETECT_SCHEMA_ON_STARTUP,
                "cache_warmup": settings.WARMUP_CACHE_ON_STARTUP,
                "rate_limiting": settings.ENABLE_RATE_LIMITING
            },
            "timeouts": {
                "database_timeout": settings.DATABASE_TIMEOUT_SECONDS,
                "ai_timeout": settings.AI_REQUEST_TIMEOUT_SECONDS
            }
        }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Configuration for development server
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
        access_log=True
    )