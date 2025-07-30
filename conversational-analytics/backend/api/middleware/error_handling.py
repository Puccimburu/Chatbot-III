# api/middleware/error_handling.py

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware for the conversational analytics API
    """
    
    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug
        self.error_stats = {
            "total_errors": 0,
            "error_types": {},
            "last_reset": datetime.now()
        }
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and handle any errors that occur
        """
        start_time = time.time()
        request_id = self._generate_request_id()
        
        # Add request ID to headers for tracking
        request.state.request_id = request_id
        
        try:
            # Log incoming request
            self._log_request(request, request_id)
            
            # Process the request
            response = await call_next(request)
            
            # Log successful response
            processing_time = time.time() - start_time
            self._log_response(request, response, processing_time, request_id)
            
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return await self._handle_http_exception(e, request, request_id)
            
        except Exception as e:
            # Handle unexpected errors
            return await self._handle_unexpected_error(e, request, request_id, start_time)
    
    def _generate_request_id(self) -> str:
        """
        Generate unique request ID for tracking
        """
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _log_request(self, request: Request, request_id: str) -> None:
        """
        Log incoming request details
        """
        logger.info(
            f"📥 {request_id} {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
    
    def _log_response(self, request: Request, response, processing_time: float, request_id: str) -> None:
        """
        Log response details
        """
        logger.info(
            f"📤 {request_id} {response.status_code} "
            f"processed in {processing_time:.3f}s"
        )
    
    async def _handle_http_exception(self, exc: HTTPException, request: Request, request_id: str) -> JSONResponse:
        """
        Handle FastAPI HTTP exceptions with consistent formatting
        """
        self._update_error_stats("HTTPException")
        
        error_response = {
            "success": False,
            "error": {
                "type": "http_error",
                "code": exc.status_code,
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
        
        # Add debug info if in debug mode
        if self.debug:
            error_response["error"]["debug_info"] = {
                "headers": dict(exc.headers) if exc.headers else {},
                "method": request.method,
                "query_params": str(request.query_params)
            }
        
        logger.warning(
            f"❌ {request_id} HTTP {exc.status_code}: {exc.detail} "
            f"at {request.url.path}"
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    async def _handle_unexpected_error(self, exc: Exception, request: Request, 
                                     request_id: str, start_time: float) -> JSONResponse:
        """
        Handle unexpected errors with detailed logging and user-friendly response
        """
        self._update_error_stats(type(exc).__name__)
        
        processing_time = time.time() - start_time
        
        # Determine error category and user message
        error_category = self._categorize_error(exc)
        user_message = self._get_user_friendly_message(error_category, exc)
        
        error_response = {
            "success": False,
            "error": {
                "type": "internal_error",
                "category": error_category,
                "message": user_message,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path)
            }
        }
        
        # Add debug info if in debug mode
        if self.debug:
            error_response["error"]["debug_info"] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "processing_time": round(processing_time, 3),
                "method": request.method,
                "headers": dict(request.headers),
                "query_params": str(request.query_params)
            }
        
        # Log detailed error information
        logger.error(
            f"💥 {request_id} Unexpected error in {processing_time:.3f}s:\n"
            f"Type: {type(exc).__name__}\n"
            f"Message: {str(exc)}\n"
            f"Path: {request.url.path}\n"
            f"Method: {request.method}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        
        # Return appropriate status code
        status_code = self._get_status_code_for_error(error_category)
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    def _categorize_error(self, exc: Exception) -> str:
        """
        Categorize error types for better handling
        """
        exc_type = type(exc).__name__
        exc_message = str(exc).lower()
        
        # Database errors
        if any(keyword in exc_type.lower() for keyword in ['mongo', 'database', 'connection']):
            return "database_error"
        
        if any(keyword in exc_message for keyword in ['connection', 'timeout', 'unreachable']):
            return "database_error"
        
        # AI/Gemini errors
        if any(keyword in exc_type.lower() for keyword in ['gemini', 'generative', 'api']):
            return "ai_error"
        
        if any(keyword in exc_message for keyword in ['api key', 'quota', 'rate limit']):
            return "ai_error"
        
        # Validation errors
        if any(keyword in exc_type.lower() for keyword in ['validation', 'value', 'type']):
            return "validation_error"
        
        # Authentication/Authorization errors
        if any(keyword in exc_type.lower() for keyword in ['auth', 'permission', 'forbidden']):
            return "auth_error"
        
        # Schema/Query errors
        if any(keyword in exc_message for keyword in ['schema', 'field', 'collection', 'query']):
            return "query_error"
        
        # JSON/Parsing errors
        if any(keyword in exc_type.lower() for keyword in ['json', 'parse', 'decode']):
            return "parsing_error"
        
        # Memory/Resource errors
        if any(keyword in exc_type.lower() for keyword in ['memory', 'resource', 'timeout']):
            return "resource_error"
        
        return "unknown_error"
    
    def _get_user_friendly_message(self, category: str, exc: Exception) -> str:
        """
        Get user-friendly error message based on category
        """
        messages = {
            "database_error": "We're having trouble connecting to the database. Please try again in a moment.",
            "ai_error": "Our AI service is temporarily unavailable. Please try again later.",
            "validation_error": "There was an issue with your request format. Please check your input and try again.",
            "auth_error": "Authentication failed. Please check your credentials.",
            "query_error": "We couldn't process your query. Please try rephrasing your question.",
            "parsing_error": "There was an issue processing your request data. Please check the format.",
            "resource_error": "The server is temporarily overloaded. Please try again in a few moments.",
            "unknown_error": "An unexpected error occurred. Our team has been notified."
        }
        
        base_message = messages.get(category, messages["unknown_error"])
        
        # Add specific hints for common issues
        if category == "ai_error" and "api key" in str(exc).lower():
            base_message += " (API key issue)"
        elif category == "database_error" and "timeout" in str(exc).lower():
            base_message += " (Connection timeout)"
        elif category == "query_error" and "collection" in str(exc).lower():
            base_message += " (Collection not found)"
        
        return base_message
    
    def _get_status_code_for_error(self, category: str) -> int:
        """
        Get appropriate HTTP status code for error category
        """
        status_codes = {
            "database_error": 503,  # Service Unavailable
            "ai_error": 503,        # Service Unavailable
            "validation_error": 400, # Bad Request
            "auth_error": 401,      # Unauthorized
            "query_error": 400,     # Bad Request
            "parsing_error": 400,   # Bad Request
            "resource_error": 503,  # Service Unavailable
            "unknown_error": 500    # Internal Server Error
        }
        
        return status_codes.get(category, 500)
    
    def _update_error_stats(self, error_type: str) -> None:
        """
        Update internal error statistics for monitoring
        """
        self.error_stats["total_errors"] += 1
        self.error_stats["error_types"][error_type] = (
            self.error_stats["error_types"].get(error_type, 0) + 1
        )
        
        # Reset stats daily
        if (datetime.now() - self.error_stats["last_reset"]).days >= 1:
            self.error_stats = {
                "total_errors": 1,
                "error_types": {error_type: 1},
                "last_reset": datetime.now()
            }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get current error statistics for monitoring
        """
        return self.error_stats.copy()


class CustomExceptionHandler:
    """
    Custom exception handlers for specific error types
    """
    
    @staticmethod
    async def validation_exception_handler(request: Request, exc) -> JSONResponse:
        """
        Handle Pydantic validation errors
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        error_details = []
        if hasattr(exc, 'errors'):
            for error in exc.errors():
                error_details.append({
                    "field": ".".join(str(x) for x in error.get("loc", [])),
                    "message": error.get("msg", "Validation error"),
                    "type": error.get("type", "unknown")
                })
        
        logger.warning(f"🔍 {request_id} Validation error: {error_details}")
        
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "type": "validation_error",
                    "message": "Request validation failed",
                    "details": error_details,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    
    @staticmethod
    async def rate_limit_exception_handler(request: Request, exc) -> JSONResponse:
        """
        Handle rate limiting errors
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.warning(f"⚡ {request_id} Rate limit exceeded from {request.client.host if request.client else 'unknown'}")
        
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": {
                    "type": "rate_limit_error",
                    "message": "Too many requests. Please slow down.",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "retry_after": 60
                }
            },
            headers={"Retry-After": "60"}
        )


class ContextualErrorLogger:
    """
    Enhanced error logging with context information
    """
    
    def __init__(self):
        self.error_contexts = {}
    
    def log_error_with_context(self, request_id: str, error: Exception, 
                             context: Dict[str, Any]) -> None:
        """
        Log error with additional context information
        """
        error_info = {
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        # Store for analysis
        self.error_contexts[request_id] = error_info
        
        # Log with structured information
        logger.error(
            f"🔍 Contextual Error Analysis for {request_id}:\n"
            f"Error: {error_info['error_type']} - {error_info['error_message']}\n"
            f"Context: {context}\n"
            f"Traceback: {error_info['traceback']}"
        )
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """
        Analyze error patterns for insights
        """
        if not self.error_contexts:
            return {"message": "No error data available"}
        
        error_types = {}
        context_patterns = {}
        
        for error_info in self.error_contexts.values():
            # Count error types
            error_type = error_info["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Analyze context patterns
            context = error_info.get("context", {})
            for key, value in context.items():
                if key not in context_patterns:
                    context_patterns[key] = {}
                context_patterns[key][str(value)] = context_patterns[key].get(str(value), 0) + 1
        
        return {
            "total_errors": len(self.error_contexts),
            "error_types": error_types,
            "context_patterns": context_patterns,
            "most_common_error": max(error_types.items(), key=lambda x: x[1]) if error_types else None
        }


def create_error_response(error_type: str, message: str, status_code: int = 500, 
                         request_id: Optional[str] = None, **kwargs) -> JSONResponse:
    """
    Utility function to create standardized error responses
    """
    error_response = {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
    }
    
    if request_id:
        error_response["error"]["request_id"] = request_id
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


def setup_error_handlers(app):
    """
    Setup all error handlers for the FastAPI application
    """
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException
    
    # Add validation error handler
    app.add_exception_handler(
        RequestValidationError,
        CustomExceptionHandler.validation_exception_handler
    )
    
    # Add rate limit handler (if using slowapi or similar)
    try:
        from slowapi.errors import RateLimitExceeded
        app.add_exception_handler(
            RateLimitExceeded,
            CustomExceptionHandler.rate_limit_exception_handler
        )
    except ImportError:
        pass  # Rate limiting not available
    
    logger.info("✅ Error handlers configured successfully")