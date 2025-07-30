"""
Analytics Service - Main orchestrator for conversational analytics
Handles the complete flow from natural language to insights with charts
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import json

from models.schema_models import DatabaseSchema
from models.query_models import AnalyticsRequest, AnalyticsResponse, ChartData
from core.query_generation.gemini_client import GeminiClient
from core.query_generation.fallback_generator import FallbackQueryGenerator
from core.visualization.chart_analyzer import ChartAnalyzer
from services.database_service import DatabaseService
from services.schema_service import SchemaService
from services.cache_service import CacheService
from config.settings import settings
from utils.logging_config import monitor_performance, log_query_execution

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Main analytics service orchestrating the complete conversational analytics flow"""
    
    def __init__(self, db_manager, schema_service: SchemaService, cache_service: CacheService):
        self.db_manager = db_manager
        self.schema_service = schema_service
        self.cache_service = cache_service
        self.database_service = DatabaseService(db_manager)
        
        # AI Components
        self.gemini_client = GeminiClient()
        self.fallback_generator = FallbackQueryGenerator()
        self.chart_analyzer = ChartAnalyzer()
        
        # Performance tracking
        self.total_queries = 0
        self.successful_queries = 0
        self.ai_powered_queries = 0
        self.fallback_queries = 0
        
    async def initialize(self):
        """Initialize the analytics service"""
        logger.info("🔧 Initializing Analytics Service...")
        
        # Initialize AI components
        await self.gemini_client.initialize()
        
        # Initialize fallback generator with schema
        database_schema = await self.schema_service.get_database_schema()
        self.fallback_generator.initialize(database_schema)
        
        logger.info("✅ Analytics Service initialized")
    
    @monitor_performance("process_analytics_request")
    async def process_request(self, request: AnalyticsRequest) -> AnalyticsResponse:
        """
        Process a complete analytics request from natural language to insights
        
        Args:
            request: AnalyticsRequest containing user question and options
            
        Returns:
            AnalyticsResponse with text answer and optional chart data
        """
        
        start_time = time.perf_counter()
        self.total_queries += 1
        
        try:
            logger.info(f"🚀 Processing analytics request: '{request.question}'")
            
            # Check cache first
            if settings.ENABLE_QUERY_CACHE:
                cached_response = await self._get_cached_response(request)
                if cached_response:
                    logger.info("📋 Returning cached response")
                    return cached_response
            
            # Get database schema
            database_schema = await self.schema_service.get_database_schema()
            
            # Stage 1: Generate MongoDB query
            query_result = await self._generate_query(request.question, database_schema)
            
            if not query_result["success"]:
                return self._create_error_response(
                    f"Failed to generate query: {query_result.get('error', 'Unknown error')}",
                    execution_time=(time.perf_counter() - start_time) * 1000
                )
            
            # Stage 2: Execute database query
            execution_result = await self._execute_query(
                query_result["data"], 
                request.max_results or settings.MAX_QUERY_RESULTS
            )
            
            if not execution_result["success"]:
                return self._create_error_response(
                    f"Query execution failed: {execution_result.get('error', 'Unknown error')}",
                    execution_time=(time.perf_counter() - start_time) * 1000
                )
            
            raw_data = execution_result["data"]
            
            # Stage 3: Generate text summary
            summary = await self._generate_summary(
                request.question, 
                raw_data, 
                query_result["data"]
            )
            
            # Stage 4: Analyze chart worthiness
            chart_analysis = await self._analyze_chart_worthiness(
                request.question,
                raw_data,
                query_result["data"]
            )
            
            # Stage 5: Generate chart data if worthy
            chart_data = None
            if chart_analysis.get("chart_worthy", False):
                chart_data = await self._generate_chart_data(
                    raw_data,
                    chart_analysis,
                    query_result["data"]
                )
            
            # Create response
            response = AnalyticsResponse(
                success=True,
                summary=summary,
                chart_data=chart_data,
                chart_worthy=chart_analysis.get("chart_worthy", False),
                chart_suggestion=chart_analysis.get("suggested_chart_type"),
                insights=chart_analysis.get("insights", []),
                data_count=len(raw_data),
                execution_time=(time.perf_counter() - start_time) * 1000,
                ai_powered=query_result.get("ai_powered", False),
                cache_used=False
            )
            
            # Update statistics
            self.successful_queries += 1
            if response.ai_powered:
                self.ai_powered_queries += 1
            else:
                self.fallback_queries += 1
            
            # Cache the response
            if settings.ENABLE_QUERY_CACHE:
                await self._cache_response(request, response)
            
            logger.info(
                f"✅ Analytics request completed in {response.execution_time:.2f}ms: "
                f"{response.data_count} results, chart_worthy={response.chart_worthy}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Analytics request failed: {str(e)}")
            return self._create_error_response(
                f"Internal error: {str(e)}",
                execution_time=(time.perf_counter() - start_time) * 1000
            )
    
    async def _generate_query(
        self, 
        user_question: str, 
        database_schema: DatabaseSchema
    ) -> Dict[str, Any]:
        """Generate MongoDB query using AI or fallback methods"""
        
        # Try Gemini first
        if self.gemini_client.available:
            logger.debug("🤖 Attempting query generation with Gemini AI")
            
            gemini_result = await self.gemini_client.generate_mongodb_query(
                user_question, database_schema
            )
            
            if gemini_result["success"]:
                logger.info("✅ Query generated successfully with Gemini AI")
                return {
                    "success": True,
                    "data": gemini_result["data"],
                    "ai_powered": True,
                    "attempts": gemini_result.get("attempts", 1)
                }
        
        # Fallback to rule-based generation
        logger.info("🔄 Falling back to rule-based query generation")
        
        fallback_result = await self.fallback_generator.generate_query(
            user_question, database_schema
        )
        
        if fallback_result["success"]:
            return {
                "success": True,
                "data": fallback_result["data"],
                "ai_powered": False,
                "method": "fallback"
            }
        
        return {
            "success": False,
            "error": "Both AI and fallback query generation failed",
            "ai_powered": False
        }
    
    async def _execute_query(
        self, 
        query_data: Dict[str, Any], 
        max_results: int
    ) -> Dict[str, Any]:
        """Execute MongoDB query with error handling"""
        
        try:
            collection_name = query_data["collection"]
            pipeline = query_data["pipeline"]
            
            # Add result limit if not present
            if not any("$limit" in stage for stage in pipeline):
                pipeline.append({"$limit": max_results})
            
            # Execute query
            start_time = time.perf_counter()
            
            results = await self.database_service.execute_aggregation(
                collection_name, pipeline
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            log_query_execution(
                collection_name, 
                "aggregation", 
                len(results), 
                execution_time
            )
            
            return {
                "success": True,
                "data": results,
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_summary(
        self, 
        user_question: str, 
        raw_data: List[Dict[str, Any]], 
        query_context: Dict[str, Any]
    ) -> str:
        """Generate human-readable summary of results"""
        
        if not raw_data:
            return "No data found matching your query."
        
        # Try AI-powered summary first
        if self.gemini_client.available:
            try:
                # Simple prompt for summary generation
                summary_prompt = f"""
Generate a concise, business-friendly summary for this data analysis.

USER QUESTION: "{user_question}"
RESULTS COUNT: {len(raw_data)}
SAMPLE DATA: {json.dumps(raw_data[:3], default=str, indent=2)}

Requirements:
- 1-2 sentences maximum
- Include specific numbers/values
- Business-focused insights
- Direct answer to the question

Response format: Just the summary text, no JSON or formatting.
"""
                
                summary_response = await self.gemini_client._make_request(
                    summary_prompt, 
                    self.gemini_client.query_generation_config
                )
                
                if summary_response and len(summary_response) < 500:
                    return summary_response.strip()
                    
            except Exception as e:
                logger.warning(f"⚠️ AI summary generation failed: {str(e)}")
        
        # Fallback to rule-based summary
        return self._generate_fallback_summary(user_question, raw_data, query_context)
    
    def _generate_fallback_summary(
        self, 
        user_question: str, 
        raw_data: List[Dict[str, Any]], 
        query_context: Dict[str, Any]
    ) -> str:
        """Generate fallback summary using simple rules"""
        
        result_count = len(raw_data)
        
        # Single result
        if result_count == 1:
            first_result = raw_data[0]
            if len(first_result) == 1:
                key, value = next(iter(first_result.items()))
                return f"The result is {value}."
            else:
                return f"Found 1 matching record."
        
        # Multiple results
        elif result_count <= 10:
            # Try to identify the key field
            if raw_data:
                first_item = raw_data[0]
                if '_id' in first_item and len(first_item) == 2:
                    # Looks like aggregation result
                    other_field = [k for k in first_item.keys() if k != '_id'][0]
                    return f"Found {result_count} results. Top result: {first_item['_id']} with {other_field} of {first_item[other_field]}."
            
            return f"Found {result_count} matching records."
        
        # Large result set
        else:
            return f"Found {result_count} matching records. Showing summary of results."
    
    async def _analyze_chart_worthiness(
        self, 
        user_question: str, 
        raw_data: List[Dict[str, Any]], 
        query_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze if data should be visualized and suggest chart type"""
        
        # Try AI analysis first
        if self.gemini_client.available:
            ai_analysis = await self.gemini_client.analyze_chart_worthiness(
                user_question, raw_data, query_context
            )
            
            if ai_analysis.get("success"):
                return ai_analysis["data"]
        
        # Fallback to rule-based analysis
        return self.chart_analyzer.analyze_chart_worthiness(
            user_question, raw_data, query_context
        )
    
    async def _generate_chart_data(
        self, 
        raw_data: List[Dict[str, Any]], 
        chart_analysis: Dict[str, Any], 
        query_context: Dict[str, Any]
    ) -> Optional[ChartData]:
        """Generate Chart.js configuration from raw data"""
        
        try:
            chart_config = self.chart_analyzer.generate_chart_config(
                raw_data, 
                chart_analysis.get("suggested_chart_type", "bar"),
                chart_analysis.get("chart_config", {})
            )
            
            if chart_config:
                return ChartData(
                    type=chart_analysis.get("suggested_chart_type", "bar"),
                    config=chart_config,
                    title=chart_config.get("options", {}).get("plugins", {}).get("title", {}).get("text", "Data Visualization")
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Chart generation failed: {str(e)}")
        
        return None
    
    async def _get_cached_response(self, request: AnalyticsRequest) -> Optional[AnalyticsResponse]:
        """Get cached response for request"""
        
        cache_key = f"analytics:{hash(request.question)}:{request.max_results}"
        
        try:
            cached_data = await self.cache_service.get(cache_key)
            if cached_data:
                response_dict = json.loads(cached_data)
                response = AnalyticsResponse(**response_dict)
                response.cache_used = True
                return response
        except Exception as e:
            logger.warning(f"⚠️ Cache retrieval failed: {str(e)}")
        
        return None
    
    async def _cache_response(self, request: AnalyticsRequest, response: AnalyticsResponse):
        """Cache analytics response"""
        
        cache_key = f"analytics:{hash(request.question)}:{request.max_results}"
        
        try:
            # Don't cache error responses
            if not response.success:
                return
            
            # Create cacheable version (remove cache_used flag)
            cache_response = response.copy()
            cache_response.cache_used = False
            
            await self.cache_service.set(
                cache_key, 
                cache_response.json(), 
                ttl=settings.CACHE_DEFAULT_TTL
            )
        except Exception as e:
            logger.warning(f"⚠️ Response caching failed: {str(e)}")
    
    def _create_error_response(self, error_message: str, execution_time: float) -> AnalyticsResponse:
        """Create error response"""
        
        return AnalyticsResponse(
            success=False,
            error=error_message,
            summary="Sorry, I couldn't process your request. Please try rephrasing your question.",
            chart_data=None,
            chart_worthy=False,
            insights=[],
            data_count=0,
            execution_time=execution_time,
            ai_powered=False,
            cache_used=False
        )
    
    async def get_suggestions(self, partial_question: str = None) -> List[Dict[str, Any]]:
        """Get query suggestions based on schema and user context"""
        
        suggestions = []
        
        try:
            # Get recommended collections
            recommended = await self.schema_service.get_recommended_collections(limit=5)
            
            for collection_info in recommended:
                collection_name = collection_info["name"]
                
                # Generate example questions for this collection
                examples = self._generate_example_questions(collection_info)
                
                for example in examples:
                    suggestions.append({
                        "question": example,
                        "collection": collection_name,
                        "analytics_value": collection_info["analytics_value"],
                        "expected_chart": collection_info.get("suggested_charts", ["bar"])[0] if collection_info.get("suggested_charts") else "bar"
                    })
            
            # If partial question provided, filter relevant suggestions
            if partial_question:
                partial_lower = partial_question.lower()
                suggestions = [
                    s for s in suggestions
                    if any(word in s["question"].lower() for word in partial_lower.split())
                ]
            
            return suggestions[:10]  # Top 10 suggestions
            
        except Exception as e:
            logger.error(f"❌ Failed to generate suggestions: {str(e)}")
            return []
    
    def _generate_example_questions(self, collection_info: Dict[str, Any]) -> List[str]:
        """Generate example questions for a collection"""
        
        collection_name = collection_info["name"]
        classification = collection_info.get("classification", "unknown")
        
        examples = []
        
        # Collection-specific examples
        if "sales" in collection_name.lower():
            examples.extend([
                f"What are the top selling products?",
                f"Show me sales by month",
                f"Which region has the highest sales?"
            ])
        
        elif "orders" in collection_name.lower():
            examples.extend([
                f"How many orders were placed today?",
                f"What's the average order value?",
                f"Show me orders by status"
            ])
        
        elif "customers" in collection_name.lower() or "users" in collection_name.lower():
            examples.extend([
                f"How many active customers do we have?",
                f"Show me customer distribution by city",
                f"What's our customer growth rate?"
            ])
        
        # Generic examples based on classification
        if classification == "transactional":
            examples.extend([
                f"Show me {collection_name} trends over time",
                f"What are the top categories in {collection_name}?"
            ])
        
        elif classification == "reference":
            examples.extend([
                f"How many {collection_name} do we have?",
                f"Show me {collection_name} breakdown"
            ])
        
        return examples[:3]  # Limit to 3 examples per collection
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analytics service statistics"""
        
        success_rate = (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0
        ai_usage_rate = (self.ai_powered_queries / self.successful_queries * 100) if self.successful_queries > 0 else 0
        
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate_percent": round(success_rate, 2),
            "ai_powered_queries": self.ai_powered_queries,
            "fallback_queries": self.fallback_queries,
            "ai_usage_rate_percent": round(ai_usage_rate, 2),
            "gemini_stats": self.gemini_client.get_stats(),
            "database_service_available": self.database_service is not None,
            "schema_service_available": self.schema_service is not None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        
        health_status = {
            "status": "healthy",
            "components": {},
            "overall_health": True
        }
        
        # Check Gemini client
        gemini_health = await self.gemini_client.health_check()
        health_status["components"]["gemini"] = gemini_health
        if gemini_health.get("status") not in ["healthy", "unavailable"]:
            health_status["overall_health"] = False
        
        # Check database service
        try:
            db_health = await self.db_manager.health_check()
            health_status["components"]["database"] = db_health
            if db_health.get("status") != "healthy":
                health_status["overall_health"] = False
        except Exception as e:
            health_status["components"]["database"] = {"status": "error", "error": str(e)}
            health_status["overall_health"] = False
        
        # Check schema service
        try:
            schema_summary = await self.schema_service.get_schema_summary()
            health_status["components"]["schema"] = {
                "status": "healthy" if schema_summary.get("status") == "ready" else "degraded",
                "details": schema_summary
            }
        except Exception as e:
            health_status["components"]["schema"] = {"status": "error", "error": str(e)}
        
        # Check cache service
        cache_health = await self.cache_service.health_check()
        health_status["components"]["cache"] = cache_health
        
        # Set overall status
        if not health_status["overall_health"]:
            health_status["status"] = "degraded"
        
        return health_status