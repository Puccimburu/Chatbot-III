"""
Gemini AI Client for query generation and visualization suggestions
Enhanced with retry logic, response validation, and intelligent prompting
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from config.settings import settings
from models.schema_models import DatabaseSchema, CollectionSchema
from utils.logging_config import log_gemini_interaction, monitor_performance

logger = logging.getLogger(__name__)


class GeminiClient:
    """Enhanced Gemini client for conversational analytics with bulletproof reliability"""
    
    def __init__(self):
        self.model = None
        self.available = False
        self.request_count = 0
        self.error_count = 0
        self.total_tokens_used = 0
        
        # Configuration for different stages
        self.query_generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Very low for consistent query generation
            max_output_tokens=1500,
            top_p=0.8,
            top_k=40
        ) if GEMINI_AVAILABLE else None
        
        self.visualization_config = genai.types.GenerationConfig(
            temperature=0.2,  # Slightly higher for creative visualizations
            max_output_tokens=2000,
            top_p=0.9,
            top_k=50
        ) if GEMINI_AVAILABLE else None
    
    async def initialize(self):
        """Initialize Gemini client with API key validation"""
        
        if not GEMINI_AVAILABLE:
            logger.warning("⚠️ Google Generative AI library not available")
            return
        
        if not settings.GEMINI_API_KEY:
            logger.warning("⚠️ Gemini API key not configured")
            return
        
        try:
            logger.info("🤖 Initializing Gemini AI client...")
            
            # Configure Gemini
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Initialize model
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            
            # Test the connection
            await self._test_connection()
            
            self.available = True
            logger.info("✅ Gemini AI client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini client: {str(e)}")
            self.available = False
    
    async def _test_connection(self):
        """Test Gemini API connection with a simple request"""
        
        try:
            test_prompt = "Respond with just 'OK' if you can see this message."
            response = await asyncio.wait_for(
                self._make_request(test_prompt, self.query_generation_config),
                timeout=10
            )
            
            if not response or "ok" not in response.lower():
                raise Exception("Invalid test response")
            
            logger.debug("✅ Gemini connection test successful")
            
        except Exception as e:
            raise Exception(f"Gemini connection test failed: {str(e)}")
    
    @monitor_performance("generate_mongodb_query")
    async def generate_mongodb_query(
        self, 
        user_question: str, 
        database_schema: DatabaseSchema,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate MongoDB aggregation query from natural language
        
        Args:
            user_question: Natural language question
            database_schema: Complete database schema
            max_retries: Maximum retry attempts
            
        Returns:
            Dict containing query information or error
        """
        
        if not self.available:
            return {
                "success": False,
                "error": "Gemini AI not available",
                "fallback_required": True
            }
        
        start_time = time.perf_counter()
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"🔍 Generating MongoDB query (attempt {attempt + 1})")
                
                # Build enhanced prompt
                prompt = self._build_query_generation_prompt(user_question, database_schema)
                
                # Make request with timeout
                response_text = await asyncio.wait_for(
                    self._make_request(prompt, self.query_generation_config),
                    timeout=settings.GEMINI_TIMEOUT
                )
                
                if not response_text:
                    raise ValueError("Empty response from Gemini")
                
                # Extract and validate JSON
                query_data = self._extract_json_from_response(response_text)
                
                if not query_data:
                    raise ValueError("Could not extract valid JSON from response")
                
                # Validate query structure
                validation_result = self._validate_query_response(query_data, database_schema)
                
                if not validation_result["valid"]:
                    raise ValueError(f"Query validation failed: {validation_result['error']}")
                
                # Success
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.request_count += 1
                
                # Estimate tokens (rough approximation)
                estimated_tokens = len(prompt.split()) + len(response_text.split())
                self.total_tokens_used += estimated_tokens
                
                log_gemini_interaction("query_generation", estimated_tokens, duration_ms, True)
                
                return {
                    "success": True,
                    "data": query_data,
                    "attempts": attempt + 1,
                    "duration_ms": duration_ms
                }
                
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Gemini request timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.warning(f"⚠️ Gemini query generation error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 + attempt)
        
        # All attempts failed
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.error_count += 1
        
        log_gemini_interaction("query_generation", 0, duration_ms, False)
        
        return {
            "success": False,
            "error": f"Failed after {max_retries} attempts",
            "fallback_required": True,
            "attempts": max_retries
        }
    
    @monitor_performance("analyze_chart_worthiness")
    async def analyze_chart_worthiness(
        self, 
        user_question: str, 
        query_results: List[Dict[str, Any]],
        query_context: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze if data is worth visualizing and suggest chart types
        
        Args:
            user_question: Original user question
            query_results: Results from database query
            query_context: Context about the query executed
            max_retries: Maximum retry attempts
            
        Returns:
            Dict containing chart analysis and suggestions
        """
        
        if not self.available:
            return self._create_fallback_chart_analysis(query_results)
        
        if not query_results or len(query_results) == 0:
            return {
                "chart_worthy": False,
                "reason": "No data to visualize"
            }
        
        start_time = time.perf_counter()
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"📊 Analyzing chart worthiness (attempt {attempt + 1})")
                
                # Build chart analysis prompt
                prompt = self._build_chart_analysis_prompt(
                    user_question, query_results, query_context
                )
                
                # Make request
                response_text = await asyncio.wait_for(
                    self._make_request(prompt, self.visualization_config),
                    timeout=settings.GEMINI_TIMEOUT
                )
                
                if not response_text:
                    raise ValueError("Empty response from Gemini")
                
                # Extract and validate JSON
                chart_data = self._extract_json_from_response(response_text)
                
                if not chart_data:
                    raise ValueError("Could not extract valid JSON from response")
                
                # Validate chart analysis structure
                if not self._validate_chart_analysis_response(chart_data):
                    raise ValueError("Invalid chart analysis structure")
                
                # Success
                duration_ms = (time.perf_counter() - start_time) * 1000
                estimated_tokens = len(prompt.split()) + len(response_text.split())
                self.total_tokens_used += estimated_tokens
                
                log_gemini_interaction("chart_analysis", estimated_tokens, duration_ms, True)
                
                return {
                    "success": True,
                    "data": chart_data,
                    "attempts": attempt + 1
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Gemini chart analysis error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 + attempt)
        
        # All attempts failed, return fallback
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_gemini_interaction("chart_analysis", 0, duration_ms, False)
        
        return self._create_fallback_chart_analysis(query_results)
    
    async def _make_request(self, prompt: str, config: Any) -> str:
        """Make a request to Gemini with error handling"""
        
        try:
            response = self.model.generate_content(prompt, generation_config=config)
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini API")
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"❌ Gemini API request failed: {str(e)}")
            raise
    
    def _build_query_generation_prompt(
        self, 
        user_question: str, 
        database_schema: DatabaseSchema
    ) -> str:
        """Build comprehensive prompt for MongoDB query generation"""
        
        # Get top collections by analytics value
        top_collections = database_schema.get_collection_by_value(min_analytics_value=0.4)[:5]
        
        # Build schema context
        schema_context = {}
        for collection in top_collections:
            schema_context[collection.name] = {
                "fields": list(collection.fields.keys())[:10],  # Limit fields for prompt size
                "metric_fields": collection.metric_fields[:5],
                "dimension_fields": collection.dimension_fields[:5],
                "temporal_fields": collection.temporal_fields,
                "document_count": collection.document_count,
                "classification": collection.classification.value
            }
        
        prompt = f"""You are an expert MongoDB query generator for conversational analytics.

USER QUESTION: "{user_question}"

AVAILABLE COLLECTIONS AND SCHEMA:
{json.dumps(schema_context, indent=2)}

CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON - no markdown, no explanations, no extra text
2. Use exact field names from the schema provided
3. Create efficient MongoDB aggregation pipelines
4. Always include $sort and $limit for performance
5. Choose the most relevant collection based on the question

RESPONSE FORMAT (JSON only):
{{
  "collection": "collection_name",
  "pipeline": [
    {{"$match": {{"field": "value"}}}},
    {{"$group": {{"_id": "$field", "metric": {{"$sum": "$value"}}}}}},
    {{"$sort": {{"metric": -1}}}},
    {{"$limit": 10}}
  ],
  "query_intent": "Brief description of what this query achieves",
  "expected_result_type": "aggregation|list|single_value",
  "confidence": 0.85
}}

EXAMPLES:

For "What are our top selling products?":
{{
  "collection": "sales",
  "pipeline": [
    {{"$group": {{"_id": "$product_name", "total_sales": {{"$sum": "$total_amount"}}}}}},
    {{"$sort": {{"total_sales": -1}}}},
    {{"$limit": 10}}
  ],
  "query_intent": "Find products with highest total sales revenue",
  "expected_result_type": "aggregation",
  "confidence": 0.9
}}

For "How many orders were placed today?":
{{
  "collection": "orders",
  "pipeline": [
    {{"$match": {{"order_date": {{"$gte": new Date(new Date().setHours(0,0,0,0))}}}}}},
    {{"$count": "today_orders"}}
  ],
  "query_intent": "Count orders placed today",
  "expected_result_type": "single_value",
  "confidence": 0.95
}}

JSON only - no other text:"""
        
        return prompt
    
    def _build_chart_analysis_prompt(
        self, 
        user_question: str, 
        query_results: List[Dict[str, Any]], 
        query_context: Dict[str, Any]
    ) -> str:
        """Build prompt for chart worthiness analysis"""
        
        # Sample the data for analysis
        sample_data = query_results[:5] if query_results else []
        
        prompt = f"""You are a data visualization expert. Analyze if this data should be visualized and suggest the best chart type.

USER QUESTION: "{user_question}"
QUERY CONTEXT: {json.dumps(query_context, indent=2)}
SAMPLE DATA: {json.dumps(sample_data, indent=2)}
TOTAL RECORDS: {len(query_results)}

CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON - no markdown, no explanations
2. Determine if this data is worth visualizing
3. If yes, suggest the best chart type and provide Chart.js configuration
4. If no, explain why in the reason field

RESPONSE FORMAT (JSON only):
{{
  "chart_worthy": true|false,
  "reason": "explanation for chart worthiness decision",
  "suggested_chart_type": "bar|line|pie|doughnut|scatter|histogram|table|metric",
  "chart_config": {{
    "type": "bar",
    "data": {{
      "labels": ["label1", "label2"],
      "datasets": [{{
        "label": "Dataset Label",
        "data": [10, 20],
        "backgroundColor": ["#3B82F6", "#EF4444"]
      }}]
    }},
    "options": {{
      "responsive": true,
      "plugins": {{
        "title": {{
          "display": true,
          "text": "Chart Title"
        }}
      }}
    }}
  }},
  "summary": "Brief text summary of the data",
  "insights": ["insight 1", "insight 2"],
  "confidence": 0.85
}}

CHART WORTHINESS CRITERIA:
- Multiple data points (>1): Usually chart-worthy
- Single value: Use metric display, not a chart
- Comparisons: Perfect for bar/pie charts
- Time series: Perfect for line charts
- Distributions: Good for pie/doughnut charts
- Large datasets (>50 items): Consider table or simplified chart

JSON only - no other text:"""
        
        return prompt
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from Gemini response with multiple fallback methods"""
        
        # Method 1: Direct JSON parsing
        try:
            cleaned_text = response_text.strip()
            result = json.loads(cleaned_text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract from code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
        
        # Method 3: Find JSON object in text
        brace_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        matches = re.findall(brace_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and len(result) > 1:  # Must have multiple keys
                    return result
            except json.JSONDecodeError:
                continue
        
        logger.warning("⚠️ Could not extract valid JSON from Gemini response")
        return None
    
    def _validate_query_response(
        self, 
        query_data: Dict[str, Any], 
        database_schema: DatabaseSchema
    ) -> Dict[str, Any]:
        """Validate MongoDB query response structure"""
        
        required_fields = ["collection", "pipeline", "query_intent"]
        
        # Check required fields
        for field in required_fields:
            if field not in query_data:
                return {
                    "valid": False,
                    "error": f"Missing required field: {field}"
                }
        
        # Validate collection exists
        collection_name = query_data["collection"]
        if collection_name not in database_schema.collections:
            return {
                "valid": False,
                "error": f"Collection '{collection_name}' not found in schema"
            }
        
        # Validate pipeline is a list
        pipeline = query_data["pipeline"]
        if not isinstance(pipeline, list):
            return {
                "valid": False,
                "error": "Pipeline must be a list"
            }
        
        # Basic pipeline validation
        if len(pipeline) == 0:
            return {
                "valid": False,
                "error": "Pipeline cannot be empty"
            }
        
        return {"valid": True}
    
    def _validate_chart_analysis_response(self, chart_data: Dict[str, Any]) -> bool:
        """Validate chart analysis response structure"""
        
        required_fields = ["chart_worthy", "reason"]
        
        # Check required fields
        for field in required_fields:
            if field not in chart_data:
                return False
        
        # If chart worthy, need additional fields
        if chart_data.get("chart_worthy"):
            chart_fields = ["suggested_chart_type", "summary"]
            for field in chart_fields:
                if field not in chart_data:
                    return False
        
        return True
    
    def _create_fallback_chart_analysis(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback chart analysis when Gemini fails"""
        
        if not query_results:
            return {
                "chart_worthy": False,
                "reason": "No data to visualize",
                "confidence": 1.0
            }
        
        # Simple heuristics for chart worthiness
        result_count = len(query_results)
        
        if result_count == 1:
            # Single value - not chart worthy
            return {
                "chart_worthy": False,
                "reason": "Single value result - better displayed as a metric",
                "confidence": 0.9
            }
        
        elif result_count <= 10:
            # Small dataset - good for pie or bar chart
            return {
                "chart_worthy": True,
                "reason": f"Dataset with {result_count} items suitable for visualization",
                "suggested_chart_type": "bar",
                "summary": f"Displaying {result_count} data points",
                "confidence": 0.7
            }
        
        else:
            # Larger dataset - bar chart or table
            return {
                "chart_worthy": True,
                "reason": f"Dataset with {result_count} items suitable for chart or table",
                "suggested_chart_type": "bar" if result_count <= 50 else "table",
                "summary": f"Displaying {result_count} data points",
                "confidence": 0.6
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Gemini client statistics"""
        
        return {
            "available": self.available,
            "model": settings.GEMINI_MODEL if self.available else None,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.request_count - self.error_count) / self.request_count * 100
                if self.request_count > 0 else 0
            ),
            "total_tokens_used": self.total_tokens_used,
            "api_key_configured": settings.GEMINI_API_KEY is not None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Gemini client"""
        
        if not self.available:
            return {
                "status": "unavailable",
                "reason": "Gemini client not initialized or API key missing"
            }
        
        try:
            # Quick test request
            test_response = await asyncio.wait_for(
                self._make_request(
                    "Respond with just 'healthy' if you receive this.", 
                    self.query_generation_config
                ),
                timeout=5
            )
            
            if "healthy" in test_response.lower():
                return {
                    "status": "healthy",
                    **self.get_stats()
                }
            else:
                return {
                    "status": "unhealthy",
                    "reason": "Unexpected test response",
                    **self.get_stats()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                **self.get_stats()
            }