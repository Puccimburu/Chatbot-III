"""
Data models for query processing and responses
Request/Response models for the analytics API
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class AnalyticsRequest(BaseModel):
    """Request model for analytics queries"""
    question: str = Field(description="Natural language question")
    max_results: Optional[int] = Field(default=None, description="Maximum number of results")
    force_refresh: bool = Field(default=False, description="Force fresh data, ignore cache")
    preferred_chart_type: Optional[str] = Field(default=None, description="User preferred chart type")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        if len(v) > 500:
            raise ValueError("Question too long (max 500 characters)")
        return v.strip()
    
    @validator('max_results')
    def validate_max_results(cls, v):
        if v is not None and (v < 1 or v > 1000):
            raise ValueError("max_results must be between 1 and 1000")
        return v


class ChartData(BaseModel):
    """Chart configuration data"""
    type: str = Field(description="Chart type (bar, line, pie, etc.)")
    config: Dict[str, Any] = Field(description="Complete Chart.js configuration")
    title: str = Field(description="Chart title")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "bar",
                "config": {
                    "type": "bar",
                    "data": {
                        "labels": ["Product A", "Product B"],
                        "datasets": [{
                            "label": "Sales",
                            "data": [100, 200],
                            "backgroundColor": ["#3B82F6", "#EF4444"]
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "title": {
                                "display": True,
                                "text": "Sales by Product"
                            }
                        }
                    }
                },
                "title": "Sales by Product"
            }
        }


class AnalyticsResponse(BaseModel):
    """Response model for analytics queries"""
    success: bool = Field(description="Whether the request was successful")
    summary: str = Field(description="Human-readable summary of results")
    chart_data: Optional[ChartData] = Field(default=None, description="Chart configuration if applicable")
    chart_worthy: bool = Field(default=False, description="Whether data is suitable for visualization")
    chart_suggestion: Optional[str] = Field(default=None, description="Suggested chart type")
    insights: List[str] = Field(default_factory=list, description="Key insights from the data")
    data_count: int = Field(description="Number of data points returned")
    execution_time: float = Field(description="Query execution time in milliseconds")
    ai_powered: bool = Field(description="Whether AI was used for query generation")
    cache_used: bool = Field(default=False, description="Whether result came from cache")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "summary": "Your top selling product is MacBook Pro with $15,999 in revenue, followed by iPhone 15 Pro at $8,999.",
                "chart_data": {
                    "type": "bar",
                    "config": {"type": "bar", "data": {}, "options": {}},
                    "title": "Top Selling Products"
                },
                "chart_worthy": True,
                "chart_suggestion": "bar",
                "insights": ["MacBook Pro leads sales", "Strong performance in premium category"],
                "data_count": 10,
                "execution_time": 245.5,
                "ai_powered": True,
                "cache_used": False
            }
        }


class ChartRequest(BaseModel):
    """Request model for chart generation from existing data"""
    data: List[Dict[str, Any]] = Field(description="Raw data to visualize")
    chart_type: str = Field(description="Desired chart type")
    title: Optional[str] = Field(default=None, description="Chart title")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional chart options")


class ChartResponse(BaseModel):
    """Response model for chart generation"""
    success: bool = Field(description="Whether chart generation was successful")
    chart_data: Optional[ChartData] = Field(default=None, description="Generated chart configuration")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class SuggestionRequest(BaseModel):
    """Request model for getting query suggestions"""
    partial_question: Optional[str] = Field(default=None, description="Partial question for filtering")
    collection_hint: Optional[str] = Field(default=None, description="Hint about which collection to focus on")
    limit: int = Field(default=10, description="Maximum number of suggestions")


class QuerySuggestion(BaseModel):
    """Individual query suggestion"""
    question: str = Field(description="Suggested question")
    collection: str = Field(description="Target collection")
    analytics_value: float = Field(description="Analytics value score")
    expected_chart: str = Field(description="Expected chart type for this query")
    confidence: float = Field(default=0.5, description="Confidence in suggestion relevance")


class SuggestionResponse(BaseModel):
    """Response model for query suggestions"""
    success: bool = Field(description="Whether suggestion generation was successful")
    suggestions: List[QuerySuggestion] = Field(description="List of suggested queries")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ValidationRequest(BaseModel):
    """Request model for query validation"""
    collection_name: str = Field(description="Target collection name")
    requested_fields: List[str] = Field(description="Fields requested in query")
    operation_type: str = Field(default="aggregation", description="Type of operation")


class ValidationResult(BaseModel):
    """Field validation result"""
    field_name: str = Field(description="Field name")
    exists: bool = Field(description="Whether field exists in collection")
    suggestions: List[str] = Field(default_factory=list, description="Alternative field suggestions")
    field_info: Optional[Dict[str, Any]] = Field(default=None, description="Field metadata if exists")


class ValidationResponse(BaseModel):
    """Response model for query validation"""
    valid: bool = Field(description="Whether query is valid")
    collection_exists: bool = Field(description="Whether collection exists")
    field_results: List[ValidationResult] = Field(description="Validation results for each field")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Query improvement suggestions")
    error: Optional[str] = Field(default=None, description="Error message if validation failed")


class SystemStatusRequest(BaseModel):
    """Request model for system status"""
    include_detailed_stats: bool = Field(default=False, description="Include detailed component statistics")
    check_connectivity: bool = Field(default=True, description="Perform connectivity checks")


class ComponentStatus(BaseModel):
    """Status of individual system component"""
    name: str = Field(description="Component name")
    status: str = Field(description="Status (healthy, degraded, error)")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Detailed status information")
    last_check: datetime = Field(description="Last status check timestamp")


class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    overall_status: str = Field(description="Overall system status")
    components: List[ComponentStatus] = Field(description="Status of individual components")
    statistics: Optional[Dict[str, Any]] = Field(default=None, description="System statistics")
    timestamp: datetime = Field(description="Status check timestamp")


class SchemaRequest(BaseModel):
    """Request model for schema operations"""
    collections: Optional[List[str]] = Field(default=None, description="Specific collections to analyze")
    force_refresh: bool = Field(default=False, description="Force fresh schema detection")
    include_samples: bool = Field(default=True, description="Include sample data in response")


class CollectionInfo(BaseModel):
    """Collection information for schema response"""
    name: str = Field(description="Collection name")
    classification: str = Field(description="Business classification")
    document_count: int = Field(description="Number of documents")
    field_count: int = Field(description="Number of fields")
    analytics_value: float = Field(description="Analytics value score")
    suggested_charts: List[str] = Field(description="Suggested chart types")
    key_fields: Dict[str, List[str]] = Field(description="Key fields by role")


class SchemaResponse(BaseModel):
    """Response model for schema information"""
    success: bool = Field(description="Whether schema retrieval was successful")
    database_name: str = Field(description="Database name")
    collections: List[CollectionInfo] = Field(description="Collection information")
    recommended_collections: List[str] = Field(description="Collections recommended for analytics")
    detection_time: Optional[float] = Field(default=None, description="Schema detection time in seconds")
    cached: bool = Field(default=False, description="Whether result came from cache")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthCheckResponse(BaseModel):
    """Response model for health checks"""
    status: str = Field(description="Overall health status")
    timestamp: datetime = Field(description="Health check timestamp")
    components: Dict[str, Any] = Field(description="Component health details")
    version: str = Field(default="1.0.0", description="System version")
    uptime_seconds: Optional[float] = Field(default=None, description="System uptime in seconds")


class ExampleQuestionsRequest(BaseModel):
    """Request model for getting example questions"""
    collection_name: Optional[str] = Field(default=None, description="Generate examples for specific collection")
    category: Optional[str] = Field(default=None, description="Category of examples (sales, customers, etc.)")
    limit: int = Field(default=10, description="Maximum number of examples")


class ExampleQuestion(BaseModel):
    """Individual example question"""
    question: str = Field(description="Example question")
    collection: str = Field(description="Target collection")
    expected_result_type: str = Field(description="Expected result type (chart, metric, list)")
    difficulty: str = Field(description="Question difficulty (easy, medium, hard)")
    category: str = Field(description="Question category")


class ExampleQuestionsResponse(BaseModel):
    """Response model for example questions"""
    success: bool = Field(description="Whether example generation was successful")
    examples: List[ExampleQuestion] = Field(description="List of example questions")
    categories: List[str] = Field(description="Available question categories")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ChatMessage(BaseModel):
    """Individual chat message"""
    id: str = Field(description="Message ID")
    type: str = Field(description="Message type (user, assistant, system)")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional message metadata")


class ChatSession(BaseModel):
    """Chat session with message history"""
    session_id: str = Field(description="Session ID")
    messages: List[ChatMessage] = Field(description="Message history")
    created_at: datetime = Field(description="Session creation time")
    last_activity: datetime = Field(description="Last activity timestamp")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Session context")


class ChatRequest(BaseModel):
    """Request model for chat-based analytics"""
    session_id: Optional[str] = Field(default=None, description="Existing session ID")
    message: str = Field(description="User message/question")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Response model for chat-based analytics"""
    session_id: str = Field(description="Session ID")
    message_id: str = Field(description="Response message ID")
    response: AnalyticsResponse = Field(description="Analytics response")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    session_context: Optional[Dict[str, Any]] = Field(default=None, description="Updated session context")


class BulkAnalyticsRequest(BaseModel):
    """Request model for bulk analytics processing"""
    questions: List[str] = Field(description="List of questions to process")
    max_results_per_query: Optional[int] = Field(default=None, description="Max results per individual query")
    parallel_processing: bool = Field(default=True, description="Whether to process queries in parallel")
    return_errors: bool = Field(default=True, description="Whether to return failed queries in response")


class BulkAnalyticsResult(BaseModel):
    """Individual result in bulk processing"""
    question: str = Field(description="Original question")
    response: Optional[AnalyticsResponse] = Field(default=None, description="Analytics response if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_order: int = Field(description="Order in which this query was processed")


class BulkAnalyticsResponse(BaseModel):
    """Response model for bulk analytics processing"""
    success: bool = Field(description="Whether bulk processing completed")
    results: List[BulkAnalyticsResult] = Field(description="Results for each question")
    total_questions: int = Field(description="Total number of questions processed")
    successful_count: int = Field(description="Number of successful queries")
    failed_count: int = Field(description="Number of failed queries")
    total_execution_time: float = Field(description="Total execution time in milliseconds")
    parallel_processed: bool = Field(description="Whether queries were processed in parallel")


class ExportRequest(BaseModel):
    """Request model for exporting analytics results"""
    query_result: AnalyticsResponse = Field(description="Analytics result to export")
    format: str = Field(description="Export format (csv, json, excel)")
    include_chart: bool = Field(default=True, description="Whether to include chart image")
    filename: Optional[str] = Field(default=None, description="Custom filename")


class ExportResponse(BaseModel):
    """Response model for analytics export"""
    success: bool = Field(description="Whether export was successful")
    download_url: Optional[str] = Field(default=None, description="Download URL for exported file")
    filename: str = Field(description="Generated filename")
    format: str = Field(description="Export format used")
    file_size_bytes: Optional[int] = Field(default=None, description="File size in bytes")
    expires_at: Optional[datetime] = Field(default=None, description="Download link expiration")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DatabaseStatsRequest(BaseModel):
    """Request model for database statistics"""
    include_collection_details: bool = Field(default=True, description="Include per-collection statistics")
    sample_data: bool = Field(default=False, description="Include sample data in statistics")


class CollectionStats(BaseModel):
    """Statistics for individual collection"""
    name: str = Field(description="Collection name")
    document_count: int = Field(description="Number of documents")
    size_bytes: int = Field(description="Storage size in bytes")
    avg_document_size: float = Field(description="Average document size")
    indexes: List[str] = Field(description="Available indexes")
    last_modified: Optional[datetime] = Field(default=None, description="Last modification time")


class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics"""
    success: bool = Field(description="Whether stats retrieval was successful")
    database_name: str = Field(description="Database name")
    total_collections: int = Field(description="Total number of collections")
    total_documents: int = Field(description="Total number of documents across all collections")
    total_size_bytes: int = Field(description="Total database size in bytes")
    collections: List[CollectionStats] = Field(description="Per-collection statistics")
    server_info: Optional[Dict[str, Any]] = Field(default=None, description="Database server information")
    generated_at: datetime = Field(description="Statistics generation timestamp")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# Utility classes for internal use
class QueryExecutionContext(BaseModel):
    """Internal context for query execution"""
    collection_name: str
    pipeline: List[Dict[str, Any]]
    query_intent: str
    user_question: str
    execution_start: datetime
    ai_generated: bool = False
    schema_used: Optional[str] = None


class ChartAnalysisContext(BaseModel):
    """Internal context for chart analysis"""
    user_question: str
    data_sample: List[Dict[str, Any]]
    total_count: int
    query_context: QueryExecutionContext
    analysis_start: datetime


class CacheKey(BaseModel):
    """Cache key structure for consistent caching"""
    type: str = Field(description="Cache type (query, schema, chart)")
    identifier: str = Field(description="Unique identifier")
    version: str = Field(default="1.0", description="Cache version")
    
    def to_string(self) -> str:
        """Convert to cache key string"""
        return f"{self.type}:{self.identifier}:v{self.version}"


# Response metadata for API versioning and tracking
class ResponseMetadata(BaseModel):
    """Metadata included in API responses"""
    api_version: str = Field(default="1.0", description="API version")
    request_id: str = Field(description="Unique request identifier")
    processing_time_ms: float = Field(description="Total processing time")
    cache_hit: bool = Field(default=False, description="Whether response came from cache")
    ai_powered: bool = Field(default=False, description="Whether AI was used in processing")
    timestamp: datetime = Field(description="Response timestamp")


# Error response models
class APIError(BaseModel):
    """Standardized API error response"""
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix or alternative")
    metadata: ResponseMetadata = Field(description="Response metadata")


# Query processing states for advanced workflows
class QueryProcessingState(BaseModel):
    """State tracking for complex query processing"""
    query_id: str = Field(description="Unique query identifier")
    status: str = Field(description="Processing status (pending, processing, completed, failed)")
    stages_completed: List[str] = Field(description="List of completed processing stages")
    current_stage: Optional[str] = Field(default=None, description="Currently executing stage")
    progress_percent: float = Field(description="Processing progress percentage")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    created_at: datetime = Field(description="Query creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class AsyncQueryRequest(BaseModel):
    """Request model for asynchronous query processing"""
    question: str = Field(description="Natural language question")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL for completion notification")
    priority: str = Field(default="normal", description="Processing priority (low, normal, high)")
    timeout_seconds: Optional[int] = Field(default=None, description="Query timeout in seconds")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class AsyncQueryResponse(BaseModel):
    """Response model for asynchronous query submission"""
    success: bool = Field(description="Whether query was successfully queued")
    query_id: str = Field(description="Unique identifier for tracking")
    status: str = Field(description="Initial processing status")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    polling_url: str = Field(description="URL to check query status")
    error: Optional[str] = Field(default=None, description="Error message if submission failed")


# Advanced analytics models
class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis"""
    metrics: List[str] = Field(description="Metrics to analyze for trends")
    time_field: str = Field(description="Field to use for time-based analysis")
    time_range: Optional[Dict[str, Any]] = Field(default=None, description="Time range filter")
    granularity: str = Field(default="daily", description="Time granularity (hourly, daily, weekly, monthly)")


class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis"""
    success: bool = Field(description="Whether trend analysis was successful")
    trends: List[Dict[str, Any]] = Field(description="Detected trends for each metric")
    seasonality: Optional[Dict[str, Any]] = Field(default=None, description="Seasonality patterns detected")
    forecasts: Optional[List[Dict[str, Any]]] = Field(default=None, description="Future trend forecasts")
    insights: List[str] = Field(description="Key insights from trend analysis")
    chart_data: Optional[ChartData] = Field(default=None, description="Trend visualization")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ComparisonRequest(BaseModel):
    """Request model for comparative analysis"""
    primary_metric: str = Field(description="Primary metric to compare")
    comparison_dimensions: List[str] = Field(description="Dimensions to compare across")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filters to apply")
    time_range: Optional[Dict[str, Any]] = Field(default=None, description="Time range for comparison")


class ComparisonResponse(BaseModel):
    """Response model for comparative analysis"""
    success: bool = Field(description="Whether comparison was successful")
    comparisons: List[Dict[str, Any]] = Field(description="Comparison results")
    rankings: Optional[List[Dict[str, str]]] = Field(default=None, description="Ranked results")
    statistical_significance: Optional[Dict[str, Any]] = Field(default=None, description="Statistical significance tests")
    insights: List[str] = Field(description="Key insights from comparison")
    chart_data: Optional[ChartData] = Field(default=None, description="Comparison visualization")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# User preference and personalization models
class UserPreferences(BaseModel):
    """User preferences for analytics"""
    preferred_chart_types: List[str] = Field(default_factory=list, description="Preferred chart types")
    default_result_limit: int = Field(default=50, description="Default number of results")
    favorite_collections: List[str] = Field(default_factory=list, description="Frequently used collections")
    dashboard_layout: Optional[Dict[str, Any]] = Field(default=None, description="Custom dashboard configuration")
    notification_settings: Optional[Dict[str, Any]] = Field(default=None, description="Notification preferences")


class PersonalizationRequest(BaseModel):
    """Request model for personalized analytics"""
    user_id: str = Field(description="User identifier")
    question: str = Field(description="Natural language question")
    use_preferences: bool = Field(default=True, description="Whether to apply user preferences")
    context_history: Optional[List[str]] = Field(default=None, description="Recent query history for context")


class PersonalizationResponse(BaseModel):
    """Response model for personalized analytics"""
    response: AnalyticsResponse = Field(description="Personalized analytics response")
    personalization_applied: List[str] = Field(description="List of personalizations applied")
    suggested_preferences: Optional[Dict[str, Any]] = Field(default=None, description="Suggested preference updates")
    context_used: bool = Field(description="Whether historical context was used")
    cached_result: bool = Field(description="Whether result came from cache")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthCheckResponse(BaseModel):
    """Response model for health checks"""
    status: str = Field(description="Overall health status")
    timestamp: datetime = Field(description="Health check timestamp")
    components: Dict[str, Any] = Field(description="Component health details")
    version: str = Field(default="1.0.0", description="System version")
    uptime_seconds: Optional[float] = Field(default=None, description="System uptime in seconds")


class ExampleQuestionsRequest(BaseModel):
    """Request model for getting example questions"""
    collection_name: Optional[str] = Field(default=None, description="Generate examples for specific collection")
    category: Optional[str] = Field(default=None, description="Category of examples (sales, customers, etc.)")
    limit: int = Field(default=10, description="Maximum number of examples")


class ExampleQuestion(BaseModel):
    """Individual example question"""
    question: str = Field(description="Example question")
    collection: str = Field(description="Target collection")
    expected_result_type: str = Field(description="Expected result type (chart, metric, list)")
    difficulty: str = Field(description="Question difficulty (easy, medium, hard)")
    category: str = Field(description="Question category")


class ExampleQuestionsResponse(BaseModel):
    """Response model for example questions"""
    success: bool = Field(description="Whether example generation was successful")
    examples: List[ExampleQuestion] = Field(description="List of example questions")
    categories: List[str] = Field(description="Available question categories")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ChatMessage(BaseModel):
    """Individual chat message"""
    id: str = Field(description="Message ID")
    type: str = Field(description="Message type (user, assistant, system)")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional message metadata")


class ChatSession(BaseModel):
    """Chat session with message history"""
    session_id: str = Field(description="Session ID")
    messages: List[ChatMessage] = Field(description="Message history")
    created_at: datetime = Field(description="Session creation time")
    last_activity: datetime = Field(description="Last activity timestamp")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Session context")


class ChatRequest(BaseModel):
    """Request model for chat-based analytics"""
    session_id: Optional[str] = Field(default=None, description="Existing session ID")
    message: str = Field(description="User message/question")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Response model for chat-based analytics"""
    session_id: str = Field(description="Session ID")
    message_id: str = Field(description="Response message ID")
    response: AnalyticsResponse = Field(description="Analytics response")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    session_context: Optional[Dict[str, Any]] = Field(default=None, description="Updated session context")


class BulkAnalyticsRequest(BaseModel):
    """Request model for bulk analytics processing"""
    questions: List[str] = Field(description="List of questions to process")
    max_results_per_query: Optional[int] = Field(default=None, description="Max results per individual query")
    parallel_processing: bool = Field(default=True, description="Whether to process queries in parallel")
    return_errors: bool = Field(default=True, description="Whether to return failed queries in response")


class BulkAnalyticsResult(BaseModel):
    """Individual result in bulk processing"""
    question: str = Field(description="Original question")
    response: Optional[AnalyticsResponse] = Field(default=None, description="Analytics response if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_order: int = Field(description="Order in which this query was processed")


class BulkAnalyticsResponse(BaseModel):
    """Response model for bulk analytics processing"""
    success: bool = Field(description="Whether bulk processing completed")
    results: List[BulkAnalyticsResult] = Field(description="Results for each question")
    total_questions: int = Field(description="Total number of questions processed")
    successful_count: int = Field(description="Number of successful queries")
    failed_count: int = Field(description="Number of failed queries")
    total_execution_time: float = Field(description="Total execution time in milliseconds")
    parallel_processed: bool = Field(description="Whether queries were processed in parallel")


class ExportRequest(BaseModel):
    """Request model for exporting analytics results"""
    query_result: AnalyticsResponse = Field(description="Analytics result to export")
    format: str = Field(description="Export format (csv, json, excel)")
    include_chart: bool = Field(default=True, description="Whether to include chart image")
    filename: Optional[str] = Field(default=None, description="Custom filename")


class ExportResponse(BaseModel):
    """Response model for analytics export"""
    success: bool = Field(description="Whether export was successful")
    download_url: Optional[str] = Field(default=None, description="Download URL for exported file")
    filename: str = Field(description="Generated filename")
    format: str = Field(description="Export format used")
    file_size_bytes: Optional[int] = Field(default=None, description="File size in bytes")
    expires_at: Optional[datetime] = Field(default=None, description="Download link expiration")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DatabaseStatsRequest(BaseModel):
    """Request model for database statistics"""
    include_collection_details: bool = Field(default=True, description="Include per-collection statistics")
    sample_data: bool = Field(default=False, description="Include sample data in statistics")


class CollectionStats(BaseModel):
    """Statistics for individual collection"""
    name: str = Field(description="Collection name")
    document_count: int = Field(description="Number of documents")
    size_bytes: int = Field(description="Storage size in bytes")
    avg_document_size: float = Field(description="Average document size")
    indexes: List[str] = Field(description="Available indexes")
    last_modified: Optional[datetime] = Field(default=None, description="Last modification time")


class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics"""
    success: bool = Field(description="Whether stats retrieval was successful")
    database_name: str = Field(description="Database name")
    total_collections: int = Field(description="Total number of collections")
    total_documents: int = Field(description="Total number of documents across all collections")
    total_size_bytes: int = Field(description="Total database size in bytes")
    collections: List[CollectionStats] = Field(description="Per-collection statistics")
    server_info: Optional[Dict[str, Any]] = Field(default=None, description="Database server information")
    generated_at: datetime = Field(description="Statistics generation timestamp")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# Utility classes for internal use
class QueryExecutionContext(BaseModel):
    """Internal context for query execution"""
    collection_name: str
    pipeline: List[Dict[str, Any]]
    query_intent: str
    user_question: str
    execution_start: datetime
    ai_generated: bool = False
    schema_used: Optional[str] = None


class ChartAnalysisContext(BaseModel):
    """Internal context for chart analysis"""
    user_question: str
    data_sample: List[Dict[str, Any]]
    total_count: int
    query_context: QueryExecutionContext
    analysis_start: datetime


class CacheKey(BaseModel):
    """Cache key structure for consistent caching"""
    type: str = Field(description="Cache type (query, schema, chart)")
    identifier: str = Field(description="Unique identifier")
    version: str = Field(default="1.0", description="Cache version")
    
    def to_string(self) -> str:
        """Convert to cache key string"""
        return f"{self.type}:{self.identifier}:v{self.version}"


# Response metadata for API versioning and tracking
class ResponseMetadata(BaseModel):
    """Metadata included in API responses"""
    api_version: str = Field(default="1.0", description="API version")
    request_id: str = Field(description="Unique request identifier")
    processing_time_ms: float = Field(description="Total processing time")
    cache_hit: bool = Field(default=False, description="Whether response came from cache")
    ai_powered: bool = Field(default=False, description="Whether AI was used in processing")
    timestamp: datetime = Field(description="Response timestamp")


# Error response models
class APIError(BaseModel):
    """Standardized API error response"""
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix or alternative")
    metadata: ResponseMetadata = Field(description="Response metadata")