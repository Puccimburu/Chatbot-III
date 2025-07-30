"""
Data models for schema detection and representation
Pydantic models for type safety and validation
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class FieldType(str, Enum):
    """Enumeration of detected field types"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"
    OBJECTID = "objectid"
    NULL = "null"
    MIXED = "mixed"


class FieldRole(str, Enum):
    """Role of a field in analytics context"""
    DIMENSION = "dimension"      # Categorical, good for grouping
    METRIC = "metric"           # Numeric, good for aggregation
    IDENTIFIER = "identifier"    # Unique identifiers, foreign keys
    TEMPORAL = "temporal"       # Date/time fields
    METADATA = "metadata"       # System fields like createdAt, _id
    TEXT = "text"              # Text content, descriptions
    UNKNOWN = "unknown"         # Cannot determine role


class ChartType(str, Enum):
    """Suggested chart types for visualization"""
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    LINE = "line"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    AREA = "area"
    TABLE = "table"
    METRIC = "metric"


class FieldStatistics(BaseModel):
    """Statistical information about a field"""
    total_count: int = Field(description="Total documents analyzed")
    non_null_count: int = Field(description="Documents with non-null values")
    null_count: int = Field(description="Documents with null values")
    unique_count: int = Field(description="Number of unique values")
    cardinality: float = Field(description="Uniqueness ratio (unique/total)")
    
    # For numeric fields
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    avg_value: Optional[float] = None
    median_value: Optional[float] = None
    
    # For string fields
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # Sample values
    sample_values: List[Any] = Field(default_factory=list, description="Sample values for analysis")
    most_common: List[Dict[str, Any]] = Field(default_factory=list, description="Most common values with counts")
    
    @validator('cardinality')
    def validate_cardinality(cls, v):
        return max(0.0, min(1.0, v))


class FieldSchema(BaseModel):
    """Schema information for a single field"""
    name: str = Field(description="Field name")
    field_type: FieldType = Field(description="Detected field type")
    role: FieldRole = Field(description="Analytics role of the field")
    
    # Type detection confidence
    type_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in type detection")
    
    # Statistical information
    statistics: FieldStatistics = Field(description="Field statistics")
    
    # Patterns and formats
    detected_patterns: List[str] = Field(default_factory=list, description="Detected patterns (e.g., email, phone)")
    date_format: Optional[str] = None
    
    # Analytics metadata
    is_groupable: bool = Field(description="Suitable for grouping operations")
    is_aggregatable: bool = Field(description="Suitable for aggregation operations")
    is_filterable: bool = Field(description="Suitable for filtering operations")
    suggested_chart_types: List[ChartType] = Field(default_factory=list, description="Suggested visualization types")
    
    # Relationship hints
    possible_foreign_key: bool = Field(default=False, description="Might be a foreign key")
    referenced_collection: Optional[str] = None
    
    @validator('type_confidence')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))


class RelationshipType(str, Enum):
    """Types of relationships between collections"""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"
    EMBEDDED = "embedded"
    REFERENCE = "reference"


class CollectionRelationship(BaseModel):
    """Relationship between two collections"""
    from_collection: str = Field(description="Source collection")
    to_collection: str = Field(description="Target collection")
    from_field: str = Field(description="Field in source collection")
    to_field: str = Field(description="Field in target collection")
    relationship_type: RelationshipType = Field(description="Type of relationship")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in relationship detection")
    sample_matches: int = Field(description="Number of sample matches found")


class QueryPattern(BaseModel):
    """Common query pattern for a collection"""
    name: str = Field(description="Pattern name")
    description: str = Field(description="Human-readable description")
    aggregation_pipeline: List[Dict[str, Any]] = Field(description="MongoDB aggregation pipeline")
    result_type: str = Field(description="Type of result (metric, list, distribution)")
    suggested_chart_type: ChartType = Field(description="Best chart for this pattern")
    example_question: str = Field(description="Example natural language question")


class CollectionClassification(str, Enum):
    """Classification of collection by business purpose"""
    TRANSACTIONAL = "transactional"     # Core business transactions
    REFERENCE = "reference"             # Lookup tables, catalogs
    AUDIT = "audit"                     # Logs, history, tracking
    SYSTEM = "system"                   # Internal system collections
    STAGING = "staging"                 # Temporary, processing data
    ARCHIVE = "archive"                 # Historical data
    UNKNOWN = "unknown"                 # Cannot classify


class CollectionSchema(BaseModel):
    """Complete schema information for a collection"""
    name: str = Field(description="Collection name")
    classification: CollectionClassification = Field(description="Business classification")
    
    # Basic statistics
    document_count: int = Field(description="Total number of documents")
    avg_document_size: float = Field(description="Average document size in bytes")
    sample_size: int = Field(description="Number of documents sampled for analysis")
    
    # Schema analysis
    fields: Dict[str, FieldSchema] = Field(description="Field schemas by field name")
    detected_at: datetime = Field(description="When schema was detected")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall schema confidence")
    
    # Analytics metadata
    analytics_value: float = Field(ge=0.0, le=1.0, description="Estimated analytics value")
    query_patterns: List[QueryPattern] = Field(default_factory=list, description="Common query patterns")
    relationships: List[CollectionRelationship] = Field(default_factory=list, description="Relationships to other collections")
    
    # Performance hints
    recommended_indexes: List[Dict[str, Any]] = Field(default_factory=list, description="Recommended indexes")
    estimated_query_time: Optional[float] = None
    
    # Field categorization for quick access
    dimension_fields: List[str] = Field(default_factory=list, description="Fields suitable for dimensions")
    metric_fields: List[str] = Field(default_factory=list, description="Fields suitable for metrics")
    temporal_fields: List[str] = Field(default_factory=list, description="Date/time fields")
    identifier_fields: List[str] = Field(default_factory=list, description="ID fields")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._categorize_fields()
    
    def _categorize_fields(self):
        """Categorize fields by role for quick access"""
        self.dimension_fields = [
            name for name, field in self.fields.items() 
            if field.role == FieldRole.DIMENSION
        ]
        self.metric_fields = [
            name for name, field in self.fields.items() 
            if field.role == FieldRole.METRIC
        ]
        self.temporal_fields = [
            name for name, field in self.fields.items() 
            if field.role == FieldRole.TEMPORAL
        ]
        self.identifier_fields = [
            name for name, field in self.fields.items() 
            if field.role == FieldRole.IDENTIFIER
        ]
    
    def get_groupable_fields(self) -> List[str]:
        """Get fields suitable for grouping operations"""
        return [
            name for name, field in self.fields.items() 
            if field.is_groupable
        ]
    
    def get_aggregatable_fields(self) -> List[str]:
        """Get fields suitable for aggregation operations"""
        return [
            name for name, field in self.fields.items() 
            if field.is_aggregatable
        ]
    
    def get_time_series_potential(self) -> bool:
        """Check if collection has time series analysis potential"""
        return len(self.temporal_fields) > 0 and len(self.metric_fields) > 0
    
    def get_suggested_chart_types(self) -> List[ChartType]:
        """Get all suggested chart types for the collection"""
        chart_types = set()
        for field in self.fields.values():
            chart_types.update(field.suggested_chart_types)
        return list(chart_types)


class DatabaseSchema(BaseModel):
    """Complete database schema information"""
    database_name: str = Field(description="Database name")
    collections: Dict[str, CollectionSchema] = Field(description="Collection schemas by name")
    relationships: List[CollectionRelationship] = Field(default_factory=list, description="Cross-collection relationships")
    
    # Detection metadata
    detected_at: datetime = Field(description="When schema was detected")
    detection_duration: float = Field(description="Time taken for detection in seconds")
    total_documents_analyzed: int = Field(description="Total documents sampled across all collections")
    
    # Analytics insights
    recommended_collections: List[str] = Field(default_factory=list, description="Best collections for analytics")
    cross_collection_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Patterns across collections")
    
    def get_collection_by_value(self, min_analytics_value: float = 0.5) -> List[CollectionSchema]:
        """Get collections sorted by analytics value"""
        return sorted(
            [col for col in self.collections.values() if col.analytics_value >= min_analytics_value],
            key=lambda x: x.analytics_value,
            reverse=True
        )
    
    def get_collections_by_classification(self, classification: CollectionClassification) -> List[CollectionSchema]:
        """Get collections by classification type"""
        return [
            col for col in self.collections.values() 
            if col.classification == classification
        ]
    
    def find_related_collections(self, collection_name: str) -> List[str]:
        """Find collections related to the given collection"""
        related = set()
        for rel in self.relationships:
            if rel.from_collection == collection_name:
                related.add(rel.to_collection)
            elif rel.to_collection == collection_name:
                related.add(rel.from_collection)
        return list(related)


# Request/Response models for API
class SchemaDetectionRequest(BaseModel):
    """Request model for schema detection"""
    collections: Optional[List[str]] = None
    sample_size: Optional[int] = None
    force_refresh: bool = False
    include_relationships: bool = True


class SchemaDetectionResponse(BaseModel):
    """Response model for schema detection"""
    success: bool
    schema: Optional[DatabaseSchema] = None
    error: Optional[str] = None
    detection_time: float
    collections_analyzed: int
    total_documents_sampled: int