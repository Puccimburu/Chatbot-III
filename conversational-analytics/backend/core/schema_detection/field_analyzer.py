"""
Field type detection and analysis
Intelligent analysis of field types, patterns, and analytics roles
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
from collections import Counter
import statistics

from models.schema_models import (
    FieldSchema, FieldType, FieldRole, FieldStatistics, ChartType
)

logger = logging.getLogger(__name__)


class FieldAnalyzer:
    """Intelligent field analysis for schema detection"""
    
    def __init__(self):
        # Patterns for field detection
        self.date_patterns = [
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO datetime
            r'^\d{4}-\d{2}-\d{2}$',                     # Date YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',                     # Date MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',                     # Date MM-DD-YYYY
        ]
        
        self.email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        self.phone_pattern = r'^[\+]?[1-9]?[\d\s\-\(\)]{7,}$'
        self.url_pattern = r'^https?://[^\s]+$'
        self.ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        
        # System field names
        self.system_fields = {
            '_id', '__v', 'createdAt', 'updatedAt', 'created_at', 'updated_at',
            'deletedAt', 'deleted_at', 'timestamp', 'last_modified', 'version'
        }
        
        # ID field patterns
        self.id_field_patterns = [
            r'.*[Ii]d$', r'.*[Ii][Dd]$', r'.*_id$', r'.*ID$', r'^id$', r'^ID$'
        ]
    
    async def analyze_fields(self, documents: List[Dict[str, Any]]) -> Dict[str, FieldSchema]:
        """
        Analyze all fields in a collection of documents
        
        Args:
            documents: List of sample documents
            
        Returns:
            Dict mapping field names to FieldSchema objects
        """
        
        if not documents:
            return {}
        
        logger.debug(f"🔍 Analyzing fields in {len(documents)} documents")
        
        # Collect all field information
        field_data = self._collect_field_data(documents)
        
        # Analyze each field
        field_schemas = {}
        for field_name, field_info in field_data.items():
            try:
                schema = self._analyze_single_field(field_name, field_info, len(documents))
                field_schemas[field_name] = schema
            except Exception as e:
                logger.warning(f"⚠️ Failed to analyze field {field_name}: {str(e)}")
        
        logger.debug(f"✅ Successfully analyzed {len(field_schemas)} fields")
        return field_schemas
    
    def _collect_field_data(self, documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Collect comprehensive data about each field across all documents"""
        
        field_data = {}
        
        for doc in documents:
            for field_name, value in doc.items():
                if field_name not in field_data:
                    field_data[field_name] = {
                        'values': [],
                        'types': [],
                        'nulls': 0,
                        'total': 0
                    }
                
                field_data[field_name]['total'] += 1
                
                if value is None:
                    field_data[field_name]['nulls'] += 1
                else:
                    field_data[field_name]['values'].append(value)
                    field_data[field_name]['types'].append(type(value).__name__)
        
        return field_data
    
    def _analyze_single_field(
        self, 
        field_name: str, 
        field_info: Dict[str, Any], 
        total_documents: int
    ) -> FieldSchema:
        """Analyze a single field comprehensively"""
        
        values = field_info['values']
        types = field_info['types']
        null_count = field_info['nulls']
        
        # Detect field type
        field_type, type_confidence = self._detect_field_type(values, types)
        
        # Calculate statistics
        statistics = self._calculate_field_statistics(values, null_count, total_documents)
        
        # Determine field role
        role = self._determine_field_role(field_name, field_type, values, statistics)
        
        # Detect patterns
        patterns = self._detect_patterns(values)
        
        # Determine analytics capabilities
        is_groupable = self._is_groupable(field_type, statistics)
        is_aggregatable = self._is_aggregatable(field_type, role)
        is_filterable = self._is_filterable(field_type, statistics)
        
        # Suggest chart types
        suggested_charts = self._suggest_chart_types(field_type, role, statistics)
        
        # Check for foreign key potential
        possible_fk, referenced_collection = self._check_foreign_key_potential(
            field_name, field_type, values
        )
        
        return FieldSchema(
            name=field_name,
            field_type=field_type,
            role=role,
            type_confidence=type_confidence,
            statistics=statistics,
            detected_patterns=patterns,
            date_format=self._detect_date_format(values) if field_type in [FieldType.DATE, FieldType.DATETIME] else None,
            is_groupable=is_groupable,
            is_aggregatable=is_aggregatable,
            is_filterable=is_filterable,
            suggested_chart_types=suggested_charts,
            possible_foreign_key=possible_fk,
            referenced_collection=referenced_collection
        )
    
    def _detect_field_type(self, values: List[Any], types: List[str]) -> tuple[FieldType, float]:
        """Detect the field type with confidence score"""
        
        if not values:
            return FieldType.NULL, 1.0
        
        # Count type occurrences
        type_counter = Counter(types)
        most_common_type = type_counter.most_common(1)[0][0]
        type_consistency = type_counter[most_common_type] / len(values)
        
        # Sample values for analysis
        sample_values = values[:100]  # Analyze first 100 values
        
        # Check for specific types
        
        # ObjectId detection
        if self._is_objectid(sample_values):
            return FieldType.OBJECTID, 0.95
        
        # Date/DateTime detection
        if self._is_datetime(sample_values):
            return FieldType.DATETIME, 0.9
        
        if self._is_date(sample_values):
            return FieldType.DATE, 0.9
        
        # Boolean detection
        if self._is_boolean(sample_values):
            return FieldType.BOOLEAN, 0.95
        
        # Numeric detection
        if most_common_type in ['int', 'float']:
            if all(isinstance(v, int) for v in sample_values if v is not None):
                return FieldType.INTEGER, type_consistency
            else:
                return FieldType.NUMBER, type_consistency
        
        # String analysis
        if most_common_type == 'str':
            # Check if all values are numeric strings
            if all(self._is_numeric_string(str(v)) for v in sample_values if v is not None):
                return FieldType.NUMBER, type_consistency * 0.8
            
            return FieldType.STRING, type_consistency
        
        # Array detection
        if most_common_type == 'list':
            return FieldType.ARRAY, type_consistency
        
        # Object detection
        if most_common_type == 'dict':
            return FieldType.OBJECT, type_consistency
        
        # Mixed types
        if len(type_counter) > 2:
            return FieldType.MIXED, 0.5
        
        # Default to string
        return FieldType.STRING, 0.3
    
    def _is_objectid(self, values: List[Any]) -> bool:
        """Check if values are MongoDB ObjectIds"""
        if not values:
            return False
        
        sample = [str(v) for v in values[:10] if v is not None]
        if not sample:
            return False
        
        # ObjectId pattern: 24 character hex string
        objectid_pattern = r'^[a-fA-F0-9]{24}$'
        matches = sum(1 for v in sample if re.match(objectid_pattern, v))
        
        return matches / len(sample) > 0.8
    
    def _is_datetime(self, values: List[Any]) -> bool:
        """Check if values are datetime objects or datetime strings"""
        if not values:
            return False
        
        sample = values[:10]
        datetime_count = 0
        
        for value in sample:
            if value is None:
                continue
            
            # Check if it's a datetime object
            if isinstance(value, datetime):
                datetime_count += 1
                continue
            
            # Check if it's a datetime string
            if isinstance(value, str):
                for pattern in self.date_patterns:
                    if re.match(pattern, value):
                        datetime_count += 1
                        break
        
        return datetime_count / len([v for v in sample if v is not None]) > 0.8
    
    def _is_date(self, values: List[Any]) -> bool:
        """Check if values are date-only strings"""
        if not values:
            return False
        
        sample = [str(v) for v in values[:10] if v is not None]
        if not sample:
            return False
        
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        matches = sum(1 for v in sample if re.match(date_pattern, v))
        
        return matches / len(sample) > 0.8
    
    def _is_boolean(self, values: List[Any]) -> bool:
        """Check if values are boolean"""
        if not values:
            return False
        
        sample = values[:20]
        non_null_values = [v for v in sample if v is not None]
        
        if not non_null_values:
            return False
        
        # Check for actual booleans
        if all(isinstance(v, bool) for v in non_null_values):
            return True
        
        # Check for boolean-like strings
        boolean_strings = {'true', 'false', 'yes', 'no', '1', '0', 'y', 'n'}
        if all(str(v).lower() in boolean_strings for v in non_null_values):
            return True
        
        return False
    
    def _is_numeric_string(self, value: str) -> bool:
        """Check if a string represents a number"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _calculate_field_statistics(
        self, 
        values: List[Any], 
        null_count: int, 
        total_count: int
    ) -> FieldStatistics:
        """Calculate comprehensive field statistics"""
        
        non_null_values = [v for v in values if v is not None]
        unique_values = list(set(str(v) for v in non_null_values))
        
        stats = FieldStatistics(
            total_count=total_count,
            non_null_count=len(non_null_values),
            null_count=null_count,
            unique_count=len(unique_values),
            cardinality=len(unique_values) / len(non_null_values) if non_null_values else 0
        )
        
        # Numeric statistics
        numeric_values = []
        for value in non_null_values:
            try:
                if isinstance(value, (int, float)):
                    numeric_values.append(float(value))
                elif isinstance(value, str) and self._is_numeric_string(value):
                    numeric_values.append(float(value))
            except (ValueError, TypeError):
                continue
        
        if numeric_values:
            stats.min_value = min(numeric_values)
            stats.max_value = max(numeric_values)
            stats.avg_value = statistics.mean(numeric_values)
            if len(numeric_values) > 1:
                stats.median_value = statistics.median(numeric_values)
        
        # String length statistics
        string_values = [str(v) for v in non_null_values if v is not None]
        if string_values:
            lengths = [len(s) for s in string_values]
            stats.min_length = min(lengths)
            stats.max_length = max(lengths)
            stats.avg_length = statistics.mean(lengths)
        
        # Sample values (up to 10)
        stats.sample_values = non_null_values[:10]
        
        # Most common values
        if non_null_values:
            value_counter = Counter(str(v) for v in non_null_values)
            stats.most_common = [
                {"value": value, "count": count}
                for value, count in value_counter.most_common(5)
            ]
        
        return stats
    
    def _determine_field_role(
        self, 
        field_name: str, 
        field_type: FieldType, 
        values: List[Any], 
        statistics: FieldStatistics
    ) -> FieldRole:
        """Determine the analytics role of a field"""
        
        field_name_lower = field_name.lower()
        
        # System metadata fields
        if field_name in self.system_fields or field_name_lower in self.system_fields:
            return FieldRole.METADATA
        
        # Temporal fields
        if field_type in [FieldType.DATE, FieldType.DATETIME]:
            return FieldRole.TEMPORAL
        
        # Identifier fields
        if (field_type == FieldType.OBJECTID or 
            any(re.match(pattern, field_name) for pattern in self.id_field_patterns) or
            statistics.cardinality > 0.95):
            return FieldRole.IDENTIFIER
        
        # Metric fields (numeric and aggregatable)
        if (field_type in [FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT] and
            field_name_lower not in ['id', 'index', 'order', 'rank']):
            return FieldRole.METRIC
        
        # Dimension fields (categorical with reasonable cardinality)
        if (field_type in [FieldType.STRING, FieldType.BOOLEAN] and
            0.01 <= statistics.cardinality <= 0.5):
            return FieldRole.DIMENSION
        
        # Text content fields
        if (field_type == FieldType.STRING and 
            statistics.avg_length and statistics.avg_length > 50):
            return FieldRole.TEXT
        
        # Default to unknown
        return FieldRole.UNKNOWN
    
    def _detect_patterns(self, values: List[Any]) -> List[str]:
        """Detect specific patterns in field values"""
        
        patterns = []
        
        if not values:
            return patterns
        
        # Sample string values for pattern detection
        string_values = [str(v) for v in values[:50] if v is not None]
        
        if not string_values:
            return patterns
        
        # Email pattern
        if sum(1 for v in string_values if re.match(self.email_pattern, v)) / len(string_values) > 0.8:
            patterns.append("email")
        
        # Phone pattern
        if sum(1 for v in string_values if re.match(self.phone_pattern, v)) / len(string_values) > 0.7:
            patterns.append("phone")
        
        # URL pattern
        if sum(1 for v in string_values if re.match(self.url_pattern, v)) / len(string_values) > 0.8:
            patterns.append("url")
        
        # IP address pattern
        if sum(1 for v in string_values if re.match(self.ip_pattern, v)) / len(string_values) > 0.8:
            patterns.append("ip_address")
        
        # UUID pattern
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
        
        
        if sum(1 for v in string_values if re.match(uuid_pattern, v.lower())) / len(string_values) > 0.8:
            patterns.append("uuid")
        
        return patterns
    
    def _detect_date_format(self, values: List[Any]) -> Optional[str]:
        """Detect the date format used in string date fields"""
        
        string_values = [str(v) for v in values[:10] if v is not None]
        
        format_patterns = {
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}': 'ISO 8601 datetime',
            r'^\d{4}-\d{2}-\d{2}'
        
        : 'YYYY-MM-DD',
            r'^\d{2}/\d{2}/\d{4}'
        
        : 'MM/DD/YYYY',
            r'^\d{2}-\d{2}-\d{4}'
        
        : 'MM-DD-YYYY'
        }
        
        for pattern, format_name in format_patterns.items():
            if all(re.match(pattern, v) for v in string_values):
                return format_name
        
        return None
    
    def _is_groupable(self, field_type: FieldType, statistics: FieldStatistics) -> bool:
        """Determine if field is suitable for grouping operations"""
        
        # High cardinality fields are not good for grouping
        if statistics.cardinality > 0.8:
            return False
        
        # Good grouping field types
        if field_type in [
            FieldType.STRING, FieldType.BOOLEAN, FieldType.DATE, 
            FieldType.DATETIME, FieldType.INTEGER
        ]:
            return True
        
        # Numeric fields with low cardinality can be grouped
        if field_type in [FieldType.NUMBER, FieldType.FLOAT] and statistics.cardinality < 0.1:
            return True
        
        return False
    
    def _is_aggregatable(self, field_type: FieldType, role: FieldRole) -> bool:
        """Determine if field is suitable for aggregation operations"""
        
        # Numeric fields are aggregatable
        if field_type in [FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT]:
            return True
        
        # Arrays can be counted
        if field_type == FieldType.ARRAY:
            return True
        
        # Count aggregations work on any field
        return True
    
    def _is_filterable(self, field_type: FieldType, statistics: FieldStatistics) -> bool:
        """Determine if field is suitable for filtering operations"""
        
        # Most field types are filterable
        if field_type in [
            FieldType.STRING, FieldType.NUMBER, FieldType.INTEGER, 
            FieldType.FLOAT, FieldType.BOOLEAN, FieldType.DATE, 
            FieldType.DATETIME, FieldType.OBJECTID
        ]:
            return True
        
        # Very high cardinality fields might not be efficient for filtering
        if statistics.cardinality > 0.95 and statistics.unique_count > 1000:
            return False
        
        return True
    
    def _suggest_chart_types(
        self, 
        field_type: FieldType, 
        role: FieldRole, 
        statistics: FieldStatistics
    ) -> List[ChartType]:
        """Suggest appropriate chart types for this field"""
        
        suggested = []
        
        # Based on field role
        if role == FieldRole.DIMENSION:
            if statistics.unique_count <= 10:
                suggested.extend([ChartType.PIE, ChartType.DOUGHNUT, ChartType.BAR])
            else:
                suggested.append(ChartType.BAR)
        
        elif role == FieldRole.METRIC:
            suggested.extend([ChartType.BAR, ChartType.LINE, ChartType.HISTOGRAM])
            if statistics.unique_count == 1:
                suggested.append(ChartType.METRIC)
        
        elif role == FieldRole.TEMPORAL:
            suggested.extend([ChartType.LINE, ChartType.AREA, ChartType.BAR])
        
        # Based on field type
        if field_type == FieldType.BOOLEAN:
            suggested.extend([ChartType.PIE, ChartType.DOUGHNUT])
        
        elif field_type in [FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT]:
            suggested.extend([ChartType.HISTOGRAM, ChartType.SCATTER])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggested = []
        for chart_type in suggested:
            if chart_type not in seen:
                seen.add(chart_type)
                unique_suggested.append(chart_type)
        
        return unique_suggested[:5]  # Limit to top 5 suggestions
    
    def _check_foreign_key_potential(
        self, 
        field_name: str, 
        field_type: FieldType, 
        values: List[Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if field might be a foreign key reference"""
        
        # Check field name patterns
        is_id_field = any(re.match(pattern, field_name) for pattern in self.id_field_patterns)
        
        # ObjectId fields are likely foreign keys (except _id)
        if field_type == FieldType.OBJECTID and field_name != '_id':
            referenced_collection = self._guess_referenced_collection(field_name)
            return True, referenced_collection
        
        # ID-named fields with high cardinality
        if is_id_field and field_name != '_id':
            referenced_collection = self._guess_referenced_collection(field_name)
            return True, referenced_collection
        
        return False, None
    
    def _guess_referenced_collection(self, field_name: str) -> Optional[str]:
        """Guess the referenced collection name from field name"""
        
        # Common patterns
        if field_name.endswith('_id'):
            return field_name[:-3] + 's'  # user_id -> users
        
        if field_name.endswith('Id'):
            return field_name[:-2].lower() + 's'  # userId -> users
        
        if field_name.lower().endswith('id'):
            base = field_name[:-2]
            if base:
                return base.lower() + 's'
        
        return None