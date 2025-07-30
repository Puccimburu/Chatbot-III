# core/schema_detection/statistics_engine.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import math

logger = logging.getLogger(__name__)

@dataclass
class FieldStatistics:
    field_name: str
    data_type: str
    total_count: int
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    cardinality: float
    sample_values: List[Any]
    
    # Numeric statistics
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    
    # String statistics
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # Categorical statistics
    most_common_values: Optional[List[Tuple[Any, int]]] = None
    
    # Temporal statistics
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None
    date_range_days: Optional[int] = None

@dataclass
class CollectionStatistics:
    collection_name: str
    document_count: int
    average_document_size: float
    field_statistics: Dict[str, FieldStatistics]
    estimated_storage_size: int
    update_frequency: str
    data_quality_score: float
    
class StatisticsEngine:
    """
    Computes comprehensive statistics for MongoDB collections and fields
    """
    
    def __init__(self, database_service):
        self.db_service = database_service
        self.sample_size = 1000
        self.max_categorical_values = 50
        
    async def compute_collection_statistics(self, collection_name: str) -> CollectionStatistics:
        """
        Compute comprehensive statistics for a collection
        """
        try:
            logger.info(f"📊 Computing statistics for collection: {collection_name}")
            
            # Get basic collection info
            document_count = await self._get_document_count(collection_name)
            
            if document_count == 0:
                return self._empty_collection_stats(collection_name)
            
            # Get sample documents for analysis
            sample_docs = await self._get_sample_documents(collection_name, min(self.sample_size, document_count))
            
            if not sample_docs:
                return self._empty_collection_stats(collection_name)
            
            # Analyze field statistics
            field_stats = await self._analyze_field_statistics(sample_docs, collection_name, document_count)
            
            # Compute collection-level metrics
            avg_doc_size = await self._estimate_average_document_size(sample_docs)
            storage_size = await self._estimate_storage_size(document_count, avg_doc_size)
            update_frequency = await self._analyze_update_frequency(collection_name, sample_docs)
            data_quality_score = self._calculate_data_quality_score(field_stats)
            
            collection_stats = CollectionStatistics(
                collection_name=collection_name,
                document_count=document_count,
                average_document_size=avg_doc_size,
                field_statistics=field_stats,
                estimated_storage_size=storage_size,
                update_frequency=update_frequency,
                data_quality_score=data_quality_score
            )
            
            logger.info(f"✅ Statistics computed for {collection_name}: {document_count} docs, {len(field_stats)} fields")
            
            return collection_stats
            
        except Exception as e:
            logger.error(f"Error computing statistics for {collection_name}: {e}")
            return self._empty_collection_stats(collection_name)
    
    async def _get_document_count(self, collection_name: str) -> int:
        """
        Get total document count for collection
        """
        try:
            # Use countDocuments for accuracy
            pipeline = [{"$count": "total"}]
            result = await self.db_service.execute_aggregation(collection_name, pipeline)
            
            if result and len(result) > 0:
                return result[0].get("total", 0)
            
            return 0
            
        except Exception as e:
            logger.warning(f"Could not get document count for {collection_name}: {e}")
            return 0
    
    async def _get_sample_documents(self, collection_name: str, sample_size: int) -> List[Dict]:
        """
        Get a representative sample of documents
        """
        try:
            # Use $sample for random sampling
            pipeline = [{"$sample": {"size": sample_size}}]
            result = await self.db_service.execute_aggregation(collection_name, pipeline)
            
            return result if result else []
            
        except Exception as e:
            logger.warning(f"Could not get sample documents from {collection_name}: {e}")
            return []
    
    async def _analyze_field_statistics(self, sample_docs: List[Dict], 
                                      collection_name: str, total_count: int) -> Dict[str, FieldStatistics]:
        """
        Analyze statistics for all fields in the sample
        """
        # Collect all field names and values
        field_data = {}
        
        for doc in sample_docs:
            self._extract_fields_recursive(doc, field_data)
        
        # Compute statistics for each field
        field_stats = {}
        
        for field_name, values in field_data.items():
            stats = await self._compute_field_statistics(field_name, values, len(sample_docs), total_count)
            field_stats[field_name] = stats
        
        return field_stats
    
    def _extract_fields_recursive(self, doc: Dict, field_data: Dict, prefix: str = "") -> None:
        """
        Recursively extract fields from nested documents
        """
        for key, value in doc.items():
            field_name = f"{prefix}.{key}" if prefix else key
            
            if field_name not in field_data:
                field_data[field_name] = []
            
            field_data[field_name].append(value)
            
            # Recurse into nested objects (but not arrays)
            if isinstance(value, dict):
                self._extract_fields_recursive(value, field_data, field_name)
    
    async def _compute_field_statistics(self, field_name: str, values: List[Any], 
                                      sample_count: int, total_count: int) -> FieldStatistics:
        """
        Compute comprehensive statistics for a single field
        """
        # Basic counts
        non_null_values = [v for v in values if v is not None]
        null_count = len(values) - len(non_null_values)
        null_percentage = (null_count / len(values)) * 100 if values else 0
        
        # Unique count and cardinality
        unique_values = list(set(str(v) for v in non_null_values))
        unique_count = len(unique_values)
        cardinality = unique_count / len(non_null_values) if non_null_values else 0
        
        # Determine data type
        data_type = self._determine_field_type(non_null_values)
        
        # Sample values (limit to avoid memory issues)
        sample_values = unique_values[:10]
        
        # Create base statistics object
        stats = FieldStatistics(
            field_name=field_name,
            data_type=data_type,
            total_count=sample_count,
            non_null_count=len(non_null_values),
            null_count=null_count,
            null_percentage=round(null_percentage, 2),
            unique_count=unique_count,
            cardinality=round(cardinality, 3),
            sample_values=sample_values
        )
        
        # Type-specific statistics
        if data_type == "numeric":
            self._compute_numeric_statistics(stats, non_null_values)
        elif data_type == "string":
            self._compute_string_statistics(stats, non_null_values)
        elif data_type == "datetime":
            self._compute_temporal_statistics(stats, non_null_values)
        elif data_type == "categorical":
            self._compute_categorical_statistics(stats, non_null_values)
        
        return stats
    
    def _determine_field_type(self, values: List[Any]) -> str:
        """
        Determine the primary data type of field values
        """
        if not values:
            return "unknown"
        
        # Count different types
        type_counts = {
            "numeric": 0,
            "string": 0,
            "boolean": 0,
            "datetime": 0,
            "array": 0,
            "object": 0,
            "null": 0
        }
        
        for value in values[:100]:  # Sample first 100 values for performance
            if value is None:
                type_counts["null"] += 1
            elif isinstance(value, bool):
                type_counts["boolean"] += 1
            elif isinstance(value, (int, float)):
                type_counts["numeric"] += 1
            elif isinstance(value, str):
                if self._is_datetime_string(value):
                    type_counts["datetime"] += 1
                else:
                    type_counts["string"] += 1
            elif isinstance(value, list):
                type_counts["array"] += 1
            elif isinstance(value, dict):
                type_counts["object"] += 1
            else:
                type_counts["string"] += 1  # Default to string
        
        # Return the most common type
        primary_type = max(type_counts, key=type_counts.get)
        
        # Special handling for categorical vs string
        if primary_type == "string":
            unique_ratio = len(set(str(v) for v in values)) / len(values)
            if unique_ratio < 0.5:  # Low uniqueness suggests categorical
                return "categorical"
        
        return primary_type
    
    def _is_datetime_string(self, value: str) -> bool:
        """
        Check if a string value represents a datetime
        """
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
        ]
        
        import re
        return any(re.search(pattern, value) for pattern in datetime_patterns)
    
    def _compute_numeric_statistics(self, stats: FieldStatistics, values: List[Any]) -> None:
        """
        Compute statistics specific to numeric fields
        """
        try:
            numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(float(v))
                elif isinstance(v, str):
                    try:
                        numeric_values.append(float(v))
                    except ValueError:
                        continue
            
            if not numeric_values:
                return
            
            stats.min_value = min(numeric_values)
            stats.max_value = max(numeric_values)
            stats.mean = statistics.mean(numeric_values)
            
            if len(numeric_values) >= 2:
                stats.std_dev = statistics.stdev(numeric_values)
                stats.median = statistics.median(numeric_values)
            
        except Exception as e:
            logger.warning(f"Error computing numeric statistics: {e}")
    
    def _compute_string_statistics(self, stats: FieldStatistics, values: List[Any]) -> None:
        """
        Compute statistics specific to string fields
        """
        try:
            string_values = [str(v) for v in values if v is not None]
            
            if not string_values:
                return
            
            lengths = [len(s) for s in string_values]
            
            stats.min_length = min(lengths)
            stats.max_length = max(lengths)
            stats.avg_length = statistics.mean(lengths)
            
        except Exception as e:
            logger.warning(f"Error computing string statistics: {e}")
    
    def _compute_temporal_statistics(self, stats: FieldStatistics, values: List[Any]) -> None:
        """
        Compute statistics specific to datetime fields
        """
        try:
            datetime_values = []
            
            for v in values:
                if isinstance(v, datetime):
                    datetime_values.append(v)
                elif isinstance(v, str):
                    try:
                        # Try to parse common datetime formats
                        from dateutil import parser
                        dt = parser.parse(v)
                        datetime_values.append(dt)
                    except Exception:
                        continue
            
            if not datetime_values:
                return
            
            stats.earliest_date = min(datetime_values)
            stats.latest_date = max(datetime_values)
            
            date_range = stats.latest_date - stats.earliest_date
            stats.date_range_days = date_range.days
            
        except Exception as e:
            logger.warning(f"Error computing temporal statistics: {e}")
    
    def _compute_categorical_statistics(self, stats: FieldStatistics, values: List[Any]) -> None:
        """
        Compute statistics specific to categorical fields
        """
        try:
            from collections import Counter
            
            string_values = [str(v) for v in values if v is not None]
            
            if not string_values:
                return
            
            value_counts = Counter(string_values)
            stats.most_common_values = value_counts.most_common(self.max_categorical_values)
            
        except Exception as e:
            logger.warning(f"Error computing categorical statistics: {e}")
    
    async def _estimate_average_document_size(self, sample_docs: List[Dict]) -> float:
        """
        Estimate average document size in bytes
        """
        try:
            import json
            
            if not sample_docs:
                return 0.0
            
            total_size = 0
            for doc in sample_docs:
                # Rough estimation using JSON serialization
                doc_json = json.dumps(doc, default=str)
                total_size += len(doc_json.encode('utf-8'))
            
            return total_size / len(sample_docs)
            
        except Exception as e:
            logger.warning(f"Error estimating document size: {e}")
            return 1000.0  # Default estimate
    
    async def _estimate_storage_size(self, document_count: int, avg_doc_size: float) -> int:
        """
        Estimate total storage size for collection
        """
        # Add overhead for indexes and MongoDB storage overhead (roughly 30%)
        overhead_factor = 1.3
        
        return int(document_count * avg_doc_size * overhead_factor)
    
    async def _analyze_update_frequency(self, collection_name: str, sample_docs: List[Dict]) -> str:
        """
        Analyze how frequently the collection is updated
        """
        try:
            # Look for timestamp fields
            timestamp_fields = ["updatedAt", "updated_at", "lastModified", "modifiedDate"]
            
            timestamps = []
            for doc in sample_docs:
                for field in timestamp_fields:
                    if field in doc and doc[field]:
                        try:
                            if isinstance(doc[field], datetime):
                                timestamps.append(doc[field])
                            elif isinstance(doc[field], str):
                                from dateutil import parser
                                timestamps.append(parser.parse(doc[field]))
                        except Exception:
                            continue
            
            if not timestamps:
                return "unknown"
            
            # Analyze recency
            now = datetime.now()
            recent_updates = sum(1 for ts in timestamps if (now - ts).days <= 7)
            
            recent_ratio = recent_updates / len(timestamps)
            
            if recent_ratio > 0.5:
                return "high"
            elif recent_ratio > 0.2:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.warning(f"Error analyzing update frequency: {e}")
            return "unknown"
    
    def _calculate_data_quality_score(self, field_stats: Dict[str, FieldStatistics]) -> float:
        """
        Calculate an overall data quality score (0-100)
        """
        if not field_stats:
            return 0.0
        
        quality_factors = []
        
        for stats in field_stats.values():
            # Factor 1: Completeness (non-null percentage)
            completeness = (stats.non_null_count / stats.total_count) * 100 if stats.total_count > 0 else 0
            
            # Factor 2: Consistency (based on data type uniformity)
            consistency = 100 if stats.data_type != "unknown" else 50
            
            # Factor 3: Uniqueness appropriateness
            if "_id" in stats.field_name.lower() or "id" in stats.field_name.lower():
                # ID fields should have high cardinality
                uniqueness = min(100, stats.cardinality * 100)
            else:
                # Regular fields: moderate cardinality is good
                if 0.1 <= stats.cardinality <= 0.8:
                    uniqueness = 100
                else:
                    uniqueness = max(50, 100 - abs(stats.cardinality - 0.4) * 100)
            
            field_quality = (completeness * 0.5) + (consistency * 0.3) + (uniqueness * 0.2)
            quality_factors.append(field_quality)
        
        return round(statistics.mean(quality_factors), 1)
    
    def _empty_collection_stats(self, collection_name: str) -> CollectionStatistics:
        """
        Return empty statistics for collections with no data
        """
        return CollectionStatistics(
            collection_name=collection_name,
            document_count=0,
            average_document_size=0.0,
            field_statistics={},
            estimated_storage_size=0,
            update_frequency="unknown",
            data_quality_score=0.0
        )
    
    def generate_statistics_summary(self, collection_stats: CollectionStatistics) -> Dict[str, Any]:
        """
        Generate a human-readable summary of collection statistics
        """
        field_types = {}
        high_cardinality_fields = []
        potential_ids = []
        date_fields = []
        
        for field_name, stats in collection_stats.field_statistics.items():
            # Count field types
            field_types[stats.data_type] = field_types.get(stats.data_type, 0) + 1
            
            # Identify interesting fields
            if stats.cardinality > 0.9:
                high_cardinality_fields.append(field_name)
            
            if "id" in field_name.lower() or stats.cardinality > 0.95:
                potential_ids.append(field_name)
            
            if stats.data_type == "datetime":
                date_fields.append(field_name)
        
        return {
            "collection": collection_stats.collection_name,
            "document_count": collection_stats.document_count,
            "field_count": len(collection_stats.field_statistics),
            "field_types": field_types,
            "data_quality_score": collection_stats.data_quality_score,
            "update_frequency": collection_stats.update_frequency,
            "storage_size_mb": round(collection_stats.estimated_storage_size / 1024 / 1024, 2),
            "potential_id_fields": potential_ids[:5],
            "date_fields": date_fields,
            "analytics_potential": self._assess_analytics_potential(collection_stats)
        }
    
    def _assess_analytics_potential(self, collection_stats: CollectionStatistics) -> str:
        """
        Assess how suitable the collection is for analytics
        """
        if collection_stats.document_count < 10:
            return "low"
        
        numeric_fields = sum(1 for stats in collection_stats.field_statistics.values() 
                           if stats.data_type == "numeric")
        
        date_fields = sum(1 for stats in collection_stats.field_statistics.values() 
                         if stats.data_type == "datetime")
        
        categorical_fields = sum(1 for stats in collection_stats.field_statistics.values() 
                               if stats.data_type == "categorical")
        
        # High potential if has mix of numeric, categorical, and date fields
        if numeric_fields >= 2 and categorical_fields >= 1 and date_fields >= 1:
            return "high"
        elif numeric_fields >= 1 and (categorical_fields >= 1 or date_fields >= 1):
            return "medium"
        else:
            return "low"