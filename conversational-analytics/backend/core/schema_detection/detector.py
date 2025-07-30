"""
Main schema detection engine
Automatically discovers and analyzes database schema for conversational analytics
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from models.schema_models import (
    DatabaseSchema, CollectionSchema, FieldSchema, 
    CollectionClassification, FieldType, FieldRole
)
from config.settings import settings
from utils.logging_config import monitor_performance, log_schema_detection
from .field_analyzer import FieldAnalyzer
from .relationship_finder import RelationshipFinder
from .statistics_engine import StatisticsEngine
from .collection_classifier import CollectionClassifier

logger = logging.getLogger(__name__)


class SchemaDetector:
    """Main schema detection engine with intelligent analysis"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.field_analyzer = FieldAnalyzer()
        self.relationship_finder = RelationshipFinder()
        self.statistics_engine = StatisticsEngine()
        self.collection_classifier = CollectionClassifier()
        
        self._detection_cache: Dict[str, DatabaseSchema] = {}
        self._last_detection: Optional[datetime] = None
    
    @monitor_performance("schema_detection")
    async def detect_schema(
        self, 
        collections: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        force_refresh: bool = False
    ) -> DatabaseSchema:
        """
        Detect complete database schema with intelligent analysis
        
        Args:
            collections: Specific collections to analyze (None for all)
            sample_size: Number of documents to sample per collection
            force_refresh: Force fresh detection ignoring cache
            
        Returns:
            DatabaseSchema: Complete schema information
        """
        
        start_time = time.perf_counter()
        
        try:
            logger.info("🔍 Starting intelligent schema detection...")
            
            # Check cache first
            cache_key = f"{settings.DATABASE_NAME}_{collections}_{sample_size}"
            if not force_refresh and cache_key in self._detection_cache:
                cached_schema = self._detection_cache[cache_key]
                cache_age = (datetime.utcnow() - cached_schema.detected_at).total_seconds()
                
                if cache_age < settings.SCHEMA_CACHE_TTL:
                    logger.info(f"📋 Using cached schema (age: {cache_age:.0f}s)")
                    return cached_schema
            
            # Get collections to analyze
            if collections is None:
                collections = await self._get_collections_to_analyze()
            
            logger.info(f"📊 Analyzing {len(collections)} collections...")
            
            # Analyze collections in parallel
            collection_schemas = await self._analyze_collections_parallel(
                collections, sample_size or settings.SCHEMA_SAMPLE_SIZE
            )
            
            # Filter out empty or invalid collections
            valid_schemas = {
                name: schema for name, schema in collection_schemas.items()
                if schema and schema.document_count > 0
            }
            
            logger.info(f"✅ Successfully analyzed {len(valid_schemas)} collections")
            
            # Find cross-collection relationships
            logger.info("🔗 Analyzing relationships between collections...")
            relationships = await self.relationship_finder.find_relationships(
                list(valid_schemas.values()), self.db_manager
            )
            
            # Create complete database schema
            database_schema = DatabaseSchema(
                database_name=settings.DATABASE_NAME,
                collections=valid_schemas,
                relationships=relationships,
                detected_at=datetime.utcnow(),
                detection_duration=time.perf_counter() - start_time,
                total_documents_analyzed=sum(
                    schema.sample_size for schema in valid_schemas.values()
                ),
                recommended_collections=self._get_recommended_collections(valid_schemas),
                cross_collection_patterns=self._find_cross_collection_patterns(valid_schemas, relationships)
            )
            
            # Cache the result
            self._detection_cache[cache_key] = database_schema
            self._last_detection = datetime.utcnow()
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"🎯 Schema detection completed in {duration_ms:.2f}ms: "
                f"{len(valid_schemas)} collections, "
                f"{len(relationships)} relationships"
            )
            
            return database_schema
            
        except Exception as e:
            logger.error(f"❌ Schema detection failed: {str(e)}")
            raise
    
    async def _get_collections_to_analyze(self) -> List[str]:
        """Get list of collections worth analyzing"""
        
        all_collections = await self.db_manager.get_collections()
        
        # Filter out system collections and empty collections
        filtered_collections = []
        
        for collection_name in all_collections:
            # Skip obvious system collections
            if self._is_system_collection(collection_name):
                logger.debug(f"⏭️ Skipping system collection: {collection_name}")
                continue
            
            # Check if collection has documents
            try:
                stats = await self.db_manager.get_collection_stats(collection_name)
                if stats.get('document_count', 0) == 0:
                    logger.debug(f"⏭️ Skipping empty collection: {collection_name}")
                    continue
                
                filtered_collections.append(collection_name)
                
            except Exception as e:
                logger.warning(f"⚠️ Could not check collection {collection_name}: {str(e)}")
        
        # Limit to max collections for performance
        if len(filtered_collections) > settings.SCHEMA_MAX_COLLECTIONS:
            logger.warning(
                f"⚠️ Too many collections ({len(filtered_collections)}), "
                f"limiting to {settings.SCHEMA_MAX_COLLECTIONS}"
            )
            # Sort by document count and take the largest
            collection_stats = []
            for name in filtered_collections:
                try:
                    stats = await self.db_manager.get_collection_stats(name)
                    collection_stats.append((name, stats.get('document_count', 0)))
                except:
                    collection_stats.append((name, 0))
            
            collection_stats.sort(key=lambda x: x[1], reverse=True)
            filtered_collections = [name for name, _ in collection_stats[:settings.SCHEMA_MAX_COLLECTIONS]]
        
        logger.info(f"📋 Selected {len(filtered_collections)} collections for analysis")
        return filtered_collections
    
    def _is_system_collection(self, collection_name: str) -> bool:
        """Check if collection is a system collection"""
        
        system_prefixes = ['system.', 'fs.', '__', 'tmp_', 'temp_']
        system_names = ['sessions', 'logs', 'cache', 'oplog']
        
        collection_lower = collection_name.lower()
        
        # Check prefixes
        for prefix in system_prefixes:
            if collection_lower.startswith(prefix):
                return True
        
        # Check exact names
        if collection_lower in system_names:
            return True
        
        return False
    
    async def _analyze_collections_parallel(
        self, 
        collections: List[str], 
        sample_size: int
    ) -> Dict[str, CollectionSchema]:
        """Analyze multiple collections in parallel"""
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent analyses
        
        async def analyze_single_collection(collection_name: str) -> tuple[str, Optional[CollectionSchema]]:
            async with semaphore:
                try:
                    schema = await self._analyze_single_collection(collection_name, sample_size)
                    return collection_name, schema
                except Exception as e:
                    logger.error(f"❌ Failed to analyze collection {collection_name}: {str(e)}")
                    return collection_name, None
        
        # Execute all analyses in parallel
        tasks = [analyze_single_collection(name) for name in collections]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        collection_schemas = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Collection analysis task failed: {str(result)}")
                continue
            
            collection_name, schema = result
            if schema:
                collection_schemas[collection_name] = schema
        
        return collection_schemas
    
    @monitor_performance("single_collection_analysis")
    async def _analyze_single_collection(
        self, 
        collection_name: str, 
        sample_size: int
    ) -> Optional[CollectionSchema]:
        """Analyze a single collection comprehensively"""
        
        start_time = time.perf_counter()
        
        try:
            logger.debug(f"🔍 Analyzing collection: {collection_name}")
            
            # Get collection statistics
            stats = await self.db_manager.get_collection_stats(collection_name)
            document_count = stats.get('document_count', 0)
            
            if document_count == 0:
                logger.debug(f"⏭️ Skipping empty collection: {collection_name}")
                return None
            
            # Adjust sample size based on collection size
            effective_sample_size = min(sample_size, document_count)
            
            # Sample documents
            sample_documents = await self.db_manager.sample_documents(
                collection_name, effective_sample_size
            )
            
            if not sample_documents:
                logger.warning(f"⚠️ No documents sampled from {collection_name}")
                return None
            
            # Analyze fields
            logger.debug(f"📊 Analyzing {len(sample_documents)} documents from {collection_name}")
            field_schemas = await self.field_analyzer.analyze_fields(sample_documents)
            
            # Calculate collection-level statistics
            collection_stats = self.statistics_engine.calculate_collection_stats(
                sample_documents, field_schemas
            )
            
            # Classify collection
            classification = self.collection_classifier.classify_collection(
                collection_name, field_schemas, sample_documents
            )
            
            # Calculate analytics value
            analytics_value = self._calculate_analytics_value(
                field_schemas, classification, document_count
            )
            
            # Generate query patterns
            query_patterns = self._generate_query_patterns(collection_name, field_schemas)
            
            # Create collection schema
            collection_schema = CollectionSchema(
                name=collection_name,
                classification=classification,
                document_count=document_count,
                avg_document_size=stats.get('avg_obj_size', 0),
                sample_size=len(sample_documents),
                fields=field_schemas,
                detected_at=datetime.utcnow(),
                confidence_score=self._calculate_confidence_score(field_schemas),
                analytics_value=analytics_value,
                query_patterns=query_patterns,
                relationships=[],  # Will be populated later
                recommended_indexes=self._suggest_indexes(field_schemas)
            )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_schema_detection(
                collection_name, 
                len(field_schemas), 
                len(sample_documents), 
                duration_ms
            )
            
            return collection_schema
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze collection {collection_name}: {str(e)}")
            return None
    
    def _calculate_analytics_value(
        self, 
        field_schemas: Dict[str, FieldSchema], 
        classification: CollectionClassification,
        document_count: int
    ) -> float:
        """Calculate the analytics value of a collection (0.0 to 1.0)"""
        
        score = 0.0
        
        # Base score by classification
        classification_scores = {
            CollectionClassification.TRANSACTIONAL: 0.9,
            CollectionClassification.REFERENCE: 0.6,
            CollectionClassification.AUDIT: 0.7,
            CollectionClassification.SYSTEM: 0.1,
            CollectionClassification.STAGING: 0.3,
            CollectionClassification.ARCHIVE: 0.5,
            CollectionClassification.UNKNOWN: 0.4
        }
        score += classification_scores.get(classification, 0.4)
        
        # Document count factor (more documents = higher value)
        if document_count > 1000:
            score += 0.2
        elif document_count > 100:
            score += 0.1
        
        # Field diversity bonus
        metric_fields = [f for f in field_schemas.values() if f.role == FieldRole.METRIC]
        dimension_fields = [f for f in field_schemas.values() if f.role == FieldRole.DIMENSION]
        temporal_fields = [f for f in field_schemas.values() if f.role == FieldRole.TEMPORAL]
        
        if len(metric_fields) > 0:
            score += 0.2
        if len(dimension_fields) > 2:
            score += 0.1
        if len(temporal_fields) > 0:
            score += 0.1
        
        # Time series potential bonus
        if len(temporal_fields) > 0 and len(metric_fields) > 0:
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def _calculate_confidence_score(self, field_schemas: Dict[str, FieldSchema]) -> float:
        """Calculate overall confidence in schema detection"""
        
        if not field_schemas:
            return 0.0
        
        total_confidence = sum(field.type_confidence for field in field_schemas.values())
        return total_confidence / len(field_schemas)
    
    def _generate_query_patterns(
        self, 
        collection_name: str, 
        field_schemas: Dict[str, FieldSchema]
    ) -> List[Any]:
        """Generate common query patterns for a collection"""
        
        # This is a placeholder - will be implemented based on field analysis
        # TODO: Implement intelligent query pattern generation
        patterns = []
        
        return patterns
    
    def _suggest_indexes(self, field_schemas: Dict[str, FieldSchema]) -> List[Dict[str, Any]]:
        """Suggest indexes based on field analysis"""
        
        indexes = []
        
        # Suggest indexes for high-cardinality identifier fields
        for field_name, field_schema in field_schemas.items():
            if (field_schema.role == FieldRole.IDENTIFIER and 
                field_schema.statistics.cardinality > 0.8):
                indexes.append({
                    "fields": {field_name: 1},
                    "reason": "High cardinality identifier field"
                })
        
        # Suggest compound indexes for common grouping combinations
        temporal_fields = [name for name, schema in field_schemas.items() 
                          if schema.role == FieldRole.TEMPORAL]
        dimension_fields = [name for name, schema in field_schemas.items() 
                           if schema.role == FieldRole.DIMENSION and schema.is_groupable]
        
        if temporal_fields and dimension_fields:
            for temporal_field in temporal_fields[:1]:  # Just the first temporal field
                for dimension_field in dimension_fields[:2]:  # Top 2 dimension fields
                    indexes.append({
                        "fields": {temporal_field: 1, dimension_field: 1},
                        "reason": "Time series analysis optimization"
                    })
        
        return indexes
    
    def _get_recommended_collections(
        self, 
        collection_schemas: Dict[str, CollectionSchema]
    ) -> List[str]:
        """Get collections recommended for analytics"""
        
        # Sort by analytics value and return top collections
        sorted_collections = sorted(
            collection_schemas.items(),
            key=lambda x: x[1].analytics_value,
            reverse=True
        )
        
        # Return top 10 or collections with value > 0.6
        recommended = []
        for name, schema in sorted_collections:
            if schema.analytics_value > 0.6 or len(recommended) < 5:
                recommended.append(name)
            if len(recommended) >= 10:
                break
        
        return recommended
    
    def _find_cross_collection_patterns(
        self, 
        collection_schemas: Dict[str, CollectionSchema],
        relationships: List[Any]
    ) -> List[Dict[str, Any]]:
        """Find patterns that span multiple collections"""
        
        # This is a placeholder for cross-collection pattern detection
        # TODO: Implement cross-collection analytics pattern detection
        patterns = []
        
        return patterns
    
    async def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the current schema"""
        
        if not self._last_detection:
            return {"status": "no_schema_detected"}
        
        try:
            # Get basic database stats
            collections = await self.db_manager.get_collections()
            total_collections = len(collections)
            
            cache_key = list(self._detection_cache.keys())[-1] if self._detection_cache else None
            schema = self._detection_cache.get(cache_key) if cache_key else None
            
            if not schema:
                return {"status": "schema_cache_empty"}
            
            return {
                "status": "ready",
                "database_name": schema.database_name,
                "total_collections": total_collections,
                "analyzed_collections": len(schema.collections),
                "recommended_collections": len(schema.recommended_collections),
                "total_relationships": len(schema.relationships),
                "last_detection": self._last_detection.isoformat(),
                "detection_duration": schema.detection_duration,
                "top_collections": [
                    {
                        "name": name,
                        "analytics_value": schema.analytics_value,
                        "document_count": schema.document_count,
                        "field_count": len(schema.fields)
                    }
                    for name, schema in sorted(
                        schema.collections.items(),
                        key=lambda x: x[1].analytics_value,
                        reverse=True
                    )[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get schema summary: {str(e)}")
            return {"status": "error", "error": str(e)}