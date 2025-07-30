"""
Schema Service - Main orchestrator for schema detection and management
Handles caching, background updates, and schema API operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from models.schema_models import DatabaseSchema, CollectionSchema, SchemaDetectionResponse
from core.schema_detection.detector import SchemaDetector
from services.cache_service import CacheService
from config.settings import settings
from utils.logging_config import monitor_performance

logger = logging.getLogger(__name__)


class SchemaService:
    """Main schema service with intelligent caching and background updates"""
    
    def __init__(self, db_manager, cache_service: CacheService):
        self.db_manager = db_manager
        self.cache_service = cache_service
        self.schema_detector = SchemaDetector(db_manager)
        
        # Background task management
        self._background_task: Optional[asyncio.Task] = None
        self._last_background_update: Optional[datetime] = None
        self._background_running = False
        
        # Cache keys
        self.SCHEMA_CACHE_KEY = f"schema:{settings.DATABASE_NAME}"
        self.COLLECTION_CACHE_PREFIX = f"collection:{settings.DATABASE_NAME}:"
        self.STATS_CACHE_KEY = f"stats:{settings.DATABASE_NAME}"
    
    async def initialize(self):
        """Initialize the schema service"""
        logger.info("🔧 Initializing Schema Service...")
        
        # Start background schema detection if enabled
        if settings.ENABLE_ASYNC_SCHEMA_DETECTION:
            await self.start_background_detection()
        
        logger.info("✅ Schema Service initialized")
    
    @monitor_performance("get_database_schema")
    async def get_database_schema(
        self, 
        force_refresh: bool = False,
        collections: Optional[List[str]] = None
    ) -> DatabaseSchema:
        """
        Get complete database schema with intelligent caching
        
        Args:
            force_refresh: Force fresh detection ignoring cache
            collections: Specific collections to analyze
            
        Returns:
            DatabaseSchema: Complete schema information
        """
        
        cache_key = self._get_schema_cache_key(collections)
        
        # Try cache first (unless force refresh)
        if not force_refresh:
            cached_schema = await self._get_cached_schema(cache_key)
            if cached_schema:
                logger.info("📋 Returning cached database schema")
                return cached_schema
        
        # Detect fresh schema
        logger.info("🔍 Detecting fresh database schema...")
        schema = await self.schema_detector.detect_schema(
            collections=collections,
            force_refresh=force_refresh
        )
        
        # Cache the result
        await self._cache_schema(cache_key, schema)
        
        logger.info(f"✅ Database schema detected and cached: {len(schema.collections)} collections")
        return schema
    
    @monitor_performance("get_collection_schema")
    async def get_collection_schema(
        self, 
        collection_name: str,
        force_refresh: bool = False
    ) -> Optional[CollectionSchema]:
        """
        Get schema for a specific collection
        
        Args:
            collection_name: Name of the collection
            force_refresh: Force fresh detection
            
        Returns:
            CollectionSchema or None if not found
        """
        
        cache_key = f"{self.COLLECTION_CACHE_PREFIX}{collection_name}"
        
        # Try cache first
        if not force_refresh:
            cached_collection = await self._get_cached_collection(cache_key)
            if cached_collection:
                logger.debug(f"📋 Returning cached schema for {collection_name}")
                return cached_collection
        
        # Get from full schema detection
        database_schema = await self.get_database_schema(force_refresh=force_refresh)
        collection_schema = database_schema.collections.get(collection_name)
        
        if collection_schema:
            # Cache individual collection
            await self._cache_collection(cache_key, collection_schema)
        
        return collection_schema
    
    async def get_recommended_collections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get collections recommended for analytics"""
        
        schema = await self.get_database_schema()
        
        # Sort by analytics value
        sorted_collections = sorted(
            schema.collections.items(),
            key=lambda x: x[1].analytics_value,
            reverse=True
        )
        
        recommendations = []
        for name, collection_schema in sorted_collections[:limit]:
            recommendations.append({
                "name": name,
                "analytics_value": collection_schema.analytics_value,
                "classification": collection_schema.classification.value,
                "document_count": collection_schema.document_count,
                "field_count": len(collection_schema.fields),
                "metric_fields": len(collection_schema.metric_fields),
                "dimension_fields": len(collection_schema.dimension_fields),
                "temporal_fields": len(collection_schema.temporal_fields),
                "time_series_potential": collection_schema.get_time_series_potential(),
                "suggested_charts": [ct.value for ct in collection_schema.get_suggested_chart_types()]
            })
        
        return recommendations
    
    async def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the current schema state"""
        
        try:
            # Try to get from detector first (includes cache info)
            detector_summary = await self.schema_detector.get_schema_summary()
            
            if detector_summary.get("status") == "ready":
                return detector_summary
            
            # Fallback to basic info
            collections = await self.db_manager.get_collections()
            
            return {
                "status": "basic_info",
                "database_name": settings.DATABASE_NAME,
                "total_collections": len(collections),
                "schema_detection_enabled": settings.ENABLE_AUTO_SCHEMA,
                "background_detection_enabled": settings.ENABLE_ASYNC_SCHEMA_DETECTION,
                "last_update": self._last_background_update.isoformat() if self._last_background_update else None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get schema summary: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics for monitoring"""
        
        cache_key = self.STATS_CACHE_KEY
        
        # Try cache first
        cached_stats = await self.cache_service.get(cache_key)
        if cached_stats:
            return json.loads(cached_stats)
        
        # Calculate fresh stats
        try:
            collections = await self.db_manager.get_collections()
            total_documents = 0
            total_size = 0
            
            # Sample a few collections for stats
            for collection_name in collections[:5]:  # Limit for performance
                try:
                    stats = await self.db_manager.get_collection_stats(collection_name)
                    total_documents += stats.get('document_count', 0)
                    total_size += stats.get('size_bytes', 0)
                except:
                    continue
            
            system_stats = {
                "database_name": settings.DATABASE_NAME,
                "total_collections": len(collections),
                "estimated_documents": total_documents,
                "estimated_size_mb": round(total_size / (1024 * 1024), 2),
                "schema_cache_enabled": settings.ENABLE_CACHING,
                "auto_detection_enabled": settings.ENABLE_AUTO_SCHEMA,
                "last_calculated": datetime.utcnow().isoformat()
            }
            
            # Cache for 10 minutes
            await self.cache_service.set(
                cache_key, 
                json.dumps(system_stats), 
                ttl=600
            )
            
            return system_stats
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate system stats: {str(e)}")
            return {"error": str(e)}
    
    async def refresh_schema(self, collections: Optional[List[str]] = None) -> SchemaDetectionResponse:
        """Force refresh of schema detection"""
        
        start_time = datetime.utcnow()
        
        try:
            logger.info("🔄 Force refreshing schema...")
            
            # Clear relevant caches
            await self._clear_schema_caches(collections)
            
            # Detect fresh schema
            schema = await self.schema_detector.detect_schema(
                collections=collections,
                force_refresh=True
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return SchemaDetectionResponse(
                success=True,
                schema=schema,
                detection_time=duration,
                collections_analyzed=len(schema.collections),
                total_documents_sampled=schema.total_documents_analyzed
            )
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"❌ Schema refresh failed: {str(e)}")
            
            return SchemaDetectionResponse(
                success=False,
                error=str(e),
                detection_time=duration,
                collections_analyzed=0,
                total_documents_sampled=0
            )
    
    async def start_background_detection(self):
        """Start background schema detection task"""
        
        if self._background_running:
            logger.warning("⚠️ Background detection already running")
            return
        
        logger.info("🔄 Starting background schema detection...")
        self._background_running = True
        self._background_task = asyncio.create_task(self._background_detection_loop())
    
    async def stop_background_detection(self):
        """Stop background schema detection task"""
        
        if not self._background_running:
            return
        
        logger.info("🛑 Stopping background schema detection...")
        self._background_running = False
        
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Background detection stopped")
    
    async def _background_detection_loop(self):
        """Background loop for periodic schema detection"""
        
        while self._background_running:
            try:
                # Wait for the interval
                await asyncio.sleep(settings.BACKGROUND_TASK_INTERVAL)
                
                if not self._background_running:
                    break
                
                logger.info("🔄 Running background schema detection...")
                
                # Detect schema without forcing refresh (uses cache if recent)
                await self.get_database_schema(force_refresh=False)
                
                self._last_background_update = datetime.utcnow()
                logger.info("✅ Background schema detection completed")
                
            except asyncio.CancelledError:
                logger.info("🛑 Background detection cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Background detection error: {str(e)}")
                # Continue running despite errors
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _get_cached_schema(self, cache_key: str) -> Optional[DatabaseSchema]:
        """Get schema from cache"""
        
        if not settings.ENABLE_CACHING:
            return None
        
        try:
            cached_data = await self.cache_service.get(cache_key)
            if cached_data:
                schema_dict = json.loads(cached_data)
                return DatabaseSchema(**schema_dict)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load schema from cache: {str(e)}")
        
        return None
    
    async def _cache_schema(self, cache_key: str, schema: DatabaseSchema):
        """Cache schema with TTL"""
        
        if not settings.ENABLE_CACHING:
            return
        
        try:
            schema_json = schema.json()
            await self.cache_service.set(
                cache_key, 
                schema_json, 
                ttl=settings.SCHEMA_CACHE_TTL
            )
            logger.debug(f"💾 Cached schema: {cache_key}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cache schema: {str(e)}")
    
    async def _get_cached_collection(self, cache_key: str) -> Optional[CollectionSchema]:
        """Get collection schema from cache"""
        
        if not settings.ENABLE_CACHING:
            return None
        
        try:
            cached_data = await self.cache_service.get(cache_key)
            if cached_data:
                collection_dict = json.loads(cached_data)
                return CollectionSchema(**collection_dict)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load collection from cache: {str(e)}")
        
        return None
    
    async def _cache_collection(self, cache_key: str, collection_schema: CollectionSchema):
        """Cache collection schema"""
        
        if not settings.ENABLE_CACHING:
            return
        
        try:
            collection_json = collection_schema.json()
            await self.cache_service.set(
                cache_key, 
                collection_json, 
                ttl=settings.SCHEMA_CACHE_TTL
            )
            logger.debug(f"💾 Cached collection: {cache_key}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cache collection: {str(e)}")
    
    def _get_schema_cache_key(self, collections: Optional[List[str]]) -> str:
        """Generate cache key for schema"""
        
        base_key = self.SCHEMA_CACHE_KEY
        
        if collections:
            collection_hash = hash(tuple(sorted(collections)))
            base_key += f":{collection_hash}"
        
        return base_key
    
    async def _clear_schema_caches(self, collections: Optional[List[str]] = None):
        """Clear schema-related caches"""
        
        if not settings.ENABLE_CACHING:
            return
        
        try:
            # Clear main schema cache
            cache_key = self._get_schema_cache_key(collections)
            await self.cache_service.delete(cache_key)
            
            # If clearing all, also clear individual collection caches
            if collections is None:
                # Clear all collection caches (this is a bit brute force)
                # In a real implementation, you might want to track cache keys
                await self.cache_service.delete_pattern(f"{self.COLLECTION_CACHE_PREFIX}*")
            else:
                # Clear specific collection caches
                for collection_name in collections:
                    collection_cache_key = f"{self.COLLECTION_CACHE_PREFIX}{collection_name}"
                    await self.cache_service.delete(collection_cache_key)
            
            # Clear stats cache
            await self.cache_service.delete(self.STATS_CACHE_KEY)
            
            logger.info("🗑️ Schema caches cleared")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to clear schema caches: {str(e)}")
    
    async def get_collection_suggestions(self, query_intent: str) -> List[Dict[str, Any]]:
        """Get collection suggestions based on query intent"""
        
        schema = await self.get_database_schema()
        
        # Simple keyword-based matching for now
        # TODO: Implement more sophisticated NLP-based matching
        query_lower = query_intent.lower()
        
        suggestions = []
        
        for name, collection_schema in schema.collections.items():
            score = 0.0
            reasons = []
            
            # Name matching
            if any(word in name.lower() for word in query_lower.split()):
                score += 0.5
                reasons.append("Collection name matches query keywords")
            
            # Classification matching
            if "sales" in query_lower and "sales" in name.lower():
                score += 0.8
                reasons.append("Sales-related query matched to sales collection")
            
            if "user" in query_lower and "user" in name.lower():
                score += 0.8
                reasons.append("User-related query matched to user collection")
            
            if "order" in query_lower and "order" in name.lower():
                score += 0.8
                reasons.append("Order-related query matched to order collection")
            
            # Analytics value bonus
            score += collection_schema.analytics_value * 0.3
            
            if score > 0.2:
                suggestions.append({
                    "collection_name": name,
                    "relevance_score": round(score, 2),
                    "analytics_value": collection_schema.analytics_value,
                    "document_count": collection_schema.document_count,
                    "field_count": len(collection_schema.fields),
                    "reasons": reasons,
                    "suggested_charts": [ct.value for ct in collection_schema.get_suggested_chart_types()[:3]]
                })
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
    
    async def validate_query_compatibility(
        self, 
        collection_name: str, 
        requested_fields: List[str],
        operation_type: str = "aggregation"
    ) -> Dict[str, Any]:
        """Validate if a query is compatible with the collection schema"""
        
        collection_schema = await self.get_collection_schema(collection_name)
        
        if not collection_schema:
            return {
                "valid": False,
                "error": f"Collection '{collection_name}' not found or not analyzed"
            }
        
        validation_result = {
            "valid": True,
            "warnings": [],
            "suggestions": [],
            "field_mapping": {}
        }
        
        available_fields = set(collection_schema.fields.keys())
        
        for field in requested_fields:
            if field not in available_fields:
                # Try to find similar field names
                similar_fields = [
                    f for f in available_fields 
                    if field.lower() in f.lower() or f.lower() in field.lower()
                ]
                
                if similar_fields:
                    validation_result["warnings"].append(
                        f"Field '{field}' not found. Did you mean: {', '.join(similar_fields[:3])}?"
                    )
                    validation_result["field_mapping"][field] = similar_fields[0]
                else:
                    validation_result["warnings"].append(
                        f"Field '{field}' not found in collection '{collection_name}'"
                    )
            else:
                field_schema = collection_schema.fields[field]
                validation_result["field_mapping"][field] = {
                    "type": field_schema.field_type.value,
                    "role": field_schema.role.value,
                    "is_groupable": field_schema.is_groupable,
                    "is_aggregatable": field_schema.is_aggregatable
                }
        
        # Operation-specific validations
        if operation_type == "aggregation":
            groupable_fields = [
                f for f in requested_fields 
                if f in collection_schema.fields and collection_schema.fields[f].is_groupable
            ]
            
            if not groupable_fields:
                validation_result["suggestions"].append(
                    f"Consider using groupable fields: {', '.join(collection_schema.get_groupable_fields()[:5])}"
                )
        
        return validation_result
    
    async def get_field_suggestions(self, collection_name: str, field_role: str = None) -> List[Dict[str, Any]]:
        """Get field suggestions for a collection"""
        
        collection_schema = await self.get_collection_schema(collection_name)
        
        if not collection_schema:
            return []
        
        suggestions = []
        
        for field_name, field_schema in collection_schema.fields.items():
            # Filter by role if specified
            if field_role and field_schema.role.value != field_role:
                continue
            
            suggestion = {
                "field_name": field_name,
                "type": field_schema.field_type.value,
                "role": field_schema.role.value,
                "is_groupable": field_schema.is_groupable,
                "is_aggregatable": field_schema.is_aggregatable,
                "cardinality": field_schema.statistics.cardinality,
                "sample_values": field_schema.statistics.sample_values[:3],
                "suggested_charts": [ct.value for ct in field_schema.suggested_chart_types]
            }
            
            # Add usage hints
            if field_schema.role.value == "dimension":
                suggestion["usage_hint"] = "Good for grouping and filtering"
            elif field_schema.role.value == "metric":
                suggestion["usage_hint"] = "Good for aggregation and calculations"
            elif field_schema.role.value == "temporal":
                suggestion["usage_hint"] = "Good for time-based analysis"
            
            suggestions.append(suggestion)
        
        # Sort by analytics usefulness
        def sort_key(item):
            role_priority = {
                "metric": 4,
                "dimension": 3,
                "temporal": 3,
                "identifier": 1,
                "metadata": 0,
                "text": 1,
                "unknown": 0
            }
            return role_priority.get(item["role"], 0)
        
        suggestions.sort(key=sort_key, reverse=True)
        
        return suggestions
    
    async def close(self):
        """Clean shutdown of schema service"""
        
        logger.info("🛑 Shutting down Schema Service...")
        
        # Stop background detection
        await self.stop_background_detection()
        
        logger.info("✅ Schema Service shutdown complete")