"""
Database Service - MongoDB operations wrapper with analytics optimizations
Handles query execution, data sampling, and performance monitoring
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time

from config.settings import settings
from utils.logging_config import monitor_performance, log_query_execution

logger = logging.getLogger(__name__)


class DatabaseService:
    """Enhanced database service for analytics operations"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.query_cache = {}
        self.performance_stats = {
            "total_queries": 0,
            "total_execution_time": 0,
            "average_execution_time": 0,
            "slow_queries": 0,
            "failed_queries": 0
        }
    
    @monitor_performance("execute_aggregation")
    async def execute_aggregation(
        self, 
        collection_name: str, 
        pipeline: List[Dict[str, Any]],
        timeout: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute MongoDB aggregation pipeline with optimization and monitoring
        
        Args:
            collection_name: Target collection
            pipeline: MongoDB aggregation pipeline
            timeout: Query timeout in seconds
            
        Returns:
            List of aggregation results
        """
        
        start_time = time.perf_counter()
        
        try:
            # Validate collection exists
            if not await self._collection_exists(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            # Optimize pipeline
            optimized_pipeline = self._optimize_pipeline(pipeline)
            
            # Add safety limits
            safe_pipeline = self._add_safety_limits(optimized_pipeline)
            
            # Execute with timeout
            timeout_seconds = timeout or settings.QUERY_TIMEOUT
            
            results = await asyncio.wait_for(
                self.db_manager.execute_aggregation(collection_name, safe_pipeline),
                timeout=timeout_seconds
            )
            
            # Update performance stats
            execution_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(execution_time, success=True)
            
            log_query_execution(
                collection_name, 
                "aggregation", 
                len(results), 
                execution_time
            )
            
            logger.debug(
                f"📊 Aggregation completed: {collection_name}, "
                f"{len(results)} results, {execution_time:.2f}ms"
            )
            
            return results
            
        except asyncio.TimeoutError:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(execution_time, success=False)
            
            logger.error(f"⏱️ Query timeout after {timeout_seconds}s: {collection_name}")
            raise Exception(f"Query timeout after {timeout_seconds} seconds")
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(execution_time, success=False)
            
            logger.error(f"❌ Aggregation failed on {collection_name}: {str(e)}")
            raise
    
    async def execute_find(
        self, 
        collection_name: str, 
        filter_query: Dict[str, Any] = None,
        projection: Dict[str, Any] = None,
        sort: Dict[str, Any] = None,
        limit: int = None,
        skip: int = None
    ) -> List[Dict[str, Any]]:
        """
        Execute MongoDB find operation with optimization
        
        Args:
            collection_name: Target collection
            filter_query: MongoDB filter query
            projection: Fields to include/exclude
            sort: Sort specification
            limit: Maximum number of results
            skip: Number of documents to skip
            
        Returns:
            List of matching documents
        """
        
        start_time = time.perf_counter()
        
        try:
            if not await self._collection_exists(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            collection = self.db_manager.get_collection(collection_name)
            
            # Build query
            cursor = collection.find(filter_query or {})
            
            if projection:
                cursor = cursor.projection(projection)
            
            if sort:
                cursor = cursor.sort(list(sort.items()))
            
            if skip:
                cursor = cursor.skip(skip)
            
            if limit:
                cursor = cursor.limit(min(limit, settings.MAX_QUERY_RESULTS))
            else:
                cursor = cursor.limit(settings.MAX_QUERY_RESULTS)
            
            # Execute query
            results = []
            async for doc in cursor:
                # Convert ObjectId to string
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                results.append(doc)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(execution_time, success=True)
            
            log_query_execution(collection_name, "find", len(results), execution_time)
            
            return results
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(execution_time, success=False)
            
            logger.error(f"❌ Find operation failed on {collection_name}: {str(e)}")
            raise
    
    async def count_documents(
        self, 
        collection_name: str, 
        filter_query: Dict[str, Any] = None
    ) -> int:
        """
        Count documents in collection with optional filter
        
        Args:
            collection_name: Target collection
            filter_query: MongoDB filter query
            
        Returns:
            Number of matching documents
        """
        
        try:
            if not await self._collection_exists(collection_name):
                return 0
            
            collection = self.db_manager.get_collection(collection_name)
            count = await collection.count_documents(filter_query or {})
            
            logger.debug(f"📊 Document count: {collection_name} = {count}")
            return count
            
        except Exception as e:
            logger.error(f"❌ Count operation failed on {collection_name}: {str(e)}")
            return 0
    
    async def get_distinct_values(
        self, 
        collection_name: str, 
        field_name: str,
        filter_query: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[Any]:
        """
        Get distinct values for a field
        
        Args:
            collection_name: Target collection
            field_name: Field to get distinct values for
            filter_query: Optional filter
            limit: Maximum number of distinct values
            
        Returns:
            List of distinct values
        """
        
        try:
            if not await self._collection_exists(collection_name):
                return []
            
            # Use aggregation for better control
            pipeline = []
            
            if filter_query:
                pipeline.append({"$match": filter_query})
            
            pipeline.extend([
                {"$group": {"_id": f"${field_name}"}},
                {"$limit": limit},
                {"$sort": {"_id": 1}}
            ])
            
            results = await self.execute_aggregation(collection_name, pipeline)
            
            # Extract distinct values
            distinct_values = [doc["_id"] for doc in results if doc["_id"] is not None]
            
            logger.debug(f"📊 Distinct values: {collection_name}.{field_name} = {len(distinct_values)}")
            return distinct_values
            
        except Exception as e:
            logger.error(f"❌ Distinct operation failed: {collection_name}.{field_name}: {str(e)}")
            return []
    
    async def get_field_statistics(
        self, 
        collection_name: str, 
        field_name: str,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Get statistical information about a field
        
        Args:
            collection_name: Target collection
            field_name: Field to analyze
            sample_size: Number of documents to sample
            
        Returns:
            Statistical information about the field
        """
        
        try:
            if not await self._collection_exists(collection_name):
                return {"error": "Collection not found"}
            
            # Sample documents and analyze field
            pipeline = [
                {"$sample": {"size": sample_size}},
                {"$project": {field_name: 1}},
                {"$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                    "null_count": {
                        "$sum": {
                            "$cond": [{"$eq": [f"${field_name}", None]}, 1, 0]
                        }
                    },
                    "sample_values": {"$addToSet": f"${field_name}"}
                }}
            ]
            
            results = await self.execute_aggregation(collection_name, pipeline)
            
            if not results:
                return {"error": "No data found"}
            
            stats = results[0]
            sample_values = [v for v in stats.get("sample_values", []) if v is not None]
            
            # Calculate additional statistics
            field_stats = {
                "total_count": stats.get("count", 0),
                "null_count": stats.get("null_count", 0),
                "non_null_count": stats.get("count", 0) - stats.get("null_count", 0),
                "unique_count": len(sample_values),
                "sample_values": sample_values[:10],  # First 10 values
                "cardinality": len(sample_values) / max(1, stats.get("count", 1))
            }
            
            # Detect data type
            if sample_values:
                field_stats["detected_type"] = self._detect_field_type(sample_values)
            
            return field_stats
            
        except Exception as e:
            logger.error(f"❌ Field statistics failed: {collection_name}.{field_name}: {str(e)}")
            return {"error": str(e)}
    
    async def sample_documents(
        self, 
        collection_name: str, 
        sample_size: int = 100,
        filter_query: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get random sample of documents from collection
        
        Args:
            collection_name: Target collection
            sample_size: Number of documents to sample
            filter_query: Optional filter to apply before sampling
            
        Returns:
            List of sampled documents
        """
        
        try:
            if not await self._collection_exists(collection_name):
                return []
            
            pipeline = []
            
            if filter_query:
                pipeline.append({"$match": filter_query})
            
            pipeline.append({"$sample": {"size": sample_size}})
            
            results = await self.execute_aggregation(collection_name, pipeline)
            
            logger.debug(f"📊 Sampled documents: {collection_name} = {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Document sampling failed: {collection_name}: {str(e)}")
            return []
    
    async def validate_pipeline(
        self, 
        collection_name: str, 
        pipeline: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate aggregation pipeline without executing it
        
        Args:
            collection_name: Target collection
            pipeline: Pipeline to validate
            
        Returns:
            Validation result with details
        """
        
        try:
            if not await self._collection_exists(collection_name):
                return {
                    "valid": False,
                    "error": f"Collection '{collection_name}' does not exist"
                }
            
            # Basic validation
            if not isinstance(pipeline, list):
                return {
                    "valid": False,
                    "error": "Pipeline must be a list"
                }
            
            if len(pipeline) == 0:
                return {
                    "valid": False,
                    "error": "Pipeline cannot be empty"
                }
            
            # Check for valid stages
            valid_stages = {
                "$match", "$group", "$sort", "$limit", "$skip", "$project", 
                "$unwind", "$lookup", "$addFields", "$sample", "$count",
                "$facet", "$bucket", "$bucketAuto", "$sortByCount"
            }
            
            for i, stage in enumerate(pipeline):
                if not isinstance(stage, dict):
                    return {
                        "valid": False,
                        "error": f"Stage {i} must be a dictionary"
                    }
                
                stage_names = list(stage.keys())
                if len(stage_names) != 1:
                    return {
                        "valid": False,
                        "error": f"Stage {i} must have exactly one operation"
                    }
                
                stage_name = stage_names[0]
                if stage_name not in valid_stages:
                    return {
                        "valid": False,
                        "error": f"Unknown stage '{stage_name}' at position {i}"
                    }
            
            # Try to explain the pipeline (dry run)
            try:
                collection = self.db_manager.get_collection(collection_name)
                
                # Add explain stage for validation
                explain_pipeline = pipeline + [{"$limit": 0}]
                
                # This will validate the pipeline without returning data
                cursor = collection.aggregate(explain_pipeline)
                await cursor.to_list(length=0)
                
                return {
                    "valid": True,
                    "stages": len(pipeline),
                    "estimated_performance": self._estimate_pipeline_performance(pipeline)
                }
                
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Pipeline validation failed: {str(e)}"
                }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    def _optimize_pipeline(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize aggregation pipeline for better performance"""
        
        optimized = []
        
        # Move $match stages to the beginning
        match_stages = [stage for stage in pipeline if "$match" in stage]
        other_stages = [stage for stage in pipeline if "$match" not in stage]
        
        # Add match stages first
        optimized.extend(match_stages)
        
        # Add other stages
        for stage in other_stages:
            # Add $sort before $limit for efficiency
            if "$limit" in stage and optimized and "$sort" not in str(optimized[-1]):
                # If we have a group stage before limit, add sort by count
                if optimized and "$group" in str(optimized[-1]):
                    # Try to add a smart sort based on grouped field
                    group_stage = optimized[-1]["$group"]
                    if "_id" in group_stage and any(key.startswith("count") or key.startswith("total") for key in group_stage.keys()):
                        count_field = next((key for key in group_stage.keys() if key.startswith("count") or key.startswith("total")), None)
                        if count_field:
                            optimized.append({"$sort": {count_field: -1}})
            
            optimized.append(stage)
        
        return optimized
    
    def _add_safety_limits(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add safety limits to prevent runaway queries"""
        
        safe_pipeline = pipeline.copy()
        
        # Check if there's already a limit
        has_limit = any("$limit" in stage for stage in safe_pipeline)
        
        if not has_limit:
            safe_pipeline.append({"$limit": settings.MAX_QUERY_RESULTS})
        else:
            # Ensure limit is not too high
            for stage in safe_pipeline:
                if "$limit" in stage:
                    current_limit = stage["$limit"]
                    if current_limit > settings.MAX_QUERY_RESULTS:
                        stage["$limit"] = settings.MAX_QUERY_RESULTS
        
        return safe_pipeline
    
    async def _collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        
        try:
            collections = await self.db_manager.get_collections()
            return collection_name in collections
        except Exception:
            return False
    
    def _detect_field_type(self, sample_values: List[Any]) -> str:
        """Detect field type from sample values"""
        
        if not sample_values:
            return "unknown"
        
        # Count types
        type_counts = {}
        for value in sample_values[:50]:  # Check first 50 values
            value_type = type(value).__name__
            type_counts[value_type] = type_counts.get(value_type, 0) + 1
        
        # Return most common type
        if type_counts:
            most_common_type = max(type_counts, key=type_counts.get)
            
            # Map Python types to our field types
            type_mapping = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
                "datetime": "datetime",
                "NoneType": "null"
            }
            
            return type_mapping.get(most_common_type, most_common_type)
        
        return "unknown"
    
    def _estimate_pipeline_performance(self, pipeline: List[Dict[str, Any]]) -> str:
        """Estimate pipeline performance based on stages"""
        
        # Simple heuristic based on stage types
        expensive_stages = ["$lookup", "$unwind", "$facet", "$bucket"]
        moderate_stages = ["$group", "$sort"]
        cheap_stages = ["$match", "$project", "$limit", "$skip"]
        
        score = 0
        for stage in pipeline:
            stage_name = list(stage.keys())[0]
            
            if stage_name in expensive_stages:
                score += 3
            elif stage_name in moderate_stages:
                score += 2
            elif stage_name in cheap_stages:
                score += 1
        
        if score <= 3:
            return "fast"
        elif score <= 6:
            return "moderate"
        else:
            return "slow"
    
    def _update_performance_stats(self, execution_time: float, success: bool):
        """Update internal performance statistics"""
        
        self.performance_stats["total_queries"] += 1
        
        if success:
            self.performance_stats["total_execution_time"] += execution_time
            self.performance_stats["average_execution_time"] = (
                self.performance_stats["total_execution_time"] / 
                max(1, self.performance_stats["total_queries"] - self.performance_stats["failed_queries"])
            )
            
            if execution_time > 5000:  # 5 seconds
                self.performance_stats["slow_queries"] += 1
        else:
            self.performance_stats["failed_queries"] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        
        total_queries = self.performance_stats["total_queries"]
        success_rate = 0
        
        if total_queries > 0:
            successful_queries = total_queries - self.performance_stats["failed_queries"]
            success_rate = (successful_queries / total_queries) * 100
        
        return {
            **self.performance_stats,
            "success_rate_percent": round(success_rate, 2),
            "slow_query_rate_percent": round(
                (self.performance_stats["slow_queries"] / max(1, total_queries)) * 100, 2
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database service health check"""
        
        try:
            # Test basic connectivity
            collections = await self.db_manager.get_collections()
            
            # Test a simple query if collections exist
            if collections:
                test_collection = collections[0]
                await self.count_documents(test_collection)
            
            return {
                "status": "healthy",
                "collections_available": len(collections),
                "performance_stats": self.get_performance_stats(),
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }