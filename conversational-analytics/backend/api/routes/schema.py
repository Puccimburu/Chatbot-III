# api/routes/schema.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.schema_service import SchemaService
from services.database_service import DatabaseService
from core.schema_detection.detector import SchemaDetector
from core.schema_detection.statistics_engine import StatisticsEngine
from models.query_models import (
    SchemaAnalysisRequest, SchemaAnalysisResponse,
    CollectionInfoRequest, CollectionInfoResponse
)
from utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/schema", tags=["schema"])

# Dependency injection
async def get_schema_service() -> SchemaService:
    """Get schema service instance"""
    from app import get_schema_service as _get_schema_service
    return await _get_schema_service()

async def get_database_service() -> DatabaseService:
    """Get database service instance"""
    from app import get_database_service as _get_database_service
    return await _get_database_service()

# Request/Response Models
class RefreshSchemaRequest(BaseModel):
    collections: Optional[List[str]] = Field(None, description="Specific collections to refresh")
    force_refresh: bool = Field(False, description="Force refresh even if cache is valid")
    deep_analysis: bool = Field(False, description="Perform deep statistical analysis")

class SchemaOverviewResponse(BaseModel):
    success: bool
    collections: Dict[str, Any]
    total_collections: int
    analysis_timestamp: datetime
    cache_status: str
    recommendations: List[str]

class CollectionDetailRequest(BaseModel):
    collection_name: str
    include_statistics: bool = Field(True, description="Include detailed statistics")
    include_relationships: bool = Field(True, description="Include relationship information")
    include_sample_data: bool = Field(False, description="Include sample documents")

# Routes
@router.get("/collections", response_model=SchemaOverviewResponse)
async def get_collections_overview(
    background_tasks: BackgroundTasks,
    refresh: bool = Query(False, description="Force refresh schema cache"),
    schema_service: SchemaService = Depends(get_schema_service)
) -> JSONResponse:
    """
    Get overview of all collections with basic schema information
    """
    try:
        logger.info(f"📋 Getting collections overview (refresh={refresh})")
        
        # Get schema information
        if refresh:
            schema_info = await schema_service.refresh_schema_cache()
        else:
            schema_info = await schema_service.get_schema_info()
        
        if not schema_info:
            # Trigger background schema detection
            background_tasks.add_task(schema_service.detect_and_cache_schema)
            
            return JSONResponse(
                status_code=202,
                content={
                    "success": True,
                    "message": "Schema detection initiated. Please check back in a few moments.",
                    "collections": {},
                    "total_collections": 0,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "cache_status": "detecting",
                    "recommendations": ["Schema analysis in progress"]
                }
            )
        
        # Format collections for overview
        formatted_collections = {}
        recommendations = []
        
        for collection_name, collection_info in schema_info.items():
            if isinstance(collection_info, dict):
                analytics_potential = collection_info.get("analytics_potential", "unknown")
                field_count = len(collection_info.get("fields", {}))
                
                formatted_collections[collection_name] = {
                    "field_count": field_count,
                    "analytics_potential": analytics_potential,
                    "data_types": collection_info.get("field_types", {}),
                    "document_count": collection_info.get("document_count", 0),
                    "last_analyzed": collection_info.get("last_analyzed", "never")
                }
                
                # Generate recommendations
                if analytics_potential == "high":
                    recommendations.append(f"'{collection_name}' has high analytics potential")
                elif field_count > 20:
                    recommendations.append(f"'{collection_name}' has many fields ({field_count}) - consider focusing on key metrics")
        
        response_data = {
            "success": True,
            "collections": formatted_collections,
            "total_collections": len(formatted_collections),
            "analysis_timestamp": datetime.now(),
            "cache_status": "fresh" if not refresh else "refreshed",
            "recommendations": recommendations[:5]  # Limit to top 5
        }
        
        logger.info(f"✅ Collections overview returned: {len(formatted_collections)} collections")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error getting collections overview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collections overview: {str(e)}"
        )

@router.get("/collections/{collection_name}", response_model=CollectionInfoResponse)
async def get_collection_details(
    collection_name: str,
    include_statistics: bool = Query(True, description="Include detailed statistics"),
    include_relationships: bool = Query(True, description="Include relationship info"),
    include_sample_data: bool = Query(False, description="Include sample documents"),
    schema_service: SchemaService = Depends(get_schema_service),
    db_service: DatabaseService = Depends(get_database_service)
) -> JSONResponse:
    """
    Get detailed information about a specific collection
    """
    try:
        logger.info(f"🔍 Getting details for collection: {collection_name}")
        
        # Get schema information
        schema_info = await schema_service.get_schema_info()
        
        if not schema_info or collection_name not in schema_info:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found or not analyzed"
            )
        
        collection_info = schema_info[collection_name]
        
        # Build detailed response
        response_data = {
            "success": True,
            "collection_name": collection_name,
            "basic_info": {
                "field_count": len(collection_info.get("fields", {})),
                "document_count": collection_info.get("document_count", 0),
                "analytics_potential": collection_info.get("analytics_potential", "unknown"),
                "last_analyzed": collection_info.get("last_analyzed", "never")
            },
            "fields": collection_info.get("fields", {}),
            "field_types": collection_info.get("field_types", {}),
            "metrics": collection_info.get("metrics", []),
            "dimensions": collection_info.get("dimensions", [])
        }
        
        # Add statistics if requested
        if include_statistics:
            stats_engine = StatisticsEngine(db_service)
            collection_stats = await stats_engine.compute_collection_statistics(collection_name)
            
            response_data["statistics"] = {
                "document_count": collection_stats.document_count,
                "average_document_size": collection_stats.average_document_size,
                "estimated_storage_size": collection_stats.estimated_storage_size,
                "data_quality_score": collection_stats.data_quality_score,
                "update_frequency": collection_stats.update_frequency,
                "field_statistics": {
                    name: {
                        "data_type": stats.data_type,
                        "null_percentage": stats.null_percentage,
                        "unique_count": stats.unique_count,
                        "cardinality": stats.cardinality,
                        "sample_values": stats.sample_values[:5]
                    }
                    for name, stats in collection_stats.field_statistics.items()
                }
            }
        
        # Add relationships if requested
        if include_relationships:
            relationships = await schema_service.get_collection_relationships(collection_name)
            response_data["relationships"] = [
                {
                    "related_collection": rel.related_collection,
                    "relationship_strength": rel.relationship_strength,
                    "suggested_join_strategy": rel.suggested_join_strategy,
                    "field_relationships": [
                        {
                            "from_field": field_rel.from_field,
                            "to_field": field_rel.to_field,
                            "relationship_type": field_rel.relationship_type.value,
                            "confidence": field_rel.confidence
                        }
                        for field_rel in rel.relationships
                    ]
                }
                for rel in relationships
            ]
        
        # Add sample data if requested
        if include_sample_data:
            try:
                sample_docs = await db_service.execute_aggregation(
                    collection_name, 
                    [{"$sample": {"size": 3}}]
                )
                response_data["sample_documents"] = sample_docs or []
            except Exception as e:
                logger.warning(f"Could not get sample data: {e}")
                response_data["sample_documents"] = []
        
        logger.info(f"✅ Collection details returned for: {collection_name}")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection details: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection details: {str(e)}"
        )

@router.post("/refresh")
async def refresh_schema(
    request: RefreshSchemaRequest,
    background_tasks: BackgroundTasks,
    schema_service: SchemaService = Depends(get_schema_service)
) -> JSONResponse:
    """
    Refresh schema information for specific collections or all collections
    """
    try:
        logger.info(f"🔄 Refreshing schema (collections={request.collections}, force={request.force_refresh})")
        
        if request.collections:
            # Refresh specific collections
            refreshed_collections = []
            for collection_name in request.collections:
                try:
                    await schema_service.refresh_collection_schema(
                        collection_name, 
                        force=request.force_refresh,
                        deep_analysis=request.deep_analysis
                    )
                    refreshed_collections.append(collection_name)
                except Exception as e:
                    logger.warning(f"Failed to refresh {collection_name}: {e}")
            
            response_data = {
                "success": True,
                "message": f"Refreshed {len(refreshed_collections)} collections",
                "refreshed_collections": refreshed_collections,
                "failed_collections": [c for c in request.collections if c not in refreshed_collections],
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            # Refresh all collections in background
            background_tasks.add_task(
                schema_service.detect_and_cache_schema,
                force_refresh=request.force_refresh,
                deep_analysis=request.deep_analysis
            )
            
            response_data = {
                "success": True,
                "message": "Full schema refresh initiated in background",
                "timestamp": datetime.now().isoformat(),
                "estimated_completion": "2-5 minutes"
            }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error refreshing schema: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh schema: {str(e)}"
        )

@router.get("/analysis/suggestions")
async def get_analysis_suggestions(
    collection: Optional[str] = Query(None, description="Specific collection to analyze"),
    schema_service: SchemaService = Depends(get_schema_service)
) -> JSONResponse:
    """
    Get suggestions for data analysis based on schema
    """
    try:
        logger.info(f"💡 Getting analysis suggestions (collection={collection})")
        
        schema_info = await schema_service.get_schema_info()
        
        if not schema_info:
            raise HTTPException(
                status_code=404,
                detail="No schema information available. Please refresh schema first."
            )
        
        suggestions = []
        
        if collection:
            # Suggestions for specific collection
            if collection not in schema_info:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{collection}' not found"
                )
            
            collection_info = schema_info[collection]
            suggestions.extend(
                await _generate_collection_suggestions(collection, collection_info)
            )
        
        else:
            # Suggestions for all collections
            for coll_name, coll_info in schema_info.items():
                if isinstance(coll_info, dict):
                    coll_suggestions = await _generate_collection_suggestions(coll_name, coll_info)
                    suggestions.extend(coll_suggestions[:2])  # Limit per collection
        
        # Sort suggestions by priority/relevance
        suggestions.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        response_data = {
            "success": True,
            "suggestions": suggestions[:20],  # Limit total suggestions
            "total_available": len(suggestions),
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis suggestions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analysis suggestions: {str(e)}"
        )

@router.get("/relationships")
async def get_all_relationships(
    min_confidence: float = Query(0.7, description="Minimum relationship confidence"),
    schema_service: SchemaService = Depends(get_schema_service)
) -> JSONResponse:
    """
    Get all discovered relationships between collections
    """
    try:
        logger.info(f"🔗 Getting all relationships (min_confidence={min_confidence})")
        
        all_relationships = await schema_service.get_all_relationships()
        
        # Filter by confidence and format for response
        formatted_relationships = []
        
        for collection_name, relationships in all_relationships.items():
            for relationship in relationships:
                if relationship.relationship_strength >= min_confidence:
                    formatted_relationships.append({
                        "from_collection": collection_name,
                        "to_collection": relationship.related_collection,
                        "strength": relationship.relationship_strength,
                        "join_strategy": relationship.suggested_join_strategy,
                        "field_relationships": [
                            {
                                "from_field": fr.from_field,
                                "to_field": fr.to_field,
                                "type": fr.relationship_type.value,
                                "confidence": fr.confidence,
                                "cardinality_ratio": fr.cardinality_ratio
                            }
                            for fr in relationship.relationships
                        ]
                    })
        
        # Sort by relationship strength
        formatted_relationships.sort(key=lambda x: x["strength"], reverse=True)
        
        response_data = {
            "success": True,
            "relationships": formatted_relationships,
            "total_relationships": len(formatted_relationships),
            "min_confidence_filter": min_confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error getting relationships: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get relationships: {str(e)}"
        )

@router.post("/analyze")
async def analyze_schema(
    request: SchemaAnalysisRequest,
    background_tasks: BackgroundTasks,
    schema_service: SchemaService = Depends(get_schema_service),
    db_service: DatabaseService = Depends(get_database_service)
) -> JSONResponse:
    """
    Perform comprehensive schema analysis
    """
    try:
        logger.info(f"🔬 Starting comprehensive schema analysis")
        
        # Start analysis in background
        background_tasks.add_task(
            _perform_comprehensive_analysis,
            request, schema_service, db_service
        )
        
        response_data = {
            "success": True,
            "message": "Comprehensive schema analysis initiated",
            "analysis_id": f"analysis_{int(datetime.now().timestamp())}",
            "estimated_completion": "5-10 minutes",
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error starting schema analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start schema analysis: {str(e)}"
        )

@router.get("/status")
async def get_schema_status(
    schema_service: SchemaService = Depends(get_schema_service)
) -> JSONResponse:
    """
    Get current status of schema detection and caching
    """
    try:
        cache_info = await schema_service.get_cache_info()
        schema_info = await schema_service.get_schema_info()
        
        status = "empty"
        if schema_info:
            if cache_info.get("is_fresh", False):
                status = "fresh"
            elif cache_info.get("is_stale", False):
                status = "stale"
            else:
                status = "available"
        
        collections_analyzed = len(schema_info) if schema_info else 0
        
        response_data = {
            "success": True,
            "status": status,
            "collections_analyzed": collections_analyzed,
            "cache_info": cache_info,
            "last_update": cache_info.get("last_update"),
            "next_refresh": cache_info.get("next_refresh"),
            "is_analysis_running": schema_service.is_analysis_running(),
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error getting schema status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get schema status: {str(e)}"
        )

# Helper functions
async def _generate_collection_suggestions(collection_name: str, 
                                         collection_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate analysis suggestions for a specific collection
    """
    suggestions = []
    
    # Get collection characteristics
    field_count = len(collection_info.get("fields", {}))
    analytics_potential = collection_info.get("analytics_potential", "unknown")
    field_types = collection_info.get("field_types", {})
    metrics = collection_info.get("metrics", [])
    dimensions = collection_info.get("dimensions", [])
    
    # Suggest based on analytics potential
    if analytics_potential == "high":
        suggestions.append({
            "type": "high_value_analysis",
            "collection": collection_name,
            "title": f"Explore {collection_name} - High Analytics Potential",
            "description": f"This collection has rich data perfect for analysis with {len(metrics)} metrics and {len(dimensions)} dimensions",
            "suggested_questions": [
                f"What are the trends in {collection_name}?",
                f"Show me the top performers in {collection_name}",
                f"How does {collection_name} data break down by category?"
            ],
            "priority": 90
        })
    
    # Suggest based on numeric fields
    numeric_fields = field_types.get("numeric", 0)
    if numeric_fields >= 2:
        suggestions.append({
            "type": "comparative_analysis",
            "collection": collection_name,
            "title": f"Compare Metrics in {collection_name}",
            "description": f"With {numeric_fields} numeric fields, this collection is great for comparisons",
            "suggested_questions": [
                f"Compare different metrics in {collection_name}",
                f"What are the correlations in {collection_name}?",
                f"Show me the distribution of values in {collection_name}"
            ],
            "priority": 80
        })
    
    # Suggest based on categorical fields
    categorical_fields = field_types.get("categorical", 0)
    if categorical_fields >= 1:
        suggestions.append({
            "type": "segmentation_analysis",
            "collection": collection_name,
            "title": f"Segment Analysis for {collection_name}",
            "description": f"Analyze different segments with {categorical_fields} categorical fields",
            "suggested_questions": [
                f"Break down {collection_name} by category",
                f"Which segments perform best in {collection_name}?",
                f"Show me the distribution across categories in {collection_name}"
            ],
            "priority": 70
        })
    
    # Suggest based on temporal fields
    temporal_fields = field_types.get("datetime", 0)
    if temporal_fields >= 1:
        suggestions.append({
            "type": "time_series_analysis",
            "collection": collection_name,
            "title": f"Time-based Analysis for {collection_name}",
            "description": f"Analyze trends over time with {temporal_fields} date fields",
            "suggested_questions": [
                f"Show me trends over time in {collection_name}",
                f"What are the seasonal patterns in {collection_name}?",
                f"How has {collection_name} changed month by month?"
            ],
            "priority": 85
        })
    
    return suggestions

async def _perform_comprehensive_analysis(request: SchemaAnalysisRequest,
                                        schema_service: SchemaService,
                                        db_service: DatabaseService) -> None:
    """
    Perform comprehensive schema analysis in background
    """
    try:
        logger.info("🔬 Starting comprehensive analysis")
        
        # Refresh schema with deep analysis
        await schema_service.detect_and_cache_schema(
            force_refresh=True,
            deep_analysis=True
        )
        
        # Analyze relationships
        await schema_service.discover_and_cache_relationships()
        
        # Generate analysis report
        schema_info = await schema_service.get_schema_info()
        relationships = await schema_service.get_all_relationships()
        
        # Create comprehensive report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "collections_analyzed": len(schema_info) if schema_info else 0,
            "total_relationships": sum(len(rels) for rels in relationships.values()),
            "high_value_collections": [
                name for name, info in schema_info.items()
                if isinstance(info, dict) and info.get("analytics_potential") == "high"
            ],
            "recommendations": await _generate_comprehensive_recommendations(schema_info, relationships)
        }
        
        # Cache the report
        await schema_service.cache_analysis_report(report)
        
        logger.info("✅ Comprehensive analysis completed")
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")

async def _generate_comprehensive_recommendations(schema_info: Dict[str, Any],
                                                relationships: Dict[str, Any]) -> List[str]:
    """
    Generate comprehensive recommendations based on analysis
    """
    recommendations = []
    
    if not schema_info:
        return ["No schema data available for recommendations"]
    
    # Count collection types
    high_value_count = sum(
        1 for info in schema_info.values()
        if isinstance(info, dict) and info.get("analytics_potential") == "high"
    )
    
    total_collections = len(schema_info)
    relationship_count = sum(len(rels) for rels in relationships.values())
    
    # Generate recommendations
    if high_value_count > 0:
        recommendations.append(f"Focus analysis on {high_value_count} high-value collections for best insights")
    
    if relationship_count > 5:
        recommendations.append(f"Strong data relationships detected ({relationship_count}) - consider cross-collection analysis")
    
    if total_collections > 10:
        recommendations.append("Large schema detected - use collection filtering for focused analysis")
    
    # Add specific collection recommendations
    for name, info in schema_info.items():
        if isinstance(info, dict):
            field_count = len(info.get("fields", {}))
            if field_count > 20:
                recommendations.append(f"'{name}' has many fields ({field_count}) - consider field selection for queries")
    
    return recommendations[:10]  # Limit to top 10