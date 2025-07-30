# api/routes/charts.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.visualization.chart_analyzer import ChartAnalyzer
from core.visualization.chart_suggester import ChartSuggester, ChartSuggestion
from core.visualization.chart_generator import ChartGenerator
from core.visualization.response_formatter import ResponseFormatter
from services.database_service import DatabaseService
from models.query_models import ChartGenerationRequest, ChartGenerationResponse
from utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/charts", tags=["charts"])

# Dependency injection
async def get_database_service() -> DatabaseService:
    """Get database service instance"""
    from app import get_database_service as _get_database_service
    return await _get_database_service()

# Request/Response Models
class ChartSuggestionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Data to analyze for chart suggestions")
    user_question: str = Field("", description="Original user question for context")
    query_context: Optional[Dict[str, Any]] = Field(None, description="Additional query context")

class ChartAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Data to analyze")
    question: str = Field("", description="User's question")
    preferred_chart_types: Optional[List[str]] = Field(None, description="User's preferred chart types")

class ChartConfigRequest(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    data: List[Dict[str, Any]] = Field(..., description="Data for the chart")
    title: Optional[str] = Field(None, description="Chart title")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional chart options")

class ChartWorthinessResponse(BaseModel):
    success: bool
    is_chart_worthy: bool
    reasoning: str
    suggested_chart_types: List[str]
    confidence: float
    alternatives: List[str]

class ChartSuggestionsResponse(BaseModel):
    success: bool
    suggestions: List[Dict[str, Any]]
    best_suggestion: Dict[str, Any]
    analysis_summary: str
    timestamp: datetime

# Routes
@router.post("/analyze", response_model=ChartWorthinessResponse)
async def analyze_chart_worthiness(
    request: ChartAnalysisRequest
) -> JSONResponse:
    """
    Analyze if data warrants visualization and suggest chart types
    """
    try:
        logger.info(f"📊 Analyzing chart worthiness for {len(request.data)} data points")
        
        if not request.data:
            return JSONResponse(content={
                "success": True,
                "is_chart_worthy": False,
                "reasoning": "No data provided for analysis",
                "suggested_chart_types": [],
                "confidence": 0.0,
                "alternatives": ["Collect data before creating visualizations"]
            })
        
        # Initialize analyzers
        chart_analyzer = ChartAnalyzer()
        chart_suggester = ChartSuggester()
        
        # Analyze chart worthiness
        is_worthy = chart_analyzer.is_chart_worthy(request.data, request.question)
        
        if not is_worthy["worthy"]:
            return JSONResponse(content={
                "success": True,
                "is_chart_worthy": False,
                "reasoning": is_worthy["reasoning"],
                "suggested_chart_types": [],
                "confidence": is_worthy["confidence"],
                "alternatives": is_worthy.get("alternatives", [])
            })
        
        # Get chart suggestions
        query_context = {
            "user_question": request.question,
            "data_size": len(request.data),
            "preferred_types": request.preferred_chart_types or []
        }
        
        suggestion = chart_suggester.suggest_chart_type(
            request.data, 
            query_context, 
            request.question
        )
        
        # Format response
        response_data = {
            "success": True,
            "is_chart_worthy": True,
            "reasoning": suggestion.reasoning,
            "suggested_chart_types": [suggestion.chart_type.value] + [t.value for t in suggestion.alternative_charts],
            "confidence": suggestion.confidence,
            "alternatives": []
        }
        
        logger.info(f"✅ Chart analysis complete: worthy={True}, type={suggestion.chart_type.value}")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing chart worthiness: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze chart worthiness: {str(e)}"
        )

@router.post("/suggest", response_model=ChartSuggestionsResponse)
async def get_chart_suggestions(
    request: ChartSuggestionRequest
) -> JSONResponse:
    """
    Get multiple chart suggestions with detailed analysis
    """
    try:
        logger.info(f"💡 Getting chart suggestions for {len(request.data)} data points")
        
        if not request.data:
            raise HTTPException(
                status_code=400,
                detail="No data provided for chart suggestions"
            )
        
        chart_suggester = ChartSuggester()
        
        # Get multiple suggestions
        query_context = request.query_context or {}
        query_context.update({
            "user_question": request.user_question,
            "data_size": len(request.data)
        })
        
        # Get primary suggestion
        primary_suggestion = chart_suggester.suggest_chart_type(
            request.data, 
            query_context, 
            request.user_question
        )
        
        # Generate alternative suggestions
        suggestions = await _generate_multiple_chart_suggestions(
            request.data, 
            request.user_question, 
            query_context
        )
        
        # Format suggestions
        formatted_suggestions = []
        for i, suggestion in enumerate(suggestions):
            formatted_suggestions.append({
                "rank": i + 1,
                "chart_type": suggestion.chart_type.value,
                "title": suggestion.title,
                "confidence": suggestion.confidence,
                "reasoning": suggestion.reasoning,
                "alternative_charts": [t.value for t in suggestion.alternative_charts],
                "preview_config": suggestion.chart_config
            })
        
        # Create analysis summary
        data_summary = _analyze_data_characteristics(request.data)
        
        response_data = {
            "success": True,
            "suggestions": formatted_suggestions,
            "best_suggestion": formatted_suggestions[0] if formatted_suggestions else {},
            "analysis_summary": data_summary,
            "timestamp": datetime.now()
        }
        
        logger.info(f"✅ Generated {len(formatted_suggestions)} chart suggestions")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart suggestions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get chart suggestions: {str(e)}"
        )

@router.post("/generate", response_model=ChartGenerationResponse)
async def generate_chart_config(
    request: ChartConfigRequest
) -> JSONResponse:
    """
    Generate complete Chart.js configuration for specified chart type
    """
    try:
        logger.info(f"🎨 Generating {request.chart_type} chart config for {len(request.data)} data points")
        
        if not request.data:
            raise HTTPException(
                status_code=400,
                detail="No data provided for chart generation"
            )
        
        # Validate chart type
        valid_chart_types = ["bar", "line", "pie", "doughnut", "horizontalBar", "scatter", "bubble", "radar", "area"]
        if request.chart_type not in valid_chart_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chart type '{request.chart_type}'. Valid types: {', '.join(valid_chart_types)}"
            )
        
        # Generate chart configuration
        chart_generator = ChartGenerator()
        
        context = {
            "chart_type": request.chart_type,
            "user_options": request.options or {}
        }
        
        chart_config = chart_generator.generate_chart_config(
            request.data,
            request.chart_type,
            request.title or f"{request.chart_type.title()} Chart",
            context
        )
        
        # Generate formatted response
        response_formatter = ResponseFormatter()
        
        # Create mock query context for response formatting
        query_context = {
            "chart_type": request.chart_type,
            "user_generated": True
        }
        
        formatted_response = response_formatter.format_response(
            request.data,
            {"chart_type": request.chart_type, "confidence": 1.0},
            f"Generated {request.chart_type} chart with {len(request.data)} data points",
            query_context
        )
        
        # Merge configurations
        final_config = {
            "success": True,
            "chart_config": chart_config,
            "chart_type": request.chart_type,
            "data_summary": formatted_response["summary"],
            "insights": formatted_response["insights"],
            "metadata": {
                "data_points": len(request.data),
                "generated_at": datetime.now().isoformat(),
                "chart_version": "1.0"
            }
        }
        
        logger.info(f"✅ Chart config generated successfully for {request.chart_type}")
        
        return JSONResponse(content=final_config)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating chart config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate chart config: {str(e)}"
        )

@router.get("/types")
async def get_available_chart_types() -> JSONResponse:
    """
    Get list of available chart types with descriptions
    """
    try:
        chart_types = {
            "bar": {
                "name": "Bar Chart",
                "description": "Great for comparing categories or showing rankings",
                "best_for": ["Comparisons", "Rankings", "Categorical data"],
                "data_requirements": "Categorical labels with numeric values",
                "max_recommended_items": 20
            },
            "horizontalBar": {
                "name": "Horizontal Bar Chart", 
                "description": "Better for long category names or many categories",
                "best_for": ["Long labels", "Many categories", "Rankings"],
                "data_requirements": "Categorical labels with numeric values",
                "max_recommended_items": 30
            },
            "line": {
                "name": "Line Chart",
                "description": "Perfect for showing trends over time",
                "best_for": ["Time series", "Trends", "Continuous data"],
                "data_requirements": "Sequential data points (usually time-based)",
                "min_recommended_points": 3
            },
            "pie": {
                "name": "Pie Chart",
                "description": "Shows parts of a whole as percentages",
                "best_for": ["Proportions", "Market share", "Distribution"],
                "data_requirements": "Categories that sum to a meaningful total",
                "max_recommended_slices": 8
            },
            "doughnut": {
                "name": "Doughnut Chart",
                "description": "Like pie chart but with center space for additional info",
                "best_for": ["Proportions", "Modern design", "Distribution"],
                "data_requirements": "Categories that sum to a meaningful total",
                "max_recommended_slices": 8
            },
            "scatter": {
                "name": "Scatter Plot",
                "description": "Shows relationships between two numeric variables",
                "best_for": ["Correlations", "Relationships", "Outlier detection"],
                "data_requirements": "Two numeric variables per data point",
                "min_recommended_points": 5
            },
            "area": {
                "name": "Area Chart",
                "description": "Line chart with filled area underneath",
                "best_for": ["Volume over time", "Cumulative data", "Emphasis on magnitude"],
                "data_requirements": "Sequential data points with positive values",
                "min_recommended_points": 3
            }
        }
        
        response_data = {
            "success": True,
            "chart_types": chart_types,
            "total_types": len(chart_types),
            "usage_tips": [
                "Consider your data structure when choosing chart types",
                "Bar charts work best for comparisons",
                "Line charts are ideal for time-based data",
                "Pie charts should be used sparingly and with few categories",
                "Scatter plots reveal relationships between variables"
            ]
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error getting chart types: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get chart types: {str(e)}"
        )

@router.post("/validate")
async def validate_chart_data(
    chart_type: str = Query(..., description="Chart type to validate for"),
    data: List[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Validate if data is suitable for specific chart type
    """
    try:
        logger.info(f"🔍 Validating data for {chart_type} chart")
        
        if not data:
            raise HTTPException(
                status_code=400,
                detail="No data provided for validation"
            )
        
        validation_result = await _validate_data_for_chart_type(chart_type, data)
        
        response_data = {
            "success": True,
            "chart_type": chart_type,
            "is_valid": validation_result["valid"],
            "issues": validation_result["issues"],
            "warnings": validation_result["warnings"],
            "suggestions": validation_result["suggestions"],
            "data_summary": {
                "total_points": len(data),
                "fields": list(data[0].keys()) if data else [],
                "sample_data": data[:3] if data else []
            }
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating chart data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate chart data: {str(e)}"
        )

# Helper functions
async def _generate_multiple_chart_suggestions(data: List[Dict], question: str, 
                                             context: Dict) -> List[ChartSuggestion]:
    """
    Generate multiple chart suggestions for comparison
    """
    chart_suggester = ChartSuggester()
    
    # Get primary suggestion
    primary = chart_suggester.suggest_chart_type(data, context, question)
    suggestions = [primary]
    
    # Generate alternatives based on data characteristics
    data_analysis = chart_suggester._analyze_data_structure(data)
    
    # Add time series suggestion if applicable
    if data_analysis.get("has_time_series") and primary.chart_type.value != "line":
        try:
            time_context = context.copy()
            time_context["force_chart_type"] = "line"
            time_suggestion = chart_suggester.suggest_chart_type(data, time_context, question + " over time")
            if time_suggestion.confidence > 0.6:
                suggestions.append(time_suggestion)
        except:
            pass
    
    # Add comparison suggestion if applicable
    if data_analysis["count"] <= 15 and primary.chart_type.value not in ["bar", "horizontalBar"]:
        try:
            comp_context = context.copy()
            comp_context["force_chart_type"] = "bar"
            comp_suggestion = chart_suggester.suggest_chart_type(data, comp_context, "compare " + question)
            if comp_suggestion.confidence > 0.6:
                suggestions.append(comp_suggestion)
        except:
            pass
    
    # Add distribution suggestion if applicable
    if data_analysis["count"] <= 8 and primary.chart_type.value not in ["pie", "doughnut"]:
        try:
            dist_context = context.copy()
            dist_context["force_chart_type"] = "pie"
            dist_suggestion = chart_suggester.suggest_chart_type(data, dist_context, "distribution of " + question)
            if dist_suggestion.confidence > 0.6:
                suggestions.append(dist_suggestion)
        except:
            pass
    
    # Sort by confidence
    suggestions.sort(key=lambda x: x.confidence, reverse=True)
    
    return suggestions[:5]  # Return top 5 suggestions

def _analyze_data_characteristics(data: List[Dict]) -> str:
    """
    Analyze data characteristics and create summary
    """
    if not data:
        return "No data provided for analysis"
    
    data_size = len(data)
    field_count = len(data[0]) if data else 0
    
    # Analyze field types
    numeric_fields = 0
    text_fields = 0
    date_fields = 0
    
    if data:
        sample_row = data[0]
        for value in sample_row.values():
            if isinstance(value, (int, float)):
                numeric_fields += 1
            elif isinstance(value, str):
                # Simple date detection
                if any(keyword in str(value).lower() for keyword in ['date', 'time', '-', '/']):
                    date_fields += 1
                else:
                    text_fields += 1
    
    summary_parts = [
        f"Dataset contains {data_size} records with {field_count} fields"
    ]
    
    if numeric_fields > 0:
        summary_parts.append(f"{numeric_fields} numeric fields suitable for quantitative analysis")
    
    if text_fields > 0:
        summary_parts.append(f"{text_fields} categorical fields for grouping and segmentation")
    
    if date_fields > 0:
        summary_parts.append(f"{date_fields} temporal fields enabling time-based analysis")
    
    # Chart recommendations based on characteristics
    if data_size <= 8:
        summary_parts.append("Small dataset - pie charts and bar charts work well")
    elif data_size <= 20:
        summary_parts.append("Medium dataset - bar charts and line charts recommended")
    else:
        summary_parts.append("Large dataset - consider horizontal bar charts or data aggregation")
    
    return ". ".join(summary_parts) + "."

async def _validate_data_for_chart_type(chart_type: str, data: List[Dict]) -> Dict[str, Any]:
    """
    Validate if data is suitable for specific chart type
    """
    validation = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "suggestions": []
    }
    
    if not data:
        validation["valid"] = False
        validation["issues"].append("No data provided")
        return validation
    
    data_size = len(data)
    sample_row = data[0] if data else {}
    field_count = len(sample_row)
    
    # Chart-specific validations
    if chart_type in ["pie", "doughnut"]:
        if data_size > 8:
            validation["warnings"].append(f"Pie charts work best with ≤8 slices, you have {data_size}")
            validation["suggestions"].append("Consider using a bar chart for better readability")
        
        if data_size == 1:
            validation["valid"] = False
            validation["issues"].append("Pie charts need multiple categories to show distribution")
        
        # Check for negative values
        numeric_fields = [k for k, v in sample_row.items() if isinstance(v, (int, float))]
        if numeric_fields:
            for row in data[:10]:  # Check first 10 rows
                for field in numeric_fields:
                    if row.get(field, 0) < 0:
                        validation["warnings"].append("Pie charts work best with positive values")
                        break
    
    elif chart_type == "line":
        if data_size < 3:
            validation["warnings"].append(f"Line charts work best with ≥3 data points, you have {data_size}")
        
        # Check for sequential/time data
        has_temporal_field = any(
            'date' in str(k).lower() or 'time' in str(k).lower() 
            for k in sample_row.keys()
        )
        
        if not has_temporal_field:
            validation["warnings"].append("Line charts are most effective with time-based or sequential data")
            validation["suggestions"].append("Consider if your data has a natural sequence or order")
    
    elif chart_type in ["bar", "horizontalBar"]:
        if data_size > 30:
            validation["warnings"].append(f"Bar charts can become crowded with {data_size} items")
            if chart_type == "bar":
                validation["suggestions"].append("Consider using horizontal bar chart for better label readability")
            else:
                validation["suggestions"].append("Consider filtering to top/bottom items or grouping categories")
        
        if field_count < 2:
            validation["valid"] = False
            validation["issues"].append("Bar charts need both category labels and numeric values")
    
    elif chart_type == "scatter":
        numeric_fields = sum(1 for v in sample_row.values() if isinstance(v, (int, float)))
        
        if numeric_fields < 2:
            validation["valid"] = False
            validation["issues"].append("Scatter plots require at least 2 numeric fields")
        
        if data_size < 5:
            validation["warnings"].append(f"Scatter plots work best with ≥5 data points, you have {data_size}")
    
    # General validations
    if field_count == 0:
        validation["valid"] = False
        validation["issues"].append("Data rows are empty")
    
    elif field_count == 1:
        validation["warnings"].append("Single field data limits visualization options")
        validation["suggestions"].append("Consider adding categorical or grouping fields")
    
    # Check for data consistency
    if data_size > 1:
        first_keys = set(data[0].keys())
        inconsistent_rows = []
        
        for i, row in enumerate(data[1:6], 1):  # Check first 5 rows
            if set(row.keys()) != first_keys:
                inconsistent_rows.append(i)
        
        if inconsistent_rows:
            validation["warnings"].append(f"Inconsistent field structure in rows: {inconsistent_rows}")
            validation["suggestions"].append("Ensure all data rows have the same fields")
    
    return validation