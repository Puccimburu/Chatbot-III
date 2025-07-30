# core/visualization/chart_suggester.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ChartType(Enum):
    BAR = "bar"
    HORIZONTAL_BAR = "horizontalBar"
    LINE = "line"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    SCATTER = "scatter"
    AREA = "area"
    RADAR = "radar"
    BUBBLE = "bubble"

@dataclass
class ChartSuggestion:
    chart_type: ChartType
    confidence: float
    title: str
    reasoning: str
    alternative_charts: List[ChartType]
    chart_config: Dict[str, Any]

class ChartSuggester:
    """
    Intelligent chart type suggestion based on data patterns, query intent, and best practices
    """
    
    def __init__(self):
        self.chart_patterns = self._initialize_chart_patterns()
        self.max_pie_slices = 8
        self.max_bar_items = 20
        self.min_line_points = 3
    
    def suggest_chart_type(self, data: List[Dict], query_context: Dict, 
                          user_question: str) -> ChartSuggestion:
        """
        Main method to suggest the best chart type for given data and context
        """
        try:
            # Analyze data structure
            data_analysis = self._analyze_data_structure(data)
            
            # Analyze query intent
            query_intent = self._analyze_query_intent(user_question, query_context)
            
            # Get chart suggestions based on analysis
            suggestions = self._generate_chart_suggestions(data_analysis, query_intent)
            
            # Select best suggestion
            best_suggestion = self._select_best_chart(suggestions, data_analysis)
            
            # Generate chart configuration
            chart_config = self._generate_chart_config(best_suggestion.chart_type, data, data_analysis)
            
            best_suggestion.chart_config = chart_config
            
            logger.info(f"📊 Suggested {best_suggestion.chart_type.value} chart with {best_suggestion.confidence:.2f} confidence")
            
            return best_suggestion
            
        except Exception as e:
            logger.error(f"Error in chart suggestion: {e}")
            return self._fallback_suggestion(data)
    
    def _analyze_data_structure(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the structure and patterns in the data
        """
        if not data:
            return {"type": "empty", "count": 0}
        
        first_row = data[0]
        field_types = {}
        field_patterns = {}
        
        # Analyze each field
        for field, value in first_row.items():
            field_types[field] = self._detect_field_type(field, value, data)
            field_patterns[field] = self._analyze_field_pattern(field, data)
        
        # Determine data structure type
        structure_type = self._determine_structure_type(data, field_types)
        
        return {
            "type": structure_type,
            "count": len(data),
            "fields": field_types,
            "patterns": field_patterns,
            "numeric_fields": [f for f, t in field_types.items() if t == "numeric"],
            "categorical_fields": [f for f, t in field_types.items() if t == "categorical"],
            "temporal_fields": [f for f, t in field_types.items() if t == "temporal"],
            "has_time_series": self._has_time_series_pattern(data, field_types)
        }
    
    def _analyze_query_intent(self, question: str, context: Dict) -> Dict[str, Any]:
        """
        Analyze the user's question to understand their intent
        """
        question_lower = question.lower()
        
        # Intent keywords
        comparison_keywords = ['compare', 'vs', 'versus', 'between', 'difference']
        trend_keywords = ['trend', 'over time', 'monthly', 'yearly', 'growth', 'change']
        distribution_keywords = ['distribution', 'breakdown', 'split', 'percentage', 'share']
        ranking_keywords = ['top', 'bottom', 'best', 'worst', 'highest', 'lowest', 'rank']
        correlation_keywords = ['relationship', 'correlation', 'association', 'impact']
        
        intent = {
            "is_comparison": any(keyword in question_lower for keyword in comparison_keywords),
            "is_trend": any(keyword in question_lower for keyword in trend_keywords),
            "is_distribution": any(keyword in question_lower for keyword in distribution_keywords),
            "is_ranking": any(keyword in question_lower for keyword in ranking_keywords),
            "is_correlation": any(keyword in question_lower for keyword in correlation_keywords),
            "query_complexity": self._assess_query_complexity(question),
            "aggregation_type": context.get("aggregation_type", "unknown")
        }
        
        return intent
    
    def _generate_chart_suggestions(self, data_analysis: Dict, query_intent: Dict) -> List[ChartSuggestion]:
        """
        Generate multiple chart suggestions based on data and intent analysis
        """
        suggestions = []
        
        # Time series data
        if data_analysis.get("has_time_series") and query_intent.get("is_trend"):
            suggestions.append(ChartSuggestion(
                chart_type=ChartType.LINE,
                confidence=0.9,
                title="Time Series Trend",
                reasoning="Data contains temporal patterns perfect for line charts",
                alternative_charts=[ChartType.AREA, ChartType.BAR],
                chart_config={}
            ))
        
        # Distribution analysis
        if query_intent.get("is_distribution") and data_analysis["count"] <= self.max_pie_slices:
            suggestions.append(ChartSuggestion(
                chart_type=ChartType.PIE,
                confidence=0.85,
                title="Distribution Breakdown",
                reasoning="Perfect for showing parts of a whole",
                alternative_charts=[ChartType.DOUGHNUT, ChartType.BAR],
                chart_config={}
            ))
        
        # Comparison analysis
        if query_intent.get("is_comparison") or query_intent.get("is_ranking"):
            chart_type = ChartType.HORIZONTAL_BAR if data_analysis["count"] > 8 else ChartType.BAR
            suggestions.append(ChartSuggestion(
                chart_type=chart_type,
                confidence=0.8,
                title="Comparison Analysis",
                reasoning="Bar charts excel at comparing values across categories",
                alternative_charts=[ChartType.LINE, ChartType.HORIZONTAL_BAR],
                chart_config={}
            ))
        
        # Correlation analysis
        if (query_intent.get("is_correlation") and 
            len(data_analysis["numeric_fields"]) >= 2):
            suggestions.append(ChartSuggestion(
                chart_type=ChartType.SCATTER,
                confidence=0.75,
                title="Correlation Analysis",
                reasoning="Scatter plots reveal relationships between numeric variables",
                alternative_charts=[ChartType.BUBBLE, ChartType.LINE],
                chart_config={}
            ))
        
        # Fallback suggestions based on data structure
        if not suggestions:
            suggestions.extend(self._generate_fallback_suggestions(data_analysis))
        
        return suggestions
    
    def _select_best_chart(self, suggestions: List[ChartSuggestion], 
                          data_analysis: Dict) -> ChartSuggestion:
        """
        Select the best chart suggestion based on confidence and data suitability
        """
        if not suggestions:
            return self._fallback_suggestion([])
        
        # Apply data constraints
        filtered_suggestions = []
        for suggestion in suggestions:
            if self._is_chart_suitable_for_data(suggestion.chart_type, data_analysis):
                filtered_suggestions.append(suggestion)
        
        if not filtered_suggestions:
            filtered_suggestions = suggestions
        
        # Sort by confidence and return best
        filtered_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return filtered_suggestions[0]
    
    def _generate_chart_config(self, chart_type: ChartType, data: List[Dict], 
                              data_analysis: Dict) -> Dict[str, Any]:
        """
        Generate Chart.js configuration for the selected chart type
        """
        base_config = {
            "type": chart_type.value,
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {
                        "display": chart_type in [ChartType.PIE, ChartType.DOUGHNUT, ChartType.LINE]
                    },
                    "tooltip": {
                        "enabled": True
                    }
                }
            }
        }
        
        # Chart-specific configurations
        if chart_type in [ChartType.BAR, ChartType.HORIZONTAL_BAR, ChartType.LINE]:
            base_config["options"]["scales"] = {
                "y": {
                    "beginAtZero": True,
                    "grid": {"display": True}
                },
                "x": {
                    "grid": {"display": False}
                }
            }
        
        if chart_type == ChartType.HORIZONTAL_BAR:
            base_config["options"]["indexAxis"] = "y"
        
        if chart_type in [ChartType.PIE, ChartType.DOUGHNUT]:
            base_config["options"]["plugins"]["legend"]["position"] = "right"
            
        if chart_type == ChartType.DOUGHNUT:
            base_config["options"]["cutout"] = "50%"
        
        # Add data-specific optimizations
        if data_analysis["count"] > 15:
            base_config["options"]["plugins"]["legend"]["display"] = False
        
        return base_config
    
    def _detect_field_type(self, field_name: str, sample_value: Any, data: List[Dict]) -> str:
        """
        Detect the type of a field based on its name and values
        """
        # Temporal field detection
        temporal_indicators = ['date', 'time', 'created', 'updated', 'month', 'year', 'day']
        if any(indicator in field_name.lower() for indicator in temporal_indicators):
            return "temporal"
        
        # Sample multiple values for better type detection
        sample_values = [row.get(field_name) for row in data[:10] if row.get(field_name) is not None]
        
        if not sample_values:
            return "unknown"
        
        # Numeric detection
        if all(isinstance(v, (int, float)) for v in sample_values):
            return "numeric"
        
        # Categorical detection (string with limited unique values)
        if all(isinstance(v, str) for v in sample_values):
            unique_count = len(set(sample_values))
            if unique_count <= min(10, len(data) * 0.5):
                return "categorical"
            return "text"
        
        return "mixed"
    
    def _analyze_field_pattern(self, field_name: str, data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze patterns in field values
        """
        values = [row.get(field_name) for row in data if row.get(field_name) is not None]
        
        if not values:
            return {"pattern": "empty"}
        
        unique_values = list(set(values))
        
        return {
            "pattern": "varied" if len(unique_values) > len(values) * 0.8 else "repeating",
            "unique_count": len(unique_values),
            "cardinality": len(unique_values) / len(values),
            "sample_values": unique_values[:5]
        }
    
    def _determine_structure_type(self, data: List[Dict], field_types: Dict) -> str:
        """
        Determine the overall structure type of the data
        """
        if not data:
            return "empty"
        
        if len(data) == 1:
            return "single_record"
        
        numeric_fields = sum(1 for t in field_types.values() if t == "numeric")
        categorical_fields = sum(1 for t in field_types.values() if t == "categorical")
        temporal_fields = sum(1 for t in field_types.values() if t == "temporal")
        
        if temporal_fields > 0 and numeric_fields > 0:
            return "time_series"
        elif categorical_fields > 0 and numeric_fields > 0:
            return "categorical_numeric"
        elif numeric_fields > 1:
            return "multi_numeric"
        elif categorical_fields > 0:
            return "categorical"
        else:
            return "mixed"
    
    def _has_time_series_pattern(self, data: List[Dict], field_types: Dict) -> bool:
        """
        Check if data has time series characteristics
        """
        temporal_fields = [f for f, t in field_types.items() if t == "temporal"]
        if not temporal_fields:
            return False
        
        # Check if we have sequential temporal data
        temporal_field = temporal_fields[0]
        values = sorted([row.get(temporal_field) for row in data if row.get(temporal_field)])
        
        return len(values) >= self.min_line_points
    
    def _assess_query_complexity(self, question: str) -> str:
        """
        Assess the complexity of the user's question
        """
        complex_indicators = ['correlation', 'relationship', 'impact', 'regression', 'predict']
        medium_indicators = ['compare', 'analyze', 'breakdown', 'distribution']
        
        question_lower = question.lower()
        
        if any(indicator in question_lower for indicator in complex_indicators):
            return "high"
        elif any(indicator in question_lower for indicator in medium_indicators):
            return "medium"
        else:
            return "simple"
    
    def _generate_fallback_suggestions(self, data_analysis: Dict) -> List[ChartSuggestion]:
        """
        Generate fallback suggestions based purely on data structure
        """
        suggestions = []
        
        # Small datasets - pie chart
        if data_analysis["count"] <= 6:
            suggestions.append(ChartSuggestion(
                chart_type=ChartType.PIE,
                confidence=0.6,
                title="Simple Distribution",
                reasoning="Small dataset suitable for pie chart",
                alternative_charts=[ChartType.BAR, ChartType.DOUGHNUT],
                chart_config={}
            ))
        
        # Medium datasets - bar chart
        elif data_analysis["count"] <= 15:
            suggestions.append(ChartSuggestion(
                chart_type=ChartType.BAR,
                confidence=0.7,
                title="Category Comparison",
                reasoning="Medium dataset perfect for bar chart comparison",
                alternative_charts=[ChartType.HORIZONTAL_BAR, ChartType.LINE],
                chart_config={}
            ))
        
        # Large datasets - horizontal bar
        else:
            suggestions.append(ChartSuggestion(
                chart_type=ChartType.HORIZONTAL_BAR,
                confidence=0.65,
                title="Ranked Comparison",
                reasoning="Large dataset better displayed horizontally",
                alternative_charts=[ChartType.BAR, ChartType.LINE],
                chart_config={}
            ))
        
        return suggestions
    
    def _is_chart_suitable_for_data(self, chart_type: ChartType, data_analysis: Dict) -> bool:
        """
        Check if a chart type is suitable for the given data
        """
        data_count = data_analysis["count"]
        
        # Pie charts shouldn't have too many slices
        if chart_type in [ChartType.PIE, ChartType.DOUGHNUT] and data_count > self.max_pie_slices:
            return False
        
        # Line charts need enough points
        if chart_type == ChartType.LINE and data_count < self.min_line_points:
            return False
        
        # Scatter plots need numeric data
        if chart_type == ChartType.SCATTER and len(data_analysis["numeric_fields"]) < 2:
            return False
        
        return True
    
    def _fallback_suggestion(self, data: List[Dict]) -> ChartSuggestion:
        """
        Provide a safe fallback suggestion when all else fails
        """
        return ChartSuggestion(
            chart_type=ChartType.BAR,
            confidence=0.5,
            title="Default Visualization",
            reasoning="Fallback to versatile bar chart",
            alternative_charts=[ChartType.PIE, ChartType.LINE],
            chart_config=self._generate_chart_config(ChartType.BAR, data, {"count": len(data)})
        )
    
    def _initialize_chart_patterns(self) -> Dict[str, Any]:
        """
        Initialize chart pattern templates and rules
        """
        return {
            "time_series": [ChartType.LINE, ChartType.AREA],
            "comparison": [ChartType.BAR, ChartType.HORIZONTAL_BAR],
            "distribution": [ChartType.PIE, ChartType.DOUGHNUT],
            "correlation": [ChartType.SCATTER, ChartType.BUBBLE],
            "ranking": [ChartType.HORIZONTAL_BAR, ChartType.BAR],
            "multi_series": [ChartType.LINE, ChartType.BAR]
        }