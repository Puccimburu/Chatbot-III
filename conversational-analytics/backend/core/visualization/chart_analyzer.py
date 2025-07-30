"""
Chart Analyzer - Determines chart worthiness and generates Chart.js configurations
Intelligent analysis of data patterns for optimal visualization
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import statistics
from datetime import datetime

logger = logging.getLogger(__name__)


class ChartAnalyzer:
    """Intelligent chart analysis and Chart.js configuration generation"""
    
    def __init__(self):
        self.analysis_count = 0
        self.chart_worthy_count = 0
        self.chart_type_usage = {}
    
    def analyze_chart_worthiness(
        self, 
        user_question: str, 
        raw_data: List[Dict[str, Any]], 
        query_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze if data is worth visualizing and suggest best chart type
        
        Args:
            user_question: Original user question
            raw_data: Query results to analyze
            query_context: Context about the query
            
        Returns:
            Analysis result with chart recommendations
        """
        
        self.analysis_count += 1
        
        try:
            logger.debug(f"📊 Analyzing chart worthiness for {len(raw_data)} data points")
            
            # Quick checks for non-chart-worthy data
            if not raw_data:
                return self._create_not_worthy_response("No data to visualize")
            
            if len(raw_data) == 1:
                return self._create_not_worthy_response(
                    "Single value result - better displayed as a metric"
                )
            
            # Analyze data structure
            data_analysis = self._analyze_data_structure(raw_data)
            
            # Determine chart worthiness
            chart_worthy = self._is_chart_worthy(data_analysis, user_question)
            
            if not chart_worthy["worthy"]:
                return self._create_not_worthy_response(chart_worthy["reason"])
            
            # Suggest best chart type
            chart_type = self._suggest_chart_type(data_analysis, user_question, query_context)
            
            # Generate insights
            insights = self._generate_insights(raw_data, data_analysis, chart_type)
            
            self.chart_worthy_count += 1
            self.chart_type_usage[chart_type] = self.chart_type_usage.get(chart_type, 0) + 1
            
            return {
                "chart_worthy": True,
                "suggested_chart_type": chart_type,
                "summary": self._generate_summary(raw_data, data_analysis, user_question),
                "insights": insights,
                "confidence": chart_worthy["confidence"],
                "data_analysis": data_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Chart analysis failed: {str(e)}")
            return self._create_not_worthy_response(f"Analysis failed: {str(e)}")
    
    def generate_chart_config(
        self, 
        raw_data: List[Dict[str, Any]], 
        chart_type: str,
        existing_config: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate Chart.js configuration from raw data
        
        Args:
            raw_data: Data to visualize
            chart_type: Type of chart to generate
            existing_config: Existing config to merge with
            
        Returns:
            Complete Chart.js configuration
        """
        
        try:
            if not raw_data:
                return None
            
            logger.debug(f"📈 Generating {chart_type} chart config for {len(raw_data)} data points")
            
            # Route to specific chart generators
            if chart_type == "bar":
                return self._generate_bar_chart_config(raw_data, existing_config)
            elif chart_type == "horizontal_bar":
                return self._generate_horizontal_bar_chart_config(raw_data, existing_config)
            elif chart_type == "line":
                return self._generate_line_chart_config(raw_data, existing_config)
            elif chart_type == "pie":
                return self._generate_pie_chart_config(raw_data, existing_config)
            elif chart_type == "doughnut":
                return self._generate_doughnut_chart_config(raw_data, existing_config)
            elif chart_type == "scatter":
                return self._generate_scatter_chart_config(raw_data, existing_config)
            elif chart_type == "histogram":
                return self._generate_histogram_chart_config(raw_data, existing_config)
            else:
                # Default to bar chart
                return self._generate_bar_chart_config(raw_data, existing_config)
            
        except Exception as e:
            logger.error(f"❌ Chart config generation failed: {str(e)}")
            return None
    
    def _analyze_data_structure(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the structure and patterns in the data"""
        
        if not raw_data:
            return {"empty": True}
        
        first_item = raw_data[0]
        field_count = len(first_item)
        total_records = len(raw_data)
        
        # Analyze fields
        fields = {}
        for field_name, value in first_item.items():
            field_info = {
                "name": field_name,
                "type": type(value).__name__,
                "sample_value": value,
                "is_numeric": self._is_numeric(value),
                "is_temporal": self._is_temporal_field(field_name) or self._is_date_value(value),
                "is_identifier": field_name == "_id" or "id" in field_name.lower()
            }
            fields[field_name] = field_info
        
        # Identify patterns
        numeric_fields = [name for name, info in fields.items() if info["is_numeric"]]
        categorical_fields = [name for name, info in fields.items() if not info["is_numeric"] and not info["is_temporal"]]
        temporal_fields = [name for name, info in fields.items() if info["is_temporal"]]
        
        # Calculate value ranges for numeric fields
        numeric_stats = {}
        for field_name in numeric_fields:
            values = []
            for record in raw_data:
                if field_name in record and self._is_numeric(record[field_name]):
                    values.append(float(record[field_name]))
            
            if values:
                numeric_stats[field_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "range": max(values) - min(values)
                }
        
        # Detect if data is time series
        is_time_series = (
            len(temporal_fields) > 0 and 
            len(numeric_fields) > 0 and 
            total_records > 2
        )
        
        # Detect if data is categorical breakdown
        is_categorical_breakdown = (
            len(categorical_fields) > 0 and 
            len(numeric_fields) > 0 and
            field_count == 2
        )
        
        return {
            "total_records": total_records,
            "field_count": field_count,
            "fields": fields,
            "numeric_fields": numeric_fields,
            "categorical_fields": categorical_fields,
            "temporal_fields": temporal_fields,
            "numeric_stats": numeric_stats,
            "is_time_series": is_time_series,
            "is_categorical_breakdown": is_categorical_breakdown,
            "has_aggregation_pattern": self._detect_aggregation_pattern(raw_data)
        }
    
    def _is_chart_worthy(
        self, 
        data_analysis: Dict[str, Any], 
        user_question: str
    ) -> Dict[str, Any]:
        """Determine if data is suitable for visualization"""
        
        if data_analysis.get("empty"):
            return {"worthy": False, "reason": "No data available", "confidence": 1.0}
        
        total_records = data_analysis["total_records"]
        
        # Too many records for effective visualization
        if total_records > 100:
            return {
                "worthy": True, 
                "reason": f"Large dataset ({total_records} records) - consider table view", 
                "confidence": 0.6
            }
        
        # Single record - not chart worthy
        if total_records == 1:
            return {"worthy": False, "reason": "Single value - use metric display", "confidence": 1.0}
        
        # Check for suitable data patterns
        has_numeric = len(data_analysis["numeric_fields"]) > 0
        has_categorical = len(data_analysis["categorical_fields"]) > 0
        has_temporal = len(data_analysis["temporal_fields"]) > 0
        
        # Excellent chart candidates
        if data_analysis["is_time_series"]:
            return {"worthy": True, "reason": "Time series data ideal for line chart", "confidence": 0.95}
        
        if data_analysis["is_categorical_breakdown"] and 2 <= total_records <= 20:
            return {"worthy": True, "reason": "Categorical breakdown perfect for bar/pie chart", "confidence": 0.9}
        
        # Good chart candidates
        if has_numeric and has_categorical and total_records <= 50:
            return {"worthy": True, "reason": "Numeric data by category suitable for visualization", "confidence": 0.8}
        
        if has_numeric and total_records <= 30:
            return {"worthy": True, "reason": "Numeric data suitable for charts", "confidence": 0.75}
        
        # Questionable chart candidates
        if total_records > 50:
            return {"worthy": True, "reason": "Large dataset - table might be more appropriate", "confidence": 0.5}
        
        # Check question intent
        question_lower = user_question.lower()
        chart_indicating_words = ["show", "chart", "graph", "plot", "visualize", "compare", "trend", "distribution"]
        
        if any(word in question_lower for word in chart_indicating_words):
            return {"worthy": True, "reason": "User question suggests visualization intent", "confidence": 0.7}
        
        # Default: small datasets are usually chart-worthy
        if total_records <= 20:
            return {"worthy": True, "reason": "Small dataset suitable for visualization", "confidence": 0.6}
        
        return {"worthy": False, "reason": "Data not well-suited for visualization", "confidence": 0.8}
    
    def _suggest_chart_type(
        self, 
        data_analysis: Dict[str, Any], 
        user_question: str,
        query_context: Dict[str, Any]
    ) -> str:
        """Suggest the best chart type for the data"""
        
        total_records = data_analysis["total_records"]
        
        # Time series data
        if data_analysis["is_time_series"]:
            return "line"
        
        # Small categorical breakdown - pie/doughnut charts
        if data_analysis["is_categorical_breakdown"] and total_records <= 8:
            return "pie"
        
        # Medium categorical breakdown - bar charts
        if data_analysis["is_categorical_breakdown"] and total_records <= 20:
            return "bar"
        
        # Large categorical breakdown - horizontal bar for readability
        if data_analysis["is_categorical_breakdown"] and total_records > 20:
            return "horizontal_bar"
        
        # Question-based hints
        question_lower = user_question.lower()
        
        if any(word in question_lower for word in ["trend", "over time", "monthly", "daily", "yearly"]):
            return "line"
        
        if any(word in question_lower for word in ["compare", "comparison", "vs", "versus"]):
            return "bar"
        
        if any(word in question_lower for word in ["distribution", "breakdown", "split", "share"]):
            if total_records <= 8:
                return "pie"
            else:
                return "bar"
        
        if any(word in question_lower for word in ["top", "ranking", "best", "worst"]):
            return "bar"
        
        # Default based on data characteristics
        if len(data_analysis["numeric_fields"]) >= 2:
            return "scatter"
        
        if total_records <= 10:
            return "pie"
        
        return "bar"  # Safe default
    
    def _generate_bar_chart_config(
        self, 
        raw_data: List[Dict[str, Any]], 
        existing_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate bar chart configuration"""
        
        # Find label and data fields
        label_field, data_field = self._identify_label_and_data_fields(raw_data)
        
        if not label_field or not data_field:
            return None
        
        # Extract labels and data
        labels = []
        data = []
        
        for record in raw_data:
            if label_field in record and data_field in record:
                labels.append(str(record[label_field]))
                data_value = record[data_field]
                data.append(float(data_value) if self._is_numeric(data_value) else 0)
        
        # Generate colors
        colors = self._generate_colors(len(data))
        
        config = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": self._format_field_name(data_field),
                    "data": data,
                    "backgroundColor": colors["background"],
                    "borderColor": colors["border"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{self._format_field_name(data_field)} by {self._format_field_name(label_field)}"
                    },
                    "legend": {
                        "display": False
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": self._format_field_name(data_field)
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": self._format_field_name(label_field)
                        }
                    }
                }
            }
        }
        
        # Merge with existing config if provided
        if existing_config:
            config = self._merge_configs(config, existing_config)
        
        return config
    
    def _generate_horizontal_bar_chart_config(
        self, 
        raw_data: List[Dict[str, Any]], 
        existing_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate horizontal bar chart configuration"""
        
        config = self._generate_bar_chart_config(raw_data, existing_config)
        
        if config:
            config["type"] = "bar"
            config["options"]["indexAxis"] = "y"
            
            # Swap x and y axis configurations
            if "scales" in config["options"]:
                x_config = config["options"]["scales"].get("x", {})
                y_config = config["options"]["scales"].get("y", {})
                
                config["options"]["scales"]["x"] = y_config
                config["options"]["scales"]["y"] = x_config
        
        return config
    
    def _generate_line_chart_config(
        self, 
        raw_data: List[Dict[str, Any]], 
        existing_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate line chart configuration"""
        
        label_field, data_field = self._identify_label_and_data_fields(raw_data)
        
        if not label_field or not data_field:
            return None
        
        # Extract and sort data by label (assuming temporal data)
        sorted_data = sorted(raw_data, key=lambda x: x.get(label_field, ""))
        
        labels = []
        data = []
        
        for record in sorted_data:
            if label_field in record and data_field in record:
                labels.append(str(record[label_field]))
                data_value = record[data_field]
                data.append(float(data_value) if self._is_numeric(data_value) else 0)
        
        config = {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": self._format_field_name(data_field),
                    "data": data,
                    "borderColor": "#3B82F6",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                    "borderWidth": 2,
                    "fill": True,
                    "tension": 0.4
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{self._format_field_name(data_field)} Over Time"
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": self._format_field_name(data_field)
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": self._format_field_name(label_field)
                        }
                    }
                }
            }
        }
        
        if existing_config:
            config = self._merge_configs(config, existing_config)
        
        return config
    
    def _generate_pie_chart_config(
        self, 
        raw_data: List[Dict[str, Any]], 
        existing_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate pie chart configuration"""
        
        label_field, data_field = self._identify_label_and_data_fields(raw_data)
        
        if not label_field or not data_field:
            return None
        
        labels = []
        data = []
        
        for record in raw_data:
            if label_field in record and data_field in record:
                labels.append(str(record[label_field]))
                data_value = record[data_field]
                data.append(float(data_value) if self._is_numeric(data_value) else 0)
        
        colors = self._generate_colors(len(data))
        
        config = {
            "type": "pie",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": colors["background"],
                    "borderColor": colors["border"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{self._format_field_name(data_field)} Distribution"
                    },
                    "legend": {
                        "display": True,
                        "position": "bottom"
                    }
                }
            }
        }
        
        if existing_config:
            config = self._merge_configs(config, existing_config)
        
        return config
    
    def _generate_doughnut_chart_config(
        self, 
        raw_data: List[Dict[str, Any]], 
        existing_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate doughnut chart configuration"""
        
        config = self._generate_pie_chart_config(raw_data, existing_config)
        
        if config:
            config["type"] = "doughnut"
        
        return config
    
    def _generate_scatter_chart_config(
        self, 
        raw_data: List[Dict[str, Any]], 
        existing_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate scatter plot configuration"""
        
        # Find two numeric fields
        numeric_fields = []
        for field_name, value in raw_data[0].items():
            if self._is_numeric(value):
                numeric_fields.append(field_name)
        
        if len(numeric_fields) < 2:
            return None
        
        x_field = numeric_fields[0]
        y_field = numeric_fields[1]
        
        # Extract data points
        data_points = []
        for record in raw_data:
            if x_field in record and y_field in record:
                x_val = record[x_field]
                y_val = record[y_field]
                
                if self._is_numeric(x_val) and self._is_numeric(y_val):
                    data_points.append({
                        "x": float(x_val),
                        "y": float(y_val)
                    })
        
        config = {
            "type": "scatter",
            "data": {
                "datasets": [{
                    "label": f"{self._format_field_name(y_field)} vs {self._format_field_name(x_field)}",
                    "data": data_points,
                    "backgroundColor": "rgba(59, 130, 246, 0.6)",
                    "borderColor": "#3B82F6",
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{self._format_field_name(y_field)} vs {self._format_field_name(x_field)}"
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": self._format_field_name(x_field)
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": self._format_field_name(y_field)
                        }
                    }
                }
            }
        }
        
        if existing_config:
            config = self._merge_configs(config, existing_config)
        
        return config
    
    def _generate_histogram_chart_config(
        self, 
        raw_data: List[Dict[str, Any]], 
        existing_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate histogram configuration (using bar chart)"""
        
        # Find numeric field
        numeric_field = None
        for field_name, value in raw_data[0].items():
            if self._is_numeric(value):
                numeric_field = field_name
                break
        
        if not numeric_field:
            return None
        
        # Extract values and create bins
        values = []
        for record in raw_data:
            if numeric_field in record and self._is_numeric(record[numeric_field]):
                values.append(float(record[numeric_field]))
        
        if not values:
            return None
        
        # Create histogram bins
        bins, frequencies = self._create_histogram_bins(values)
        
        config = {
            "type": "bar",
            "data": {
                "labels": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(frequencies))],
                "datasets": [{
                    "label": "Frequency",
                    "data": frequencies,
                    "backgroundColor": "rgba(59, 130, 246, 0.6)",
                    "borderColor": "#3B82F6",
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"Distribution of {self._format_field_name(numeric_field)}"
                    },
                    "legend": {
                        "display": False
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Frequency"
                        }
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": self._format_field_name(numeric_field)
                        }
                    }
                }
            }
        }
        
        if existing_config:
            config = self._merge_configs(config, existing_config)
        
        return config
    
    def _identify_label_and_data_fields(self, raw_data: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """Identify which fields to use for labels and data"""
        
        if not raw_data:
            return None, None
        
        first_record = raw_data[0]
        
        # Separate fields by type
        categorical_fields = []
        numeric_fields = []
        
        for field_name, value in first_record.items():
            if field_name == "_id":
                categorical_fields.insert(0, field_name)  # Prefer _id as label
            elif self._is_numeric(value):
                numeric_fields.append(field_name)
            else:
                categorical_fields.append(field_name)
        
        # Choose label field (categorical/identifier)
        label_field = None
        if categorical_fields:
            label_field = categorical_fields[0]
        
        # Choose data field (numeric)
        data_field = None
        if numeric_fields:
            # Prefer fields with meaningful names
            preferred_names = ["total", "count", "sum", "value", "amount", "revenue", "sales"]
            
            for preferred in preferred_names:
                for field in numeric_fields:
                    if preferred in field.lower():
                        data_field = field
                        break
                if data_field:
                    break
            
            # If no preferred name found, use first numeric field
            if not data_field:
                data_field = numeric_fields[0]
        
        return label_field, data_field
    
    def _generate_summary(
        self, 
        raw_data: List[Dict[str, Any]], 
        data_analysis: Dict[str, Any], 
        user_question: str
    ) -> str:
        """Generate a brief summary of the data for chart context"""
        
        total_records = len(raw_data)
        
        if data_analysis["is_time_series"]:
            return f"Time series data showing {total_records} data points over time."
        
        if data_analysis["is_categorical_breakdown"]:
            label_field, data_field = self._identify_label_and_data_fields(raw_data)
            return f"Breakdown showing {self._format_field_name(data_field)} across {total_records} categories."
        
        return f"Data visualization showing {total_records} data points."
    
    def _generate_insights(
        self, 
        raw_data: List[Dict[str, Any]], 
        data_analysis: Dict[str, Any], 
        chart_type: str
    ) -> List[str]:
        """Generate insights about the data"""
        
        insights = []
        
        # Basic insights
        total_records = len(raw_data)
        insights.append(f"Dataset contains {total_records} data points")
        
        # Numeric field insights
        if data_analysis["numeric_stats"]:
            for field_name, stats in data_analysis["numeric_stats"].items():
                if stats["range"] > 0:
                    insights.append(
                        f"{self._format_field_name(field_name)} ranges from {stats['min']:.1f} to {stats['max']:.1f}"
                    )
        
        # Chart-specific insights
        if chart_type in ["pie", "doughnut"] and total_records <= 10:
            insights.append("Pie chart ideal for showing proportional relationships")
        
        if chart_type == "line" and data_analysis["is_time_series"]:
            insights.append("Line chart reveals trends and patterns over time")
        
        if chart_type == "bar" and total_records > 10:
            insights.append("Bar chart effectively compares values across categories")
        
        return insights[:3]  # Limit to top 3 insights
    
    def _create_not_worthy_response(self, reason: str) -> Dict[str, Any]:
        """Create response for non-chart-worthy data"""
        
        return {
            "chart_worthy": False,
            "reason": reason,
            "suggested_chart_type": None,
            "summary": reason,
            "insights": [],
            "confidence": 0.9
        }
    
    def _detect_aggregation_pattern(self, raw_data: List[Dict[str, Any]]) -> bool:
        """Detect if data looks like aggregation results"""
        
        if not raw_data:
            return False
        
        # Common aggregation field patterns
        first_record = raw_data[0]
        aggregation_indicators = ["count", "total", "sum", "avg", "average", "min", "max"]
        
        for field_name in first_record.keys():
            field_lower = field_name.lower()
            if any(indicator in field_lower for indicator in aggregation_indicators):
                return True
        
        return False
    
    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric"""
        
        if isinstance(value, (int, float)):
            return True
        
        if isinstance(value, str):
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        
        return False
    
    def _is_temporal_field(self, field_name: str) -> bool:
        """Check if field name suggests temporal data"""
        
        temporal_indicators = ["date", "time", "created", "updated", "timestamp", "at"]
        field_lower = field_name.lower()
        
        return any(indicator in field_lower for indicator in temporal_indicators)
    
    def _is_date_value(self, value: Any) -> bool:
        """Check if value looks like a date"""
        
        if isinstance(value, datetime):
            return True
        
        if isinstance(value, str):
            # Simple date pattern check
            import re
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{4}-\d{2}',        # YYYY-MM
            ]
            
            for pattern in date_patterns:
                if re.match(pattern, value):
                    return True
        
        return False
    
    def _format_field_name(self, field_name: str) -> str:
        """Format field name for display"""
        
        if not field_name:
            return ""
        
        # Remove underscores and capitalize words
        formatted = field_name.replace("_", " ").title()
        
        # Handle common abbreviations
        replacements = {
            "Id": "ID",
            "Url": "URL",
            "Api": "API",
            "Uuid": "UUID"
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _generate_colors(self, count: int) -> Dict[str, List[str]]:
        """Generate color palette for charts"""
        
        # Base color palette
        base_colors = [
            "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
            "#06B6D4", "#84CC16", "#F97316", "#EC4899", "#6B7280"
        ]
        
        background_colors = []
        border_colors = []
        
        for i in range(count):
            base_color = base_colors[i % len(base_colors)]
            background_colors.append(base_color + "80")  # Add transparency
            border_colors.append(base_color)
        
        return {
            "background": background_colors,
            "border": border_colors
        }
    
    def _create_histogram_bins(self, values: List[float], num_bins: int = 10) -> Tuple[List[float], List[int]]:
        """Create histogram bins from numeric values"""
        
        if not values:
            return [], []
        
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            return [min_val, max_val], [len(values)]
        
        # Calculate bin edges
        bin_width = (max_val - min_val) / num_bins
        bins = [min_val + i * bin_width for i in range(num_bins + 1)]
        
        # Count frequencies
        frequencies = [0] * num_bins
        
        for value in values:
            bin_index = min(int((value - min_val) / bin_width), num_bins - 1)
            frequencies[bin_index] += 1
        
        return bins, frequencies
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge chart configurations"""
        
        def merge_dict(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        return merge_dict(base_config, override_config)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chart analyzer statistics"""
        
        chart_worthy_rate = 0
        if self.analysis_count > 0:
            chart_worthy_rate = (self.chart_worthy_count / self.analysis_count) * 100
        
        most_used_chart_type = None
        if self.chart_type_usage:
            most_used_chart_type = max(self.chart_type_usage, key=self.chart_type_usage.get)
        
        return {
            "total_analyses": self.analysis_count,
            "chart_worthy_count": self.chart_worthy_count,
            "chart_worthy_rate_percent": round(chart_worthy_rate, 2),
            "chart_type_usage": self.chart_type_usage,
            "most_used_chart_type": most_used_chart_type
        }