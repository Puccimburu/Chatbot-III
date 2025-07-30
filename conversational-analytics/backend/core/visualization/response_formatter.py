# core/visualization/response_formatter.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """
    Formats raw database results into user-friendly responses with Chart.js configurations
    """
    
    def __init__(self):
        self.number_formatter = self._initialize_number_formatter()
        
    def format_response(self, raw_data: List[Dict], chart_suggestion: Dict, 
                       summary: str, query_context: Dict) -> Dict[str, Any]:
        """
        Main method to format a complete analytics response
        """
        try:
            # Format the data for Chart.js
            chart_data = self._format_chart_data(raw_data, chart_suggestion)
            
            # Generate insights from the data
            insights = self._generate_insights(raw_data, query_context)
            
            # Format summary with numbers
            formatted_summary = self._format_summary(summary, raw_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(raw_data, query_context)
            
            # Determine if chart is worthy
            chart_worthy = self._is_chart_worthy(raw_data, chart_suggestion)
            
            response = {
                "success": True,
                "summary": formatted_summary,
                "chart_worthy": chart_worthy,
                "chart_data": chart_data if chart_worthy else None,
                "chart_suggestion": chart_suggestion.get("chart_type", "bar") if chart_worthy else None,
                "insights": insights,
                "recommendations": recommendations,
                "raw_data_count": len(raw_data),
                "formatted_data": self._format_table_data(raw_data),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_type": query_context.get("query_type", "unknown"),
                    "processing_method": query_context.get("processing_method", "ai")
                }
            }
            
            logger.info(f"📊 Formatted response: {len(raw_data)} records, chart_worthy={chart_worthy}")
            return response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return self._format_error_response(str(e), raw_data)
    
    def _format_chart_data(self, raw_data: List[Dict], chart_suggestion: Dict) -> Dict[str, Any]:
        """
        Format data specifically for Chart.js consumption
        """
        if not raw_data:
            return {}
        
        chart_type = chart_suggestion.get("chart_type", "bar")
        
        # Extract labels and data
        if len(raw_data[0]) >= 2:
            # Get the first two fields as label and value
            fields = list(raw_data[0].keys())
            label_field = fields[0]
            value_field = fields[1]
            
            labels = [self._format_label(row.get(label_field, "Unknown")) for row in raw_data]
            values = [self._format_number(row.get(value_field, 0)) for row in raw_data]
        else:
            # Single field data
            field = list(raw_data[0].keys())[0]
            labels = [str(i+1) for i in range(len(raw_data))]
            values = [self._format_number(row.get(field, 0)) for row in raw_data]
        
        # Create Chart.js data structure
        chart_data = {
            "type": chart_type,
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": values,
                    "backgroundColor": self._generate_colors(len(values), chart_type),
                    "borderColor": self._generate_border_colors(len(values), chart_type),
                    "borderWidth": 1
                }]
            },
            "options": self._get_chart_options(chart_type, chart_suggestion)
        }
        
        # Add dataset label for non-pie charts
        if chart_type not in ['pie', 'doughnut']:
            chart_data["data"]["datasets"][0]["label"] = self._generate_dataset_label(raw_data, value_field if 'value_field' in locals() else "Value")
        
        return chart_data
    
    def _generate_insights(self, raw_data: List[Dict], query_context: Dict) -> List[str]:
        """
        Generate intelligent insights from the data
        """
        insights = []
        
        if not raw_data:
            return ["No data available for analysis."]
        
        # Basic data insights
        insights.append(f"Analysis includes {len(raw_data)} data points")
        
        # Numerical insights
        numeric_fields = self._get_numeric_fields(raw_data[0])
        for field in numeric_fields[:2]:  # Limit to first 2 numeric fields
            values = [row.get(field, 0) for row in raw_data if isinstance(row.get(field), (int, float))]
            if values:
                insights.extend(self._analyze_numeric_field(field, values))
        
        # Top performer insights
        if len(raw_data) > 1:
            insights.extend(self._find_top_performers(raw_data))
        
        # Trend insights
        insights.extend(self._detect_trends(raw_data))
        
        return insights[:5]  # Limit to 5 most important insights
    
    def _generate_recommendations(self, raw_data: List[Dict], query_context: Dict) -> List[str]:
        """
        Generate actionable recommendations based on the data
        """
        recommendations = []
        
        if not raw_data:
            return ["Gather more data to enable meaningful analysis."]
        
        # Performance-based recommendations
        recommendations.extend(self._performance_recommendations(raw_data))
        
        # Data quality recommendations
        recommendations.extend(self._data_quality_recommendations(raw_data))
        
        # Business recommendations
        recommendations.extend(self._business_recommendations(raw_data, query_context))
        
        return recommendations[:3]  # Limit to 3 most actionable recommendations
    
    def _is_chart_worthy(self, raw_data: List[Dict], chart_suggestion: Dict) -> bool:
        """
        Determine if the data warrants a chart visualization
        """
        if not raw_data:
            return False
        
        # Single data point rarely needs a chart
        if len(raw_data) == 1:
            return False
        
        # Check if we have meaningful variation
        numeric_fields = self._get_numeric_fields(raw_data[0])
        if numeric_fields:
            field = numeric_fields[0]
            values = [row.get(field, 0) for row in raw_data]
            # If all values are the same, chart isn't helpful
            if len(set(values)) == 1:
                return False
        
        # Check chart suggestion confidence
        confidence = chart_suggestion.get("confidence", 0)
        if confidence < 0.5:
            return False
        
        return True
    
    def _format_table_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Format raw data for table display
        """
        if not raw_data:
            return []
        
        formatted_data = []
        for row in raw_data[:50]:  # Limit table display to 50 rows
            formatted_row = {}
            for key, value in row.items():
                formatted_row[self._format_header(key)] = self._format_cell_value(value)
            formatted_data.append(formatted_row)
        
        return formatted_data
    
    def _format_summary(self, summary: str, raw_data: List[Dict]) -> str:
        """
        Enhance summary with formatted numbers and context
        """
        if not summary:
            return self._generate_default_summary(raw_data)
        
        # Add context if missing
        if len(raw_data) > 1 and "records" not in summary.lower():
            summary += f" (Based on {len(raw_data)} records)"
        
        return summary
    
    def _generate_default_summary(self, raw_data: List[Dict]) -> str:
        """
        Generate a default summary when none is provided
        """
        if not raw_data:
            return "No data found for your query."
        
        if len(raw_data) == 1:
            return f"Found 1 result matching your criteria."
        
        return f"Analysis complete. Found {len(raw_data)} results matching your criteria."
    
    def _format_label(self, label: Any) -> str:
        """
        Format a label for chart display
        """
        if label is None:
            return "Unknown"
        
        label_str = str(label)
        
        # Truncate long labels
        if len(label_str) > 20:
            return label_str[:17] + "..."
        
        return label_str
    
    def _format_number(self, value: Any) -> float:
        """
        Format a number for chart data
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        try:
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return 0.0
    
    def _format_cell_value(self, value: Any) -> str:
        """
        Format a cell value for table display
        """
        if value is None:
            return ""
        
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        
        if isinstance(value, str) and len(value) > 50:
            return value[:47] + "..."
        
        return str(value)
    
    def _format_header(self, header: str) -> str:
        """
        Format a header for table display
        """
        # Convert from snake_case or camelCase to Title Case
        formatted = header.replace('_', ' ').replace('-', ' ')
        words = formatted.split()
        return ' '.join(word.capitalize() for word in words)
    
    def _get_numeric_fields(self, sample_row: Dict) -> List[str]:
        """
        Identify numeric fields in the data
        """
        numeric_fields = []
        for key, value in sample_row.items():
            if isinstance(value, (int, float)):
                numeric_fields.append(key)
        return numeric_fields
    
    def _analyze_numeric_field(self, field_name: str, values: List[float]) -> List[str]:
        """
        Analyze a numeric field and generate insights
        """
        insights = []
        
        if not values:
            return insights
        
        total = sum(values)
        avg = total / len(values)
        max_val = max(values)
        min_val = min(values)
        
        # Format field name for display
        display_name = self._format_header(field_name)
        
        insights.append(f"{display_name} ranges from {self._format_display_number(min_val)} to {self._format_display_number(max_val)}")
        
        if len(values) > 1:
            insights.append(f"Average {display_name.lower()}: {self._format_display_number(avg)}")
        
        return insights
    
    def _find_top_performers(self, raw_data: List[Dict]) -> List[str]:
        """
        Find and report top performers in the data
        """
        insights = []
        
        if len(raw_data) < 2:
            return insights
        
        # Find the first categorical and numeric field
        sample_row = raw_data[0]
        categorical_field = None
        numeric_field = None
        
        for key, value in sample_row.items():
            if categorical_field is None and isinstance(value, str):
                categorical_field = key
            if numeric_field is None and isinstance(value, (int, float)):
                numeric_field = key
        
        if categorical_field and numeric_field:
            # Sort by numeric field and get top performer
            sorted_data = sorted(raw_data, key=lambda x: x.get(numeric_field, 0), reverse=True)
            top_item = sorted_data[0]
            
            top_name = top_item.get(categorical_field, "Unknown")
            top_value = top_item.get(numeric_field, 0)
            
            insights.append(f"Top performer: {top_name} with {self._format_display_number(top_value)}")
        
        return insights
    
    def _detect_trends(self, raw_data: List[Dict]) -> List[str]:
        """
        Detect trends in the data
        """
        insights = []
        
        if len(raw_data) < 3:
            return insights
        
        # Look for temporal fields
        temporal_fields = []
        numeric_fields = []
        
        sample_row = raw_data[0]
        for key, value in sample_row.items():
            if any(indicator in key.lower() for indicator in ['date', 'time', 'month', 'year', 'day']):
                temporal_fields.append(key)
            elif isinstance(value, (int, float)):
                numeric_fields.append(key)
        
        if temporal_fields and numeric_fields:
            temporal_field = temporal_fields[0]
            numeric_field = numeric_fields[0]
            
            # Sort by temporal field
            try:
                sorted_data = sorted(raw_data, key=lambda x: x.get(temporal_field, ""))
                
                # Compare first and last values
                first_value = sorted_data[0].get(numeric_field, 0)
                last_value = sorted_data[-1].get(numeric_field, 0)
                
                if last_value > first_value * 1.1:
                    insights.append(f"Positive trend detected: {self._format_header(numeric_field)} increased over time")
                elif last_value < first_value * 0.9:
                    insights.append(f"Declining trend detected: {self._format_header(numeric_field)} decreased over time")
                else:
                    insights.append(f"{self._format_header(numeric_field)} remains relatively stable over time")
                    
            except Exception as e:
                logger.debug(f"Could not analyze trend: {e}")
        
        return insights
    
    def _performance_recommendations(self, raw_data: List[Dict]) -> List[str]:
        """
        Generate performance-based recommendations
        """
        recommendations = []
        
        if len(raw_data) < 2:
            return recommendations
        
        # Find numeric fields for performance analysis
        numeric_fields = self._get_numeric_fields(raw_data[0])
        
        if numeric_fields:
            field = numeric_fields[0]
            values = [row.get(field, 0) for row in raw_data]
            sorted_values = sorted(values, reverse=True)
            
            # Identify underperformers
            if len(sorted_values) >= 3:
                top_quartile = sorted_values[len(sorted_values)//4]
                bottom_value = sorted_values[-1]
                
                if top_quartile > bottom_value * 2:
                    recommendations.append(f"Focus on improving bottom performers to increase overall {self._format_header(field).lower()}")
        
        return recommendations
    
    def _data_quality_recommendations(self, raw_data: List[Dict]) -> List[str]:
        """
        Generate data quality recommendations
        """
        recommendations = []
        
        if not raw_data:
            return recommendations
        
        # Check for missing data
        total_fields = len(raw_data[0])
        null_count = 0
        
        for row in raw_data:
            null_count += sum(1 for value in row.values() if value is None or value == "")
        
        null_percentage = (null_count / (len(raw_data) * total_fields)) * 100
        
        if null_percentage > 10:
            recommendations.append("Consider improving data collection to reduce missing values")
        
        return recommendations
    
    def _business_recommendations(self, raw_data: List[Dict], query_context: Dict) -> List[str]:
        """
        Generate business-focused recommendations
        """
        recommendations = []
        
        query_type = query_context.get("query_type", "")
        
        if "sales" in query_type.lower():
            recommendations.append("Consider analyzing seasonal patterns to optimize inventory and marketing")
        elif "customer" in query_type.lower():
            recommendations.append("Segment customers by behavior to create targeted marketing campaigns")
        elif "product" in query_type.lower():
            recommendations.append("Analyze product performance to guide inventory and pricing decisions")
        
        return recommendations
    
    def _generate_colors(self, count: int, chart_type: str) -> List[str]:
        """
        Generate appropriate colors for charts
        """
        # Color palette
        colors = [
            '#3B82F6',  # Blue
            '#EF4444',  # Red
            '#10B981',  # Green
            '#F59E0B',  # Yellow
            '#8B5CF6',  # Purple
            '#EC4899',  # Pink
            '#6B7280',  # Gray
            '#F97316',  # Orange
            '#06B6D4',  # Cyan
            '#84CC16'   # Lime
        ]
        
        if chart_type in ['pie', 'doughnut']:
            # Use distinct colors for pie charts
            return [colors[i % len(colors)] for i in range(count)]
        else:
            # Use consistent color for bar charts
            return [colors[0]] * count
    
    def _generate_border_colors(self, count: int, chart_type: str) -> List[str]:
        """
        Generate border colors for charts
        """
        if chart_type in ['pie', 'doughnut']:
            return ['#FFFFFF'] * count
        else:
            return ['#1F2937'] * count
    
    def _get_chart_options(self, chart_type: str, chart_suggestion: Dict) -> Dict[str, Any]:
        """
        Get chart-specific options
        """
        base_options = {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {
                    "display": chart_type in ['pie', 'doughnut', 'line'],
                    "position": "top"
                },
                "tooltip": {
                    "enabled": True,
                    "backgroundColor": 'rgba(0, 0, 0, 0.8)',
                    "titleColor": '#FFFFFF',
                    "bodyColor": '#FFFFFF'
                }
            }
        }
        
        # Chart-specific options
        if chart_type in ['bar', 'line']:
            base_options["scales"] = {
                "y": {
                    "beginAtZero": True,
                    "grid": {"color": "#E5E7EB"},
                    "ticks": {"color": "#6B7280"}
                },
                "x": {
                    "grid": {"display": False},
                    "ticks": {"color": "#6B7280"}
                }
            }
        
        if chart_type == 'horizontalBar':
            base_options["indexAxis"] = 'y'
            base_options["scales"] = {
                "x": {
                    "beginAtZero": True,
                    "grid": {"color": "#E5E7EB"},
                    "ticks": {"color": "#6B7280"}
                },
                "y": {
                    "grid": {"display": False},
                    "ticks": {"color": "#6B7280"}
                }
            }
        
        if chart_type in ['pie', 'doughnut']:
            base_options["plugins"]["legend"]["position"] = "right"
            
        if chart_type == 'doughnut':
            base_options["cutout"] = "60%"
        
        return base_options
    
    def _generate_dataset_label(self, raw_data: List[Dict], field_name: str) -> str:
        """
        Generate an appropriate label for the dataset
        """
        # Convert field name to human readable
        formatted_name = self._format_header(field_name)
        
        # Add context based on data
        if "amount" in field_name.lower() or "price" in field_name.lower():
            return formatted_name
        elif "count" in field_name.lower() or "quantity" in field_name.lower():
            return formatted_name
        else:
            return formatted_name
    
    def _format_display_number(self, number: float) -> str:
        """
        Format numbers for display in insights and summaries
        """
        if number >= 1_000_000:
            return f"{number/1_000_000:.1f}M"
        elif number >= 1_000:
            return f"{number/1_000:.1f}K"
        elif number == int(number):
            return str(int(number))
        else:
            return f"{number:.2f}"
    
    def _initialize_number_formatter(self) -> Dict[str, Any]:
        """
        Initialize number formatting configurations
        """
        return {
            "currency_symbols": ["$", "€", "£", "¥"],
            "percentage_indicators": ["%", "percent", "rate"],
            "large_number_threshold": 1000
        }
    
    def _format_error_response(self, error_message: str, raw_data: List[Dict]) -> Dict[str, Any]:
        """
        Format an error response when something goes wrong
        """
        return {
            "success": False,
            "summary": "An error occurred while processing your request.",
            "chart_worthy": False,
            "chart_data": None,
            "chart_suggestion": None,
            "insights": [f"Error: {error_message}"],
            "recommendations": ["Please try rephrasing your question or contact support."],
            "raw_data_count": len(raw_data),
            "formatted_data": [],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "error": error_message,
                "processing_method": "error_fallback"
            }
        }