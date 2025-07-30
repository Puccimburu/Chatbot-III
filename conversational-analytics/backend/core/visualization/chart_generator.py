# core/visualization/chart_generator.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Generates complete Chart.js configurations optimized for web display
    """
    
    def __init__(self):
        self.color_palettes = self._initialize_color_palettes()
        self.chart_defaults = self._initialize_chart_defaults()
        
    def generate_chart_config(self, data: List[Dict], chart_type: str, 
                            title: str = "", context: Dict = None) -> Dict[str, Any]:
        """
        Generate a complete Chart.js configuration
        """
        try:
            if not data:
                return self._empty_chart_config(chart_type, title)
            
            # Process the data
            processed_data = self._process_data_for_chart(data, chart_type)
            
            # Generate the configuration
            config = {
                "type": chart_type,
                "data": self._generate_chart_data(processed_data, chart_type),
                "options": self._generate_chart_options(chart_type, title, context or {})
            }
            
            # Add chart-specific enhancements
            config = self._enhance_chart_config(config, processed_data, context or {})
            
            logger.info(f"📊 Generated {chart_type} chart config with {len(data)} data points")
            return config
            
        except Exception as e:
            logger.error(f"Error generating chart config: {e}")
            return self._fallback_chart_config(chart_type, title)
    
    def _process_data_for_chart(self, data: List[Dict], chart_type: str) -> Dict[str, Any]:
        """
        Process raw data into chart-ready format
        """
        if not data:
            return {"labels": [], "values": [], "metadata": {}}
        
        # Determine labels and values based on data structure
        sample_row = data[0]
        fields = list(sample_row.keys())
        
        if len(fields) >= 2:
            # Two or more fields - use first as label, second as value
            label_field = fields[0]
            value_field = fields[1]
            
            labels = []
            values = []
            
            for row in data:
                label = self._format_label(row.get(label_field, "Unknown"))
                value = self._extract_numeric_value(row.get(value_field, 0))
                labels.append(label)
                values.append(value)
                
        else:
            # Single field - create sequential labels
            field = fields[0]
            labels = [f"Item {i+1}" for i in range(len(data))]
            values = [self._extract_numeric_value(row.get(field, 0)) for row in data]
        
        # Sort data if needed
        if chart_type in ["bar", "horizontalBar"] and len(values) > 1:
            # Sort by value for bar charts (highest first)
            sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
            labels, values = zip(*sorted_pairs) if sorted_pairs else ([], [])
            labels, values = list(labels), list(values)
        
        return {
            "labels": labels,
            "values": values,
            "metadata": {
                "total_records": len(data),
                "label_field": label_field if 'label_field' in locals() else fields[0],
                "value_field": value_field if 'value_field' in locals() else fields[0],
                "chart_type": chart_type
            }
        }
    
    def _generate_chart_data(self, processed_data: Dict, chart_type: str) -> Dict[str, Any]:
        """
        Generate the data section of Chart.js config
        """
        labels = processed_data["labels"]
        values = processed_data["values"]
        
        # Select color palette
        colors = self._get_colors_for_chart(chart_type, len(values))
        
        chart_data = {
            "labels": labels,
            "datasets": [{
                "data": values,
                "backgroundColor": colors["background"],
                "borderColor": colors["border"],
                "borderWidth": colors.get("borderWidth", 1)
            }]
        }
        
        # Add dataset label for non-pie charts
        if chart_type not in ['pie', 'doughnut']:
            value_field = processed_data["metadata"].get("value_field", "Value")
            chart_data["datasets"][0]["label"] = self._format_dataset_label(value_field)
        
        # Chart-specific data modifications
        if chart_type == "line":
            chart_data["datasets"][0]["fill"] = False
            chart_data["datasets"][0]["tension"] = 0.1
            
        elif chart_type == "area":
            chart_data["type"] = "line"  # Area is a line chart with fill
            chart_data["datasets"][0]["fill"] = True
            chart_data["datasets"][0]["tension"] = 0.1
            
        elif chart_type in ["pie", "doughnut"]:
            # Pie charts need distinct colors for each slice
            chart_data["datasets"][0]["backgroundColor"] = colors["background"]
            
        return chart_data
    
    def _generate_chart_options(self, chart_type: str, title: str, context: Dict) -> Dict[str, Any]:
        """
        Generate the options section of Chart.js config
        """
        # Base options
        options = {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": self._get_legend_config(chart_type),
                "tooltip": self._get_tooltip_config(chart_type),
            }
        }
        
        # Add title if provided
        if title:
            options["plugins"]["title"] = {
                "display": True,
                "text": title,
                "font": {"size": 16, "weight": "bold"},
                "padding": 20,
                "color": "#1F2937"
            }
        
        # Add scales for charts that need them
        if chart_type in ["bar", "line", "area"]:
            options["scales"] = self._get_scales_config(chart_type, context)
            
        elif chart_type == "horizontalBar":
            options["indexAxis"] = "y"
            options["scales"] = self._get_horizontal_scales_config(context)
        
        # Chart-specific options
        if chart_type == "doughnut":
            options["cutout"] = "60%"
            
        elif chart_type == "pie":
            options["cutout"] = "0%"
            
        elif chart_type == "line":
            options["elements"] = {
                "point": {"radius": 4, "hoverRadius": 6},
                "line": {"borderWidth": 2}
            }
        
        return options
    
    def _enhance_chart_config(self, config: Dict, processed_data: Dict, context: Dict) -> Dict[str, Any]:
        """
        Add final enhancements to the chart configuration
        """
        chart_type = config["type"]
        data_count = len(processed_data["values"])
        
        # Adjust for large datasets
        if data_count > 20:
            if chart_type in ["pie", "doughnut"]:
                # Convert to bar chart for better readability
                config["type"] = "horizontalBar"
                config["options"]["indexAxis"] = "y"
                config["options"]["scales"] = self._get_horizontal_scales_config(context)
                
            elif chart_type in ["bar", "horizontalBar"]:
                # Hide x-axis labels for crowded charts
                if "scales" in config["options"]:
                    axis_key = "x" if chart_type == "bar" else "y"
                    config["options"]["scales"][axis_key]["ticks"]["display"] = False
        
        # Add animation
        config["options"]["animation"] = {
            "duration": 750,
            "easing": "easeInOutQuart"
        }
        
        # Add interaction options
        config["options"]["interaction"] = {
            "intersect": False,
            "mode": "index"
        }
        
        return config
    
    def _get_colors_for_chart(self, chart_type: str, count: int) -> Dict[str, Any]:
        """
        Get appropriate colors for the chart type
        """
        if chart_type in ["pie", "doughnut"]:
            # Use distinct colors for each slice
            palette = self.color_palettes["vibrant"]
            background = [palette[i % len(palette)] for i in range(count)]
            border = ["#FFFFFF"] * count
            border_width = 2
            
        else:
            # Use consistent color scheme for bar/line charts
            primary_color = self.color_palettes["primary"][0]
            background = [primary_color] * count
            border = [self._darken_color(primary_color)] * count
            border_width = 1
        
        return {
            "background": background,
            "border": border,
            "borderWidth": border_width
        }
    
    def _get_legend_config(self, chart_type: str) -> Dict[str, Any]:
        """
        Get legend configuration for the chart type
        """
        if chart_type in ["pie", "doughnut"]:
            return {
                "display": True,
                "position": "right",
                "labels": {
                    "usePointStyle": True,
                    "padding": 15,
                    "font": {"size": 12}
                }
            }
        else:
            return {
                "display": False  # Bar and line charts don't need legends for single datasets
            }
    
    def _get_tooltip_config(self, chart_type: str) -> Dict[str, Any]:
        """
        Get tooltip configuration
        """
        return {
            "enabled": True,
            "backgroundColor": "rgba(0, 0, 0, 0.8)",
            "titleColor": "#FFFFFF",
            "bodyColor": "#FFFFFF",
            "borderColor": "#374151",
            "borderWidth": 1,
            "cornerRadius": 6,
            "displayColors": chart_type in ["pie", "doughnut"],
            "callbacks": {
                "label": self._get_tooltip_label_callback(chart_type)
            }
        }
    
    def _get_scales_config(self, chart_type: str, context: Dict) -> Dict[str, Any]:
        """
        Get scales configuration for vertical charts
        """
        return {
            "y": {
                "beginAtZero": True,
                "grid": {
                    "color": "#E5E7EB",
                    "borderDash": [2, 2]
                },
                "ticks": {
                    "color": "#6B7280",
                    "font": {"size": 11},
                    "callback": "function(value) { return this.getLabelForValue(value); }"
                }
            },
            "x": {
                "grid": {"display": False},
                "ticks": {
                    "color": "#6B7280",
                    "font": {"size": 11},
                    "maxRotation": 45
                }
            }
        }
    
    def _get_horizontal_scales_config(self, context: Dict) -> Dict[str, Any]:
        """
        Get scales configuration for horizontal charts
        """
        return {
            "x": {
                "beginAtZero": True,
                "grid": {
                    "color": "#E5E7EB",
                    "borderDash": [2, 2]
                },
                "ticks": {
                    "color": "#6B7280",
                    "font": {"size": 11}
                }
            },
            "y": {
                "grid": {"display": False},
                "ticks": {
                    "color": "#6B7280",
                    "font": {"size": 11}
                }
            }
        }
    
    def _get_tooltip_label_callback(self, chart_type: str) -> str:
        """
        Get tooltip label callback function
        """
        if chart_type in ["pie", "doughnut"]:
            return """function(context) {
                let label = context.label || '';
                let value = context.parsed;
                let total = context.dataset.data.reduce((a, b) => a + b, 0);
                let percentage = ((value / total) * 100).toFixed(1);
                return label + ': ' + value.toLocaleString() + ' (' + percentage + '%)';
            }"""
        else:
            return """function(context) {
                let label = context.dataset.label || '';
                let value = context.parsed.y || context.parsed;
                return label + ': ' + value.toLocaleString();
            }"""
    
    def _format_label(self, label: Any) -> str:
        """
        Format labels for display
        """
        if label is None:
            return "Unknown"
        
        label_str = str(label)
        
        # Truncate very long labels
        if len(label_str) > 25:
            return label_str[:22] + "..."
        
        return label_str
    
    def _format_dataset_label(self, field_name: str) -> str:
        """
        Format dataset label from field name
        """
        # Convert snake_case to Title Case
        formatted = field_name.replace('_', ' ').replace('-', ' ')
        words = formatted.split()
        return ' '.join(word.capitalize() for word in words)
    
    def _extract_numeric_value(self, value: Any) -> float:
        """
        Extract numeric value from various data types
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Try to extract number from string
            try:
                # Remove common non-numeric characters
                cleaned = value.replace(',', '').replace('$', '').replace('%', '')
                return float(cleaned)
            except ValueError:
                return 0.0
        
        return 0.0
    
    def _darken_color(self, color: str) -> str:
        """
        Darken a hex color for borders
        """
        # Simple darkening by reducing each RGB component
        if color.startswith('#') and len(color) == 7:
            try:
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                
                # Darken by 20%
                r = max(0, int(r * 0.8))
                g = max(0, int(g * 0.8))
                b = max(0, int(b * 0.8))
                
                return f"#{r:02x}{g:02x}{b:02x}"
            except ValueError:
                pass
        
        return "#374151"  # Default dark gray
    
    def _empty_chart_config(self, chart_type: str, title: str) -> Dict[str, Any]:
        """
        Generate config for empty data
        """
        return {
            "type": chart_type,
            "data": {
                "labels": ["No Data"],
                "datasets": [{
                    "data": [0],
                    "backgroundColor": ["#E5E7EB"],
                    "borderColor": ["#9CA3AF"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {"display": False},
                    "title": {
                        "display": bool(title),
                        "text": title or "No Data Available"
                    }
                }
            }
        }
    
    def _fallback_chart_config(self, chart_type: str, title: str) -> Dict[str, Any]:
        """
        Generate fallback config when error occurs
        """
        return {
            "type": "bar",
            "data": {
                "labels": ["Error"],
                "datasets": [{
                    "data": [1],
                    "backgroundColor": ["#EF4444"],
                    "borderColor": ["#DC2626"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {"display": False},
                    "title": {
                        "display": True,
                        "text": "Chart Generation Error"
                    }
                }
            }
        }
    
    def _initialize_color_palettes(self) -> Dict[str, List[str]]:
        """
        Initialize color palettes for different chart types
        """
        return {
            "primary": ["#3B82F6", "#1D4ED8", "#1E40AF"],
            "vibrant": [
                "#3B82F6",  # Blue
                "#EF4444",  # Red
                "#10B981",  # Green
                "#F59E0B",  # Amber
                "#8B5CF6",  # Purple
                "#EC4899",  # Pink
                "#06B6D4",  # Cyan
                "#F97316",  # Orange
                "#84CC16",  # Lime
                "#6B7280"   # Gray
            ],
            "professional": [
                "#1F2937",  # Dark Gray
                "#374151",  # Gray
                "#4B5563",  # Medium Gray
                "#6B7280",  # Light Gray
                "#9CA3AF"   # Very Light Gray
            ]
        }
    
    def _initialize_chart_defaults(self) -> Dict[str, Any]:
        """
        Initialize default chart configurations
        """
        return {
            "font_family": "'Inter', 'Helvetica', 'Arial', sans-serif",
            "font_sizes": {
                "title": 16,
                "legend": 12,
                "ticks": 11,
                "tooltip": 12
            },
            "colors": {
                "text": "#1F2937",
                "grid": "#E5E7EB",
                "border": "#D1D5DB"
            },
            "animation_duration": 750
        }