# core/query_generation/prompt_builder.py

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Builds sophisticated prompts for Gemini AI to generate MongoDB queries and visualizations
    """
    
    def __init__(self):
        self.query_examples = self._initialize_query_examples()
        self.visualization_examples = self._initialize_visualization_examples()
        
    def build_query_generation_prompt(self, user_question: str, schema_info: Dict, 
                                    context: Optional[Dict] = None) -> str:
        """
        Build a comprehensive prompt for Stage 1: Query Generation
        """
        schema_context = self._format_schema_context(schema_info)
        examples = self._select_relevant_examples(user_question, schema_info)
        validation_rules = self._get_validation_rules()
        
        prompt = f"""You are an expert MongoDB query generator and data analyst. Convert natural language questions into MongoDB aggregation pipelines.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no markdown, no explanations, no extra text
2. Use exact field names from the provided schema
3. Create efficient aggregation pipelines with proper operators
4. Include validation fields for quality assurance
5. Always add $sort and $limit for performance optimization

{schema_context}

USER QUESTION: "{user_question}"

{examples}

{validation_rules}

RESPONSE FORMAT (JSON ONLY):
{{
  "collection": "target_collection_name",
  "pipeline": [
    {{"$match": {{"field": "value"}}}},
    {{"$group": {{"_id": "$field", "total": {{"$sum": "$amount"}}}}}},
    {{"$sort": {{"total": -1}}}},
    {{"$limit": 20}}
  ],
  "chart_hint": "bar|pie|line|doughnut|horizontalBar",
  "query_intent": "Brief description of what this query achieves",
  "expected_fields": ["field1", "field2", "field3"],
  "data_summary": "Description of what the results should contain",
  "confidence": 0.95
}}

JSON ONLY - NO OTHER TEXT:"""

        return prompt.strip()
    
    def build_visualization_prompt(self, user_question: str, raw_data: List[Dict], 
                                 query_context: Dict) -> str:
        """
        Build a comprehensive prompt for Stage 2: Visualization Generation
        """
        data_sample = self._format_data_sample(raw_data)
        chart_guidance = self._get_chart_type_guidance()
        viz_examples = self._get_visualization_examples()
        
        prompt = f"""You are a data visualization expert. Create Chart.js configurations and business insights from query results.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no markdown, no explanations
2. Choose optimal chart type based on data patterns
3. Generate business-focused insights and recommendations
4. Create complete Chart.js configurations
5. Provide actionable summary with specific numbers

USER QUESTION: "{user_question}"

QUERY CONTEXT:
{json.dumps(query_context, indent=2)}

DATA SAMPLE (showing {min(len(raw_data), 5)} of {len(raw_data)} total records):
{data_sample}

{chart_guidance}

{viz_examples}

RESPONSE FORMAT (JSON ONLY):
{{
  "chart_type": "bar|pie|line|doughnut|horizontalBar",
  "chart_config": {{
    "type": "bar",
    "data": {{
      "labels": ["Label1", "Label2"],
      "datasets": [{{
        "data": [100, 200],
        "backgroundColor": ["#3B82F6", "#EF4444"],
        "borderColor": ["#1D4ED8", "#DC2626"],
        "borderWidth": 1,
        "label": "Dataset Name"
      }}]
    }},
    "options": {{
      "responsive": true,
      "maintainAspectRatio": false,
      "plugins": {{
        "legend": {{"display": true}},
        "title": {{"display": true, "text": "Chart Title"}}
      }},
      "scales": {{
        "y": {{"beginAtZero": true}},
        "x": {{"display": true}}
      }}
    }}
  }},
  "summary": "Business-focused summary with specific numbers and key findings",
  "insights": [
    "Specific insight about the data pattern",
    "Comparison or trend observation",
    "Business implication of the results"
  ],
  "recommendations": [
    "Actionable business recommendation",
    "Strategic suggestion based on data",
    "Next steps for analysis"
  ],
  "chart_worthy": true,
  "confidence": 0.9
}}

JSON ONLY - NO OTHER TEXT:"""

        return prompt.strip()
    
    def build_schema_analysis_prompt(self, collections_info: Dict) -> str:
        """
        Build prompt for analyzing database schema for better query generation
        """
        prompt = f"""Analyze this database schema and provide insights for query generation.

SCHEMA INFORMATION:
{json.dumps(collections_info, indent=2)}

Analyze and return insights about:
1. Collection relationships and join patterns
2. Common query patterns this schema supports
3. Recommended aggregation approaches
4. Performance considerations
5. Business context from field names

RESPONSE FORMAT (JSON ONLY):
{{
  "collection_analysis": {{
    "primary_collections": ["most important collections for analytics"],
    "lookup_collections": ["reference/lookup tables"],
    "transaction_collections": ["fact tables with metrics"]
  }},
  "relationship_patterns": [
    {{"from": "collection1", "to": "collection2", "field": "foreign_key", "type": "one_to_many"}}
  ],
  "common_queries": [
    {{"pattern": "aggregation_pattern", "description": "what it analyzes", "collections": ["involved collections"]}}
  ],
  "performance_tips": [
    "indexing recommendations",
    "aggregation optimization suggestions"
  ]
}}

JSON ONLY:"""

        return prompt.strip()
    
    def _format_schema_context(self, schema_info: Dict) -> str:
        """
        Format schema information for the prompt
        """
        if not schema_info:
            return "SCHEMA: No schema information available"
        
        context = "DATABASE SCHEMA:\n"
        
        for collection_name, collection_info in schema_info.items():
            if isinstance(collection_info, dict):
                context += f"\nCOLLECTION: {collection_name}\n"
                
                if "description" in collection_info:
                    context += f"Description: {collection_info['description']}\n"
                
                if "fields" in collection_info:
                    context += f"Fields: {', '.join(collection_info['fields'])}\n"
                
                if "sample_data" in collection_info:
                    context += f"Sample values: {collection_info['sample_data']}\n"
                
                if "metrics" in collection_info:
                    context += f"Metrics: {', '.join(collection_info['metrics'])}\n"
                
                if "dimensions" in collection_info:
                    context += f"Dimensions: {', '.join(collection_info['dimensions'])}\n"
        
        return context
    
    def _format_data_sample(self, raw_data: List[Dict]) -> str:
        """
        Format data sample for visualization prompt
        """
        if not raw_data:
            return "[]"
        
        # Take first 5 records for context
        sample = raw_data[:5]
        
        try:
            return json.dumps(sample, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Error formatting data sample: {e}")
            return str(sample)
    
    def _select_relevant_examples(self, user_question: str, schema_info: Dict) -> str:
        """
        Select relevant query examples based on the user question and schema
        """
        question_lower = user_question.lower()
        relevant_examples = []
        
        # Select examples based on query intent
        if any(word in question_lower for word in ['top', 'best', 'highest', 'most']):
            relevant_examples.append(self.query_examples['ranking'])
        
        if any(word in question_lower for word in ['compare', 'vs', 'versus', 'between']):
            relevant_examples.append(self.query_examples['comparison'])
        
        if any(word in question_lower for word in ['trend', 'over time', 'monthly', 'yearly']):
            relevant_examples.append(self.query_examples['time_series'])
        
        if any(word in question_lower for word in ['total', 'sum', 'count', 'average']):
            relevant_examples.append(self.query_examples['aggregation'])
        
        # Default examples if none matched
        if not relevant_examples:
            relevant_examples = [
                self.query_examples['aggregation'],
                self.query_examples['ranking']
            ]
        
        examples_text = "QUERY EXAMPLES:\n\n"
        for i, example in enumerate(relevant_examples[:3]):  # Limit to 3 examples
            examples_text += f"Example {i+1}:\n{example}\n\n"
        
        return examples_text
    
    def _get_validation_rules(self) -> str:
        """
        Get validation rules for query generation
        """
        return """VALIDATION RULES:
1. Always use exact field names from schema (case-sensitive)
2. Include $match stage first when filtering is needed
3. Use $group for aggregations with proper _id field
4. Always add $sort for consistent ordering
5. Include $limit to prevent performance issues (max 1000 records)
6. Use $lookup for cross-collection queries when needed
7. Ensure aggregation operators match field types (numeric fields for $sum, $avg)
8. Set confidence based on schema field availability and query complexity"""
    
    def _get_chart_type_guidance(self) -> str:
        """
        Get guidance for chart type selection
        """
        return """CHART TYPE SELECTION GUIDE:

BAR CHART: Use for comparing categories, rankings, discrete values
- Good for: "top 10", "compare categories", "sales by region"
- Data pattern: Categorical x-axis, numeric y-axis
- Max recommended items: 20

PIE/DOUGHNUT: Use for showing parts of a whole, distributions
- Good for: "breakdown", "distribution", "market share"
- Data pattern: Categories with percentages/proportions
- Max recommended slices: 8

LINE CHART: Use for trends over time, continuous data
- Good for: "over time", "trend", "growth", "monthly/yearly"
- Data pattern: Time-based x-axis, numeric y-axis
- Min recommended points: 3

HORIZONTAL BAR: Use for long category names, large datasets
- Good for: Many categories, long labels, rankings
- Data pattern: Same as bar but better for readability
- Use when: >8 categories or long labels

CHART WORTHINESS CRITERIA:
- Single data point: NOT chart worthy
- All identical values: NOT chart worthy
- 2-50 data points with variation: Chart worthy
- Clear comparative/trend intent: Chart worthy"""
    
    def _get_visualization_examples(self) -> str:
        """
        Get visualization examples
        """
        return """VISUALIZATION EXAMPLES:

For "top selling products":
{
  "chart_type": "bar",
  "chart_config": {
    "type": "bar",
    "data": {
      "labels": ["MacBook Pro", "iPhone 15", "iPad Pro"],
      "datasets": [{
        "data": [15999, 12500, 8999],
        "backgroundColor": ["#3B82F6", "#3B82F6", "#3B82F6"],
        "label": "Revenue ($)"
      }]
    },
    "options": {
      "responsive": true,
      "scales": {"y": {"beginAtZero": true}}
    }
  },
  "summary": "MacBook Pro leads sales with $15,999 revenue, followed by iPhone 15 at $12,500",
  "insights": ["MacBook Pro dominates with 28% higher revenue than iPhone 15"],
  "chart_worthy": true
}

For "sales distribution by category":
{
  "chart_type": "pie",
  "chart_config": {
    "type": "pie",
    "data": {
      "labels": ["Laptops", "Smartphones", "Tablets"],
      "datasets": [{
        "data": [45, 35, 20],
        "backgroundColor": ["#3B82F6", "#EF4444", "#10B981"]
      }]
    }
  },
  "summary": "Laptops account for 45% of sales, smartphones 35%, tablets 20%",
  "insights": ["Laptops are the dominant category with nearly half of all sales"],
  "chart_worthy": true
}"""
    
    def _initialize_query_examples(self) -> Dict[str, str]:
        """
        Initialize query examples for different patterns
        """
        return {
            'aggregation': '''Question: "What is the total revenue by category?"
{
  "collection": "sales",
  "pipeline": [
    {"$group": {"_id": "$category", "total_revenue": {"$sum": "$total_amount"}}},
    {"$sort": {"total_revenue": -1}},
    {"$limit": 20}
  ],
  "chart_hint": "bar",
  "query_intent": "Aggregate total revenue grouped by product category",
  "expected_fields": ["_id", "total_revenue"],
  "data_summary": "Category names with their total revenue amounts",
  "confidence": 0.9
}''',
            
            'ranking': '''Question: "Show me the top 10 customers by total spending"
{
  "collection": "customers",
  "pipeline": [
    {"$match": {"total_spent": {"$gt": 0}}},
    {"$sort": {"total_spent": -1}},
    {"$limit": 10},
    {"$project": {"name": 1, "total_spent": 1, "_id": 0}}
  ],
  "chart_hint": "horizontalBar",
  "query_intent": "Find top 10 customers ranked by spending amount",
  "expected_fields": ["name", "total_spent"],
  "data_summary": "Customer names with their total spending amounts",
  "confidence": 0.95
}''',
            
            'comparison': '''Question: "Compare laptop sales vs smartphone sales"
{
  "collection": "sales",
  "pipeline": [
    {"$match": {"category": {"$in": ["Laptops", "Smartphones"]}}},
    {"$group": {"_id": "$category", "total_sales": {"$sum": "$total_amount"}, "quantity": {"$sum": "$quantity"}}},
    {"$sort": {"total_sales": -1}}
  ],
  "chart_hint": "bar",
  "query_intent": "Compare sales performance between laptop and smartphone categories",
  "expected_fields": ["_id", "total_sales", "quantity"],
  "data_summary": "Category comparison with sales amounts and quantities",
  "confidence": 0.9
}''',
            
            'time_series': '''Question: "Show monthly revenue trend for this year"
{
  "collection": "sales",
  "pipeline": [
    {"$match": {"date": {"$gte": "2024-01-01"}}},
    {"$group": {"_id": "$month", "monthly_revenue": {"$sum": "$total_amount"}}},
    {"$sort": {"_id": 1}},
    {"$limit": 12}
  ],
  "chart_hint": "line",
  "query_intent": "Show revenue trends over months for time series analysis",
  "expected_fields": ["_id", "monthly_revenue"],
  "data_summary": "Monthly revenue amounts showing temporal patterns",
  "confidence": 0.85
}'''
        }
    
    def _initialize_visualization_examples(self) -> Dict[str, str]:
        """
        Initialize visualization examples
        """
        return {
            'bar_chart': '''Data showing category comparisons should use bar charts with clear labels and consistent colors''',
            'pie_chart': '''Data showing distribution/breakdown should use pie charts when ≤8 categories''',
            'line_chart': '''Time-based data should use line charts to show trends and patterns''',
            'insights': '''Generate 3-5 specific insights focusing on: top performers, notable patterns, business implications'''
        }
    
    def build_fallback_query_prompt(self, user_question: str, schema_info: Dict) -> str:
        """
        Build a simpler prompt for fallback query generation
        """
        collections = list(schema_info.keys()) if schema_info else ["data"]
        
        prompt = f"""Generate a simple MongoDB query for: "{user_question}"

Available collections: {', '.join(collections)}

Return simple JSON:
{{
  "collection": "most_relevant_collection",
  "pipeline": [{{"$limit": 50}}],
  "chart_hint": "bar"
}}

JSON only:"""
        
        return prompt.strip()
    
    def validate_prompt_response(self, response: str, stage: str) -> Dict[str, Any]:
        """
        Validate the response from Gemini against expected format
        """
        try:
            parsed = json.loads(response)
            
            if stage == "query":
                required_fields = ["collection", "pipeline", "chart_hint"]
                missing = [field for field in required_fields if field not in parsed]
                if missing:
                    return {"valid": False, "error": f"Missing fields: {missing}"}
                
            elif stage == "visualization":
                required_fields = ["chart_type", "chart_config", "summary"]
                missing = [field for field in required_fields if field not in parsed]
                if missing:
                    return {"valid": False, "error": f"Missing fields: {missing}"}
            
            return {"valid": True, "data": parsed}
            
        except json.JSONDecodeError as e:
            return {"valid": False, "error": f"Invalid JSON: {str(e)}"}
    
    def enhance_prompt_with_context(self, base_prompt: str, context: Dict) -> str:
        """
        Enhance prompt with additional context like user preferences or conversation history
        """
        if not context:
            return base_prompt
        
        enhancements = []
        
        if "user_preferences" in context:
            prefs = context["user_preferences"]
            if "preferred_chart_types" in prefs:
                enhancements.append(f"User prefers: {', '.join(prefs['preferred_chart_types'])} charts")
        
        if "conversation_context" in context:
            conv = context["conversation_context"]
            if "recent_queries" in conv:
                enhancements.append(f"Recent queries: {', '.join(conv['recent_queries'][-3:])}")
        
        if enhancements:
            enhancement_text = "\nADDITIONAL CONTEXT:\n" + "\n".join(f"- {e}" for e in enhancements) + "\n"
            # Insert before the response format section
            format_index = base_prompt.find("RESPONSE FORMAT")
            if format_index > 0:
                return base_prompt[:format_index] + enhancement_text + base_prompt[format_index:]
        
        return base_prompt