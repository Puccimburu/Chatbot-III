"""
Fallback Query Generator - Rule-based query generation when AI fails
Intelligent pattern matching and template-based MongoDB query generation
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from models.schema_models import DatabaseSchema, CollectionSchema, FieldRole, FieldType

logger = logging.getLogger(__name__)


class FallbackQueryGenerator:
    """Rule-based query generator for when AI systems are unavailable"""
    
    def __init__(self):
        self.database_schema: Optional[DatabaseSchema] = None
        self.query_patterns = self._initialize_query_patterns()
        self.common_words = self._initialize_common_words()
        
        # Statistics
        self.total_generations = 0
        self.successful_generations = 0
        self.pattern_usage = {}
    
    def initialize(self, database_schema: DatabaseSchema):
        """Initialize with database schema"""
        self.database_schema = database_schema
        logger.info(f"🔧 Fallback generator initialized with {len(database_schema.collections)} collections")
    
    async def generate_query(
        self, 
        user_question: str, 
        database_schema: DatabaseSchema = None
    ) -> Dict[str, Any]:
        """
        Generate MongoDB query using rule-based pattern matching
        
        Args:
            user_question: Natural language question
            database_schema: Database schema (optional override)
            
        Returns:
            Dict containing query information or error
        """
        
        self.total_generations += 1
        
        try:
            # Use provided schema or default
            schema = database_schema or self.database_schema
            
            if not schema:
                return {
                    "success": False,
                    "error": "No database schema available"
                }
            
            logger.debug(f"🔍 Generating fallback query for: '{user_question}'")
            
            # Clean and analyze the question
            cleaned_question = self._clean_question(user_question)
            question_analysis = self._analyze_question(cleaned_question)
            
            # Find best matching collection
            target_collection = self._find_target_collection(question_analysis, schema)
            
            if not target_collection:
                return {
                    "success": False,
                    "error": "Could not determine target collection from question"
                }
            
            # Generate query based on pattern matching
            query_result = self._generate_query_for_collection(
                question_analysis, 
                target_collection,
                schema.collections[target_collection]
            )
            
            if query_result["success"]:
                self.successful_generations += 1
                pattern_used = query_result.get("pattern_used", "unknown")
                self.pattern_usage[pattern_used] = self.pattern_usage.get(pattern_used, 0) + 1
            
            return query_result
            
        except Exception as e:
            logger.error(f"❌ Fallback query generation failed: {str(e)}")
            return {
                "success": False,
                "error": f"Query generation failed: {str(e)}"
            }
    
    def _clean_question(self, question: str) -> str:
        """Clean and normalize the user question"""
        
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', ' ', question.lower())
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question to extract intent and components"""
        
        analysis = {
            "intent": "unknown",
            "aggregation_type": "group",
            "time_context": None,
            "comparison": False,
            "superlatives": [],
            "numbers": [],
            "keywords": [],
            "filters": []
        }
        
        words = question.split()
        analysis["keywords"] = words
        
        # Detect intent patterns
        if any(word in question for word in ["top", "best", "highest", "most", "maximum"]):
            analysis["intent"] = "ranking"
            analysis["aggregation_type"] = "group_sort_limit"
            
            # Extract numbers for limits
            numbers = re.findall(r'\b(\d+)\b', question)
            if numbers:
                analysis["numbers"] = [int(n) for n in numbers]
            
            # Find superlatives
            superlatives = ["top", "best", "highest", "most", "maximum", "largest"]
            analysis["superlatives"] = [word for word in superlatives if word in question]
        
        elif any(word in question for word in ["count", "how many", "number of", "total"]):
            analysis["intent"] = "count"
            analysis["aggregation_type"] = "count"
        
        elif any(word in question for word in ["average", "avg", "mean"]):
            analysis["intent"] = "average"
            analysis["aggregation_type"] = "average"
        
        elif any(word in question for word in ["sum", "total", "overall"]):
            analysis["intent"] = "sum"
            analysis["aggregation_type"] = "sum"
        
        elif any(word in question for word in ["compare", "vs", "versus", "difference", "between"]):
            analysis["intent"] = "comparison"
            analysis["comparison"] = True
            analysis["aggregation_type"] = "group"
        
        elif any(word in question for word in ["trend", "over time", "monthly", "daily", "yearly"]):
            analysis["intent"] = "trend"
            analysis["aggregation_type"] = "time_series"
        
        elif any(word in question for word in ["show", "list", "display", "get"]):
            analysis["intent"] = "list"
            analysis["aggregation_type"] = "find"
        
        # Detect time context
        time_indicators = {
            "today": 0,
            "yesterday": 1,
            "this week": 7,
            "last week": 14,
            "this month": 30,
            "last month": 60,
            "this year": 365,
            "last year": 730
        }
        
        for indicator, days_back in time_indicators.items():
            if indicator in question:
                analysis["time_context"] = {
                    "indicator": indicator,
                    "days_back": days_back,
                    "date_filter": datetime.utcnow() - timedelta(days=days_back)
                }
                break
        
        # Detect filters
        filter_words = ["where", "with", "having", "for", "in", "by", "of"]
        for word in filter_words:
            if word in question:
                # Simple filter detection - could be enhanced
                word_index = question.find(word)
                potential_filter = question[word_index:].split()[:3]
                analysis["filters"].append(" ".join(potential_filter))
        
        return analysis
    
    def _find_target_collection(
        self, 
        question_analysis: Dict[str, Any], 
        schema: DatabaseSchema
    ) -> Optional[str]:
        """Find the most likely target collection based on question analysis"""
        
        keywords = question_analysis["keywords"]
        
        # Score collections based on keyword matching
        collection_scores = {}
        
        for collection_name, collection_schema in schema.collections.items():
            score = 0
            
            # Direct name matching (highest score)
            collection_name_words = collection_name.lower().split('_')
            for keyword in keywords:
                if keyword in collection_name_words:
                    score += 10
                
                # Partial matching
                for name_word in collection_name_words:
                    if keyword in name_word or name_word in keyword:
                        score += 5
            
            # Field name matching
            for field_name in collection_schema.fields.keys():
                field_name_lower = field_name.lower()
                for keyword in keywords:
                    if keyword in field_name_lower:
                        score += 3
            
            # Analytics value bonus (prefer analytically valuable collections)
            score += collection_schema.analytics_value * 2
            
            # Intent-based scoring
            intent = question_analysis["intent"]
            if intent in ["ranking", "count", "average", "sum"] and collection_schema.metric_fields:
                score += 5
            
            if intent == "trend" and collection_schema.temporal_fields:
                score += 5
            
            if intent == "comparison" and collection_schema.dimension_fields:
                score += 3
            
            collection_scores[collection_name] = score
        
        # Return collection with highest score (if above threshold)
        if collection_scores:
            best_collection = max(collection_scores, key=collection_scores.get)
            best_score = collection_scores[best_collection]
            
            if best_score > 2:  # Minimum confidence threshold
                logger.debug(f"🎯 Selected collection: {best_collection} (score: {best_score})")
                return best_collection
        
        # Fallback: return collection with highest analytics value
        high_value_collections = schema.get_collection_by_value(min_analytics_value=0.5)
        if high_value_collections:
            fallback_collection = high_value_collections[0].name
            logger.debug(f"🔄 Fallback to high-value collection: {fallback_collection}")
            return fallback_collection
        
        return None
    
    def _generate_query_for_collection(
        self,
        question_analysis: Dict[str, Any],
        collection_name: str,
        collection_schema: CollectionSchema
    ) -> Dict[str, Any]:
        """Generate specific query for the target collection"""
        
        intent = question_analysis["intent"]
        
        try:
            # Route to specific query generators based on intent
            if intent == "ranking":
                return self._generate_ranking_query(question_analysis, collection_name, collection_schema)
            elif intent == "count":
                return self._generate_count_query(question_analysis, collection_name, collection_schema)
            elif intent == "average":
                return self._generate_average_query(question_analysis, collection_name, collection_schema)
            elif intent == "sum":
                return self._generate_sum_query(question_analysis, collection_name, collection_schema)
            elif intent == "comparison":
                return self._generate_comparison_query(question_analysis, collection_name, collection_schema)
            elif intent == "trend":
                return self._generate_trend_query(question_analysis, collection_name, collection_schema)
            elif intent == "list":
                return self._generate_list_query(question_analysis, collection_name, collection_schema)
            else:
                return self._generate_default_query(question_analysis, collection_name, collection_schema)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Query generation failed for intent '{intent}': {str(e)}"
            }
    
    def _generate_ranking_query(
        self,
        question_analysis: Dict[str, Any],
        collection_name: str,
        collection_schema: CollectionSchema
    ) -> Dict[str, Any]:
        """Generate ranking/top-N queries"""
        
        # Find the best metric field to rank by
        metric_field = self._find_best_metric_field(question_analysis["keywords"], collection_schema)
        
        if not metric_field:
            return {
                "success": False,
                "error": "No suitable metric field found for ranking"
            }
        
        # Find grouping field (dimension)
        group_field = self._find_best_dimension_field(question_analysis["keywords"], collection_schema)
        
        if not group_field:
            # If no dimension field, just sort by metric
            pipeline = [
                {"$sort": {metric_field: -1}},
                {"$limit": self._extract_limit(question_analysis)}
            ]
        else:
            # Group by dimension and aggregate metric
            pipeline = [
                {"$group": {
                    "_id": f"${group_field}",
                    "total": {"$sum": f"${metric_field}"}
                }},
                {"$sort": {"total": -1}},
                {"$limit": self._extract_limit(question_analysis)}
            ]
        
        # Add time filter if present
        pipeline = self._add_time_filter(pipeline, question_analysis, collection_schema)
        
        return {
            "success": True,
            "collection": collection_name,
            "pipeline": pipeline,
            "query_intent": f"Find top {group_field or 'records'} by {metric_field}",
            "expected_result_type": "aggregation",
            "confidence": 0.8,
            "pattern_used": "ranking"
        }
    
    def _generate_count_query(
        self,
        question_analysis: Dict[str, Any],
        collection_name: str,
        collection_schema: CollectionSchema
    ) -> Dict[str, Any]:
        """Generate count queries"""
        
        # Simple count or count by group
        group_field = self._find_best_dimension_field(question_analysis["keywords"], collection_schema)
        
        if group_field:
            pipeline = [
                {"$group": {
                    "_id": f"${group_field}",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
        else:
            pipeline = [
                {"$count": "total_count"}
            ]
        
        # Add time filter if present
        pipeline = self._add_time_filter(pipeline, question_analysis, collection_schema)
        
        return {
            "success": True,
            "collection": collection_name,
            "pipeline": pipeline,
            "query_intent": f"Count documents" + (f" by {group_field}" if group_field else ""),
            "expected_result_type": "single_value" if not group_field else "aggregation",
            "confidence": 0.9,
            "pattern_used": "count"
        }
    
    def _generate_average_query(
        self,
        question_analysis: Dict[str, Any],
        collection_name: str,
        collection_schema: CollectionSchema
    ) -> Dict[str, Any]:
        """Generate average/mean queries"""
        
        metric_field = self._find_best_metric_field(question_analysis["keywords"], collection_schema)
        
        if not metric_field:
            return {
                "success": False,
                "error": "No suitable numeric field found for average calculation"
            }
        
        group_field = self._find_best_dimension_field(question_analysis["keywords"], collection_schema)
        
        if group_field:
            pipeline = [
                {"$group": {
                    "_id": f"${group_field}",
                    "average": {"$avg": f"${metric_field}"}
                }},
                {"$sort": {"average": -1}}
            ]
        else:
            pipeline = [
                {"$group": {
                    "_id": None,
                    "average": {"$avg": f"${metric_field}"}
                }}
            ]
        
        pipeline = self._add_time_filter(pipeline, question_analysis, collection_schema)
        
        return {
            "success": True,
            "collection": collection_name,
            "pipeline": pipeline,
            "query_intent": f"Calculate average {metric_field}" + (f" by {group_field}" if group_field else ""),
            "expected_result_type": "single_value" if not group_field else "aggregation",
            "confidence": 0.85,
            "pattern_used": "average"
        }
    
    def _generate_sum_query(
        self,
        question_analysis: Dict[str, Any],
        collection_name: str,
        collection_schema: CollectionSchema
    ) -> Dict[str, Any]:
        """Generate sum/total queries"""
        
        metric_field = self._find_best_metric_field(question_analysis["keywords"], collection_schema)
        
        if not metric_field:
            return {
                "success": False,
                "error": "No suitable numeric field found for sum calculation"
            }
        
        group_field = self._find_best_dimension_field(question_analysis["keywords"], collection_schema)
        
        if group_field:
            pipeline = [
                {"$group": {
                    "_id": f"${group_field}",
                    "total": {"$sum": f"${metric_field}"}
                }},
                {"$sort": {"total": -1}}
            ]
        else:
            pipeline = [
                {"$group": {
                    "_id": None,
                    "total": {"$sum": f"${metric_field}"}
                }}
            ]
        
        pipeline = self._add_time_filter(pipeline, question_analysis, collection_schema)
        
        return {
            "success": True,
            "collection": collection_name,
            "pipeline": pipeline,
            "query_intent": f"Calculate total {metric_field}" + (f" by {group_field}" if group_field else ""),
            "expected_result_type": "single_value" if not group_field else "aggregation",
            "confidence": 0.85,
            "pattern_used": "sum"
        }
    
    def _generate_trend_query(
        self,
        question_analysis: Dict[str, Any],
        collection_name: str,
        collection_schema: CollectionSchema
    ) -> Dict[str, Any]:
        """Generate time-based trend queries"""
        
        temporal_field = self._find_best_temporal_field(collection_schema)
        
        if not temporal_field:
            return {
                "success": False,
                "error": "No date/time field found for trend analysis"
            }
        
        metric_field = self._find_best_metric_field(question_analysis["keywords"], collection_schema)
        
        # Determine time grouping granularity
        granularity = self._determine_time_granularity(question_analysis["keywords"])
        
        # Build date grouping expression
        if granularity == "daily":
            date_group = {"$dateToString": {"format": "%Y-%m-%d", "date": f"${temporal_field}"}}
        elif granularity == "monthly":
            date_group = {"$dateToString": {"format": "%Y-%m", "date": f"${temporal_field}"}}
        elif granularity == "yearly":
            date_group = {"$dateToString": {"format": "%Y", "date": f"${temporal_field}"}}
        else:
            date_group = {"$dateToString": {"format": "%Y-%m-%d", "date": f"${temporal_field}"}}
        
        if metric_field:
            pipeline = [
                {"$group": {
                    "_id": date_group,
                    "value": {"$sum": f"${metric_field}"},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
        else:
            pipeline = [
                {"$group": {
                    "_id": date_group,
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
        
        pipeline = self._add_time_filter(pipeline, question_analysis, collection_schema)
        
        return {
            "success": True,
            "collection": collection_name,
            "pipeline": pipeline,
            "query_intent": f"Show {granularity} trend" + (f" of {metric_field}" if metric_field else ""),
            "expected_result_type": "time_series",
            "confidence": 0.8,
            "pattern_used": "trend"
        }
    
    def _generate_comparison_query(
        self,
        question_analysis: Dict[str, Any],
        collection_name: str,
        collection_schema: CollectionSchema
    ) -> Dict[str, Any]:
        """Generate comparison queries"""
        
        dimension_field = self._find_best_dimension_field(question_analysis["keywords"], collection_schema)
        metric_field = self._find_best_metric_field(question_analysis["keywords"], collection_schema)
        
        if not dimension_field:
            return {
                "success": False,
                "error": "No suitable dimension field found for comparison"
            }
        
        if metric_field:
            pipeline = [
                {"$group": {
                    "_id": f"${dimension_field}",
                    "value": {"$sum": f"${metric_field}"},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"value": -1}}
            ]
        else:
            pipeline = [
                {"$group": {
                    "_id": f"${dimension_field}",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
        
        pipeline = self._add_time_filter(pipeline, question_analysis, collection_schema)
        
        return {
            "success": True,
            "collection": collection_name,
            "pipeline": pipeline,
            "query_intent": f"Compare by {dimension_field}" + (f" using {metric_field}" if metric_field else ""),
            "expected_result_type": "aggregation",
            "confidence": 0.75,
            "pattern_used": "comparison"
        }
    
    def _generate_list_query(
        self,
        question_analysis: Dict[str, Any],
        collection_name: str,
        collection_schema: CollectionSchema
    ) -> Dict[str, Any]:
        """Generate simple list/find queries"""
        
        pipeline = []
        
        # Add basic projection to limit fields
        important_fields = {}
        
        # Include dimension and metric fields
        for field_name in collection_schema.dimension_fields[:5]:
            important_fields[field_name] = 1
        
        for field_name in collection_schema.metric_fields[:3]:
            important_fields[field_name] = 1
        
        if important_fields:
            pipeline.append({"$project": important_fields})
        
        # Add limit
        pipeline.append({"$limit": self._extract_limit(question_analysis, default=20)})
        
        # Add time filter if present
        pipeline = self._add_time_filter(pipeline, question_analysis, collection_schema)
        
        return {
            "success": True,
            "collection": collection_name,
            "pipeline": pipeline,
            "query_intent": "List records with key fields",
            "expected_result_type": "list",
            "confidence": 0.7,
            "pattern_used": "list"
        }
    
    def _generate_default_query(
        self,
        question_analysis: Dict[str, Any],
        collection_name: str,
        collection_schema: CollectionSchema
    ) -> Dict[str, Any]:
        """Generate default query when intent is unclear"""
        
        # Simple grouping by best dimension field
        dimension_field = self._find_best_dimension_field(question_analysis["keywords"], collection_schema)
        
        if dimension_field:
            pipeline = [
                {"$group": {
                    "_id": f"${dimension_field}",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
        else:
            # Just return sample records
            pipeline = [
                {"$limit": 10}
            ]
        
        return {
            "success": True,
            "collection": collection_name,
            "pipeline": pipeline,
            "query_intent": "General analysis of data",
            "expected_result_type": "aggregation" if dimension_field else "list",
            "confidence": 0.5,
            "pattern_used": "default"
        }
    
    def _find_best_metric_field(
        self, 
        keywords: List[str], 
        collection_schema: CollectionSchema
    ) -> Optional[str]:
        """Find the best numeric field for aggregation"""
        
        metric_fields = collection_schema.metric_fields
        
        if not metric_fields:
            return None
        
        # Score fields based on keyword matching
        field_scores = {}
        
        for field in metric_fields:
            score = 0
            field_lower = field.lower()
            
            # Exact keyword match
            for keyword in keywords:
                if keyword in field_lower:
                    score += 10
                
                # Partial match
                if any(part in field_lower for part in keyword.split('_')):
                    score += 5
            
            # Common metric field patterns
            if any(pattern in field_lower for pattern in ['amount', 'total', 'sum', 'value', 'price', 'cost', 'revenue']):
                score += 5
            
            field_scores[field] = score
        
        if field_scores and max(field_scores.values()) > 0:
            return max(field_scores, key=field_scores.get)
        
        # Return first metric field as fallback
        return metric_fields[0]
    
    def _find_best_dimension_field(
        self, 
        keywords: List[str], 
        collection_schema: CollectionSchema
    ) -> Optional[str]:
        """Find the best categorical field for grouping"""
        
        dimension_fields = collection_schema.dimension_fields
        
        if not dimension_fields:
            return None
        
        # Score fields based on keyword matching
        field_scores = {}
        
        for field in dimension_fields:
            score = 0
            field_lower = field.lower()
            
            # Exact keyword match
            for keyword in keywords:
                if keyword in field_lower:
                    score += 10
                
                # Partial match
                if any(part in field_lower for part in keyword.split('_')):
                    score += 5
            
            # Common dimension field patterns
            if any(pattern in field_lower for pattern in ['name', 'type', 'category', 'status', 'region', 'city']):
                score += 3
            
            field_scores[field] = score
        
        if field_scores and max(field_scores.values()) > 0:
            return max(field_scores, key=field_scores.get)
        
        # Return first dimension field as fallback
        return dimension_fields[0]
    
    def _find_best_temporal_field(
        self, 
        collection_schema: CollectionSchema
    ) -> Optional[str]:
        """Find the best date/time field for temporal analysis"""
        
        temporal_fields = collection_schema.temporal_fields
        
        if not temporal_fields:
            return None
        
        # Prefer certain field names
        preferred_names = ['created_at', 'date', 'timestamp', 'order_date', 'updated_at']
        
        for preferred in preferred_names:
            for field in temporal_fields:
                if preferred in field.lower():
                    return field
        
        # Return first temporal field
        return temporal_fields[0]
    
    def _extract_limit(self, question_analysis: Dict[str, Any], default: int = 10) -> int:
        """Extract limit from question analysis"""
        
        numbers = question_analysis.get("numbers", [])
        
        if numbers:
            # Use first reasonable number as limit
            for num in numbers:
                if 1 <= num <= 100:
                    return num
        
        return default
    
    def _determine_time_granularity(self, keywords: List[str]) -> str:
        """Determine time granularity from keywords"""
        
        if any(word in keywords for word in ['daily', 'day', 'days']):
            return 'daily'
        elif any(word in keywords for word in ['monthly', 'month', 'months']):
            return 'monthly'
        elif any(word in keywords for word in ['yearly', 'year', 'years', 'annual']):
            return 'yearly'
        elif any(word in keywords for word in ['weekly', 'week', 'weeks']):
            return 'weekly'
        else:
            return 'daily'  # Default
    
    def _add_time_filter(
        self, 
        pipeline: List[Dict[str, Any]], 
        question_analysis: Dict[str, Any],
        collection_schema: CollectionSchema
    ) -> List[Dict[str, Any]]:
        """Add time-based filter to pipeline if time context exists"""
        
        time_context = question_analysis.get("time_context")
        
        if not time_context:
            return pipeline
        
        temporal_field = self._find_best_temporal_field(collection_schema)
        
        if not temporal_field:
            return pipeline
        
        # Create date filter
        date_filter = {
            temporal_field: {
                "$gte": time_context["date_filter"]
            }
        }
        
        # Insert match stage at the beginning
        match_stage = {"$match": date_filter}
        
        # Insert at beginning, after any existing match stages
        insert_index = 0
        for i, stage in enumerate(pipeline):
            if "$match" in stage:
                insert_index = i + 1
            else:
                break
        
        pipeline.insert(insert_index, match_stage)
        
        return pipeline
    
    def _initialize_query_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common query patterns"""
        
        return {
            "top_n": {
                "keywords": ["top", "best", "highest", "most"],
                "template": "ranking",
                "requires": ["metric_field", "dimension_field"]
            },
            "count": {
                "keywords": ["count", "how many", "number of"],
                "template": "count",
                "requires": []
            },
            "average": {
                "keywords": ["average", "avg", "mean"],
                "template": "average",
                "requires": ["metric_field"]
            },
            "total": {
                "keywords": ["total", "sum", "overall"],
                "template": "sum",
                "requires": ["metric_field"]
            },
            "comparison": {
                "keywords": ["compare", "vs", "versus", "between"],
                "template": "comparison",
                "requires": ["dimension_field"]
            },
            "trend": {
                "keywords": ["trend", "over time", "monthly", "daily"],
                "template": "trend",
                "requires": ["temporal_field"]
            }
        }
    
    def _initialize_common_words(self) -> Dict[str, List[str]]:
        """Initialize common word mappings"""
        
        return {
            "metrics": ["sales", "revenue", "profit", "amount", "total", "value", "price", "cost"],
            "dimensions": ["product", "customer", "category", "region", "type", "status", "name"],
            "temporal": ["date", "time", "month", "year", "day", "created", "updated"],
            "actions": ["show", "get", "find", "list", "display", "analyze"]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback generator statistics"""
        
        success_rate = 0
        if self.total_generations > 0:
            success_rate = (self.successful_generations / self.total_generations) * 100
        
        return {
            "total_generations": self.total_generations,
            "successful_generations": self.successful_generations,
            "success_rate_percent": round(success_rate, 2),
            "pattern_usage": self.pattern_usage,
            "most_used_pattern": max(self.pattern_usage, key=self.pattern_usage.get) if self.pattern_usage else None
        }