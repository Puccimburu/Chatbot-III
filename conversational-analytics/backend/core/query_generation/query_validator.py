# core/query_generation/query_validator.py

import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import re

logger = logging.getLogger(__name__)

class QueryValidator:
    """
    Validates MongoDB aggregation pipelines for safety, correctness, and performance
    """
    
    def __init__(self):
        self.allowed_operators = self._initialize_allowed_operators()
        self.dangerous_operators = self._initialize_dangerous_operators()
        self.max_pipeline_stages = 10
        self.max_limit_value = 1000
        
    def validate_query(self, query_data: Dict, schema_info: Dict) -> Dict[str, Any]:
        """
        Comprehensive validation of a MongoDB query
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "safety_score": 100,
            "performance_score": 100
        }
        
        try:
            # Basic structure validation
            self._validate_structure(query_data, validation_result)
            
            # Collection validation
            self._validate_collection(query_data, schema_info, validation_result)
            
            # Pipeline validation
            self._validate_pipeline(query_data.get("pipeline", []), schema_info, validation_result)
            
            # Performance validation
            self._validate_performance(query_data, validation_result)
            
            # Security validation
            self._validate_security(query_data, validation_result)
            
            # Field validation
            self._validate_fields(query_data, schema_info, validation_result)
            
            # Calculate final scores
            validation_result["safety_score"] = max(0, validation_result["safety_score"])
            validation_result["performance_score"] = max(0, validation_result["performance_score"])
            
            # Determine if query is valid
            validation_result["valid"] = (
                len(validation_result["errors"]) == 0 and
                validation_result["safety_score"] >= 70 and
                validation_result["performance_score"] >= 50
            )
            
            logger.info(f"🔍 Query validation: valid={validation_result['valid']}, "
                       f"safety={validation_result['safety_score']}, "
                       f"performance={validation_result['performance_score']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during query validation: {e}")
            validation_result.update({
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "safety_score": 0,
                "performance_score": 0
            })
            return validation_result
    
    def _validate_structure(self, query_data: Dict, result: Dict) -> None:
        """
        Validate basic query structure
        """
        required_fields = ["collection", "pipeline"]
        
        for field in required_fields:
            if field not in query_data:
                result["errors"].append(f"Missing required field: {field}")
                result["safety_score"] -= 30
        
        # Validate pipeline is a list
        if "pipeline" in query_data and not isinstance(query_data["pipeline"], list):
            result["errors"].append("Pipeline must be a list of stages")
            result["safety_score"] -= 20
        
        # Check pipeline length
        pipeline = query_data.get("pipeline", [])
        if len(pipeline) > self.max_pipeline_stages:
            result["warnings"].append(f"Pipeline has {len(pipeline)} stages (max recommended: {self.max_pipeline_stages})")
            result["performance_score"] -= 10
    
    def _validate_collection(self, query_data: Dict, schema_info: Dict, result: Dict) -> None:
        """
        Validate collection exists and is accessible
        """
        collection = query_data.get("collection")
        
        if not collection:
            result["errors"].append("Collection name is required")
            return
        
        if not isinstance(collection, str):
            result["errors"].append("Collection name must be a string")
            return
        
        # Check if collection exists in schema
        if schema_info and collection not in schema_info:
            available_collections = list(schema_info.keys())
            result["warnings"].append(f"Collection '{collection}' not found in schema. Available: {available_collections}")
            result["performance_score"] -= 15
    
    def _validate_pipeline(self, pipeline: List[Dict], schema_info: Dict, result: Dict) -> None:
        """
        Validate each stage in the aggregation pipeline
        """
        if not pipeline:
            result["warnings"].append("Empty pipeline - will return all documents")
            return
        
        has_limit = False
        has_sort = False
        
        for i, stage in enumerate(pipeline):
            if not isinstance(stage, dict):
                result["errors"].append(f"Stage {i} must be a dictionary")
                continue
            
            if len(stage) != 1:
                result["errors"].append(f"Stage {i} must have exactly one operator")
                continue
            
            operator = list(stage.keys())[0]
            stage_data = stage[operator]
            
            # Validate operator
            self._validate_operator(operator, stage_data, i, result)
            
            # Track important stages
            if operator == "$limit":
                has_limit = True
            elif operator == "$sort":
                has_sort = True
        
        # Performance recommendations
        if not has_limit:
            result["suggestions"].append("Consider adding $limit stage to prevent large result sets")
            result["performance_score"] -= 10
        
        if not has_sort and len(pipeline) > 1:
            result["suggestions"].append("Consider adding $sort stage for consistent ordering")
            result["performance_score"] -= 5
    
    def _validate_operator(self, operator: str, stage_data: Any, stage_index: int, result: Dict) -> None:
        """
        Validate a specific aggregation operator
        """
        # Check if operator is allowed
        if operator not in self.allowed_operators:
            if operator in self.dangerous_operators:
                result["errors"].append(f"Dangerous operator '{operator}' not allowed")
                result["safety_score"] -= 50
            else:
                result["warnings"].append(f"Unknown operator '{operator}' at stage {stage_index}")
                result["safety_score"] -= 10
        
        # Operator-specific validation
        if operator == "$match":
            self._validate_match_stage(stage_data, result)
        elif operator == "$group":
            self._validate_group_stage(stage_data, result)
        elif operator == "$sort":
            self._validate_sort_stage(stage_data, result)
        elif operator == "$limit":
            self._validate_limit_stage(stage_data, result)
        elif operator == "$project":
            self._validate_project_stage(stage_data, result)
        elif operator == "$lookup":
            self._validate_lookup_stage(stage_data, result)
    
    def _validate_match_stage(self, stage_data: Dict, result: Dict) -> None:
        """
        Validate $match stage
        """
        if not isinstance(stage_data, dict):
            result["errors"].append("$match stage must be a dictionary")
            return
        
        # Check for potentially expensive operations
        if self._contains_regex(stage_data):
            result["warnings"].append("$match contains regex operations which may be slow")
            result["performance_score"] -= 5
        
        # Check for full collection scans
        if not stage_data:
            result["warnings"].append("Empty $match stage will scan entire collection")
            result["performance_score"] -= 10
    
    def _validate_group_stage(self, stage_data: Dict, result: Dict) -> None:
        """
        Validate $group stage
        """
        if not isinstance(stage_data, dict):
            result["errors"].append("$group stage must be a dictionary")
            return
        
        if "_id" not in stage_data:
            result["errors"].append("$group stage must have _id field")
            return
        
        # Validate accumulator operators
        for field, operation in stage_data.items():
            if field != "_id" and isinstance(operation, dict):
                for op in operation.keys():
                    if op not in ["$sum", "$avg", "$max", "$min", "$count", "$push", "$addToSet"]:
                        result["warnings"].append(f"Unknown accumulator operator: {op}")
    
    def _validate_sort_stage(self, stage_data: Dict, result: Dict) -> None:
        """
        Validate $sort stage
        """
        if not isinstance(stage_data, dict):
            result["errors"].append("$sort stage must be a dictionary")
            return
        
        for field, direction in stage_data.items():
            if direction not in [1, -1, "asc", "desc"]:
                result["errors"].append(f"Invalid sort direction for field '{field}': {direction}")
        
        # Performance warning for complex sorts
        if len(stage_data) > 3:
            result["warnings"].append("Sorting by many fields may impact performance")
            result["performance_score"] -= 5
    
    def _validate_limit_stage(self, stage_data: Any, result: Dict) -> None:
        """
        Validate $limit stage
        """
        if not isinstance(stage_data, int):
            result["errors"].append("$limit value must be an integer")
            return
        
        if stage_data <= 0:
            result["errors"].append("$limit value must be positive")
            return
        
        if stage_data > self.max_limit_value:
            result["warnings"].append(f"$limit value {stage_data} is very high (max recommended: {self.max_limit_value})")
            result["performance_score"] -= 15
    
    def _validate_project_stage(self, stage_data: Dict, result: Dict) -> None:
        """
        Validate $project stage
        """
        if not isinstance(stage_data, dict):
            result["errors"].append("$project stage must be a dictionary")
            return
        
        # Check for mix of inclusion and exclusion (except _id)
        includes = []
        excludes = []
        
        for field, value in stage_data.items():
            if field == "_id":
                continue
                
            if value in [1, True]:
                includes.append(field)
            elif value in [0, False]:
                excludes.append(field)
        
        if includes and excludes:
            result["warnings"].append("$project mixes inclusion and exclusion (may cause errors)")
    
    def _validate_lookup_stage(self, stage_data: Dict, result: Dict) -> None:
        """
        Validate $lookup stage
        """
        required_fields = ["from", "localField", "foreignField", "as"]
        
        for field in required_fields:
            if field not in stage_data:
                result["errors"].append(f"$lookup missing required field: {field}")
        
        result["warnings"].append("$lookup operations can be expensive - consider indexing join fields")
        result["performance_score"] -= 10
    
    def _validate_performance(self, query_data: Dict, result: Dict) -> None:
        """
        Validate query performance characteristics
        """
        pipeline = query_data.get("pipeline", [])
        
        # Check for early filtering
        has_early_match = False
        if pipeline and list(pipeline[0].keys())[0] == "$match":
            has_early_match = True
        
        if not has_early_match:
            result["suggestions"].append("Consider adding $match stage early in pipeline to filter data")
            result["performance_score"] -= 10
        
        # Check for unnecessary stages
        stage_count = len(pipeline)
        if stage_count > 6:
            result["warnings"].append(f"Complex pipeline with {stage_count} stages may be slow")
            result["performance_score"] -= 5
    
    def _validate_security(self, query_data: Dict, result: Dict) -> None:
        """
        Validate query for security issues
        """
        query_str = json.dumps(query_data)
        
        # Check for injection patterns
        dangerous_patterns = [
            r'\$where',
            r'eval\(',
            r'function\s*\(',
            r'javascript:',
            r'\$ne.*null'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query_str, re.IGNORECASE):
                result["errors"].append(f"Potential security risk: {pattern}")
                result["safety_score"] -= 40
    
    def _validate_fields(self, query_data: Dict, schema_info: Dict, result: Dict) -> None:
        """
        Validate field names against schema
        """
        if not schema_info:
            return
        
        collection = query_data.get("collection")
        if collection not in schema_info:
            return
        
        collection_schema = schema_info[collection]
        available_fields = collection_schema.get("fields", [])
        
        if not available_fields:
            return
        
        # Extract field names from pipeline
        used_fields = self._extract_field_names(query_data.get("pipeline", []))
        
        # Check for non-existent fields
        for field in used_fields:
            if field not in available_fields and not field.startswith("$"):
                result["warnings"].append(f"Field '{field}' not found in {collection} schema")
                result["performance_score"] -= 5
    
    def _extract_field_names(self, pipeline: List[Dict]) -> List[str]:
        """
        Extract field names referenced in the pipeline
        """
        fields = []
        
        def extract_from_dict(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.startswith("$"):
                        continue
                    
                    full_key = f"{prefix}.{key}" if prefix else key
                    fields.append(full_key)
                    
                    if isinstance(value, (dict, list)):
                        extract_from_dict(value, full_key)
            elif isinstance(obj, list):
                for item in obj:
                    extract_from_dict(item, prefix)
        
        for stage in pipeline:
            extract_from_dict(stage)
        
        return list(set(fields))  # Remove duplicates
    
    def _contains_regex(self, obj: Any) -> bool:
        """
        Check if object contains regex operations
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "$regex" or (isinstance(value, dict) and "$regex" in value):
                    return True
                if self._contains_regex(value):
                    return True
        elif isinstance(obj, list):
            return any(self._contains_regex(item) for item in obj)
        
        return False
    
    def _initialize_allowed_operators(self) -> set:
        """
        Initialize set of allowed MongoDB operators
        """
        return {
            # Pipeline stages
            "$match", "$group", "$sort", "$limit", "$skip", "$project", 
            "$unwind", "$lookup", "$addFields", "$replaceRoot", "$count",
            "$sample", "$sortByCount", "$facet", "$bucket", "$bucketAuto",
            
            # Query operators
            "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin",
            "$and", "$or", "$not", "$nor", "$exists", "$type", "$mod",
            "$regex", "$text", "$size", "$all", "$elemMatch",
            
            # Accumulator operators
            "$sum", "$avg", "$max", "$min", "$first", "$last", "$push",
            "$addToSet", "$stdDevPop", "$stdDevSamp", "$count",
            
            # Expression operators
            "$add", "$subtract", "$multiply", "$divide", "$abs", "$ceil",
            "$floor", "$round", "$sqrt", "$pow", "$log", "$ln", "$log10",
            "$exp", "$trunc", "$concat", "$substr", "$toLower", "$toUpper",
            "$dateToString", "$dayOfYear", "$dayOfMonth", "$dayOfWeek",
            "$year", "$month", "$week", "$hour", "$minute", "$second",
            "$millisecond", "$dateFromString", "$cond", "$ifNull",
            "$switch", "$arrayElemAt", "$concatArrays", "$filter", "$map",
            "$reduce", "$zip", "$isArray", "$size", "$slice"
        }
    
    def _initialize_dangerous_operators(self) -> set:
        """
        Initialize set of dangerous operators that should be blocked
        """
        return {
            "$where",  # JavaScript execution
            "$function",  # Custom JavaScript functions
            "$accumulator",  # Custom accumulator functions
            "$jsonSchema",  # Potential schema leakage
            "$expr",  # Can be complex and slow
        }
    
    def suggest_query_improvements(self, query_data: Dict, validation_result: Dict) -> List[str]:
        """
        Suggest improvements for the query based on validation results
        """
        suggestions = []
        
        # Add suggestions from validation
        suggestions.extend(validation_result.get("suggestions", []))
        
        pipeline = query_data.get("pipeline", [])
        
        # Suggest adding indexes
        if self._needs_index_suggestion(pipeline):
            suggestions.append("Consider adding indexes on frequently queried fields")
        
        # Suggest aggregation optimizations
        if self._can_optimize_aggregation(pipeline):
            suggestions.append("Pipeline could be optimized by reordering stages")
        
        # Suggest field projection
        if not self._has_projection(pipeline):
            suggestions.append("Consider using $project to limit returned fields")
        
        return suggestions
    
    def _needs_index_suggestion(self, pipeline: List[Dict]) -> bool:
        """
        Check if query would benefit from indexes
        """
        # Look for $match stages without indexes
        for stage in pipeline:
            if "$match" in stage:
                match_fields = list(stage["$match"].keys())
                if len(match_fields) > 0:
                    return True
        return False
    
    def _can_optimize_aggregation(self, pipeline: List[Dict]) -> bool:
        """
        Check if aggregation pipeline can be optimized
        """
        # Check if $match comes after $sort (should be before)
        sort_index = -1
        match_index = -1
        
        for i, stage in enumerate(pipeline):
            if "$sort" in stage and sort_index == -1:
                sort_index = i
            elif "$match" in stage:
                match_index = i
        
        return match_index > sort_index and sort_index != -1
    
    def _has_projection(self, pipeline: List[Dict]) -> bool:
        """
        Check if pipeline has field projection
        """
        return any("$project" in stage for stage in pipeline)
    
    def sanitize_query(self, query_data: Dict) -> Dict[str, Any]:
        """
        Sanitize query by removing or modifying dangerous elements
        """
        sanitized = query_data.copy()
        
        # Remove dangerous operators
        if "pipeline" in sanitized:
            sanitized_pipeline = []
            for stage in sanitized["pipeline"]:
                sanitized_stage = {}
                for operator, data in stage.items():
                    if operator not in self.dangerous_operators:
                        sanitized_stage[operator] = data
                    else:
                        logger.warning(f"Removed dangerous operator: {operator}")
                
                if sanitized_stage:
                    sanitized_pipeline.append(sanitized_stage)
            
            sanitized["pipeline"] = sanitized_pipeline
        
        # Ensure safe limits
        for stage in sanitized.get("pipeline", []):
            if "$limit" in stage:
                if stage["$limit"] > self.max_limit_value:
                    stage["$limit"] = self.max_limit_value
                    logger.info(f"Reduced $limit to {self.max_limit_value}")
        
        return sanitized
    
    def create_safe_fallback_query(self, collection: str) -> Dict[str, Any]:
        """
        Create a safe fallback query when validation fails
        """
        return {
            "collection": collection,
            "pipeline": [
                {"$limit": 50}
            ],
            "chart_hint": "bar",
            "query_intent": "Safe fallback query - limited results",
            "expected_fields": [],
            "data_summary": "Limited dataset for safety",
            "confidence": 0.3
        }
    
    def validate_chart_hint(self, chart_hint: str) -> Tuple[bool, str]:
        """
        Validate chart type hint
        """
        valid_charts = ["bar", "horizontalBar", "line", "pie", "doughnut", "scatter", "bubble", "radar", "area"]
        
        if not chart_hint:
            return False, "Chart hint is required"
        
        if chart_hint not in valid_charts:
            return False, f"Invalid chart type: {chart_hint}. Valid types: {', '.join(valid_charts)}"
        
        return True, chart_hint
    
    def estimate_query_cost(self, query_data: Dict, collection_size: int = 1000) -> Dict[str, Any]:
        """
        Estimate the computational cost of executing the query
        """
        pipeline = query_data.get("pipeline", [])
        
        cost_factors = {
            "scan_cost": 1,  # Base cost for scanning documents
            "sort_cost": 0,
            "group_cost": 0,
            "lookup_cost": 0,
            "regex_cost": 0
        }
        
        documents_processed = collection_size
        
        for stage in pipeline:
            operator = list(stage.keys())[0]
            
            if operator == "$match":
                # Assume match reduces dataset by 80%
                documents_processed *= 0.2
                if self._contains_regex(stage[operator]):
                    cost_factors["regex_cost"] += 2
            
            elif operator == "$sort":
                cost_factors["sort_cost"] += documents_processed * 0.001
            
            elif operator == "$group":
                cost_factors["group_cost"] += documents_processed * 0.002
            
            elif operator == "$lookup":
                cost_factors["lookup_cost"] += documents_processed * 0.01
            
            elif operator == "$limit":
                documents_processed = min(documents_processed, stage[operator])
        
        total_cost = sum(cost_factors.values())
        
        return {
            "estimated_cost": total_cost,
            "cost_breakdown": cost_factors,
            "documents_processed": int(documents_processed),
            "performance_tier": self._get_performance_tier(total_cost)
        }
    
    def _get_performance_tier(self, cost: float) -> str:
        """
        Categorize query performance based on estimated cost
        """
        if cost < 1:
            return "fast"
        elif cost < 5:
            return "medium"
        elif cost < 20:
            return "slow"
        else:
            return "very_slow"