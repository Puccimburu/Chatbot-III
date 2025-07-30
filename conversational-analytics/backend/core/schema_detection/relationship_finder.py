# core/schema_detection/relationship_finder.py

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class RelationshipType(Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"
    EMBEDDED = "embedded"
    REFERENCE = "reference"

@dataclass
class FieldRelationship:
    from_collection: str
    from_field: str
    to_collection: str
    to_field: str
    relationship_type: RelationshipType
    confidence: float
    sample_values: List[Tuple[Any, Any]]
    cardinality_ratio: float

@dataclass
class CollectionRelationship:
    primary_collection: str
    related_collection: str
    relationships: List[FieldRelationship]
    relationship_strength: float
    suggested_join_strategy: str

class RelationshipFinder:
    """
    Discovers relationships between MongoDB collections through field analysis and value matching
    """
    
    def __init__(self, database_service):
        self.db_service = database_service
        self.id_patterns = self._initialize_id_patterns()
        self.name_patterns = self._initialize_name_patterns()
        self.min_confidence = 0.6
        self.sample_size = 100
        
    async def discover_relationships(self, collections_schema: Dict[str, Any]) -> Dict[str, List[CollectionRelationship]]:
        """
        Discover relationships between all collections
        """
        try:
            relationships = {}
            collection_names = list(collections_schema.keys())
            
            logger.info(f"🔍 Discovering relationships between {len(collection_names)} collections")
            
            # Analyze each pair of collections
            for i, collection_a in enumerate(collection_names):
                relationships[collection_a] = []
                
                for collection_b in collection_names[i+1:]:
                    # Find relationships between collection_a and collection_b
                    collection_relationships = await self._find_collection_relationships(
                        collection_a, collection_b, collections_schema
                    )
                    
                    if collection_relationships:
                        relationships[collection_a].extend(collection_relationships)
                        
                        # Add reverse relationships
                        if collection_b not in relationships:
                            relationships[collection_b] = []
                        
                        for rel in collection_relationships:
                            reverse_rel = self._create_reverse_relationship(rel)
                            relationships[collection_b].append(reverse_rel)
            
            # Filter by confidence and deduplicate
            filtered_relationships = self._filter_and_rank_relationships(relationships)
            
            logger.info(f"✅ Found {sum(len(rels) for rels in filtered_relationships.values())} high-confidence relationships")
            
            return filtered_relationships
            
        except Exception as e:
            logger.error(f"Error discovering relationships: {e}")
            return {}
    
    async def _find_collection_relationships(self, collection_a: str, collection_b: str, 
                                           schema_info: Dict[str, Any]) -> List[CollectionRelationship]:
        """
        Find relationships between two specific collections
        """
        try:
            # Get schema information
            schema_a = schema_info.get(collection_a, {})
            schema_b = schema_info.get(collection_b, {})
            
            fields_a = schema_a.get("fields", {})
            fields_b = schema_b.get("fields", {})
            
            if not fields_a or not fields_b:
                return []
            
            # Find potential field relationships
            field_relationships = await self._analyze_field_relationships(
                collection_a, fields_a, collection_b, fields_b
            )
            
            if not field_relationships:
                return []
            
            # Group relationships and calculate strength
            collection_relationship = CollectionRelationship(
                primary_collection=collection_a,
                related_collection=collection_b,
                relationships=field_relationships,
                relationship_strength=self._calculate_relationship_strength(field_relationships),
                suggested_join_strategy=self._suggest_join_strategy(field_relationships)
            )
            
            return [collection_relationship] if collection_relationship.relationship_strength > self.min_confidence else []
            
        except Exception as e:
            logger.error(f"Error finding relationships between {collection_a} and {collection_b}: {e}")
            return []
    
    async def _analyze_field_relationships(self, collection_a: str, fields_a: Dict, 
                                         collection_b: str, fields_b: Dict) -> List[FieldRelationship]:
        """
        Analyze field-level relationships between two collections
        """
        relationships = []
        
        # Get sample data for both collections
        try:
            sample_a = await self._get_sample_data(collection_a)
            sample_b = await self._get_sample_data(collection_b)
            
            if not sample_a or not sample_b:
                return relationships
            
        except Exception as e:
            logger.warning(f"Could not get sample data for relationship analysis: {e}")
            return relationships
        
        # Analyze each field combination
        for field_a_name, field_a_info in fields_a.items():
            for field_b_name, field_b_info in fields_b.items():
                
                # Skip non-relational fields
                if not self._is_potential_relationship_field(field_a_name, field_b_name, field_a_info, field_b_info):
                    continue
                
                # Analyze value overlap
                relationship = await self._analyze_value_overlap(
                    collection_a, field_a_name, sample_a,
                    collection_b, field_b_name, sample_b
                )
                
                if relationship and relationship.confidence >= self.min_confidence:
                    relationships.append(relationship)
        
        return relationships
    
    async def _get_sample_data(self, collection: str) -> List[Dict]:
        """
        Get sample data from a collection for relationship analysis
        """
        try:
            # Get a sample of documents
            pipeline = [{"$sample": {"size": self.sample_size}}]
            results = await self.db_service.execute_aggregation(collection, pipeline)
            return results if results else []
            
        except Exception as e:
            logger.warning(f"Could not get sample data from {collection}: {e}")
            return []
    
    def _is_potential_relationship_field(self, field_a: str, field_b: str, 
                                       info_a: Dict, info_b: Dict) -> bool:
        """
        Determine if two fields could potentially have a relationship
        """
        # Check field name patterns
        if self._match_id_patterns(field_a, field_b):
            return True
        
        if self._match_name_patterns(field_a, field_b):
            return True
        
        # Check field types
        type_a = info_a.get("type", "unknown")
        type_b = info_b.get("type", "unknown")
        
        # Both should be similar types for relationships
        if type_a != type_b:
            return False
        
        # Check cardinality hints
        cardinality_a = info_a.get("cardinality", 1.0)
        cardinality_b = info_b.get("cardinality", 1.0)
        
        # At least one should have low cardinality (potential foreign key)
        if cardinality_a > 0.8 and cardinality_b > 0.8:
            return False
        
        return True
    
    def _match_id_patterns(self, field_a: str, field_b: str) -> bool:
        """
        Check if fields match ID patterns suggesting relationships
        """
        # Direct ID matches
        if field_a == field_b and any(pattern in field_a.lower() for pattern in self.id_patterns):
            return True
        
        # Foreign key patterns
        if field_a.endswith("_id") and field_b == "_id":
            return True
        
        if field_b.endswith("_id") and field_a == "_id":
            return True
        
        # Collection name + ID patterns
        for pattern in ["id", "_id", "ID"]:
            if field_a.replace(pattern, "") == field_b.replace(pattern, ""):
                return True
        
        return False
    
    def _match_name_patterns(self, field_a: str, field_b: str) -> bool:
        """
        Check if fields match name patterns suggesting relationships
        """
        # Exact name matches
        if field_a == field_b:
            return any(pattern in field_a.lower() for pattern in self.name_patterns)
        
        # Similar name patterns
        similarity = self._calculate_field_name_similarity(field_a, field_b)
        return similarity > 0.8
    
    async def _analyze_value_overlap(self, collection_a: str, field_a: str, sample_a: List[Dict],
                                   collection_b: str, field_b: str, sample_b: List[Dict]) -> Optional[FieldRelationship]:
        """
        Analyze value overlap between two fields to determine relationship
        """
        try:
            # Extract values
            values_a = [doc.get(field_a) for doc in sample_a if doc.get(field_a) is not None]
            values_b = [doc.get(field_b) for doc in sample_b if doc.get(field_b) is not None]
            
            if not values_a or not values_b:
                return None
            
            # Convert to sets for comparison
            set_a = set(values_a)
            set_b = set(values_b)
            
            # Calculate overlap
            intersection = set_a & set_b
            union = set_a | set_b
            
            if not intersection:
                return None
            
            # Calculate relationship metrics
            overlap_ratio = len(intersection) / len(union)
            confidence = self._calculate_relationship_confidence(
                set_a, set_b, intersection, field_a, field_b
            )
            
            if confidence < self.min_confidence:
                return None
            
            # Determine relationship type
            relationship_type = self._determine_relationship_type(values_a, values_b, set_a, set_b)
            
            # Calculate cardinality ratio
            cardinality_ratio = len(set_a) / len(set_b) if len(set_b) > 0 else 0
            
            # Get sample matching values
            sample_values = [(a, b) for a in list(intersection)[:5] for b in [a]]
            
            return FieldRelationship(
                from_collection=collection_a,
                from_field=field_a,
                to_collection=collection_b,
                to_field=field_b,
                relationship_type=relationship_type,
                confidence=confidence,
                sample_values=sample_values,
                cardinality_ratio=cardinality_ratio
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing value overlap: {e}")
            return None
    
    def _calculate_relationship_confidence(self, set_a: Set, set_b: Set, intersection: Set,
                                         field_a: str, field_b: str) -> float:
        """
        Calculate confidence score for a potential relationship
        """
        # Base confidence from overlap
        jaccard_similarity = len(intersection) / len(set_a | set_b)
        
        # Boost confidence based on field name similarity
        name_similarity = self._calculate_field_name_similarity(field_a, field_b)
        
        # Boost confidence for ID-like patterns
        id_boost = 0.0
        if any(pattern in field_a.lower() or pattern in field_b.lower() for pattern in self.id_patterns):
            id_boost = 0.2
        
        # Penalize if sets are very different sizes (unlikely to be related)
        size_penalty = 0.0
        if len(set_a) > 0 and len(set_b) > 0:
            size_ratio = min(len(set_a), len(set_b)) / max(len(set_a), len(set_b))
            if size_ratio < 0.1:
                size_penalty = 0.3
        
        confidence = (jaccard_similarity * 0.6) + (name_similarity * 0.2) + id_boost - size_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _determine_relationship_type(self, values_a: List, values_b: List, 
                                   set_a: Set, set_b: Set) -> RelationshipType:
        """
        Determine the type of relationship based on cardinality
        """
        # Calculate cardinality ratios
        unique_a_ratio = len(set_a) / len(values_a) if values_a else 0
        unique_b_ratio = len(set_b) / len(values_b) if values_b else 0
        
        # High uniqueness suggests primary key, low suggests foreign key
        if unique_a_ratio > 0.9 and unique_b_ratio < 0.5:
            return RelationshipType.ONE_TO_MANY
        elif unique_a_ratio < 0.5 and unique_b_ratio > 0.9:
            return RelationshipType.MANY_TO_ONE
        elif unique_a_ratio > 0.9 and unique_b_ratio > 0.9:
            return RelationshipType.ONE_TO_ONE
        else:
            return RelationshipType.MANY_TO_MANY
    
    def _calculate_field_name_similarity(self, name_a: str, name_b: str) -> float:
        """
        Calculate similarity between field names
        """
        # Normalize names
        norm_a = name_a.lower().replace("_", "").replace("-", "")
        norm_b = name_b.lower().replace("_", "").replace("-", "")
        
        if norm_a == norm_b:
            return 1.0
        
        # Check if one is contained in the other
        if norm_a in norm_b or norm_b in norm_a:
            return 0.8
        
        # Calculate Levenshtein-style similarity
        max_len = max(len(norm_a), len(norm_b))
        if max_len == 0:
            return 0.0
        
        # Simple character overlap
        common_chars = sum(1 for c in norm_a if c in norm_b)
        return common_chars / max_len
    
    def _calculate_relationship_strength(self, field_relationships: List[FieldRelationship]) -> float:
        """
        Calculate overall relationship strength between collections
        """
        if not field_relationships:
            return 0.0
        
        # Average confidence weighted by relationship type importance
        type_weights = {
            RelationshipType.ONE_TO_ONE: 1.0,
            RelationshipType.ONE_TO_MANY: 0.9,
            RelationshipType.MANY_TO_ONE: 0.9,
            RelationshipType.MANY_TO_MANY: 0.7,
            RelationshipType.EMBEDDED: 0.8,
            RelationshipType.REFERENCE: 0.8
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for rel in field_relationships:
            weight = type_weights.get(rel.relationship_type, 0.5)
            weighted_confidence += rel.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _suggest_join_strategy(self, field_relationships: List[FieldRelationship]) -> str:
        """
        Suggest the best join strategy for the relationship
        """
        if not field_relationships:
            return "none"
        
        # Find the strongest relationship
        strongest_rel = max(field_relationships, key=lambda r: r.confidence)
        
        if strongest_rel.relationship_type in [RelationshipType.ONE_TO_ONE, RelationshipType.ONE_TO_MANY]:
            return "lookup"
        elif strongest_rel.relationship_type == RelationshipType.MANY_TO_MANY:
            return "unwind_lookup"
        else:
            return "match_join"
    
    def _create_reverse_relationship(self, relationship: CollectionRelationship) -> CollectionRelationship:
        """
        Create reverse relationship for bidirectional mapping
        """
        reverse_field_relationships = []
        
        for field_rel in relationship.relationships:
            reverse_type = self._get_reverse_relationship_type(field_rel.relationship_type)
            
            reverse_field_rel = FieldRelationship(
                from_collection=field_rel.to_collection,
                from_field=field_rel.to_field,
                to_collection=field_rel.from_collection,
                to_field=field_rel.from_field,
                relationship_type=reverse_type,
                confidence=field_rel.confidence,
                sample_values=[(b, a) for a, b in field_rel.sample_values],
                cardinality_ratio=1.0 / field_rel.cardinality_ratio if field_rel.cardinality_ratio > 0 else 0
            )
            
            reverse_field_relationships.append(reverse_field_rel)
        
        return CollectionRelationship(
            primary_collection=relationship.related_collection,
            related_collection=relationship.primary_collection,
            relationships=reverse_field_relationships,
            relationship_strength=relationship.relationship_strength,
            suggested_join_strategy=relationship.suggested_join_strategy
        )
    
    def _get_reverse_relationship_type(self, relationship_type: RelationshipType) -> RelationshipType:
        """
        Get the reverse of a relationship type
        """
        reverse_map = {
            RelationshipType.ONE_TO_ONE: RelationshipType.ONE_TO_ONE,
            RelationshipType.ONE_TO_MANY: RelationshipType.MANY_TO_ONE,
            RelationshipType.MANY_TO_ONE: RelationshipType.ONE_TO_MANY,
            RelationshipType.MANY_TO_MANY: RelationshipType.MANY_TO_MANY,
            RelationshipType.EMBEDDED: RelationshipType.REFERENCE,
            RelationshipType.REFERENCE: RelationshipType.EMBEDDED
        }
        
        return reverse_map.get(relationship_type, RelationshipType.REFERENCE)
    
    def _filter_and_rank_relationships(self, relationships: Dict[str, List[CollectionRelationship]]) -> Dict[str, List[CollectionRelationship]]:
        """
        Filter relationships by confidence and rank by strength
        """
        filtered = {}
        
        for collection, collection_relationships in relationships.items():
            # Filter by minimum confidence
            high_confidence = [
                rel for rel in collection_relationships 
                if rel.relationship_strength >= self.min_confidence
            ]
            
            # Sort by relationship strength
            high_confidence.sort(key=lambda r: r.relationship_strength, reverse=True)
            
            # Keep top relationships to avoid noise
            filtered[collection] = high_confidence[:10]
        
        return filtered
    
    def _initialize_id_patterns(self) -> List[str]:
        """
        Initialize patterns that suggest ID fields
        """
        return [
            "id", "_id", "uuid", "key", "ref", "reference",
            "foreign", "fk", "pk", "primary", "identifier"
        ]
    
    def _initialize_name_patterns(self) -> List[str]:
        """
        Initialize patterns that suggest name/title fields for relationships
        """
        return [
            "name", "title", "label", "description", "code",
            "slug", "handle", "username", "email"
        ]
    
    def get_relationship_summary(self, relationships: Dict[str, List[CollectionRelationship]]) -> Dict[str, Any]:
        """
        Generate a summary of discovered relationships
        """
        total_relationships = sum(len(rels) for rels in relationships.values())
        
        relationship_types = {}
        confidence_distribution = []
        
        for collection_rels in relationships.values():
            for rel in collection_rels:
                for field_rel in rel.relationships:
                    rel_type = field_rel.relationship_type.value
                    relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                    confidence_distribution.append(field_rel.confidence)
        
        avg_confidence = sum(confidence_distribution) / len(confidence_distribution) if confidence_distribution else 0
        
        return {
            "total_relationships": total_relationships,
            "relationship_types": relationship_types,
            "average_confidence": round(avg_confidence, 2),
            "high_confidence_count": sum(1 for c in confidence_distribution if c >= 0.8),
            "collections_with_relationships": len([c for c, rels in relationships.items() if rels]),
            "most_connected_collection": max(relationships.keys(), key=lambda k: len(relationships[k])) if relationships else None
        }