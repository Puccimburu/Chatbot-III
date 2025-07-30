"""
Collection classification for business purpose identification
Intelligent classification of collections by their business function
"""

import logging
from typing import Dict, List, Any
import re

from models.schema_models import (
    CollectionClassification, FieldSchema, FieldRole, FieldType
)

logger = logging.getLogger(__name__)


class CollectionClassifier:
    """Intelligent collection classifier for business purpose identification"""
    
    def __init__(self):
        # Patterns for different collection types
        self.transactional_patterns = {
            'names': [
                'orders', 'sales', 'transactions', 'payments', 'invoices',
                'purchases', 'bookings', 'reservations', 'appointments',
                'registrations', 'subscriptions', 'enrollments'
            ],
            'field_patterns': [
                'total', 'amount', 'price', 'cost', 'quantity', 'count',
                'revenue', 'profit', 'fee', 'charge', 'balance'
            ]
        }
        
        self.reference_patterns = {
            'names': [
                'users', 'customers', 'products', 'categories', 'brands',
                'suppliers', 'vendors', 'locations', 'regions', 'countries',
                'states', 'cities', 'departments', 'roles', 'permissions',
                'settings', 'configurations', 'templates', 'types'
            ],
            'suffixes': ['_types', '_categories', '_lookup', '_master', '_ref']
        }
        
        self.audit_patterns = {
            'names': [
                'logs', 'audit', 'history', 'events', 'activities',
                'tracking', 'monitoring', 'alerts', 'notifications'
            ],
            'field_patterns': [
                'timestamp', 'logged_at', 'created_at', 'action',
                'event_type', 'user_id', 'ip_address', 'session'
            ]
        }
        
        self.system_patterns = {
            'names': [
                'sessions', 'cache', 'temp', 'staging', 'queue',
                'jobs', 'tasks', 'migrations', 'schema', 'metadata'
            ],
            'prefixes': ['sys_', 'system_', 'tmp_', 'temp_', '_'],
            'suffixes': ['_cache', '_queue', '_staging', '_temp']
        }
        
        self.staging_patterns = {
            'names': [
                'import', 'export', 'staging', 'processing', 'batch',
                'upload', 'download', 'sync', 'backup', 'archive'
            ],
            'field_patterns': [
                'status', 'processed', 'imported', 'exported',
                'batch_id', 'file_name', 'processed_at'
            ]
        }
    
    def classify_collection(
        self, 
        collection_name: str, 
        field_schemas: Dict[str, FieldSchema],
        sample_documents: List[Dict[str, Any]]
    ) -> CollectionClassification:
        """
        Classify a collection based on its name, fields, and data patterns
        
        Args:
            collection_name: Name of the collection
            field_schemas: Analyzed field schemas
            sample_documents: Sample documents from the collection
            
        Returns:
            CollectionClassification: The determined classification
        """
        
        logger.debug(f"🏷️ Classifying collection: {collection_name}")
        
        collection_lower = collection_name.lower()
        
        # Score different classifications
        scores = {
            CollectionClassification.TRANSACTIONAL: 0.0,
            CollectionClassification.REFERENCE: 0.0,
            CollectionClassification.AUDIT: 0.0,
            CollectionClassification.SYSTEM: 0.0,
            CollectionClassification.STAGING: 0.0,
            CollectionClassification.ARCHIVE: 0.0
        }
        
        # Name-based classification
        scores.update(self._score_by_name(collection_lower))
        
        # Field-based classification
        field_scores = self._score_by_fields(field_schemas)
        for classification, score in field_scores.items():
            scores[classification] += score
        
        # Data pattern-based classification
        pattern_scores = self._score_by_data_patterns(sample_documents, field_schemas)
        for classification, score in pattern_scores.items():
            scores[classification] += score
        
        # Size-based hints
        size_scores = self._score_by_size_patterns(len(sample_documents), field_schemas)
        for classification, score in size_scores.items():
            scores[classification] += score
        
        # Find the highest scoring classification
        best_classification = max(scores, key=scores.get)
        best_score = scores[best_classification]
        
        logger.debug(f"📊 Classification scores for {collection_name}: {scores}")
        
        # If no clear winner, return UNKNOWN
        if best_score < 0.3:
            return CollectionClassification.UNKNOWN
        
        return best_classification
    
    def _score_by_name(self, collection_name: str) -> Dict[CollectionClassification, float]:
        """Score classification based on collection name patterns"""
        
        scores = {classification: 0.0 for classification in CollectionClassification}
        
        # Transactional patterns
        if collection_name in self.transactional_patterns['names']:
            scores[CollectionClassification.TRANSACTIONAL] += 0.8
        
        # Reference patterns
        if collection_name in self.reference_patterns['names']:
            scores[CollectionClassification.REFERENCE] += 0.8
        
        for suffix in self.reference_patterns['suffixes']:
            if collection_name.endswith(suffix):
                scores[CollectionClassification.REFERENCE] += 0.6
        
        # Audit patterns
        if collection_name in self.audit_patterns['names']:
            scores[CollectionClassification.AUDIT] += 0.8
        
        # System patterns
        if collection_name in self.system_patterns['names']:
            scores[CollectionClassification.SYSTEM] += 0.9
        
        for prefix in self.system_patterns['prefixes']:
            if collection_name.startswith(prefix):
                scores[CollectionClassification.SYSTEM] += 0.7
        
        for suffix in self.system_patterns['suffixes']:
            if collection_name.endswith(suffix):
                scores[CollectionClassification.SYSTEM] += 0.7
        
        # Staging patterns
        if collection_name in self.staging_patterns['names']:
            scores[CollectionClassification.STAGING] += 0.8
        
        # Archive patterns
        if 'archive' in collection_name or 'backup' in collection_name:
            scores[CollectionClassification.ARCHIVE] += 0.8
        
        return scores
    
    def _score_by_fields(self, field_schemas: Dict[str, FieldSchema]) -> Dict[CollectionClassification, float]:
        """Score classification based on field patterns"""
        
        scores = {classification: 0.0 for classification in CollectionClassification}
        
        field_names = [name.lower() for name in field_schemas.keys()]
        field_roles = [schema.role for schema in field_schemas.values()]
        
        # Transactional indicators
        transactional_fields = sum(
            1 for name in field_names
            if any(pattern in name for pattern in self.transactional_patterns['field_patterns'])
        )
        
        metric_fields = sum(1 for role in field_roles if role == FieldRole.METRIC)
        
        if transactional_fields > 0:
            scores[CollectionClassification.TRANSACTIONAL] += min(0.5, transactional_fields * 0.2)
        
        if metric_fields > 2:
            scores[CollectionClassification.TRANSACTIONAL] += 0.3
        
        # Reference indicators
        if len(field_schemas) < 10 and metric_fields < 2:
            scores[CollectionClassification.REFERENCE] += 0.3
        
        # Check for typical reference field patterns
        if any(name in ['name', 'title', 'description', 'code'] for name in field_names):
            scores[CollectionClassification.REFERENCE] += 0.2
        
        # Audit indicators
        audit_fields = sum(
            1 for name in field_names
            if any(pattern in name for pattern in self.audit_patterns['field_patterns'])
        )
        
        if audit_fields > 0:
            scores[CollectionClassification.AUDIT] += min(0.6, audit_fields * 0.2)
        
        # System indicators
        system_field_count = sum(
            1 for schema in field_schemas.values()
            if schema.role == FieldRole.METADATA
        )
        
        if system_field_count / len(field_schemas) > 0.5:
            scores[CollectionClassification.SYSTEM] += 0.4
        
        # Staging indicators
        staging_fields = sum(
            1 for name in field_names
            if any(pattern in name for pattern in self.staging_patterns['field_patterns'])
        )
        
        if staging_fields > 0:
            scores[CollectionClassification.STAGING] += min(0.5, staging_fields * 0.2)
        
        return scores
    
    def _score_by_data_patterns(
        self, 
        sample_documents: List[Dict[str, Any]], 
        field_schemas: Dict[str, FieldSchema]
    ) -> Dict[CollectionClassification, float]:
        """Score classification based on actual data patterns"""
        
        scores = {classification: 0.0 for classification in CollectionClassification}
        
        if not sample_documents:
            return scores
        
        # Analyze temporal patterns
        temporal_fields = [
            name for name, schema in field_schemas.items()
            if schema.role == FieldRole.TEMPORAL
        ]
        
        # Transactional data often has timestamps
        if len(temporal_fields) > 0:
            scores[CollectionClassification.TRANSACTIONAL] += 0.2
            scores[CollectionClassification.AUDIT] += 0.1
        
        # Check for status fields (common in transactional/staging data)
        status_fields = [
            name for name in field_schemas.keys()
            if 'status' in name.lower() or 'state' in name.lower()
        ]
        
        if status_fields:
            scores[CollectionClassification.TRANSACTIONAL] += 0.1
            scores[CollectionClassification.STAGING] += 0.2
        
        # Check for user_id patterns (common in audit/transactional)
        user_id_fields = [
            name for name in field_schemas.keys()
            if 'user' in name.lower() and 'id' in name.lower()
        ]
        
        if user_id_fields:
            scores[CollectionClassification.TRANSACTIONAL] += 0.1
            scores[CollectionClassification.AUDIT] += 0.2
        
        # Check data volatility (frequent updates suggest transactional)
        update_fields = [
            name for name in field_schemas.keys()
            if 'updated' in name.lower() or 'modified' in name.lower()
        ]
        
        if update_fields:
            scores[CollectionClassification.TRANSACTIONAL] += 0.1
        
        return scores
    
    def _score_by_size_patterns(
        self, 
        sample_size: int, 
        field_schemas: Dict[str, FieldSchema]
    ) -> Dict[CollectionClassification, float]:
        """Score classification based on size and complexity patterns"""
        
        scores = {classification: 0.0 for classification in CollectionClassification}
        
        field_count = len(field_schemas)
        
        # Reference collections tend to be smaller and simpler
        if field_count < 8:
            scores[CollectionClassification.REFERENCE] += 0.1
        
        # Transactional collections tend to have more fields
        if field_count > 10:
            scores[CollectionClassification.TRANSACTIONAL] += 0.1
        
        # System collections often have very few meaningful fields
        if field_count < 5:
            scores[CollectionClassification.SYSTEM] += 0.1
        
        # Complex nested structures might indicate staging/processing
        complex_fields = sum(
            1 for schema in field_schemas.values()
            if schema.field_type in [FieldType.OBJECT, FieldType.ARRAY]
        )
        
        if complex_fields > 2:
            scores[CollectionClassification.STAGING] += 0.1
        
        return scores
    
    def get_classification_confidence(
        self, 
        collection_name: str, 
        field_schemas: Dict[str, FieldSchema],
        sample_documents: List[Dict[str, Any]]
    ) -> float:
        """Get confidence score for the classification"""
        
        # Re-run classification to get scores
        collection_lower = collection_name.lower()
        
        scores = {classification: 0.0 for classification in CollectionClassification}
        scores.update(self._score_by_name(collection_lower))
        
        field_scores = self._score_by_fields(field_schemas)
        for classification, score in field_scores.items():
            scores[classification] += score
        
        pattern_scores = self._score_by_data_patterns(sample_documents, field_schemas)
        for classification, score in pattern_scores.items():
            scores[classification] += score
        
        size_scores = self._score_by_size_patterns(len(sample_documents), field_schemas)
        for classification, score in size_scores.items():
            scores[classification] += score
        
        # Calculate confidence based on score distribution
        sorted_scores = sorted(scores.values(), reverse=True)
        
        if len(sorted_scores) < 2:
            return 0.5
        
        best_score = sorted_scores[0]
        second_score = sorted_scores[1]
        
        # High confidence if clear winner
        if best_score > 0.8:
            return 0.9
        
        # Medium confidence if decent separation
        if best_score > 0.5 and (best_score - second_score) > 0.3:
            return 0.7
        
        # Low confidence otherwise
        if best_score > 0.3:
            return 0.5
        
        return 0.2
    
    def explain_classification(
        self, 
        collection_name: str, 
        classification: CollectionClassification,
        field_schemas: Dict[str, FieldSchema]
    ) -> List[str]:
        """Provide human-readable explanation for the classification"""
        
        explanations = []
        
        collection_lower = collection_name.lower()
        
        # Name-based explanations
        if classification == CollectionClassification.TRANSACTIONAL:
            if collection_lower in self.transactional_patterns['names']:
                explanations.append(f"Collection name '{collection_name}' indicates transactional data")
            
            metric_fields = [name for name, schema in field_schemas.items() if schema.role == FieldRole.METRIC]
            if len(metric_fields) > 2:
                explanations.append(f"Contains {len(metric_fields)} numeric fields suitable for business metrics")
        
        elif classification == CollectionClassification.REFERENCE:
            if collection_lower in self.reference_patterns['names']:
                explanations.append(f"Collection name '{collection_name}' indicates reference/lookup data")
            
            if len(field_schemas) < 10:
                explanations.append("Simple structure typical of reference data")
        
        elif classification == CollectionClassification.AUDIT:
            if collection_lower in self.audit_patterns['names']:
                explanations.append(f"Collection name '{collection_name}' indicates audit/logging data")
            
            temporal_fields = [name for name, schema in field_schemas.items() if schema.role == FieldRole.TEMPORAL]
            if temporal_fields:
                explanations.append("Contains timestamp fields typical of audit logs")
        
        elif classification == CollectionClassification.SYSTEM:
            if any(collection_lower.startswith(prefix) for prefix in self.system_patterns['prefixes']):
                explanations.append("Collection name suggests system/internal data")
            
            metadata_fields = [name for name, schema in field_schemas.items() if schema.role == FieldRole.METADATA]
            if len(metadata_fields) / len(field_schemas) > 0.5:
                explanations.append("High proportion of system metadata fields")
        
        if not explanations:
            explanations.append("Classification based on field analysis and data patterns")
        
        return explanations