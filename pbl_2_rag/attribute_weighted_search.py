"""
HybridBail: Attribute-Weighted Hybrid Search
Novel retrieval algorithm combining semantic + keyword + attribute similarity
"""

import numpy as np
from typing import List, Dict, Any
from constants import ATTRIBUTE_WEIGHTS
import logging

logger = logging.getLogger(__name__)

class AttributeWeightedSearch:
    """
    Novel hybrid search combining:
    1. Semantic similarity (Jina embeddings)
    2. Keyword matching (BM25)
    3. Attribute-based re-ranking (age, gender, custody, etc.)
    """
    
    def __init__(self, vector_db, bm25_scorer, config):
        self.vector_db = vector_db
        self.bm25_scorer = bm25_scorer
        self.config = config
        self.alpha = config.HYBRID_SEARCH_ALPHA  # 0.6 semantic + 0.4 keyword
    
    def search(self, query_embedding: np.ndarray, query_text: str, 
               query_attributes: Dict, category: str, limit: int = 10) -> List[Dict]:
        """
        Perform attribute-weighted hybrid search.
        
        Args:
            query_embedding: Query vector from Jina
            query_text: Query text for BM25
            query_attributes: Extracted attributes from query case
            category: Bail category for collection selection
            limit: Number of results
        
        Returns:
            List of results with combined scores
        """
        
        # Step 1: Semantic search (vector similarity)
        collection_name = self.config.get_collection_name(category)
        semantic_results = self.vector_db.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit * 3  # Get more for re-ranking
        )
        
        # Step 2: Keyword search (BM25)
        keyword_results = self.bm25_scorer.score(query_text)
        keyword_dict = {r[0]: r[1] for r in keyword_results[:limit * 3]}
        
        # Step 3: Combine semantic + keyword
        hybrid_scores = []
        for result in semantic_results:
            doc_id = result['id']
            semantic_score = result['score']
            keyword_score = keyword_dict.get(doc_id, 0.0)
            
            # Normalize scores
            semantic_norm = semantic_score
            keyword_norm = keyword_score / max(keyword_dict.values()) if keyword_dict else 0
            
            # Hybrid score
            hybrid_score = (self.alpha * semantic_norm) + ((1 - self.alpha) * keyword_norm)
            
            result['hybrid_score'] = hybrid_score
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score
            
            hybrid_scores.append(result)
        
        # Step 4: Attribute-weighted re-ranking (NOVEL CONTRIBUTION)
        reranked = self._attribute_rerank(
            hybrid_scores, 
            query_attributes, 
            category
        )
        
        return reranked[:limit]
    
    def _attribute_rerank(self, results: List[Dict], query_attrs: Dict, 
                         category: str) -> List[Dict]:
        """
        Re-rank results based on attribute similarity.
        This is the NOVEL CONTRIBUTION for patent/paper.
        """
        
        # Get category-specific weights
        weights = ATTRIBUTE_WEIGHTS.get(category, ATTRIBUTE_WEIGHTS['default'])
        
        for result in results:
            result_attrs = result.get('attributes', {})
            
            # Calculate attribute similarity score
            attr_similarity = self._calculate_attribute_similarity(
                query_attrs, 
                result_attrs, 
                weights
            )
            
            # Combine hybrid score with attribute similarity
            # 60% hybrid (semantic+keyword) + 40% attribute matching
            final_score = (0.6 * result['hybrid_score']) + (0.4 * attr_similarity)
            
            result['attribute_similarity'] = attr_similarity
            result['final_score'] = final_score
            result['attribute_match_breakdown'] = self._get_match_breakdown(
                query_attrs, result_attrs
            )
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
    
    def _calculate_attribute_similarity(self, query_attrs: Dict, 
                                       result_attrs: Dict, 
                                       weights: Dict) -> float:
        """
        Calculate weighted attribute similarity.
        Patent claim: "Multi-dimensional attribute matching with configurable weights"
        """
        
        total_score = 0.0
        total_weight = 0.0
        
        # Age similarity
        if 'age' in weights and query_attrs.get('age') and result_attrs.get('age'):
            age_diff = abs(query_attrs['age'] - result_attrs['age'])
            age_similarity = max(0, 1 - (age_diff / 50.0))  # Normalize by 50 years
            total_score += weights['age_group'] * age_similarity
            total_weight += weights['age_group']
        
        # Gender match (exact)
        if 'gender' in weights:
            gender_match = 1.0 if query_attrs.get('gender') == result_attrs.get('gender') else 0.0
            total_score += weights['gender'] * gender_match
            total_weight += weights['gender']
        
        # Criminal history similarity
        if 'criminal_history' in weights:
            query_history = query_attrs.get('criminal_history', {}).get('category', 'unknown')
            result_history = result_attrs.get('criminal_history', {}).get('category', 'unknown')
            history_match = 1.0 if query_history == result_history else 0.3
            total_score += weights['criminal_history'] * history_match
            total_weight += weights['criminal_history']
        
        # Custody duration similarity
        if 'custody_duration' in weights:
            query_days = query_attrs.get('custody_days', 0)
            result_days = result_attrs.get('custody_days', 0)
            if query_days and result_days:
                custody_diff = abs(query_days - result_days)
                custody_similarity = max(0, 1 - (custody_diff / 180.0))  # Normalize by 6 months
                total_score += weights['custody_duration'] * custody_similarity
                total_weight += weights['custody_duration']
        
        # Legal sections overlap
        if 'legal_sections' in weights:
            query_sections = set(query_attrs.get('legal_sections', []))
            result_sections = set(result_attrs.get('legal_sections', []))
            if query_sections:
                section_overlap = len(query_sections & result_sections) / len(query_sections)
                total_score += weights['legal_sections'] * section_overlap
                total_weight += weights['legal_sections']
        
        # Normalize by total weight
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _get_match_breakdown(self, query_attrs: Dict, result_attrs: Dict) -> Dict:
        """Get detailed breakdown of attribute matches for explainability."""
        return {
            'age_match': f"{query_attrs.get('age', 'N/A')} vs {result_attrs.get('age', 'N/A')}",
            'gender_match': query_attrs.get('gender') == result_attrs.get('gender'),
            'criminal_history_match': query_attrs.get('criminal_history', {}).get('category') == 
                                     result_attrs.get('criminal_history', {}).get('category'),
            'custody_days': f"{query_attrs.get('custody_days', 0)} vs {result_attrs.get('custody_days', 0)}"
        }
