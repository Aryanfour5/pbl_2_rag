"""
HybridBail: Bail Category Classifier
Multi-label classification into 17 bail categories
"""

import re
from typing import Dict, List, Tuple
from constants import BAIL_CATEGORIES
from legal_provision_parser import LegalProvisionParser
import logging

logger = logging.getLogger(__name__)

class BailCategoryClassifier:
    """Classifies cases into one or more of 17 bail categories."""
    
    def __init__(self, config):
        self.config = config
        self.categories = BAIL_CATEGORIES
        self.legal_parser = LegalProvisionParser()
        self.threshold = config.CLASSIFICATION_THRESHOLD
    
    def classify(self, case_text: str, legal_provisions: Dict = None) -> Dict:
        """
        Classify case into bail categories.
        
        Returns:
            {
                'primary_category': str,
                'all_categories': List[str],
                'confidence_scores': Dict[str, float],
                'reasoning': str
            }
        """
        
        if legal_provisions is None:
            legal_provisions = self.legal_parser.parse(case_text)
        
        scores = {}
        
        # Rule-based classification
        scores.update(self._classify_by_statute(legal_provisions))
        scores.update(self._classify_by_keywords(case_text))
        scores.update(self._classify_by_court_level(case_text))
        scores.update(self._classify_by_outcome(case_text))
        
        # Filter by threshold
        classified_categories = {
            cat: score for cat, score in scores.items() 
            if score >= self.threshold
        }
        
        # Get primary (highest scoring) category
        if classified_categories:
            primary = max(classified_categories.items(), key=lambda x: x[1])
            primary_category = primary[0]
        else:
            primary_category = "regular_bail"  # Default
            classified_categories["regular_bail"] = 0.5
        
        return {
            'primary_category': primary_category,
            'all_categories': list(classified_categories.keys()),
            'confidence_scores': classified_categories,
            'reasoning': self._generate_reasoning(classified_categories, legal_provisions)
        }
    
    def _classify_by_statute(self, provisions: Dict) -> Dict[str, float]:
        """Classify based on legal statutes involved."""
        scores = {}
        
        provs = provisions['provisions']
        
        # NDPS cases
        if provs['NDPS']:
            scores['ndps_bail'] = 0.95
        
        # POCSO cases
        if provs['POCSO']:
            scores['pocso_bail'] = 0.95
        
        # UAPA cases
        if provs['UAPA']:
            scores['uapa_bail'] = 0.95
        
        # SC/ST Act cases
        if provs['SC_ST_ACT']:
            scores['sc_st_act_bail'] = 0.95
        
        # Anticipatory bail (Section 438 CrPC)
        if '438' in provs['CrPC']:
            scores['anticipatory_bail'] = 0.90
        
        # Regular bail keywords
        if '437' in provs['CrPC'] or '439' in provs['CrPC']:
            scores['regular_bail'] = 0.70
        
        return scores
    
    def _classify_by_keywords(self, text: str) -> Dict[str, float]:
        """Classify based on keywords in text."""
        scores = {}
        text_lower = text.lower()
        
        keyword_mapping = {
            'anticipatory_bail': ['anticipatory', 'pre-arrest', 'section 438'],
            'interim_bail': ['interim bail', 'temporary bail', 'ad-interim'],
            'default_bail': ['default bail', 'statutory bail', 'section 167', 'section 436a'],
            'bail_conditions': ['conditions', 'subject to conditions', 'conditional bail'],
            'judicial_discretion': ['discretion', 'exceptional circumstances', 'rare case'],
            'bail_precedent': ['landmark', 'precedent', 'ratio decidendi', 'binding precedent']
        }
        
        for category, keywords in keyword_mapping.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                scores[category] = min(0.6 + (count * 0.1), 0.95)
        
        return scores
    
    def _classify_by_court_level(self, text: str) -> Dict[str, float]:
        """Classify based on court level."""
        scores = {}
        
        if re.search(r'high court', text, re.IGNORECASE):
            scores['high_court_bail'] = 0.80
        
        if re.search(r'supreme court|apex court', text, re.IGNORECASE):
            scores['supreme_court_bail'] = 0.80
        
        return scores
    
    def _classify_by_outcome(self, text: str) -> Dict[str, float]:
        """Classify based on case outcome."""
        scores = {}
        text_lower = text.lower()
        
        if re.search(r'bail\s+(?:is\s+)?granted|application\s+(?:is\s+)?allowed', text_lower):
            scores['regular_bail'] = 0.75
        
        if re.search(r'bail\s+(?:is\s+)?rejected|application\s+(?:is\s+)?dismissed|bail\s+denied', text_lower):
            scores['bail_cancellation'] = 0.75
        
        return scores
    
    def _generate_reasoning(self, categories: Dict, provisions: Dict) -> str:
        """Generate human-readable reasoning for classification."""
        reasons = []
        
        primary = max(categories.items(), key=lambda x: x[1])[0]
        primary_name = BAIL_CATEGORIES[primary]['name']
        
        reasons.append(f"Primary category: {primary_name}")
        
        if provisions['provisions']['NDPS']:
            reasons.append("NDPS Act sections detected - strict bail provisions apply")
        if provisions['provisions']['POCSO']:
            reasons.append("POCSO Act case - child protection paramount")
        if provisions['provisions']['UAPA']:
            reasons.append("UAPA case - stringent bail conditions")
        
        return "; ".join(reasons)
