"""
HybridBail: Decision Engine
Three-tier decision system with confidence calibration
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BailDecisionEngine:
    """
    Core decision engine implementing three-tier system:
    1. Auto-grant (high confidence bail)
    2. Auto-deny (high confidence denial)
    3. Human intervention required (ambiguous cases)
    """
    
    def __init__(self, config):
        self.config = config
        self.auto_grant_threshold = config.AUTO_GRANT_THRESHOLD
        self.auto_deny_threshold = config.AUTO_DENY_THRESHOLD
    
    def make_decision(self, current_case: Dict, similar_cases: List[Dict], 
                     category: str) -> Dict:
        """
        Generate bail decision with confidence scoring.
        
        Args:
            current_case: Case attributes and details
            similar_cases: List of retrieved similar precedents
            category: Bail category
            
        Returns:
            Decision dictionary with recommendation, confidence, reasoning
        """
        
        if not similar_cases:
            return self._insufficient_data_decision(current_case, category)
        
        # Step 1: Analyze precedent patterns
        precedent_analysis = self._analyze_precedents(similar_cases)
        
        # Step 2: Calculate confidence score
        confidence = self._calculate_confidence(precedent_analysis, similar_cases)
        
        # Step 3: Check intervention triggers
        needs_intervention = self._check_intervention_triggers(
            current_case, 
            precedent_analysis, 
            confidence,
            category
        )
        
        # Step 4: Generate recommendation
        recommendation = self._generate_recommendation(
            precedent_analysis,
            confidence,
            needs_intervention,
            category
        )
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "precedent_analysis": precedent_analysis,
            "needs_human_review": needs_intervention,
            "intervention_reasons": self._get_intervention_reasons(precedent_analysis, confidence),
            "similar_cases_count": len(similar_cases),
            "top_precedents": similar_cases[:5]
        }
    
    def _analyze_precedents(self, similar_cases: List[Dict]) -> Dict:
        """Analyze patterns in similar precedents."""
        
        total = len(similar_cases)
        
        # Count outcomes
        granted = sum(1 for case in similar_cases if 'grant' in str(case.get('outcome', '')).lower())
        denied = sum(1 for case in similar_cases if any(word in str(case.get('outcome', '')).lower() for word in ['denied', 'rejected', 'dismiss']))
        
        grant_rate = granted / total if total > 0 else 0.0
        deny_rate = denied / total if total > 0 else 0.0
        
        # Calculate consistency
        outcomes_binary = [1 if 'grant' in str(c.get('outcome', '')).lower() else 0 for c in similar_cases]
        consistency = 1 - np.std(outcomes_binary) if outcomes_binary else 0.0
        
        # Average similarity score
        avg_similarity = np.mean([c.get('final_score', c.get('score', 0.5)) for c in similar_cases])
        
        # Check for conflicting judgments
        has_conflicts = grant_rate > 0.3 and deny_rate > 0.3
        
        return {
            "total_precedents": total,
            "granted_count": granted,
            "denied_count": denied,
            "grant_rate": grant_rate,
            "deny_rate": deny_rate,
            "consistency_score": consistency,
            "average_similarity": avg_similarity,
            "has_conflicts": has_conflicts,
            "dominant_outcome": "granted" if grant_rate > deny_rate else "denied"
        }
    
    def _calculate_confidence(self, precedent_analysis: Dict, similar_cases: List[Dict]) -> float:
        """Calculate confidence score for decision."""
        
        # Component 1: Precedent consistency (40%)
        consistency_score = precedent_analysis['consistency_score']
        
        # Component 2: Similarity strength (30%)
        similarity_score = precedent_analysis['average_similarity']
        
        # Component 3: Sample size adequacy (15%)
        sample_size = precedent_analysis['total_precedents']
        sample_score = min(sample_size / 10.0, 1.0)
        
        # Component 4: Outcome clarity (15%)
        grant_rate = precedent_analysis['grant_rate']
        if grant_rate > 0.8 or grant_rate < 0.2:
            clarity_score = 1.0
        elif 0.6 < grant_rate < 0.8 or 0.2 < grant_rate < 0.4:
            clarity_score = 0.6
        else:
            clarity_score = 0.3
        
        # Weighted combination
        confidence = (
            0.40 * consistency_score +
            0.30 * similarity_score +
            0.15 * sample_score +
            0.15 * clarity_score
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _check_intervention_triggers(self, case: Dict, precedent_analysis: Dict, 
                                    confidence: float, category: str) -> bool:
        """Check if case needs human intervention."""
        
        # Trigger 1: Low confidence
        if confidence < 0.60:
            return True
        
        # Trigger 2: Contradictory precedents
        if precedent_analysis['has_conflicts']:
            return True
        
        # Trigger 3: Critical categories
        if category in ['ndps_bail', 'pocso_bail', 'uapa_bail']:
            if confidence < 0.90:
                return True
        
        return False
    
    def _generate_recommendation(self, precedent_analysis: Dict, confidence: float,
                                needs_intervention: bool, category: str) -> str:
        """Generate final recommendation."""
        
        if needs_intervention:
            return "HUMAN_INTERVENTION_REQUIRED"
        
        grant_rate = precedent_analysis['grant_rate']
        
        # High confidence auto-grant
        if confidence >= self.auto_grant_threshold and grant_rate > 0.70:
            return "GRANT_BAIL"
        
        # High confidence auto-deny
        if confidence >= self.auto_grant_threshold and grant_rate < 0.30:
            return "DENY_BAIL"
        
        return "HUMAN_INTERVENTION_REQUIRED"
    
    def _get_intervention_reasons(self, precedent_analysis: Dict, confidence: float) -> List[str]:
        """Get human-readable reasons for intervention."""
        reasons = []
        
        if confidence < 0.60:
            reasons.append(f"Low confidence score: {confidence:.2f}")
        
        if precedent_analysis['has_conflicts']:
            reasons.append(f"Contradictory precedents: {precedent_analysis['grant_rate']:.0%} grant rate")
        
        if precedent_analysis['total_precedents'] < 3:
            reasons.append(f"Insufficient precedents: only {precedent_analysis['total_precedents']} similar cases")
        
        return reasons
    
    def _insufficient_data_decision(self, case: Dict, category: str) -> Dict:
        """Handle cases with no similar precedents."""
        return {
            "recommendation": "HUMAN_INTERVENTION_REQUIRED",
            "confidence": 0.0,
            "precedent_analysis": {
                "total_precedents": 0,
                "grant_rate": 0.0,
                "consistency_score": 0.0
            },
            "needs_human_review": True,
            "intervention_reasons": ["No similar precedents found - requires judicial assessment"],
            "similar_cases_count": 0,
            "top_precedents": []
        }
