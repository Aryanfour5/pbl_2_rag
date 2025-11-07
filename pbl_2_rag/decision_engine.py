# decision_engine.py - CORRECTED

"""
HybridBail: Decision Engine - FIXED
Three-tier decision system with confidence calibration
"""

import numpy as np
from typing import Dict, List, Optional
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
        
        logger.info("[OK] BailDecisionEngine initialized")
        logger.info("[INFO] Grant threshold: {:.0%}".format(self.auto_grant_threshold))
        logger.info("[INFO] Deny threshold: {:.0%}".format(self.auto_deny_threshold))
    
    def make_decision(self, current_case: Dict, similar_cases: List[Dict], 
                     category: str) -> Dict:
        """
        Generate bail decision with confidence scoring.
        
        Decision Logic:
        - Similar cases show mostly GRANT (>70%) → GRANT
        - Similar cases show mostly DENY (<30%) → DENY
        - Mixed outcomes (30-70%) → HUMAN_INTERVENTION_REQUIRED
        """
        
        logger.info("[DECISION] Analyzing {} similar cases".format(len(similar_cases)))
        
        if not similar_cases:
            logger.warning("[WARN] No similar cases found")
            return self._insufficient_data_decision(current_case, category)
        
        # Step 1: Analyze precedent patterns
        precedent_analysis = self._analyze_precedents(similar_cases)
        logger.debug("[DEBUG] Grant rate: {:.0%}".format(precedent_analysis['grant_rate']))
        
        # Step 2: Calculate confidence score
        confidence = self._calculate_confidence(precedent_analysis, similar_cases)
        logger.debug("[DEBUG] Confidence: {:.0%}".format(confidence))
        
        # Step 3: Generate recommendation (SIMPLIFIED LOGIC)
        recommendation, needs_intervention, reasons = self._generate_recommendation(
            precedent_analysis,
            confidence,
            category
        )
        
        logger.info("[RESULT] Recommendation: {} (confidence: {:.0%})".format(
            recommendation, confidence))
        
        if needs_intervention:
            logger.warning("[WARN] Human review required: {}".format(reasons))
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "precedent_analysis": precedent_analysis,
            "needs_human_review": needs_intervention,
            "intervention_reasons": reasons,
            "similar_cases_count": len(similar_cases),
            "decision_logic": {
                "grant_rate": precedent_analysis['grant_rate'],
                "dominant_outcome": precedent_analysis['dominant_outcome'],
                "has_conflicts": precedent_analysis['has_conflicts']
            }
        }
    
    def _analyze_precedents(self, similar_cases: List[Dict]) -> Dict:
        """Analyze patterns in similar precedents."""
        
        total = len(similar_cases)
        
        # Count outcomes - be flexible with field names
        granted = 0
        denied = 0
        
        for case in similar_cases:
            outcome_str = str(case.get('outcome', case.get('decision', 'unknown'))).lower()
            
            if any(word in outcome_str for word in ['grant', 'allowed', 'admitted', 'accept']):
                granted += 1
            elif any(word in outcome_str for word in ['deny', 'denied', 'rejected', 'dismiss', 'refuse']):
                denied += 1
        
        grant_rate = granted / total if total > 0 else 0.5
        deny_rate = denied / total if total > 0 else 0.5
        
        logger.debug("[DEBUG] Precedent outcomes: {} grant, {} deny out of {}".format(
            granted, denied, total))
        
        # Calculate consistency (how uniform are the outcomes)
        outcomes_binary = [1 if case in [c for c in similar_cases 
                          if 'grant' in str(c.get('outcome', c.get('decision', ''))).lower()] 
                          else 0 for case in similar_cases]
        consistency = 1 - np.std(outcomes_binary) if outcomes_binary else 0.5
        
        # Average similarity score
        avg_similarity = np.mean([c.get('final_score', c.get('score', 0.5)) 
                                 for c in similar_cases])
        
        # Check for conflicting judgments (30-70% split)
        has_conflicts = (0.30 < grant_rate < 0.70)
        
        return {
            "total_precedents": total,
            "granted_count": granted,
            "denied_count": denied,
            "grant_rate": grant_rate,
            "deny_rate": deny_rate,
            "consistency_score": consistency,
            "average_similarity": avg_similarity,
            "has_conflicts": has_conflicts,
            "dominant_outcome": "granted" if grant_rate > 0.5 else "denied"
        }
    
    def _calculate_confidence(self, precedent_analysis: Dict, similar_cases: List[Dict]) -> float:
        """
        Calculate confidence score for decision.
        
        How confident are we in the recommendation?
        """
        
        # Component 1: Precedent consistency (40%)
        # How similar are the precedent outcomes?
        consistency_score = precedent_analysis['consistency_score']
        
        # Component 2: Similarity strength (30%)
        # How similar are the found cases to current case?
        similarity_score = precedent_analysis['average_similarity']
        
        # Component 3: Sample size adequacy (15%)
        # Do we have enough precedents?
        sample_size = precedent_analysis['total_precedents']
        sample_score = min(sample_size / 5.0, 1.0)  # ← Reduced from 10 to 5
        
        # Component 4: Outcome clarity (15%)
        # How clear-cut is the majority outcome?
        grant_rate = precedent_analysis['grant_rate']
        
        if grant_rate > 0.85 or grant_rate < 0.15:
            # Very clear: >85% one outcome
            clarity_score = 1.0
        elif grant_rate > 0.70 or grant_rate < 0.30:
            # Clear: >70% one outcome
            clarity_score = 0.85
        elif grant_rate > 0.60 or grant_rate < 0.40:
            # Reasonable: >60% one outcome
            clarity_score = 0.70
        else:
            # Mixed: 40-60% split
            clarity_score = 0.50
        
        # Weighted combination
        confidence = (
            0.40 * consistency_score +
            0.30 * similarity_score +
            0.15 * sample_score +
            0.15 * clarity_score
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_recommendation(self, precedent_analysis: Dict, confidence: float,
                                category: str) -> tuple:
        """
        Generate final recommendation.
        
        ✅ FIXED: Simpler logic based on grant_rate
        """
        
        grant_rate = precedent_analysis['grant_rate']
        has_conflicts = precedent_analysis['has_conflicts']
        
        logger.info("[LOGIC] Grant rate: {:.0%}, Has conflicts: {}".format(
            grant_rate, has_conflicts))
        
        # ===== DECISION LOGIC (SIMPLIFIED) =====
        
        # CASE 1: Strong evidence for GRANT (>70% similar cases granted)
        if grant_rate > 0.70:
            recommendation = "GRANT_BAIL"
            needs_intervention = False
            reasons = []
            logger.info("[DECISION] GRANT - {:.0%} of similar cases granted".format(grant_rate))
        
        # CASE 2: Strong evidence for DENY (<30% similar cases granted)
        elif grant_rate < 0.30:
            recommendation = "DENY_BAIL"
            needs_intervention = False
            reasons = []
            logger.info("[DECISION] DENY - {:.0%} of similar cases granted".format(grant_rate))
        
        # CASE 3: Mixed outcomes (30-70%)
        else:
            recommendation = "HUMAN_INTERVENTION_REQUIRED"
            needs_intervention = True
            reasons = [
                "Mixed precedent outcomes: {:.0%} granted, {:.0%} denied".format(
                    grant_rate * 100, (1 - grant_rate) * 100),
                "Similar cases show conflicting decisions - requires judicial review"
            ]
            logger.warning("[DECISION] HUMAN REVIEW - Mixed outcomes ({:.0%} grant rate)".format(
                grant_rate))
        
        return recommendation, needs_intervention, reasons
    
    def _insufficient_data_decision(self, case: Dict, category: str) -> Dict:
        """Handle cases with no similar precedents."""
        
        logger.warning("[WARN] No similar precedents available")
        
        return {
            "recommendation": "HUMAN_INTERVENTION_REQUIRED",
            "confidence": 0.0,
            "precedent_analysis": {
                "total_precedents": 0,
                "granted_count": 0,
                "denied_count": 0,
                "grant_rate": 0.5,
                "consistency_score": 0.0,
                "has_conflicts": False,
                "dominant_outcome": "unknown"
            },
            "needs_human_review": True,
            "intervention_reasons": [
                "No similar precedents found in database",
                "Requires direct judicial assessment based on case merits"
            ],
            "similar_cases_count": 0,
            "decision_logic": {
                "grant_rate": 0.5,
                "dominant_outcome": "unknown",
                "has_conflicts": False
            }
        }
