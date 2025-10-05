"""
HybridBail: Precedent Analyzer
Analyzes similar precedents for patterns
"""

from typing import List, Dict
from collections import Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PrecedentAnalyzer:
    """Analyzes precedent cases to identify patterns."""
    
    def __init__(self):
        pass
    
    def analyze(self, precedents: List[Dict]) -> Dict:
        """Comprehensive analysis of precedent cases."""
        
        if not precedents:
            return self._empty_analysis()
        
        return {
            "outcome_distribution": self._analyze_outcomes(precedents),
            "summary": self._generate_summary(precedents)
        }
    
    def _analyze_outcomes(self, precedents: List[Dict]) -> Dict:
        """Analyze distribution of outcomes."""
        outcomes = [p.get('outcome', 'unknown') for p in precedents]
        outcome_counts = Counter(outcomes)
        total = len(outcomes)
        
        granted = sum(1 for o in outcomes if 'grant' in str(o).lower())
        denied = sum(1 for o in outcomes if any(w in str(o).lower() for w in ['denied', 'rejected', 'dismiss']))
        
        return {
            "total_cases": total,
            "granted": granted,
            "denied": denied,
            "grant_percentage": (granted / total * 100) if total > 0 else 0,
            "denial_percentage": (denied / total * 100) if total > 0 else 0,
        }
    
    def _generate_summary(self, precedents: List[Dict]) -> str:
        """Generate summary of precedents."""
        outcomes = self._analyze_outcomes(precedents)
        
        summary = f"Analyzed {outcomes['total_cases']} similar precedent cases. "
        
        if outcomes['grant_percentage'] > 60:
            summary += f"Majority ({outcomes['grant_percentage']:.0f}%) resulted in bail being granted."
        elif outcomes['denial_percentage'] > 60:
            summary += f"Majority ({outcomes['denial_percentage']:.0f}%) resulted in bail being denied."
        else:
            summary += f"Mixed outcomes: {outcomes['grant_percentage']:.0f}% granted, {outcomes['denial_percentage']:.0f}% denied."
        
        return summary
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis."""
        return {
            "outcome_distribution": {"total_cases": 0},
            "summary": "No precedent cases available for analysis."
        }
