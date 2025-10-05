"""
HybridBail: Attribute Extractor
Extracts case attributes (age, gender, custody duration, etc.)
"""

import re
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AttributeExtractor:
    """Extracts structured attributes from case text."""
    
    def __init__(self):
        pass
    
    def extract_all(self, text: str, case_date: Optional[datetime] = None) -> Dict:
        """Extract all attributes from case text."""
        return {
            "age": self.extract_age(text),
            "age_group": self.categorize_age(self.extract_age(text)),
            "gender": self.extract_gender(text),
            "custody_duration": self.extract_custody_duration(text, case_date),
            "custody_days": self.calculate_custody_days(text, case_date),
            "health_status": self.extract_health_status(text),
            "criminal_history": self.extract_criminal_history(text),
            "socioeconomic_status": self.extract_socioeconomic_status(text),
            "region": self.extract_region(text),
            "employment_status": self.extract_employment(text),
            "family_circumstances": self.extract_family_circumstances(text)
        }
    
    def extract_age(self, text: str) -> Optional[int]:
        """Extract age from text."""
        patterns = [
            r'age[d]?\s+(\d{1,3})\s+years?',
            r'(\d{1,3})\s+years?\s+old',
            r'aged\s+about\s+(\d{1,3})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    age = int(match.group(1))
                    if 0 < age < 150:
                        return age
                except:
                    pass
        return None
    
    def categorize_age(self, age: Optional[int]) -> str:
        """Categorize age into groups."""
        if age is None:
            return "unknown"
        
        if age < 18:
            return "juvenile"
        elif age < 25:
            return "young_adult"
        elif age < 40:
            return "adult"
        elif age < 60:
            return "middle_aged"
        else:
            return "senior_citizen"
    
    def extract_gender(self, text: str) -> str:
        """Extract gender from text."""
        text_lower = text.lower()
        
        male_indicators = ['he ', 'his ', 'him ', 'accused himself', 'petitioner himself', 'mr.']
        female_indicators = ['she ', 'her ', 'herself', 'petitioner herself', 'mrs.', 'ms.', 'smt.']
        
        male_count = sum(1 for ind in male_indicators if ind in text_lower)
        female_count = sum(1 for ind in female_indicators if ind in text_lower)
        
        if female_count > male_count:
            return "female"
        elif male_count > 0:
            return "male"
        
        return "unknown"
    
    def extract_custody_duration(self, text: str, case_date: Optional[datetime] = None) -> Optional[str]:
        """Extract custody duration description."""
        patterns = [
            r'in\s+custody\s+(?:for\s+)?(\d+)\s+(days?|months?|years?)',
            r'(?:been|remained)\s+in\s+(?:judicial\s+)?custody\s+(?:for\s+)?(\d+)\s+(days?|months?|years?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def calculate_custody_days(self, text: str, case_date: Optional[datetime] = None) -> Optional[int]:
        """Calculate number of days in custody."""
        duration_pattern = r'(\d+)\s+(days?|months?|years?)\s+(?:in\s+)?custody'
        match = re.search(duration_pattern, text, re.IGNORECASE)
        
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()
            
            if 'day' in unit:
                return value
            elif 'month' in unit:
                return value * 30
            elif 'year' in unit:
                return value * 365
        
        return None
    
    def extract_health_status(self, text: str) -> Dict:
        """Extract health information."""
        health_keywords = ['ill', 'sick', 'disease', 'medical', 'treatment', 'hospital', 'ailment']
        
        has_health_issues = any(keyword in text.lower() for keyword in health_keywords)
        
        return {
            "has_health_issues": has_health_issues,
            "medical_bail_eligible": 'medical' in text.lower() and 'bail' in text.lower(),
            "conditions": []
        }
    
    def extract_criminal_history(self, text: str) -> Dict:
        """Extract criminal history."""
        if re.search(r'no\s+(?:criminal|prior)\s+(?:history|record)', text, re.IGNORECASE):
            return {
                "category": "first_time",
                "previous_cases": 0,
                "repeat_offender": False,
                "first_time_accused": True
            }
        
        if re.search(r'habitual\s+offender', text, re.IGNORECASE):
            return {
                "category": "habitual_offender",
                "previous_cases": None,
                "repeat_offender": True,
                "first_time_accused": False
            }
        
        return {
            "category": "unknown",
            "previous_cases": None,
            "repeat_offender": False,
            "first_time_accused": False
        }
    
    def extract_socioeconomic_status(self, text: str) -> str:
        """Extract socioeconomic indicators."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['poor', 'poverty', 'below poverty line']):
            return "low_income"
        
        return "unknown"
    
    def extract_region(self, text: str) -> Optional[str]:
        """Extract jurisdiction/region."""
        states = ["Maharashtra", "Delhi", "Karnataka", "Gujarat", "Punjab", "Haryana"]
        
        for state in states:
            if state in text:
                return state
        
        return None
    
    def extract_employment(self, text: str) -> Optional[str]:
        """Extract employment status."""
        text_lower = text.lower()
        
        if 'employed' in text_lower:
            return "employed"
        elif 'unemployed' in text_lower:
            return "unemployed"
        
        return None
    
    def extract_family_circumstances(self, text: str) -> Dict:
        """Extract family circumstances."""
        text_lower = text.lower()
        
        return {
            "has_dependents": 'dependent' in text_lower or 'family' in text_lower,
            "sole_breadwinner": 'sole breadwinner' in text_lower,
            "pregnant": 'pregnant' in text_lower,
            "elderly_parents": 'elderly parent' in text_lower,
            "minor_children": 'minor child' in text_lower
        }
