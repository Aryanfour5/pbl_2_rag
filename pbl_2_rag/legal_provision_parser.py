"""
HybridBail: Legal Provision Parser
Extracts IPC, CrPC, NDPS, POCSO, UAPA sections from case text
"""

import re
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

class LegalProvisionParser:
    """Parses legal provisions from case documents."""
    
    def __init__(self):
        # Patterns for different acts
        self.patterns = {
            'IPC': [
                r'Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?I\.?P\.?C',
                r'IPC\s+Section\s+(\d+[A-Z]?)',
                r'Section\s+(\d+[A-Z]?)\s+Indian\s+Penal\s+Code',
                r'(?:u/s|under section)\s+(\d+[A-Z]?)\s+IPC'
            ],
            'CrPC': [
                r'Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?Cr\.?P\.?C',
                r'CrPC\s+Section\s+(\d+[A-Z]?)',
                r'Section\s+(\d+[A-Z]?)\s+(?:of\s+)?Criminal\s+Procedure\s+Code',
                r'(?:u/s|under section)\s+(\d+[A-Z]?)\s+CrPC'
            ],
            'NDPS': [
                r'Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?N\.?D\.?P\.?S',
                r'NDPS\s+(?:Act\s+)?Section\s+(\d+[A-Z]?)',
                r'Section\s+(\d+[A-Z]?)\s+Narcotic\s+Drugs',
                r'(?:u/s|under section)\s+(\d+[A-Z]?)\s+NDPS'
            ],
            'POCSO': [
                r'Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?POCSO',
                r'POCSO\s+(?:Act\s+)?Section\s+(\d+[A-Z]?)',
                r'Section\s+(\d+[A-Z]?)\s+Protection\s+of\s+Children',
                r'(?:u/s|under section)\s+(\d+[A-Z]?)\s+POCSO'
            ],
            'UAPA': [
                r'Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?U\.?A\.?P\.?A',
                r'UAPA\s+Section\s+(\d+[A-Z]?)',
                r'Section\s+(\d+[A-Z]?)\s+Unlawful\s+Activities',
                r'(?:u/s|under section)\s+(\d+[A-Z]?)\s+UAPA'
            ],
            'SC_ST_ACT': [
                r'Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?SC[/\s]?ST\s+Act',
                r'SC[/\s]?ST\s+(?:Act\s+)?Section\s+(\d+[A-Z]?)',
                r'Scheduled\s+Castes.*?Section\s+(\d+[A-Z]?)'
            ]
        }
    
    def parse(self, text: str) -> Dict:
        """
        Parse legal provisions from text.
        
        Returns:
            {
                'provisions': {
                    'IPC': ['302', '307', ...],
                    'CrPC': ['437', '438', ...],
                    ...
                },
                'all_sections': ['IPC 302', 'CrPC 437', ...],
                'primary_statute': 'IPC' or 'NDPS' etc,
                'offense_nature': 'bailable' or 'non-bailable' or 'unknown'
            }
        """
        
        provisions = {
            'IPC': [],
            'CrPC': [],
            'NDPS': [],
            'POCSO': [],
            'UAPA': [],
            'SC_ST_ACT': [],
            'other': []
        }
        
        # Extract sections for each act
        for act, patterns in self.patterns.items():
            sections = self._extract_sections(text, patterns)
            provisions[act] = sorted(list(set(sections)))
        
        # Determine primary statute
        primary = self._determine_primary_statute(provisions)
        
        # Determine offense nature
        offense_nature = self._determine_offense_nature(provisions)
        
        # Create combined list
        all_sections = []
        for act, sections in provisions.items():
            if sections and act != 'other':
                all_sections.extend([f"{act} {s}" for s in sections])
        
        return {
            'provisions': provisions,
            'all_sections': all_sections,
            'primary_statute': primary,
            'offense_nature': offense_nature
        }
    
    def _extract_sections(self, text: str, patterns: List[str]) -> List[str]:
        """Extract section numbers using patterns."""
        sections = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                section = match.group(1)
                sections.append(section)
        
        return sections
    
    def _determine_primary_statute(self, provisions: Dict) -> str:
        """Determine primary statute based on sections present."""
        
        # Special acts take precedence
        if provisions['NDPS']:
            return 'NDPS'
        if provisions['POCSO']:
            return 'POCSO'
        if provisions['UAPA']:
            return 'UAPA'
        if provisions['SC_ST_ACT']:
            return 'SC_ST_ACT'
        
        # Otherwise IPC
        if provisions['IPC']:
            return 'IPC'
        
        # Fallback to CrPC if only procedural sections
        if provisions['CrPC']:
            return 'CrPC'
        
        return 'Unknown'
    
    def _determine_offense_nature(self, provisions: Dict) -> str:
        """Determine if offense is bailable or non-bailable."""
        
        # Non-bailable IPC sections (common ones)
        non_bailable_ipc = ['302', '304', '307', '376', '377', '395', '396', '397', '398', '399', '400', '401', '402']
        
        # Check IPC sections
        for section in provisions['IPC']:
            if section in non_bailable_ipc:
                return 'non-bailable'
        
        # Special acts are generally non-bailable
        if provisions['NDPS'] or provisions['POCSO'] or provisions['UAPA']:
            return 'non-bailable'
        
        # If only minor IPC sections
        if provisions['IPC'] and not any(s in non_bailable_ipc for s in provisions['IPC']):
            return 'bailable'
        
        return 'unknown'
