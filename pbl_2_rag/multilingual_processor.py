"""
HybridBail: Multilingual Processor
Language detection and cross-lingual support
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MultilingualProcessor:
    """Handles multilingual document processing and language detection."""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.ENABLE_MULTILINGUAL
        self.detector_type = None
        
        if self.enabled:
            try:
                from langdetect import detect, DetectorFactory
                DetectorFactory.seed = 0
                self.detector_type = "langdetect"
            except ImportError:
                logger.warning("langdetect not installed. Language detection disabled.")
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text."""
        
        if not self.enabled or not text:
            return self.config.DEFAULT_LANGUAGE
        
        try:
            if self.detector_type == "langdetect":
                from langdetect import detect
                lang_code = detect(text)
                return self._normalize_language_code(lang_code)
            else:
                return self._heuristic_detection(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return self.config.DEFAULT_LANGUAGE
    
    def _normalize_language_code(self, code: str) -> str:
        """Normalize language code to ISO 639-1."""
        mapping = {
            "hi": "hi",  # Hindi
            "en": "en",  # English
            "bn": "bn",  # Bengali
            "te": "te",  # Telugu
            "mr": "mr",  # Marathi
            "ta": "ta",  # Tamil
            "gu": "gu",  # Gujarati
            "kn": "kn",  # Kannada
        }
        return mapping.get(code, code)
    
    def _heuristic_detection(self, text: str) -> str:
        """Simple heuristic-based language detection."""
        
        # Check for Devanagari script (Hindi/Marathi)
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        if devanagari_count > len(text) * 0.3:
            return "hi"
        
        # Check for Bengali script
        bengali_count = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        if bengali_count > len(text) * 0.3:
            return "bn"
        
        return "en"
    
    def process_document(self, text: str, metadata: Dict) -> Dict:
        """Process document with language awareness."""
        
        detected_lang = self.detect_language(text)
        
        metadata['language'] = detected_lang
        metadata['multilingual_enabled'] = self.enabled
        metadata['requires_translation'] = False
        metadata['embedding_strategy'] = 'direct_multilingual'
        
        return {
            'text': text,
            'language': detected_lang,
            'metadata': metadata
        }
