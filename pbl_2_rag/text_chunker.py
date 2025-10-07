"""
HybridBail: Text Chunker
Splits text into overlapping chunks
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    """Chunks text into smaller pieces with overlap."""
    
    def __init__(self, config):
        """Initialize with config object."""
        self.config = config
        self.chunk_size = config.CHUNK_SIZE
        self.overlap_size = config.OVERLAP_SIZE
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        chunk_id = 0
        
        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if len(chunk_text.strip()) >= self.config.MIN_CHUNK_SIZE:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1
            
            start += (self.chunk_size - self.overlap_size)
        
        logger.info(f"Created {len(chunks)} chunks from text of length {text_length}")
        
        return chunks
