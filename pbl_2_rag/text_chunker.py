import re
import hashlib
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TextChunker:
    """Handles intelligent text chunking with overlap and semantic boundaries."""
    
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with chunking
        text = re.sub(r'[^\w\s\.\!\?\;:,\-\(\)]', ' ', text)
        return text.strip()
    
    def sentence_aware_chunking(self, text: str) -> List[str]:
        """Split text into chunks while preserving sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap by keeping the last few sentences
                if self.overlap_size > 0:
                    overlap_sentences = current_chunk.split('. ')[-2:]  # Keep last 2 sentences
                    current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single document into smaller pieces."""
        text = self.clean_text(document["text"])
        chunks = self.sentence_aware_chunking(text)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very small chunks
                continue
                
            chunk_id = hashlib.md5(f"{document['metadata']['filename']}_{i}".encode()).hexdigest()
            
            chunked_doc = {
                "id": chunk_id,
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "metadata": {
                    **document["metadata"],
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "chunk_size": len(chunk)
                }
            }
            chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {doc['metadata']['filename']}")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
