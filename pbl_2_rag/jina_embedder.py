"""
HybridBail: Jina Embedder
Generates embeddings using Jina AI with improved error handling
"""

import requests
from typing import List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class JinaEmbedder:
    """Generates embeddings using Jina AI API."""
    
    def __init__(self, config):
        """Initialize with config object."""
        self.config = config
        self.api_key = config.JINA_API_KEY
        self.model = config.JINA_MODEL
        self.dimensions = config.JINA_DIMENSIONS
        self.task = config.JINA_TASK
        self.url = "https://api.jina.ai/v1/embeddings"
        
        # Token limits for Jina models
        self.max_tokens = 8192  # Conservative limit
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        # Check text length
        if len(text) > 30000:  # Rough char limit (4 chars ≈ 1 token)
            logger.warning(f"Text too long ({len(text)} chars), truncating to 30000")
            text = text[:30000]
        
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        
        # Truncate long texts
        processed_texts = []
        for text in texts:
            if len(text) > 30000:
                logger.warning(f"Truncating text from {len(text)} to 30000 chars")
                text = text[:30000]
            processed_texts.append(text)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "input": processed_texts,
            "model": self.model,
            "task": self.task,
            "dimensions": self.dimensions
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=60)
            
            # Better error handling
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"Jina API error {response.status_code}: {error_detail}")
                
                # Try to parse error message
                try:
                    error_json = response.json()
                    if 'detail' in error_json:
                        logger.error(f"Error detail: {error_json['detail']}")
                except:
                    pass
                
                raise Exception(f"Jina API returned {response.status_code}: {error_detail}")
            
            result = response.json()
            
            # Validate response
            if 'data' not in result:
                raise Exception(f"Unexpected API response format: {result}")
            
            embeddings = [item['embedding'] for item in result['data']]
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except requests.exceptions.Timeout:
            logger.error("Jina API request timed out")
            raise Exception("Jina API timeout - text may be too long")
        except requests.exceptions.RequestException as e:
            logger.error(f"Jina API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Jina API call failed: {e}")
            raise
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def chunk_for_embedding(self, text: str, max_chars: int = 30000) -> List[str]:
        """
        Chunk text if it exceeds limits.
        Returns list of text chunks suitable for embedding.
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]
            chunks.append(chunk)
            start = end
        
        logger.info(f"Split text into {len(chunks)} chunks for embedding")
        return chunks