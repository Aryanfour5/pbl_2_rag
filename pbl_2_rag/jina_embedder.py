"""
HybridBail: Jina Embedder
Generates embeddings using Jina AI
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
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "input": texts,
            "model": self.model,
            "task": self.task,
            "dimensions": self.dimensions
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item['embedding'] for item in result['data']]
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Jina API call failed: {e}")
            raise
