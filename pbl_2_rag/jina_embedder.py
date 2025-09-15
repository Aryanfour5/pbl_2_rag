import requests
import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class JinaEmbedder:
    """Handles text embedding generation using Jina AI API."""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "jina-embeddings-v3",
                 dimensions: int = 1024,
                 task: str = "retrieval.passage"):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.task = task
        self.api_url = "https://api.jina.ai/v1/embeddings"
        
        if not api_key:
            raise ValueError("Jina API key is required. Get one from https://jina.ai/embeddings/")
        
        logger.info(f"Initialized Jina embeddings with model: {model}")
    
    def _make_api_request(self, texts: List[str], retries: int = 3) -> List[List[float]]:
        """Make API request to Jina with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "input": texts,
            "model": self.model,
            "dimensions": self.dimensions,
            "task": self.task,
            "late_chunking": True  # Enable contextual chunking
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                return embeddings
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return []
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 20) -> List[Dict[str, Any]]:
        """Generate embeddings for text chunks in batches."""
        texts = [chunk["text"] for chunk in chunks]
        total_chunks = len(texts)
        
        logger.info(f"Generating Jina embeddings for {total_chunks} chunks...")
        
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch_texts = texts[i:batch_end]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
            
            try:
                batch_embeddings = self._make_api_request(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting - be respectful to the API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to get embeddings for batch {i//batch_size + 1}: {str(e)}")
                # Add empty embeddings for failed batch
                all_embeddings.extend([[0.0] * self.dimensions] * len(batch_texts))
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = {
                **chunk,
                "embedding": all_embeddings[i]
            }
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"Generated embeddings with dimension: {self.dimensions}")
        return embedded_chunks
    
    def generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for a single query."""
        # Use retrieval.query task for queries
        original_task = self.task
        self.task = "retrieval.query"
        
        try:
            embedding = self._make_api_request([query_text])[0]
            return embedding
        finally:
            self.task = original_task
