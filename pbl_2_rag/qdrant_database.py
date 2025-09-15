import logging
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)

logger = logging.getLogger(__name__)

class QdrantVectorDB:
    """Handles vector database operations using Qdrant."""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6333,
                 url: str = None,
                 collection_name: str = "legal_documents",
                 vector_size: int = 1024):
        
        # Initialize client
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(host=host, port=port)
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Create collection
        self._create_collection()
        logger.info(f"Initialized Qdrant client for collection: {collection_name}")
    
    def _create_collection(self):
        """Create or recreate the collection."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE  # Good for text similarity
                )
            )
            logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def add_documents(self, embedded_chunks: List[Dict[str, Any]]):
        """Add embedded chunks to Qdrant."""
        points = []
        
        for chunk in embedded_chunks:
            # Prepare metadata (Qdrant payload)
            payload = {
                "text": chunk["text"],
                "filename": chunk["metadata"].get("filename", ""),
                "chunk_index": chunk["metadata"].get("chunk_index", 0),
                "chunk_size": chunk["metadata"].get("chunk_size", 0),
                "file_size": chunk["metadata"].get("file_size", 0),
                "num_pages": chunk["metadata"].get("num_pages", 0),
                "title": chunk["metadata"].get("title", ""),
                "author": chunk["metadata"].get("author", "")
            }
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),  # Generate unique UUID
                vector=chunk["embedding"],
                payload=payload
            )
            points.append(point)
        
        # Upload points in batches
        batch_size = 100
        total_points = len(points)
        
        for i in range(0, total_points, batch_size):
            batch_end = min(i + batch_size, total_points)
            batch_points = points[i:batch_end]
            
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(total_points-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
                raise
        
        logger.info(f"Successfully added {total_points} documents to Qdrant")
    
    def search(self, 
              query_embedding: List[float], 
              limit: int = 5,
              score_threshold: float = 0.7,
              filename_filter: str = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        
        # Build filter if filename specified
        query_filter = None
        if filename_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename_filter)
                    )
                ]
            )
        
        # Perform search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False  # Don't return vectors to save bandwidth
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "filename": result.payload.get("filename", ""),
                "chunk_index": result.payload.get("chunk_index", 0),
                "metadata": result.payload
            })
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
