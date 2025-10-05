"""
HybridBail: Qdrant Database Interface
Multi-collection vector database management
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

class QdrantDatabase:
    """Interface for Qdrant vector database with multi-collection support."""
    
    def __init__(self, config):
        """Initialize Qdrant client."""
        self.config = config
        
        # Connect to Qdrant
        if config.QDRANT_URL:
            # Cloud instance
            self.client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY
            )
            logger.info(f"Connected to Qdrant Cloud: {config.QDRANT_URL}")
        else:
            # Local instance
            self.client = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT
            )
            logger.info(f"Connected to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    
    def create_collection(self, collection_name: str):
        """Create a new collection."""
        try:
            # Check if exists
            collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                logger.info(f"Collection '{collection_name}' already exists")
                return
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            raise
    
    def create_category_collections(self, categories: List[str]):
        """Create separate collections for each bail category."""
        for category in categories:
            collection_name = self.config.get_collection_name(category)
            self.create_collection(collection_name)
    
    def upsert(self, collection_name: str, vectors: List[List[float]], 
               payloads: List[Dict], ids: Optional[List[str]] = None):
        """Insert or update vectors in collection."""
        try:
            # Ensure collection exists
            self.create_collection(collection_name)
            
            # Generate IDs if not provided
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in vectors]
            
            # Create points
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} points to '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error upserting to '{collection_name}': {e}")
            raise
    
    def search(self, collection_name: str, query_vector: List[float], 
               limit: int = 10, score_threshold: Optional[float] = None) -> List[Dict]:
        """Search for similar vectors."""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload,
                    **result.payload  # Unpack payload for easy access
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching in '{collection_name}': {e}")
            return []
    
    def search_multi_collection(self, query_vector: List[float], 
                               categories: List[str], limit: int = 10) -> List[Dict]:
        """Search across multiple category collections."""
        all_results = []
        
        for category in categories:
            collection_name = self.config.get_collection_name(category)
            
            try:
                results = self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit
                )
                
                # Tag with category
                for result in results:
                    result['category'] = category
                
                all_results.extend(results)
                
            except Exception as e:
                logger.warning(f"Search in '{collection_name}' failed: {e}")
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:limit]
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a collection."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Error getting info for '{collection_name}': {e}")
            return {}
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
    
    def count_points(self, collection_name: str) -> int:
        """Count points in a collection."""
        try:
            info = self.get_collection_info(collection_name)
            return info.get('points_count', 0)
        except:
            return 0
