import logging
import json
import os
from typing import Dict, Any, List
from config import RAGConfig
from pdf_ingester import PDFIngester
from text_chunker import TextChunker
from jina_embedder import JinaEmbedder
from qdrant_database import QdrantVectorDB
from hybrid_search import HybridSearcher  

logger = logging.getLogger(__name__)

class LocalRAGPreprocessor:
    """RAG preprocessing pipeline for local PDF files with Jina embeddings, Qdrant, and Hybrid Search."""
    
    def __init__(self, config: RAGConfig = None):
        if config is None:
            config = RAGConfig.from_env()
        
        # Validate configuration
        config.validate()
        self.config = config
        
        # Initialize components (no Google Drive downloader needed)
        self.ingester = PDFIngester(config.PDF_DIRECTORY)
        self.chunker = TextChunker(config.CHUNK_SIZE, config.OVERLAP_SIZE)
        
        # Initialize Jina embedder
        self.embedder = JinaEmbedder(
            api_key=config.JINA_API_KEY,
            model=config.JINA_MODEL,
            dimensions=config.JINA_DIMENSIONS,
            task=config.JINA_TASK
        )
        
        # Initialize Qdrant
        self.vector_db = QdrantVectorDB(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            url=config.QDRANT_URL,
            collection_name=config.COLLECTION_NAME,
            vector_size=config.JINA_DIMENSIONS
        )
        
        # Initialize Hybrid Searcher - ADD THIS
        self.hybrid_searcher = HybridSearcher(
            vector_db=self.vector_db,
            embedder=self.embedder,
            alpha=0.7  # 70% semantic, 30% keyword
        )
    
    def get_pdf_count(self) -> int:
        """Get the number of PDF files in the directory."""
        pdf_files = []
        for root, dirs, files in os.walk(self.config.PDF_DIRECTORY):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return len(pdf_files)
    
    def process_documents(self) -> Dict[str, Any]:
        """Run the complete preprocessing pipeline for local PDFs."""
        logger.info("Starting RAG preprocessing pipeline for local PDFs with Jina + Qdrant...")
        
        try:
            # Display folder info
            pdf_count = self.get_pdf_count()
            logger.info(f"Found {pdf_count} PDF files in: {self.config.PDF_DIRECTORY}")
            
            if pdf_count == 0:
                return {"success": False, "message": "No PDF files found in the specified directory"}
            
            # Step 1: Ingest PDFs from local directory
            logger.info("Step 1: Ingesting PDF documents from local directory...")
            documents = self.ingester.ingest_directory()
            
            if not documents:
                return {"success": False, "message": "No documents were successfully ingested"}
            
            # Step 2: Chunk documents
            logger.info("Step 2: Chunking documents...")
            chunks = self.chunker.chunk_documents(documents)
            
            if not chunks:
                return {"success": False, "message": "No chunks were created"}
            
            # Step 3: Generate embeddings with Jina
            logger.info("Step 3: Generating embeddings with Jina AI...")
            embedded_chunks = self.embedder.generate_embeddings(chunks, self.config.BATCH_SIZE)
            
            # Step 4: Store in Qdrant
            logger.info("Step 4: Storing in Qdrant vector database...")
            self.vector_db.add_documents(embedded_chunks)
            
            # Step 5: Build keyword index for hybrid search - ADD THIS
            logger.info("Step 5: Building keyword search index...")
            try:
                self.hybrid_searcher.build_keyword_index()
                hybrid_enabled = True
            except Exception as e:
                logger.warning(f"Could not build keyword index: {str(e)}")
                hybrid_enabled = False
            
            # Get final collection info
            collection_info = self.vector_db.get_collection_info()
            
            # Prepare metadata
            document_files = [doc["metadata"]["filename"] for doc in documents]
            metadata = {
                "source_directory": self.config.PDF_DIRECTORY,
                "total_documents": len(documents),
                "document_files": document_files,
                "total_chunks": len(chunks),
                "chunk_size": self.config.CHUNK_SIZE,
                "overlap_size": self.config.OVERLAP_SIZE,
                "jina_model": self.config.JINA_MODEL,
                "embedding_dimension": self.config.JINA_DIMENSIONS,
                "collection_info": collection_info,
                "hybrid_search_enabled": hybrid_enabled  # ADD THIS
            }
            
            # Save metadata
            self._save_metadata(metadata)
            
            success_msg = f"Processed {len(documents)} documents from local directory into {len(chunks)} chunks"
            if hybrid_enabled:
                success_msg += " with hybrid search enabled"
            
            logger.info("RAG preprocessing pipeline completed successfully!")
            return {
                "success": True,
                "metadata": metadata,
                "message": success_msg
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {"success": False, "message": f"Pipeline failed: {str(e)}"}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save processing metadata to file."""
        try:
            metadata_path = f"./processing_metadata_{self.config.COLLECTION_NAME}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {str(e)}")
    
    def query(self, query_text: str, limit: int = 5, score_threshold: float = 0.7, 
              filename_filter: str = None, search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Query the RAG system with hybrid search options.
        
        Args:
            query_text: Search query
            limit: Number of results to return  
            score_threshold: Minimum score threshold
            filename_filter: Filter by specific filename
            search_type: 'hybrid', 'semantic', or 'keyword'
        """
        logger.info(f"Querying ({search_type}): {query_text}")
        
        try:
            if search_type == "hybrid":
                # Use hybrid search
                if not self.hybrid_searcher.indexed:
                    logger.info("Building keyword index for hybrid search...")
                    self.hybrid_searcher.build_keyword_index()
                
                results = self.hybrid_searcher.search(
                    query=query_text,
                    limit=limit,
                    score_threshold=score_threshold,
                    filename_filter=filename_filter
                )
                
            elif search_type == "semantic":
                # Use only semantic search (your original method)
                query_embedding = self.embedder.generate_query_embedding(query_text)
                results = self.vector_db.search(
                    query_embedding=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold,
                    filename_filter=filename_filter
                )
                # Add search type info
                for result in results:
                    result['search_type'] = 'semantic'
                    
            elif search_type == "keyword":
                # Use only keyword search
                if not self.hybrid_searcher.indexed:
                    self.hybrid_searcher.build_keyword_index()
                
                bm25_scores = self.hybrid_searcher.bm25.score(query_text)
                results = []
                
                for doc_idx, bm25_score in bm25_scores[:limit]:
                    if bm25_score > score_threshold:
                        metadata = self.hybrid_searcher.bm25.metadata[doc_idx]
                        
                        if filename_filter and metadata['filename'] != filename_filter:
                            continue
                        
                        doc_text = ' '.join(self.hybrid_searcher.bm25.corpus[doc_idx])
                        
                        results.append({
                            'id': metadata['id'],
                            'score': bm25_score,
                            'text': doc_text,
                            'filename': metadata['filename'],
                            'chunk_index': metadata['chunk_index'],
                            'search_type': 'keyword'
                        })
            else:
                # Default to original semantic search for backward compatibility
                query_embedding = self.embedder.generate_query_embedding(query_text)
                results = self.vector_db.search(
                    query_embedding=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold,
                    filename_filter=filename_filter
                )
            
            logger.info(f"Found {len(results)} relevant chunks using {search_type} search")
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []
    
    # ADD THESE NEW METHODS
    def explain_result(self, query_text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get explanation for why a result was returned."""
        try:
            return self.hybrid_searcher.explain_search(query_text, result)
        except Exception as e:
            logger.warning(f"Could not explain result: {str(e)}")
            return {"error": str(e)}
    
    def compare_search_methods(self, query_text: str, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Compare results from different search methods."""
        comparison = {}
        
        for search_type in ["hybrid", "semantic", "keyword"]:
            try:
                results = self.query(query_text, limit=limit, search_type=search_type, score_threshold=0.1)
                comparison[search_type] = results
            except Exception as e:
                logger.warning(f"Failed to get {search_type} results: {str(e)}")
                comparison[search_type] = []
        
        return comparison
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status and information."""
        return {
            "config": {
                "source_directory": self.config.PDF_DIRECTORY,
                "collection_name": self.config.COLLECTION_NAME,
                "jina_model": self.config.JINA_MODEL,
                "embedding_dimensions": self.config.JINA_DIMENSIONS,
                "chunk_size": self.config.CHUNK_SIZE
            },
            "collection_info": self.vector_db.get_collection_info(),
            "pdf_count": self.get_pdf_count(),
            "hybrid_search": {
                "enabled": hasattr(self, 'hybrid_searcher'),
                "indexed": getattr(self.hybrid_searcher, 'indexed', False),
                "alpha": getattr(self.hybrid_searcher, 'alpha', 0.7)
            }
        }