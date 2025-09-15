import logging
import json
import os
from typing import Dict, Any, List
from config import RAGConfig
from pdf_ingester import PDFIngester
from text_chunker import TextChunker
from jina_embedder import JinaEmbedder
from qdrant_database import QdrantVectorDB

logger = logging.getLogger(__name__)

class LocalRAGPreprocessor:
    """RAG preprocessing pipeline for local PDF files with Jina embeddings and Qdrant."""
    
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
                "collection_info": collection_info
            }
            
            # Save metadata
            self._save_metadata(metadata)
            
            logger.info("RAG preprocessing pipeline completed successfully!")
            return {
                "success": True,
                "metadata": metadata,
                "message": f"Processed {len(documents)} documents from local directory into {len(chunks)} chunks"
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
              filename_filter: str = None) -> List[Dict[str, Any]]:
        """Query the RAG system."""
        logger.info(f"Querying: {query_text}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_query_embedding(query_text)
            
            # Search in Qdrant
            results = self.vector_db.search(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filename_filter=filename_filter
            )
            
            logger.info(f"Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []
    
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
            "pdf_count": self.get_pdf_count()
        }
