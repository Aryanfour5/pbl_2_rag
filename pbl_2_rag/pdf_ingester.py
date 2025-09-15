import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from pypdf import PdfReader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFIngester:
    """Handles PDF document ingestion and text extraction."""
    
    def __init__(self, pdf_directory: str):
        self.pdf_directory = Path(pdf_directory)
        self.processed_docs = []
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text from a single PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text_content = ""
            metadata = {
                "filename": pdf_path.name,
                "path": str(pdf_path),
                "num_pages": len(reader.pages),
                "file_size": pdf_path.stat().st_size
            }
            
            # Extract text from all pages
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Extract PDF metadata if available
            if reader.metadata:
                pdf_metadata = {
                    "title": reader.metadata.get('/Title', ''),
                    "author": reader.metadata.get('/Author', ''),
                    "subject": reader.metadata.get('/Subject', ''),
                    "creator": reader.metadata.get('/Creator', ''),
                    "creation_date": str(reader.metadata.get('/CreationDate', ''))
                }
                metadata.update(pdf_metadata)
            
            return {
                "text": text_content.strip(),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return None
    
    def ingest_directory(self) -> List[Dict[str, Any]]:
        """Ingest all PDF files from the specified directory."""
        if not self.pdf_directory.exists():
            logger.error(f"Directory {self.pdf_directory} does not exist")
            return []
        
        pdf_files = list(self.pdf_directory.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = []
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            doc_data = self.extract_text_from_pdf(pdf_file)
            if doc_data and doc_data["text"]:
                documents.append(doc_data)
                logger.info(f"Successfully processed: {pdf_file.name}")
            else:
                logger.warning(f"Failed to extract text from: {pdf_file.name}")
        
        self.processed_docs = documents
        logger.info(f"Successfully ingested {len(documents)} documents")
        return documents
