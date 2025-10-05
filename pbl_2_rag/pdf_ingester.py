"""
HybridBail: PDF Ingester
Extracts text from PDF documents
"""

import os
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class PDFIngester:
    """Ingests and processes PDF documents."""
    
    def __init__(self, config):
        """Initialize with config object."""
        self.config = config
        self.pdf_directory = Path(config.PDF_DIRECTORY)
        
        if not self.pdf_directory.exists():
            os.makedirs(self.pdf_directory, exist_ok=True)
            logger.info(f"Created PDF directory: {self.pdf_directory}")
    
    def ingest_directory(self, directory: str) -> List[Dict]:
        """Ingest all PDFs from a directory."""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return []
        
        pdf_files = list(directory_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.ingest_file(str(pdf_file))
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        return documents
    
    def ingest_file(self, pdf_path: str) -> Dict:
        """Extract text from a single PDF file."""
        try:
            text = self._extract_text(pdf_path)
            
            if not text or len(text.strip()) < 50:
                logger.warning(f"Insufficient text extracted from {pdf_path}")
                return None
            
            return {
                'filename': os.path.basename(pdf_path),
                'filepath': pdf_path,
                'text': text,
                'metadata': {
                    'source': pdf_path,
                    'pages': self._count_pages(pdf_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_path}: {e}")
            return None
    
    def _extract_text(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 and pdfplumber."""
        text = ""
        
        # Try pdfplumber first (better for complex PDFs)
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                return text
        except ImportError:
            logger.debug("pdfplumber not available, trying PyPDF2")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Fallback to PyPDF2
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
        
        return text
    
    def _count_pages(self, pdf_path: str) -> int:
        """Count number of pages in PDF."""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except:
            return 0
