#!/usr/bin/env python3
"""
Main execution file for local RAG preprocessing pipeline.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from config import RAGConfig
from rag_preprocessor import LocalRAGPreprocessor  # This matches your actual file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run RAG preprocessing for local PDFs."""
    
    # Load environment variables
    load_dotenv()
    
    print("🚀 Local RAG Preprocessing Pipeline with Jina + Qdrant")
    print("=" * 55)
    
    try:
        # Initialize config
        config = RAGConfig.from_env()
        
        # Display configuration
        print(f"📁 PDF Directory: {config.PDF_DIRECTORY}")
        print(f"🧠 Jina Model: {config.JINA_MODEL}")
        print(f"📊 Vector Dimensions: {config.JINA_DIMENSIONS}")
        print(f"🗂️  Collection: {config.COLLECTION_NAME}")
        print(f"📏 Chunk Size: {config.CHUNK_SIZE}")
        print(f"🔄 Overlap Size: {config.OVERLAP_SIZE}")
        print()
        
        # Check if directory exists
        if not os.path.exists(config.PDF_DIRECTORY):
            print(f"❌ PDF directory not found: {config.PDF_DIRECTORY}")
            print("Please check the path in your .env file")
            return 1
        
        # Initialize preprocessor
        print("🔧 Initializing preprocessor...")
        preprocessor = LocalRAGPreprocessor(config)
        
        # Display PDF count
        pdf_count = preprocessor.get_pdf_count()
        print(f"📄 Found {pdf_count} PDF files in directory")
        
        if pdf_count == 0:
            print("❌ No PDF files found in the directory")
            return 1
        
        # Run preprocessing pipeline
        print("⚡ Starting preprocessing pipeline...")
        result = preprocessor.process_documents()
        
        if result["success"]:
            print(f"✅ {result['message']}")
            print("\n📊 Processing Summary:")
            metadata = result['metadata']
            print(f"  • Source directory: {metadata['source_directory']}")
            print(f"  • Documents processed: {metadata['total_documents']}")
            print(f"  • Total chunks created: {metadata['total_chunks']}")
            print(f"  • Document files:")
            for filename in metadata['document_files'][:10]:  # Show first 10
                print(f"    - {filename}")
            if len(metadata['document_files']) > 10:
                print(f"    ... and {len(metadata['document_files']) - 10} more files")
            
            # Test queries
            print("\n🔍 Testing search functionality...")
            test_queries = [
                "bail application legal requirements",
                "surety conditions",
                "court procedures for bail"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. Query: '{query}'")
                results = preprocessor.query(query, limit=2, score_threshold=0.5)
                
                if results:
                    for j, result in enumerate(results, 1):
                        print(f"   {j}. Score: {result['score']:.3f} | File: {result['filename']}")
                        print(f"      Text: {result['text'][:100]}...")
                else:
                    print("   No results found")
            
            print(f"\n🎉 Local RAG preprocessing completed successfully!")
            print(f"📋 Metadata saved to: processing_metadata_{config.COLLECTION_NAME}.json")
            
        else:
            print(f"❌ Pipeline failed: {result['message']}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"💥 Unexpected error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
