#!/usr/bin/env python3
"""
Smart main.py that can work with existing metadata even if Qdrant is empty.
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv
from config import RAGConfig
from rag_preprocessor import LocalRAGPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_existing_metadata(config):
    """Load existing metadata file."""
    metadata_file = f"./processing_metadata_{config.COLLECTION_NAME}.json"
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Could not load metadata: {str(e)}")
        return None

def check_qdrant_data(preprocessor):
    """Check if Qdrant has actual data."""
    try:
        collection_info = preprocessor.vector_db.get_collection_info()
        points_count = collection_info.get('points_count', 0)
        return points_count > 0
    except Exception as e:
        logger.warning(f"Could not check Qdrant: {str(e)}")
        return False

def demonstrate_text_search_only(preprocessor):
    """Demonstrate text-only search when vector DB is empty."""
    print("\n🔍 TEXT-ONLY SEARCH DEMO")
    print("=" * 50)
    print("Since Qdrant is empty, we'll demo keyword search only")
    
    # Build text index from your PDF directory
    try:
        print("🔧 Building keyword search index from PDFs...")
        
        # Get documents from PDF directory directly
        documents = preprocessor.ingester.ingest_directory()
        print(f"📄 Loaded {len(documents)} documents")
        
        # Chunk them
        chunks = preprocessor.chunker.chunk_documents(documents)
        print(f"📝 Created {len(chunks)} chunks")
        
        # Build keyword index
        from hybrid_search import BM25Scorer
        bm25 = BM25Scorer()
        
        # Prepare documents for BM25
        bm25_docs = []
        for chunk in chunks:
            bm25_docs.append({
                'id': chunk['id'],
                'text': chunk['text'],
                'filename': chunk['metadata']['filename'],
                'chunk_index': chunk['metadata']['chunk_index']
            })
        
        bm25.build_index(bm25_docs)
        print("✅ Keyword search index built!")
        
        # Test queries
        test_queries = [
            # "bail application procedure",
            # "Supreme Court guidelines",
            # "surety conditions",
            # "economic offenses"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            scores = bm25.score(query)
            
            print(f"📊 Found {min(len(scores), 5)} results:")
            for i, (doc_idx, score) in enumerate(scores[:3], 1):
                if score > 0:
                    doc = bm25_docs[doc_idx]
                    filename = doc['filename'].replace('.PDF', '').replace('_', ' ')[:50]
                    text_preview = doc['text'][:100] + "..."
                    
                    print(f"  {i}. Score: {score:.4f}")
                    print(f"     Case: {filename}")
                    print(f"     Text: {text_preview}")
        
        return bm25, bm25_docs
        
    except Exception as e:
        print(f"❌ Could not build text search: {str(e)}")
        return None, None

def text_only_interactive_search(bm25, docs):
    """Interactive search with text-only (BM25)."""
    print("\n🎯 Hybrid KEYWORD SEARCH")
    print("=" * 50)
    print("Enter your queries")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Search
            scores = bm25.score(query)
            results = []
            
            for doc_idx, score in scores[:10]:  # Top 10
                if score > 0.1:  # Minimum score threshold
                    doc = docs[doc_idx]
                    results.append({
                        'score': score,
                        'filename': doc['filename'],
                        'text': doc['text'],
                        'chunk_index': doc['chunk_index']
                    })
            
            if not results:
                print("❌ No results found")
                continue
            
            print(f"\n📊 Found {len(results)} results:")
            for i, result in enumerate(results[:5], 1):
                score = result['score']
                filename = result['filename']
                case_name = filename.replace('.PDF', '').replace('_', ' ')[:100]
                text_preview = result['text'][:500] + "..."
                
                print(f"\n{i}. Score: {score:.4f}")
                print(f"   📄 Case: {case_name}")
                print(f"   📝 Preview: {text_preview}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main function with better fallback options."""
    
    load_dotenv()
    
    print("🚀 Smart RAG Pipeline with Hybrid Search")
    print("🐳 Docker Qdrant + Fallback Options")
    print("=" * 60)
    
    try:
        # Initialize
        config = RAGConfig.from_env()
        print(f"📁 PDF Directory: {config.PDF_DIRECTORY}")
        print(f"🗂️  Collection: {config.COLLECTION_NAME}")
        print(f"🐳 Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        
        preprocessor = LocalRAGPreprocessor(config)
        
        # Check existing data
        metadata = load_existing_metadata(config)
        has_qdrant_data = check_qdrant_data(preprocessor)
        
        print(f"\n🔍 Data Status:")
        print(f"   Metadata file: {'✅ Found' if metadata else '❌ Missing'}")
        print(f"   Qdrant data: {'✅ Available' if has_qdrant_data else '❌ Empty'}")
        
        if metadata and not has_qdrant_data:
            print(f"\n📋 Found metadata from previous processing:")
            print(f"   • {metadata.get('total_documents', 0)} documents processed")
            print(f"   • {metadata.get('total_chunks', 0)} chunks created")
            print(f"   • But Qdrant collection is empty (data lost)")
            
            print("\nOptions:")
            print("1. Use text-only search (fast, works immediately)")
            print("2. Reprocess to restore full vector search (slow)")
            print("3. Exit")
            
            choice = input("\nChoice (1-3): ").strip()
            
            if choice == "1":
                print("\n🔄 Setting up text-only search...")
                bm25, docs = demonstrate_text_search_only(preprocessor)
                
                if bm25 and docs:
                    interactive = input("\n🎯 Start interactive search? (y/n): ").strip().lower()
                    if interactive in ['y', 'yes']:
                        text_only_interactive_search(bm25, docs)
                return 0
                
            elif choice == "2":
                print("⚡ Will reprocess all documents...")
                # Continue to full processing
            else:
                return 0
        
        elif has_qdrant_data:
            print("✅ Full vector search available!")
            
            # Build hybrid search
            try:
                preprocessor.hybrid_searcher.build_keyword_index()
                print("✅ Hybrid search ready!")
                
                # Quick test
                results = preprocessor.query("bail application", limit=3)
                print(f"🔍 Test search: {len(results)} results found")
                
                # Interactive search
                interactive = input("\n🎯 Start interactive hybrid search? (y/n): ").strip().lower()
                if interactive in ['y', 'yes']:
                    # Use the interactive search from the previous version
                    print("\n🎯 INTERACTIVE HYBRID SEARCH")
                    print("Commands: 'method:hybrid/semantic/keyword', 'quit'")
                    
                    current_method = "hybrid"
                    while True:
                        try:
                            user_input = input(f"\n[{current_method}] 🔍 Query: ").strip()
                            
                            if user_input.lower() in ['quit', 'exit', 'q']:
                                break
                            
                            if user_input.startswith('method:'):
                                method = user_input.split(':')[1].strip()
                                if method in ['hybrid', 'semantic', 'keyword']:
                                    current_method = method
                                    print(f"✅ Changed to {method}")
                                continue
                            
                            if user_input:
                                results = preprocessor.query(user_input, limit=5, search_type=current_method)
                                print(f"\n📊 {len(results)} results:")
                                for i, r in enumerate(results[:3], 1):
                                    print(f"{i}. {r['score']:.4f} | {r['filename'][:40]}")
                        
                        except KeyboardInterrupt:
                            break
                
                return 0
                
            except Exception as e:
                print(f"⚠️  Hybrid search failed: {str(e)}")
                print("Falling back to processing...")
        
        # Full processing needed
        pdf_count = preprocessor.get_pdf_count()
        print(f"\n📄 Need to process {pdf_count} PDF files")
        
        if pdf_count == 0:
            print("❌ No PDF files found")
            return 1
        
        print(f"\n⚠️  This will take significant time for {pdf_count} documents.")
        proceed = input("Continue with full processing? (y/n): ").strip().lower()
        
        if proceed not in ['y', 'yes']:
            print("❌ Processing cancelled")
            return 0
        
        # Process
        result = preprocessor.process_documents()
        
        if result["success"]:
            print("✅ Processing completed!")
            metadata = result['metadata']
            print(f"📊 {metadata['total_documents']} docs → {metadata['total_chunks']} chunks")
            
            try_now = input("\n🎯 Try search now? (y/n): ").strip().lower()
            if try_now in ['y', 'yes']:
                results = preprocessor.query("bail application", limit=3)
                print(f"🔍 Test: {len(results)} results found")
                
                for i, r in enumerate(results, 1):
                    filename = r['filename'].replace('.PDF', '').replace('_', ' ')[:50]
                    print(f"{i}. {r['score']:.4f} | {filename}")
        
        else:
            print(f"❌ Processing failed: {result['message']}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        return 1
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())