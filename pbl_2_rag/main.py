#!/usr/bin/env python3
"""
Main execution file for local RAG preprocessing pipeline.
"""

import os
import sys
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

def demonstrate_hybrid_search(preprocessor: LocalRAGPreprocessor):
    """Demonstrate hybrid search capabilities with detailed comparison."""
    
    print("\n" + "="*80)
    print("🔍 HYBRID SEARCH DEMONSTRATION")
    print("="*80)
    
    # Test queries for legal documents
    test_queries = [
        "bail application requirements procedure",
        "surety conditions and eligibility", 
        "anticipatory bail section 438",
        "Supreme Court bail guidelines",
        "economic offenses bail restrictions"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing Query: '{query}'")
        print("-" * 60)
        
        # Get comparison of all search methods
        comparison = preprocessor.compare_search_methods(query, limit=3)
        
        # Display results for each method
        for method_name, results in comparison.items():
            print(f"\n📊 {method_name.upper()} SEARCH ({len(results)} results):")
            
            if not results:
                print("   ❌ No results found")
                continue
                
            for j, result in enumerate(results[:3], 1):  # Show top 3
                score = result.get('score', 0)
                filename = result.get('filename', 'Unknown')[:50]  # Truncate long filenames
                text_preview = result.get('text', '')[:120] + "..."
                
                print(f"   {j}. Score: {score:.4f} | File: {filename}")
                print(f"      Preview: {text_preview}")
                
                # Show detailed scores for hybrid results
                if method_name == "hybrid" and 'hybrid_score' in result:
                    sem_score = result.get('semantic_score', 0)
                    key_score = result.get('keyword_score', 0)
                    print(f"      Details: Semantic={sem_score:.3f}, Keyword={key_score:.3f}")
        
        print()

def run_interactive_search(preprocessor: LocalRAGPreprocessor):
    """Run interactive search session."""
    print("\n" + "="*80)
    print("🎯 INTERACTIVE HYBRID SEARCH")
    print("="*80)
    print("Enter queries to search your legal document database.")
    print("Commands:")
    print("  - Type your search query and press Enter")
    print("  - Use 'method:hybrid|semantic|keyword' to change search method")
    print("  - Use 'explain' after a search to get detailed explanations")
    print("  - Type 'quit' to exit")
    print("-" * 80)
    
    current_method = "hybrid"
    last_results = []
    
    while True:
        try:
            user_input = input(f"\n[{current_method}] 🔍 Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if user_input.lower() == 'explain' and last_results:
                print("\n📋 DETAILED EXPLANATIONS:")
                for i, result in enumerate(last_results[:3], 1):
                    print(f"\n{i}. Document: {result.get('filename', 'Unknown')}")
                    explanation = preprocessor.explain_result(last_query, result)
                    print(f"   Search Type: {explanation.get('search_type', 'unknown')}")
                    print(f"   Hybrid Score: {explanation['scores']['hybrid']:.4f}")
                    print(f"   - Semantic: {explanation['scores']['semantic']:.4f}")
                    print(f"   - Keyword: {explanation['scores']['keyword']:.4f}")
                    print(f"   Matching Keywords: {', '.join(explanation.get('matching_keywords', []))}")
                    print(f"   Keyword Coverage: {explanation.get('keyword_coverage', 0):.2%}")
                continue
                
            if user_input.startswith('method:'):
                new_method = user_input.split(':')[1].strip().lower()
                if new_method in ['hybrid', 'semantic', 'keyword']:
                    current_method = new_method
                    print(f"✅ Search method changed to: {current_method}")
                else:
                    print("❌ Invalid method. Use: hybrid, semantic, or keyword")
                continue
            
            if not user_input:
                continue
                
            # Perform search
            print(f"\n🔍 Searching with {current_method} method...")
            results = preprocessor.query(
                query_text=user_input,
                limit=5,
                search_type=current_method,
                score_threshold=0.1
            )
            
            if not results:
                print("❌ No results found. Try a different query or search method.")
                continue
            
            # Display results
            print(f"\n📊 Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                filename = result.get('filename', 'Unknown')
                text_preview = result.get('text', '')[:150].replace('\n', ' ') + "..."
                search_type = result.get('search_type', 'unknown')
                
                print(f"\n{i}. [{search_type.upper()}] Score: {score:.4f}")
                print(f"   📄 File: {filename}")
                print(f"   📝 Preview: {text_preview}")
                
                # Show additional details for hybrid results
                if current_method == "hybrid" and 'hybrid_score' in result:
                    sem = result.get('semantic_score', 0)
                    key = result.get('keyword_score', 0)
                    print(f"   🔢 Breakdown: Semantic={sem:.3f}, Keyword={key:.3f}")
            
            # Store results for explanation
            last_results = results
            last_query = user_input
            
            print(f"\n💡 Type 'explain' for detailed explanations or continue searching...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Search interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main function to run enhanced RAG preprocessing with hybrid search."""
    
    # Load environment variables
    load_dotenv()
    
    print("🚀 Enhanced RAG Pipeline with Hybrid Search")
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
        print("🔧 Initializing enhanced preprocessor with hybrid search...")
        preprocessor = LocalRAGPreprocessor(config)
        
        # Display PDF count
        pdf_count = preprocessor.get_pdf_count()
        print(f"📄 Found {pdf_count} PDF files in directory")
        
        if pdf_count == 0:
            print("❌ No PDF files found in the directory")
            return 1
        
        # Check if system is already processed
        metadata_file = f"./processing_metadata_{config.COLLECTION_NAME}.json"
        if os.path.exists(metadata_file):
            print("📋 Found existing metadata file. System appears to be already processed.")
            
            # Try to build keyword index if not already done
            try:
                if not preprocessor.hybrid_searcher.indexed:
                    print("🔧 Building keyword search index...")
                    preprocessor.hybrid_searcher.build_keyword_index()
                    print("✅ Keyword index built successfully!")
                else:
                    print("✅ Hybrid search system is ready!")
            except Exception as e:
                print(f"⚠️  Warning: Could not build keyword index: {str(e)}")
                print("You can still use semantic search.")
            
            # Ask user what they want to do
            print("\nChoose an option:")
            print("1. Run hybrid search demonstration")
            print("2. Start interactive search")
            print("3. Reprocess all documents (will take time)")
            print("4. Exit")
            
            try:
                choice = input("\nEnter your choice (1-4): ").strip()
                
                if choice == "1":
                    demonstrate_hybrid_search(preprocessor)
                elif choice == "2":
                    run_interactive_search(preprocessor)
                elif choice == "3":
                    print("⚡ Starting full preprocessing pipeline...")
                    # Continue to full processing below
                elif choice == "4":
                    print("👋 Goodbye!")
                    return 0
                else:
                    print("❌ Invalid choice. Starting interactive search...")
                    run_interactive_search(preprocessor)
                    return 0
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return 0
                
            if choice != "3":
                return 0
        
        # Run preprocessing pipeline
        print("⚡ Starting enhanced preprocessing pipeline...")
        result = preprocessor.process_documents()
        
        if result["success"]:
            print(f"✅ {result['message']}")
            print("\n📊 Processing Summary:")
            metadata = result['metadata']
            print(f"  • Source directory: {metadata['source_directory']}")
            print(f"  • Documents processed: {metadata['total_documents']}")
            print(f"  • Total chunks created: {metadata['total_chunks']}")
            print(f"  • Hybrid search enabled: {metadata.get('hybrid_search_enabled', False)}")
            print(f"  • Semantic weight: {metadata.get('semantic_weight', 0.7):.1%}")
            print(f"  • Document files:")
            for filename in metadata['document_files'][:10]:  # Show first 10
                print(f"    - {filename}")
            if len(metadata['document_files']) > 10:
                print(f"    ... and {len(metadata['document_files']) - 10} more files")
            
            # Demonstrate hybrid search
            demonstrate_hybrid_search(preprocessor)
            
            # Ask if user wants interactive search
            try:
                start_interactive = input("\n🎯 Start interactive search session? (y/n): ").strip().lower()
                if start_interactive in ['y', 'yes']:
                    run_interactive_search(preprocessor)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
            
            print(f"\n🎉 Enhanced RAG preprocessing with hybrid search completed successfully!")
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