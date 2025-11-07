import sys
import logging
import numpy as np
from pathlib import Path
from config import HybridBailConfig
from utils import setup_logging, ProgressTracker
from pdf_ingester import PDFIngester
from text_chunker import TextChunker
from jina_embedder import JinaEmbedder
from qdrant_database import QdrantDatabase
from legal_provision_parser import LegalProvisionParser
from attribute_extractor import AttributeExtractor
from bail_classifier import BailCategoryClassifier
from attribute_weighted_search import AttributeWeightedSearch
from hybrid_search import BM25Scorer
from decision_engine import BailDecisionEngine
from precedent_analyzer import PrecedentAnalyzer
from llm_integration import LLMIntegration
from multilingual_processor import MultilingualProcessor
from evaluator import HybridBailEvaluator
from constants import BAIL_CATEGORIES
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)

class HybridBailPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: HybridBailConfig):
        self.config = config
        
        # Initialize components
        logger.info("Initializing HybridBail components...")
        
        self.pdf_ingester = PDFIngester(config)
        self.text_chunker = TextChunker(config)
        self.embedder = JinaEmbedder(config)
        self.vector_db = QdrantDatabase(config)
        self.legal_parser = LegalProvisionParser()
        self.attribute_extractor = AttributeExtractor()
        self.classifier = BailCategoryClassifier(config)
        self.bm25_scorer = BM25Scorer()
        self.search_engine = AttributeWeightedSearch(self.vector_db, self.bm25_scorer, config)
        self.decision_engine = BailDecisionEngine(config)
        self.precedent_analyzer = PrecedentAnalyzer()
        self.llm = LLMIntegration(config)
        self.multilingual = MultilingualProcessor(config)
        self.evaluator = HybridBailEvaluator(config)
        
        logger.info("✓ All components initialized")
    
    def process_documents(self, category: str = None):
        """Process and index PDF documents."""
        
        categories = [category] if category else list(BAIL_CATEGORIES.keys())
        
        for cat in categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing category: {BAIL_CATEGORIES[cat]['name']}")
            logger.info(f"{'='*60}")
            
            # Get PDFs for this category
            pdf_dir = self.config.get_category_path(cat)
            if not Path(pdf_dir).exists():
                logger.warning(f"Directory not found: {pdf_dir}")
                continue
            
            documents = self.pdf_ingester.ingest_directory(pdf_dir)
            logger.info(f"Found {len(documents)} PDFs")
            
            if not documents:
                continue
            
            # Process each document
            tracker = ProgressTracker(len(documents), f"Processing {cat}")
            processed_docs = []
            
            for doc in documents:
                # Extract attributes
                legal = self.legal_parser.parse(doc['text'])
                attributes = self.attribute_extractor.extract_all(doc['text'])
                
                # Detect language
                lang_info = self.multilingual.process_document(doc['text'], doc.get('metadata', {}))
                
                # Chunk text
                chunks = self.text_chunker.chunk_text(doc['text'])
                
                # Embed chunks
                chunk_texts = [c['text'] for c in chunks]
                embeddings = self.embedder.embed_batch(chunk_texts)
                
                # Store in vector DB
                collection_name = self.config.get_collection_name(cat)
                for chunk, embedding in zip(chunks, embeddings):
                    chunk['legal_provisions'] = legal
                    chunk['attributes'] = attributes
                    chunk['language'] = lang_info['language']
                    chunk['category'] = cat
                    
                    self.vector_db.upsert(
                        collection_name=collection_name,
                        vectors=[embedding],
                        payloads=[chunk]
                    )
                
                processed_docs.append({
                    'filename': doc['filename'],
                    'category': cat,
                    'chunks': len(chunks)
                })
                
                tracker.update()
            
            tracker.finish()
            logger.info(f"✓ Completed {cat}: {len(processed_docs)} documents indexed")
    
    def process_bail_application(self, pdf_path: str) -> Dict:
        """Process a new bail application and generate decision."""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing bail application: {pdf_path}")
        logger.info(f"{'='*60}")
        
        # Step 1: Extract PDF
        doc = self.pdf_ingester.ingest_file(pdf_path)
        
        # Step 2: Parse legal provisions
        legal = self.legal_parser.parse(doc['text'])
        logger.info(f"Legal provisions: {legal['primary_statute']}")
        
        # Step 3: Extract attributes
        attributes = self.attribute_extractor.extract_all(doc['text'])
        logger.info(f"Accused profile: Age={attributes.get('age')}, Gender={attributes.get('gender')}")
        
        # Step 4: Classify category
        classification = self.classifier.classify(doc['text'], legal)
        category = classification['primary_category']
        logger.info(f"Category: {BAIL_CATEGORIES[category]['name']} (confidence: {classification['confidence_scores'][category]:.2f})")
        
        # Step 5: Chunk the text BEFORE embedding (THIS IS THE FIX!)
        logger.info("Chunking document for embedding...")
        chunks = self.text_chunker.chunk_text(doc['text'])
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 6: Embed chunks and aggregate
        logger.info("Generating embeddings for chunks...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Batch embed chunks to avoid overwhelming the API
        chunk_embeddings = []
        batch_size = 10  # Process 10 chunks at a time
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            try:
                batch_embeddings = self.embedder.embed_batch(batch)
                chunk_embeddings.extend(batch_embeddings)
                logger.info(f"Embedded batch {i//batch_size + 1}/{(len(chunk_texts)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                raise
        
        # Aggregate embeddings using average pooling
        query_embedding = np.mean(chunk_embeddings, axis=0).tolist()
        logger.info(f"Aggregated {len(chunk_embeddings)} chunk embeddings into query vector")
        
        # Step 7: Retrieve similar cases
        logger.info("Searching for similar precedents...")
        similar_cases = self.search_engine.search(
            query_embedding=query_embedding,
            query_text=doc['text'][:2000],  # Use first 2000 chars for keyword search
            query_attributes=attributes,
            category=category,
            limit=10
        )
        logger.info(f"Retrieved {len(similar_cases)} similar cases")
        
        # Step 8: Analyze precedents
        precedent_analysis = self.precedent_analyzer.analyze(similar_cases)
        logger.info(f"Precedent analysis: {precedent_analysis['summary']}")
        
        # Step 9: Generate decision
        current_case = {
            'text': doc['text'],
            'category': category,
            'legal_provisions': legal,
            'attributes': attributes
        }
        
        decision = self.decision_engine.make_decision(current_case, similar_cases, category)
        logger.info(f"Decision: {decision['recommendation']} (confidence: {decision['confidence']:.2%})")
        
        # Step 10: Generate detailed reasoning with LLM
        logger.info("Generating detailed legal reasoning...")
        reasoning = self.llm.generate_bail_reasoning(current_case, decision, similar_cases)
        
        # Compile final output
        output = {
            'case_summary': {
                'filename': doc['filename'],
                'category': BAIL_CATEGORIES[category]['name'],
                'language': self.multilingual.detect_language(doc['text'][:1000])
            },
            'legal_analysis': legal,
            'accused_profile': attributes,
            'similar_precedents': similar_cases[:5],
            'precedent_analysis': precedent_analysis,
            'decision': decision,
            'detailed_reasoning': reasoning
        }
        
        return output

def main():
    """Main entry point."""
    
    # Load configuration
    config = HybridBailConfig.from_env()
    
    # Setup logging
    setup_logging(config.LOG_LEVEL, config.LOG_FILE)
    
    logger.info("=" * 80)
    logger.info("HybridBail: District Court Bail Decision Support System")
    logger.info("=" * 80)
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Initialize pipeline
    pipeline = HybridBailPipeline(config)
    
    # Interactive menu
    print("\nHybridBail Menu:")
    print("1. Process and index documents")
    print("2. Process bail application")
    print("3. Run evaluation")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        pipeline.process_documents()
        print("\n✓ Document processing complete!")
    
    elif choice == "2":
        pdf_path = input("Enter path to bail application PDF: ").strip()
        
        try:
            result = pipeline.process_bail_application(pdf_path)
            
            print(f"\n{'='*60}")
            print(f"BAIL DECISION REPORT")
            print(f"{'='*60}")
            print(f"\nCase: {result['case_summary']['filename']}")
            print(f"Category: {result['case_summary']['category']}")
            print(f"Language: {result['case_summary']['language']}")
            print(f"\n--- Legal Analysis ---")
            print(f"Primary Statute: {result['legal_analysis']['primary_statute']}")
            print(f"Sections: {', '.join(result['legal_analysis']['all_sections'][:5])}")
            print(f"Offense Nature: {result['legal_analysis']['offense_nature']}")
            print(f"\n--- Accused Profile ---")
            print(f"Age: {result['accused_profile'].get('age', 'Unknown')}")
            print(f"Gender: {result['accused_profile'].get('gender', 'Unknown')}")
            print(f"Criminal History: {result['accused_profile'].get('criminal_history', {}).get('category', 'Unknown')}")
            print(f"\n--- Decision ---")
            print(f"Recommendation: {result['decision']['recommendation']}")
            print(f"Confidence: {result['decision']['confidence']:.1%}")
            print(f"Similar Cases Analyzed: {result['decision']['similar_cases_count']}")
            if result['decision']['needs_human_review']:
                print(f"\n⚠️  HUMAN REVIEW REQUIRED")
                print(f"Reasons: {', '.join(result['decision']['intervention_reasons'])}")
            print(f"\n--- Precedent Analysis ---")
            print(f"{result['precedent_analysis']['summary']}")
            print(f"\n--- Detailed Reasoning ---")
            print(result['detailed_reasoning'])
            
        except Exception as e:
            logger.error(f"Error processing bail application: {e}", exc_info=True)
            print(f"\n❌ Error: {e}")
    
    elif choice == "3":
        print("Evaluation mode - implement test set loading")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())