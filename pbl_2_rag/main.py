import sys
import logging
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
                embeddings = self.embedder.embed_batch([c['text'] for c in chunks])
                
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
        
        # Step 5: Embed query
        query_embedding = self.embedder.embed(doc['text'])
        
        # Step 6: Retrieve similar cases
        similar_cases = self.search_engine.search(
            query_embedding=query_embedding,
            query_text=doc['text'],
            query_attributes=attributes,
            category=category,
            limit=10
        )
        logger.info(f"Retrieved {len(similar_cases)} similar cases")
        
        # Step 7: Analyze precedents
        precedent_analysis = self.precedent_analyzer.analyze(similar_cases)
        logger.info(f"Precedent analysis: {precedent_analysis['summary']}")
        
        # Step 8: Generate decision
        current_case = {
            'text': doc['text'],
            'category': category,
            'legal_provisions': legal,
            'attributes': attributes
        }
        
        decision = self.decision_engine.make_decision(current_case, similar_cases, category)
        logger.info(f"Decision: {decision['recommendation']} (confidence: {decision['confidence']:.2%})")
        
        # Step 9: Generate detailed reasoning with LLM
        reasoning = self.llm.generate_bail_reasoning(current_case, decision, similar_cases)
        
        # Compile final output
        output = {
            'case_summary': {
                'filename': doc['filename'],
                'category': BAIL_CATEGORIES[category]['name'],
                'language': self.multilingual.detect_language(doc['text'])
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
        result = pipeline.process_bail_application(pdf_path)
        
        print(f"\n{'='*60}")
        print(f"BAIL DECISION REPORT")
        print(f"{'='*60}")
        print(f"\nCategory: {result['case_summary']['category']}")
        print(f"Recommendation: {result['decision']['recommendation']}")
        print(f"Confidence: {result['decision']['confidence']:.1%}")
        print(f"\n{result['detailed_reasoning']}")
    
    elif choice == "3":
        print("Evaluation mode - implement test set loading")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
