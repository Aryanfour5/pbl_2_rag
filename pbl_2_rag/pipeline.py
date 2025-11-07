import logging
from pathlib import Path
from config import HybridBailConfig
from utils import ProgressTracker
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
from constants import BAIL_CATEGORIES
from typing import Dict, List

logger = logging.getLogger(__name__)


class HybridBailPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: HybridBailConfig):
        self.config = config
        
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
        
        logger.info("✓ All components initialized")
    
    def process_documents(self, category: str = None):
        """Process and index PDF documents."""
        
        categories = [category] if category else list(BAIL_CATEGORIES.keys())
        
        for cat in categories:
            logger.info(f"Processing category: {BAIL_CATEGORIES[cat]['name']}")
            
            pdf_dir = self.config.get_category_path(cat)
            if not Path(pdf_dir).exists():
                logger.warning(f"Directory not found: {pdf_dir}")
                continue
            
            documents = self.pdf_ingester.ingest_directory(pdf_dir)
            logger.info(f"Found {len(documents)} PDFs")
            
            if not documents:
                continue
            
            tracker = ProgressTracker(len(documents), f"Processing {cat}")
            
            for doc in documents:
                legal = self.legal_parser.parse(doc['text'])
                attributes = self.attribute_extractor.extract_all(doc['text'])
                lang_info = self.multilingual.process_document(doc['text'], doc.get('metadata', {}))
                chunks = self.text_chunker.chunk_text(doc['text'])
                embeddings = self.embedder.embed_batch([c['text'] for c in chunks])
                
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
                
                tracker.update()
            
            tracker.finish()
            logger.info(f"✓ Completed {cat}")
    
    def process_bail_application(self, pdf_path: str) -> Dict:
        """Process a new bail application and generate decision."""
        
        logger.info(f"Processing bail application: {pdf_path}")
        
        doc = self.pdf_ingester.ingest_file(pdf_path)
        legal = self.legal_parser.parse(doc['text'])
        attributes = self.attribute_extractor.extract_all(doc['text'])
        classification = self.classifier.classify(doc['text'], legal)
        category = classification['primary_category']
        
        query_embedding = self.embedder.embed(doc['text'])
        
        similar_cases = self.search_engine.search(
            query_embedding=query_embedding,
            query_text=doc['text'],
            query_attributes=attributes,
            category=category,
            limit=10
        )
        
        precedent_analysis = self.precedent_analyzer.analyze(similar_cases)
        
        current_case = {
            'text': doc['text'],
            'category': category,
            'legal_provisions': legal,
            'attributes': attributes
        }
        
        decision = self.decision_engine.make_decision(current_case, similar_cases, category)
        reasoning = self.llm.generate_bail_reasoning(current_case, decision, similar_cases)
        
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
