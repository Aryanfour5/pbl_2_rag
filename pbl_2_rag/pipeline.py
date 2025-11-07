# pipeline.py

"""
HybridBail Pipeline: Complete Orchestration
End-to-end bail decision support system
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

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

logger = logging.getLogger(__name__)


class HybridBailPipeline:
    """Main pipeline orchestrator for bail decision support."""
    
    def __init__(self, config: HybridBailConfig):
        self.config = config
        
        logger.info("=" * 80)
        logger.info("Initializing HybridBail Pipeline Components")
        logger.info("=" * 80)
        
        try:
            # Initialize components in order
            self.pdf_ingester = PDFIngester(config)
            logger.info("✓ PDF Ingester initialized")
            
            self.text_chunker = TextChunker(config)
            logger.info("✓ Text Chunker initialized (chunk_size: {}, overlap: {})".format(
                config.CHUNK_SIZE, config.CHUNK_OVERLAP))
            
            self.embedder = JinaEmbedder(config)
            logger.info("✓ Jina Embedder initialized (768D vectors)")
            
            self.vector_db = QdrantDatabase(config)
            logger.info("✓ Qdrant Vector Database initialized")
            
            self.legal_parser = LegalProvisionParser()
            logger.info("✓ Legal Provision Parser initialized")
            
            self.attribute_extractor = AttributeExtractor()
            logger.info("✓ Attribute Extractor initialized")
            
            self.classifier = BailCategoryClassifier(config)
            logger.info("✓ Bail Category Classifier initialized")
            
            self.bm25_scorer = BM25Scorer()
            logger.info("✓ BM25 Scorer initialized")
            
            self.search_engine = AttributeWeightedSearch(
                self.vector_db, 
                self.bm25_scorer, 
                config
            )
            logger.info("✓ Attribute Weighted Search initialized")
            
            self.decision_engine = BailDecisionEngine(config)
            logger.info("✓ Bail Decision Engine initialized")
            
            self.precedent_analyzer = PrecedentAnalyzer()
            logger.info("✓ Precedent Analyzer initialized")
            
            self.llm = LLMIntegration(config)
            logger.info("✓ LLM Integration initialized")
            
            self.multilingual = MultilingualProcessor(config)
            logger.info("✓ Multilingual Processor initialized")
            
            self.evaluator = HybridBailEvaluator(config)
            logger.info("✓ Evaluator initialized")
            
            logger.info("=" * 80)
            logger.info("✅ All components initialized successfully")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}", exc_info=True)
            raise
    
    # ==================== DOCUMENT PROCESSING ====================
    
    def process_documents(self, category: str = None):
        """
        Process and index PDF documents from directories.
        
        Workflow:
        1. Extract PDF text
        2. Parse legal provisions
        3. Extract case attributes
        4. Chunk text (400 words + 50 overlap)
        5. Generate embeddings (768D)
        6. Store in Qdrant
        """
        
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
            
            # Ingest all PDFs
            logger.info(f"📂 Reading PDFs from: {pdf_dir}")
            documents = self.pdf_ingester.ingest_directory(pdf_dir)
            logger.info(f"Found {len(documents)} PDF documents")
            
            if not documents:
                logger.warning(f"No documents found in {pdf_dir}")
                continue
            
            # ===== MAIN PROCESSING LOOP =====
            tracker = ProgressTracker(len(documents), f"Processing {cat}")
            processed_docs = []
            
            for doc in documents:
                try:
                    # Step 1: Parse legal provisions
                    legal = self.legal_parser.parse(doc['text'])
                    
                    # Step 2: Extract attributes
                    attributes = self.attribute_extractor.extract_all(doc['text'])
                    
                    # Step 3: Detect language
                    lang_info = self.multilingual.process_document(
                        doc['text'], 
                        doc.get('metadata', {})
                    )
                    
                    # Step 4: CHUNK TEXT FIRST (THIS IS CRITICAL!)
                    logger.debug(f"  📄 Chunking: {doc['filename']}")
                    chunks = self.text_chunker.chunk_text(doc['text'])
                    logger.debug(f"  📦 Created {len(chunks)} chunks")
                    
                    # Step 5: Extract chunk texts for batch embedding
                    chunk_texts = [chunk['text'] for chunk in chunks]
                    
                    # Step 6: Batch embed chunks (avoid API overload)
                    logger.debug(f"  🔢 Embedding {len(chunk_texts)} chunks...")
                    chunk_embeddings = self._batch_embed(chunk_texts, batch_size=10)
                    
                    # Step 7: Store in Qdrant
                    collection_name = self.config.get_collection_name(cat)
                    
                    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                        # Enrich chunk with metadata
                        chunk['legal_provisions'] = legal
                        chunk['attributes'] = attributes
                        chunk['language'] = lang_info['language']
                        chunk['category'] = cat
                        chunk['document_name'] = doc['filename']
                        chunk['chunk_index'] = i
                        
                        # Store in Qdrant
                        self.vector_db.upsert(
                            collection_name=collection_name,
                            vectors=[embedding],
                            payloads=[chunk],
                            ids=[f"{doc['filename']}_chunk_{i}"]
                        )
                    
                    processed_docs.append({
                        'filename': doc['filename'],
                        'category': cat,
                        'chunks': len(chunks),
                        'legal_statute': legal['primary_statute']
                    })
                    
                    tracker.update()
                    
                except Exception as e:
                    logger.error(f"Error processing {doc['filename']}: {e}")
                    tracker.update()
                    continue
            
            tracker.finish()
            logger.info(f"✅ Completed {cat}: {len(processed_docs)} documents indexed")
    
    # ==================== BAIL APPLICATION PROCESSING ====================
    
    def process_bail_application(self, pdf_path: str) -> Dict:
        """
        Process a new bail application and generate comprehensive decision.
        
        Complete workflow:
        1. PDF extraction
        2. Legal analysis
        3. Attribute extraction
        4. Category classification
        5. Chunking and embedding
        6. Precedent search
        7. Decision generation
        8. LLM reasoning
        
        Args:
            pdf_path: Path to bail application PDF
            
        Returns:
            Comprehensive bail decision report
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing bail application: {Path(pdf_path).name}")
        logger.info(f"{'='*60}")
        
        try:
            # ===== STEP 1: EXTRACT PDF =====
            logger.info("Step 1: Extracting PDF...")
            doc = self.pdf_ingester.ingest_file(pdf_path)
            logger.info(f"  ✓ Extracted {len(doc['text'].split())} words from PDF")
            
            # ===== STEP 2: PARSE LEGAL PROVISIONS =====
            logger.info("Step 2: Parsing legal provisions...")
            legal = self.legal_parser.parse(doc['text'])
            logger.info(f"  ✓ Primary statute: {legal['primary_statute']}")
            logger.info(f"  ✓ Sections: {', '.join(legal['all_sections'][:3])}")
            logger.info(f"  ✓ Offense nature: {legal['offense_nature']}")
            
            # ===== STEP 3: EXTRACT ATTRIBUTES =====
            logger.info("Step 3: Extracting case attributes...")
            attributes = self.attribute_extractor.extract_all(doc['text'])
            logger.info(f"  ✓ Accused age: {attributes.get('age', 'Unknown')}")
            logger.info(f"  ✓ Gender: {attributes.get('gender', 'Unknown')}")
            logger.info(f"  ✓ Criminal history: {attributes.get('criminal_history', {}).get('category', 'Unknown')}")
            
            # ===== STEP 4: CLASSIFY CATEGORY =====
            logger.info("Step 4: Classifying bail category...")
            classification = self.classifier.classify(doc['text'], legal)
            category = classification['primary_category']
            confidence = classification['confidence_scores'].get(category, 0)
            logger.info(f"  ✓ Category: {BAIL_CATEGORIES[category]['name']}")
            logger.info(f"  ✓ Classification confidence: {confidence:.2%}")
            
            # ===== STEP 5: CHUNK AND EMBED =====
            logger.info("Step 5: Chunking document for semantic analysis...")
            chunks = self.text_chunker.chunk_text(doc['text'])
            logger.info(f"  ✓ Created {len(chunks)} chunks (400 words ± overlap)")
            
            logger.info("Step 6: Generating embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            chunk_embeddings = self._batch_embed(chunk_texts, batch_size=10)
            logger.info(f"  ✓ Generated {len(chunk_embeddings)} 768D embeddings")
            
            # Aggregate embeddings using mean pooling
            query_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            logger.info(f"  ✓ Aggregated embeddings for semantic search")
            
            # ===== STEP 7: SEARCH SIMILAR CASES =====
            logger.info("Step 7: Searching for similar precedents...")
            similar_cases = self.search_engine.search(
                query_embedding=query_embedding,
                query_text=doc['text'][:2000],
                query_attributes=attributes,
                category=category,
                limit=10
            )
            logger.info(f"  ✓ Retrieved {len(similar_cases)} similar cases")
            
            # ===== STEP 8: ANALYZE PRECEDENTS =====
            logger.info("Step 8: Analyzing precedent patterns...")
            precedent_analysis = self.precedent_analyzer.analyze(similar_cases)
            logger.info(f"  ✓ Precedent analysis complete")
            
            # ===== STEP 9: GENERATE DECISION =====
            logger.info("Step 9: Generating bail decision...")
            current_case = {
                'text': doc['text'],
                'category': category,
                'legal_provisions': legal,
                'attributes': attributes
            }
            
            decision = self.decision_engine.make_decision(
                current_case, 
                similar_cases, 
                category
            )
            logger.info(f"  ✓ Recommendation: {decision['recommendation']}")
            logger.info(f"  ✓ Confidence: {decision['confidence']:.1%}")
            
            if decision.get('needs_human_review'):
                logger.warning(f"  ⚠️  HUMAN REVIEW REQUIRED")
                logger.warning(f"  Reasons: {', '.join(decision.get('intervention_reasons', []))}")
            
            # ===== STEP 10: GENERATE DETAILED REASONING =====
            logger.info("Step 10: Generating detailed legal reasoning...")
            reasoning = self.llm.generate_bail_reasoning(
                current_case, 
                decision, 
                similar_cases
            )
            logger.info(f"  ✓ Generated {len(reasoning.split())} word reasoning")
            
            # ===== COMPILE OUTPUT =====
            logger.info("Compiling final report...")
            output = {
                'case_summary': {
                    'filename': doc['filename'],
                    'category': BAIL_CATEGORIES[category]['name'],
                    'language': self.multilingual.detect_language(doc['text'][:1000]),
                    'processed_at': datetime.now().isoformat()
                },
                'legal_analysis': legal,
                'accused_profile': attributes,
                'similar_precedents': similar_cases[:5],
                'precedent_analysis': precedent_analysis,
                'decision': decision,
                'detailed_reasoning': reasoning,
                'processing_metadata': {
                    'chunks_created': len(chunks),
                    'embeddings_generated': len(chunk_embeddings),
                    'precedents_analyzed': len(similar_cases)
                }
            }
            
            logger.info("=" * 60)
            logger.info("✅ Bail application processing complete")
            logger.info("=" * 60)
            
            return output
            
        except Exception as e:
            logger.error(f"❌ Error processing bail application: {e}", exc_info=True)
            raise
    
    # ==================== UTILITY METHODS ====================
    
    def _batch_embed(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Embed texts in batches to avoid overwhelming the API.
        
        Args:
            texts: List of text chunks to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embeddings (768D vectors)
        """
        embeddings = []
        total_batches = (len(texts) - 1) // batch_size + 1
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                logger.debug(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
                batch_embeddings = self.embedder.embed_batch(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to embed batch {batch_num}: {e}")
                raise
        
        return embeddings
