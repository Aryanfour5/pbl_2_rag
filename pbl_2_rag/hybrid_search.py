import logging
import math
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import re
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

class BM25Scorer:
    """Implementation of BM25 ranking algorithm for keyword search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Controls term frequency impact
        self.b = b    # Controls document length normalization
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.metadata = []
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 scoring."""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents."""
        self.corpus = []
        self.metadata = []
        
        for doc in documents:
            tokens = self.tokenize(doc['text'])
            self.corpus.append(tokens)
            self.metadata.append({
                'id': doc['id'],
                'filename': doc.get('filename', ''),
                'chunk_index': doc.get('chunk_index', 0)
            })
        
        # Calculate document frequencies
        self.doc_freqs = []
        for tokens in self.corpus:
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
        
        # Calculate IDF values
        df = defaultdict(int)
        for freq in self.doc_freqs:
            for word in freq.keys():
                df[word] += 1
        
        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log((len(self.corpus) - freq + 0.5) / (freq + 0.5))
        
        # Calculate document lengths
        self.doc_len = [len(tokens) for tokens in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        logger.info(f"Built BM25 index for {len(self.corpus)} documents")
    
    def score(self, query: str) -> List[Tuple[int, float]]:
        """Score documents for a query using BM25."""
        query_tokens = self.tokenize(query)
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            for token in query_tokens:
                if token in doc_freq:
                    tf = doc_freq[token]
                    idf = self.idf.get(token, 0)
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    score += idf * (numerator / denominator)
            
            scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class HybridSearcher:
    """Hybrid search combining semantic (vector) and keyword (BM25) search."""
    
    def __init__(self, vector_db, embedder, alpha: float = 0.7):
        """
        Initialize hybrid searcher.
        
        Args:
            vector_db: QdrantVectorDB instance
            embedder: JinaEmbedder instance  
            alpha: Weight for semantic search (1-alpha for keyword search)
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.alpha = alpha  # Balance between semantic and keyword search
        self.bm25 = BM25Scorer()
        self.indexed = False
    
    def build_keyword_index(self):
        """Build BM25 index from all documents in Qdrant."""
        logger.info("Building keyword search index...")
        
        # Get all documents from Qdrant
        try:
            # Use scroll to get all documents
            documents = []
            scroll_result = self.vector_db.client.scroll(
                collection_name=self.vector_db.collection_name,
                limit=1000,  # Adjust based on your needs
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            while points:
                for point in points:
                    doc = {
                        'id': point.id,
                        'text': point.payload.get('text', ''),
                        'filename': point.payload.get('filename', ''),
                        'chunk_index': point.payload.get('chunk_index', 0)
                    }
                    documents.append(doc)
                
                # Get next batch if available
                if len(points) < 1000:  # Last batch
                    break
                    
                scroll_result = self.vector_db.client.scroll(
                    collection_name=self.vector_db.collection_name,
                    offset=scroll_result[1],
                    limit=1000,
                    with_payload=True,
                    with_vectors=False
                )
                points = scroll_result[0]
            
            logger.info(f"Retrieved {len(documents)} documents for indexing")
            
            # Build BM25 index
            self.bm25.build_index(documents)
            self.indexed = True
            
        except Exception as e:
            logger.error(f"Failed to build keyword index: {str(e)}")
            raise
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized
    
    def search(self, 
               query: str, 
               limit: int = 10,
               semantic_limit: int = 20,
               keyword_limit: int = 20,
               score_threshold: float = 0.0,
               filename_filter: str = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            limit: Final number of results to return
            semantic_limit: Number of results from semantic search
            keyword_limit: Number of results from keyword search  
            score_threshold: Minimum score threshold
            filename_filter: Filter by specific filename
        """
        if not self.indexed:
            logger.warning("Keyword index not built. Building now...")
            self.build_keyword_index()
        
        logger.info(f"Performing hybrid search for: '{query}'")
        
        # 1. Semantic Search (Vector Search)
        logger.debug("Performing semantic search...")
        try:
            query_embedding = self.embedder.generate_query_embedding(query)
            semantic_results = self.vector_db.search(
                query_embedding=query_embedding,
                limit=semantic_limit,
                score_threshold=0.0,  # We'll filter later
                filename_filter=filename_filter
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            semantic_results = []
        
        # 2. Keyword Search (BM25)
        logger.debug("Performing keyword search...")
        try:
            bm25_scores = self.bm25.score(query)
            keyword_results = []
            
            for doc_idx, bm25_score in bm25_scores[:keyword_limit]:
                if bm25_score > 0:  # Only include documents with positive BM25 scores
                    metadata = self.bm25.metadata[doc_idx]
                    
                    # Apply filename filter if specified
                    if filename_filter and metadata['filename'] != filename_filter:
                        continue
                    
                    # Get document text from BM25 corpus
                    doc_text = ' '.join(self.bm25.corpus[doc_idx])
                    
                    keyword_results.append({
                        'id': metadata['id'],
                        'score': bm25_score,
                        'text': doc_text,
                        'filename': metadata['filename'],
                        'chunk_index': metadata['chunk_index'],
                        'search_type': 'keyword'
                    })
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            keyword_results = []
        
        # 3. Combine and Rank Results
        logger.debug("Combining search results...")
        
        # Prepare semantic results
        semantic_scores = [r['score'] for r in semantic_results]
        normalized_semantic = self.normalize_scores(semantic_scores)
        
        for i, result in enumerate(semantic_results):
            result['normalized_semantic_score'] = normalized_semantic[i] if i < len(normalized_semantic) else 0
            result['search_type'] = 'semantic'
        
        # Prepare keyword results  
        keyword_scores = [r['score'] for r in keyword_results]
        normalized_keyword = self.normalize_scores(keyword_scores)
        
        for i, result in enumerate(keyword_results):
            result['normalized_keyword_score'] = normalized_keyword[i] if i < len(normalized_keyword) else 0
        
        # Merge results by document ID
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result['id']
            combined_results[doc_id] = {
                **result,
                'semantic_score': result['score'],
                'keyword_score': 0.0,
                'normalized_semantic_score': result.get('normalized_semantic_score', 0),
                'normalized_keyword_score': 0.0
            }
        
        # Add/merge keyword results
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined_results:
                # Merge scores
                combined_results[doc_id]['keyword_score'] = result['score']
                combined_results[doc_id]['normalized_keyword_score'] = result.get('normalized_keyword_score', 0)
                combined_results[doc_id]['search_type'] = 'hybrid'
            else:
                # Add new result
                combined_results[doc_id] = {
                    **result,
                    'semantic_score': 0.0,
                    'keyword_score': result['score'],
                    'normalized_semantic_score': 0.0,
                    'normalized_keyword_score': result.get('normalized_keyword_score', 0),
                    'search_type': 'keyword'
                }
        
        # 4. Calculate Hybrid Scores
        final_results = []
        for result in combined_results.values():
            # Hybrid score = alpha * semantic_score + (1-alpha) * keyword_score
            hybrid_score = (self.alpha * result['normalized_semantic_score'] + 
                          (1 - self.alpha) * result['normalized_keyword_score'])
            
            result['hybrid_score'] = hybrid_score
            result['score'] = hybrid_score  # Use hybrid score as main score
            
            # Apply score threshold
            if hybrid_score >= score_threshold:
                final_results.append(result)
        
        # 5. Sort by hybrid score and limit results
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        final_results = final_results[:limit]
        
        logger.info(f"Hybrid search returned {len(final_results)} results")
        logger.debug(f"Search breakdown - Semantic: {len(semantic_results)}, "
                    f"Keyword: {len(keyword_results)}, Combined: {len(final_results)}")
        
        return final_results
    
    def explain_search(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Provide explanation of why a result was returned."""
        explanation = {
            'query': query,
            'document_id': result['id'],
            'filename': result.get('filename', 'Unknown'),
            'search_type': result.get('search_type', 'unknown'),
            'scores': {
                'hybrid': result.get('hybrid_score', 0),
                'semantic': result.get('semantic_score', 0), 
                'keyword': result.get('keyword_score', 0),
                'normalized_semantic': result.get('normalized_semantic_score', 0),
                'normalized_keyword': result.get('normalized_keyword_score', 0)
            },
            'alpha_weight': self.alpha
        }
        
        # Find matching keywords
        query_tokens = self.bm25.tokenize(query)
        doc_tokens = self.bm25.tokenize(result.get('text', ''))
        matching_terms = list(set(query_tokens) & set(doc_tokens))
        
        explanation['matching_keywords'] = matching_terms
        explanation['keyword_coverage'] = len(matching_terms) / len(query_tokens) if query_tokens else 0
        
        return explanation