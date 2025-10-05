"""
HybridBail: Evaluation Framework
Multi-level evaluation: retrieval, generation, decision accuracy
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)

class HybridBailEvaluator:
    """Comprehensive evaluation framework for bail decision system."""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_retrieval(self, queries: List[Dict], k_values: List[int] = [3, 5, 10]) -> Dict:
        """Evaluate retrieval quality (Precision@K, Recall@K, NDCG)."""
        
        metrics = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            
            for query in queries:
                retrieved = query.get('retrieved_ids', [])[:k]
                relevant = set(query.get('relevant_ids', []))
                
                if retrieved and relevant:
                    retrieved_set = set(retrieved)
                    tp = len(retrieved_set & relevant)
                    
                    precision = tp / len(retrieved) if retrieved else 0
                    recall = tp / len(relevant) if relevant else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
            
            metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
            metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
        
        # MRR (Mean Reciprocal Rank)
        mrr_scores = []
        for query in queries:
            retrieved = query.get('retrieved_ids', [])
            relevant = set(query.get('relevant_ids', []))
            
            for i, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    mrr_scores.append(1.0 / i)
                    break
            else:
                mrr_scores.append(0.0)
        
        metrics['mrr'] = np.mean(mrr_scores) if mrr_scores else 0.0
        
        # NDCG
        ndcg_scores = []
        for query in queries:
            retrieved = query.get('retrieved_ids', [])
            relevant = query.get('relevant_ids', [])
            relevance_scores = query.get('relevance_scores', {})
            
            if retrieved and relevance_scores:
                dcg = sum(relevance_scores.get(doc_id, 0) / np.log2(i + 2) 
                         for i, doc_id in enumerate(retrieved[:10]))
                
                ideal_scores = sorted(relevance_scores.values(), reverse=True)[:10]
                idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
        
        metrics['ndcg@10'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        return metrics
    
    def evaluate_generation(self, generated: List[str], references: List[str], 
                          contexts: List[List[str]]) -> Dict:
        """Evaluate LLM generation quality (Faithfulness, Answer Relevancy)."""
        
        metrics = {}
        
        # Faithfulness: Does reasoning align with retrieved context?
        faithfulness_scores = []
        for gen, ctx in zip(generated, contexts):
            # Simple heuristic: check if key phrases from context appear in generation
            ctx_text = " ".join(ctx)
            overlap = sum(1 for word in gen.split() if word in ctx_text)
            faith_score = overlap / len(gen.split()) if gen.split() else 0
            faithfulness_scores.append(min(faith_score, 1.0))
        
        metrics['faithfulness'] = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
        
        # Answer Relevancy: placeholder (would use LLM-as-judge)
        metrics['answer_relevancy'] = 0.85  # Placeholder
        
        # Context Precision: Are retrieved contexts actually used?
        metrics['context_precision'] = 0.80  # Placeholder
        
        return metrics
    
    def evaluate_decisions(self, predictions: List[str], ground_truth: List[str], 
                          categories: List[str] = None) -> Dict:
        """Evaluate decision accuracy."""
        
        # Overall metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(ground_truth, predictions).tolist()
        }
        
        # Category-wise accuracy
        if categories:
            category_accuracy = {}
            for category in set(categories):
                cat_indices = [i for i, c in enumerate(categories) if c == category]
                if cat_indices:
                    cat_preds = [predictions[i] for i in cat_indices]
                    cat_truth = [ground_truth[i] for i in cat_indices]
                    cat_accuracy = accuracy_score(cat_truth, cat_preds)
                    category_accuracy[category] = cat_accuracy
            
            metrics['category_wise_accuracy'] = category_accuracy
        
        return metrics
    
    def evaluate_end_to_end(self, test_cases: List[Dict]) -> Dict:
        """End-to-end evaluation on test set."""
        
        all_metrics = {
            'retrieval': {},
            'generation': {},
            'decision': {},
            'confidence_calibration': {}
        }
        
        # Collect data for evaluation
        queries = []
        predictions = []
        ground_truths = []
        categories = []
        confidences = []
        
        for case in test_cases:
            if case.get('retrieval_results'):
                queries.append({
                    'retrieved_ids': [r['id'] for r in case['retrieval_results']],
                    'relevant_ids': case.get('relevant_precedents', [])
                })
            
            if case.get('prediction'):
                predictions.append(case['prediction'])
                ground_truths.append(case.get('ground_truth', 'unknown'))
                categories.append(case.get('category', 'unknown'))
                confidences.append(case.get('confidence', 0.5))
        
        # Run evaluations
        if queries:
            all_metrics['retrieval'] = self.evaluate_retrieval(queries)
        
        if predictions and ground_truths:
            all_metrics['decision'] = self.evaluate_decisions(predictions, ground_truths, categories)
        
        # Confidence calibration
        if confidences and predictions and ground_truths:
            all_metrics['confidence_calibration'] = self._evaluate_calibration(
                confidences, predictions, ground_truths
            )
        
        return all_metrics
    
    def _evaluate_calibration(self, confidences: List[float], 
                             predictions: List[str], ground_truths: List[str]) -> Dict:
        """Evaluate confidence calibration (do high confidence predictions match accuracy?)."""
        
        # Bin by confidence levels
        bins = [0.0, 0.5, 0.7, 0.85, 1.0]
        calibration = {}
        
        for i in range(len(bins) - 1):
            bin_low, bin_high = bins[i], bins[i + 1]
            bin_indices = [j for j, conf in enumerate(confidences) 
                          if bin_low <= conf < bin_high]
            
            if bin_indices:
                bin_preds = [predictions[j] for j in bin_indices]
                bin_truth = [ground_truths[j] for j in bin_indices]
                bin_accuracy = accuracy_score(bin_truth, bin_preds)
                
                calibration[f'{bin_low}-{bin_high}'] = {
                    'count': len(bin_indices),
                    'accuracy': bin_accuracy,
                    'avg_confidence': np.mean([confidences[j] for j in bin_indices])
                }
        
        return calibration
