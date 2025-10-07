"""
HybridBail: LLM Integration
RAG-based reasoning generation using Gemini
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class LLMIntegration:
    """Integrates LLM for generating detailed bail reasoning."""
    
    def __init__(self, config):
        self.config = config
        self.provider = config.LLM_PROVIDER
        
        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config.LLM_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=config.LLM_MODEL,
                generation_config={
                    'temperature': config.LLM_TEMPERATURE,
                    'max_output_tokens': config.LLM_MAX_TOKENS,
                }
            )
            self.client = self.model
            logger.info(f"Initialized Gemini: {config.LLM_MODEL}")
    
    def generate_bail_reasoning(self, current_case: Dict, decision: Dict, 
                               precedents: List[Dict]) -> str:
        """Generate detailed reasoning for bail decision using RAG."""
        
        context = self._build_rag_context(current_case, decision, precedents)
        prompt = self._create_reasoning_prompt(current_case, decision, context)
        reasoning = self._call_llm(prompt)
        
        return reasoning
    
    def _build_rag_context(self, current_case: Dict, decision: Dict, 
                          precedents: List[Dict]) -> str:
        """Build context from retrieved precedents."""
        
        context_parts = []
        
        context_parts.append("=== CURRENT CASE ===")
        context_parts.append(f"Category: {current_case.get('category', 'Unknown')}")
        
        legal = current_case.get('legal_provisions', {})
        if legal:
            context_parts.append(f"Legal Provisions: {', '.join(legal.get('all_sections', [])[:5])}")
        
        context_parts.append(f"\n=== SIMILAR PRECEDENTS ({len(precedents)} cases) ===")
        
        for i, prec in enumerate(precedents[:3], 1):
            context_parts.append(f"\n[Precedent {i}]")
            context_parts.append(f"Similarity: {prec.get('final_score', prec.get('score', 0)):.3f}")
            context_parts.append(f"Outcome: {prec.get('outcome', 'Unknown').upper()}")
        
        prec_analysis = decision.get('precedent_analysis', {})
        context_parts.append(f"\n=== ANALYSIS ===")
        context_parts.append(f"Grant Rate: {prec_analysis.get('grant_rate', 0):.1%}")
        
        return "\n".join(context_parts)
    
    def _create_reasoning_prompt(self, current_case: Dict, decision: Dict, context: str) -> str:
        """Create prompt for LLM."""
        
        recommendation = decision['recommendation']
        confidence = decision['confidence']
        
        prompt = f"""You are a legal AI assistant analyzing a bail application. Provide structured legal analysis.

{context}

RECOMMENDATION: {recommendation}
CONFIDENCE: {confidence:.3f}

Provide analysis in this format:
1. Case Summary
2. Legal Framework
3. Precedent Analysis
4. Factors Favoring Bail
5. Factors Against Bail
6. Reasoning
7. Conclusion

Write in professional legal language."""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        
        try:
            if self.provider == "gemini":
                response = self.client.generate_content(prompt)
                return response.text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "LLM reasoning unavailable. Based on precedent analysis, see decision above."
        
        return "Reasoning generation failed."
