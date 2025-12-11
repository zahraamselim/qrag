"""
Faithfulness metrics for RAG evaluation.

Measures how well generated answers are grounded in retrieved context.
"""

from typing import List, Optional
import numpy as np


def token_overlap(prediction: str, context: str) -> float:
    """
    Measure fraction of prediction tokens that appear in context.
    
    Args:
        prediction: Generated text
        context: Retrieved context
        
    Returns:
        Overlap ratio in [0, 1]
    """
    pred_tokens = set(prediction.lower().split())
    context_tokens = set(context.lower().split())
    
    if not pred_tokens:
        return 0.0
    
    overlap = len(pred_tokens & context_tokens)
    return overlap / len(pred_tokens)


def context_precision(query: str, context: str) -> float:
    """
    Measure relevance of context to query.
    
    Args:
        query: User question
        context: Retrieved context
        
    Returns:
        Precision score in [0, 1]
    """
    query_tokens = set(query.lower().split())
    context_tokens = set(context.lower().split())
    
    if not query_tokens:
        return 0.0
    
    return len(query_tokens & context_tokens) / len(query_tokens)


def context_sufficiency(answer: str, context: str, threshold: float = 0.8) -> float:
    """
    Check if context contains sufficient information to answer.
    
    Args:
        answer: Ground truth answer
        context: Retrieved context
        threshold: Token overlap threshold for sufficiency
        
    Returns:
        1.0 if sufficient, otherwise token overlap ratio
    """
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    if answer_lower in context_lower:
        return 1.0
    
    answer_tokens = set(answer_lower.split())
    context_tokens = set(context_lower.split())
    
    if not answer_tokens:
        return 0.0
    
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return 1.0 if overlap >= threshold else overlap


def answer_coverage(answer: str, context: str) -> float:
    """
    Measure what fraction of answer terms appear in context.
    
    Similar to context_sufficiency but without threshold.
    """
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    
    if not answer_tokens:
        return 0.0
    
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def batch_faithfulness(
    predictions: List[str],
    contexts: List[str]
) -> List[float]:
    """Compute token overlap faithfulness for batch."""
    return [token_overlap(p, c) for p, c in zip(predictions, contexts)]


def nli_faithfulness(
    prediction: str,
    context: str,
    nli_model = None
) -> Optional[float]:
    """
    Use NLI model to verify if prediction is entailed by context.
    
    Requires a loaded NLI model (e.g., DeBERTa-based).
    Returns None if model not provided.
    
    Args:
        prediction: Generated text
        context: Retrieved context
        nli_model: Optional NLI model for inference
        
    Returns:
        Entailment probability or None
    """
    if nli_model is None:
        return None
    
    try:
        result = nli_model(context, prediction)
        
        if isinstance(result, dict) and 'entailment' in result:
            return float(result['entailment'])
        elif isinstance(result, list) and len(result) > 0:
            entail_score = [r for r in result if r.get('label') == 'entailment']
            if entail_score:
                return float(entail_score[0].get('score', 0.0))
        
        return 0.5
    except Exception:
        return None


def citation_precision(
    prediction: str,
    citations: List[str],
    contexts: List[str]
) -> float:
    """
    Measure precision of citations in prediction.
    
    Args:
        prediction: Generated text with citations
        citations: List of cited passage IDs/indices
        contexts: List of actual context passages
        
    Returns:
        Fraction of citations that are accurate
    """
    if not citations:
        return 0.0
    
    correct = 0
    pred_lower = prediction.lower()
    
    for cit_idx in citations:
        if 0 <= cit_idx < len(contexts):
            context_lower = contexts[cit_idx].lower()
            
            if any(token in context_lower for token in pred_lower.split()[:10]):
                correct += 1
    
    return correct / len(citations)


def hallucination_score(
    prediction: str,
    context: str,
    threshold: float = 0.3
) -> float:
    """
    Estimate hallucination by measuring unsupported content.
    
    Returns fraction of prediction NOT supported by context.
    Higher score = more hallucination.
    
    Args:
        prediction: Generated text
        context: Retrieved context
        threshold: Minimum overlap to consider supported
        
    Returns:
        Hallucination ratio in [0, 1]
    """
    overlap = token_overlap(prediction, context)
    return max(0.0, 1.0 - overlap) if overlap < threshold else 0.0