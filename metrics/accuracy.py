"""
Accuracy metrics for text generation evaluation.

All metrics return float values in [0, 1] range unless otherwise specified.
"""

from collections import Counter
from typing import List, Union


def exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
    """
    Exact match score with optional normalization.
    
    Args:
        prediction: Generated text
        reference: Ground truth text
        normalize: If True, lowercase and strip whitespace
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if normalize:
        prediction = prediction.lower().strip()
        reference = reference.lower().strip()
    
    return float(prediction == reference)


def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 score using bag-of-words overlap.
    
    Args:
        prediction: Generated text
        reference: Ground truth text
        
    Returns:
        F1 score in [0, 1]
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    
    return 2 * precision * recall / (precision + recall)


def batch_f1(predictions: List[str], references: List[str]) -> List[float]:
    """Compute F1 scores for a batch of predictions."""
    return [token_f1(p, r) for p, r in zip(predictions, references)]


def batch_exact_match(predictions: List[str], references: List[str]) -> List[float]:
    """Compute exact match scores for a batch of predictions."""
    return [exact_match(p, r) for p, r in zip(predictions, references)]


def substring_match(prediction: str, reference: str) -> float:
    """
    Check if reference appears as substring in prediction.
    
    Returns:
        1.0 if reference is substring of prediction, 0.0 otherwise
    """
    return float(reference.lower() in prediction.lower())


def best_f1(prediction: str, references: List[str]) -> float:
    """
    Compute maximum F1 score across multiple references.
    
    Useful when multiple correct answers exist.
    """
    if not references:
        return 0.0
    return max(token_f1(prediction, ref) for ref in references)


def compute_rouge_scores(predictions: List[str], references: List[str]) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
    
    Requires rouge-score library. Returns empty dict if not available.
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
        
        return {
            'rouge1': sum(s['rouge1'].fmeasure for s in scores) / len(scores),
            'rouge2': sum(s['rouge2'].fmeasure for s in scores) / len(scores),
            'rougeL': sum(s['rougeL'].fmeasure for s in scores) / len(scores)
        }
    except ImportError:
        return {}