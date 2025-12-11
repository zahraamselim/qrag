"""
Retrieval quality metrics for RAG evaluation.

Measures effectiveness of document retrieval before generation.
"""

from typing import List, Dict, Set
import numpy as np


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int = None) -> float:
    """
    Precision at k: fraction of top-k retrieved docs that are relevant.
    
    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: Set of relevant document IDs
        k: Cutoff (None for all retrieved)
        
    Returns:
        Precision score in [0, 1]
    """
    if not retrieved:
        return 0.0
    
    if k is None:
        k = len(retrieved)
    
    top_k = retrieved[:k]
    num_relevant = sum(1 for doc_id in top_k if doc_id in relevant)
    
    return num_relevant / min(k, len(top_k))


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int = None) -> float:
    """
    Recall at k: fraction of relevant docs retrieved in top-k.
    
    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: Set of relevant document IDs
        k: Cutoff (None for all retrieved)
        
    Returns:
        Recall score in [0, 1]
    """
    if not relevant:
        return 0.0
    
    if k is None:
        k = len(retrieved)
    
    top_k = retrieved[:k]
    num_retrieved = sum(1 for doc_id in top_k if doc_id in relevant)
    
    return num_retrieved / len(relevant)


def mean_reciprocal_rank(retrieved_lists: List[List[int]], relevant_sets: List[Set[int]]) -> float:
    """
    Mean Reciprocal Rank across multiple queries.
    
    Args:
        retrieved_lists: List of retrieved doc lists per query
        relevant_sets: List of relevant doc sets per query
        
    Returns:
        MRR score in [0, 1]
    """
    reciprocal_ranks = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def ndcg_at_k(
    retrieved: List[int],
    relevance_scores: Dict[int, float],
    k: int = None
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    
    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevance_scores: Dict mapping doc IDs to relevance scores
        k: Cutoff (None for all retrieved)
        
    Returns:
        NDCG score in [0, 1]
    """
    if not retrieved:
        return 0.0
    
    if k is None:
        k = len(retrieved)
    
    top_k = retrieved[:k]
    
    dcg = sum(
        relevance_scores.get(doc_id, 0.0) / np.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
    )
    
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(
        score / np.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
        if score > 0
    )
    
    return float(dcg / idcg) if idcg > 0 else 0.0


def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Average Precision for a single query.
    
    Args:
        retrieved: List of retrieved document IDs (in rank order)
        relevant: Set of relevant document IDs
        
    Returns:
        AP score in [0, 1]
    """
    if not relevant or not retrieved:
        return 0.0
    
    num_relevant = 0
    sum_precisions = 0.0
    
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            num_relevant += 1
            precision = num_relevant / rank
            sum_precisions += precision
    
    return sum_precisions / len(relevant)


def mean_average_precision(retrieved_lists: List[List[int]], relevant_sets: List[Set[int]]) -> float:
    """
    Mean Average Precision across multiple queries.
    
    Args:
        retrieved_lists: List of retrieved doc lists per query
        relevant_sets: List of relevant doc sets per query
        
    Returns:
        MAP score in [0, 1]
    """
    aps = [average_precision(ret, rel) for ret, rel in zip(retrieved_lists, relevant_sets)]
    return float(np.mean(aps)) if aps else 0.0


def retrieval_score_statistics(scores: List[float]) -> Dict[str, float]:
    """
    Compute statistics for retrieval similarity scores.
    
    Args:
        scores: List of retrieval scores (e.g., cosine similarities)
        
    Returns:
        Dict with mean, std, min, max scores
    """
    if not scores:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores))
    }


def context_length_statistics(contexts: List[str]) -> Dict[str, float]:
    """
    Compute statistics for retrieved context lengths.
    
    Args:
        contexts: List of retrieved context strings
        
    Returns:
        Dict with mean, std, min, max word counts
    """
    if not contexts:
        return {'mean_words': 0.0, 'std_words': 0.0, 'min_words': 0, 'max_words': 0}
    
    word_counts = [len(ctx.split()) for ctx in contexts]
    
    return {
        'mean_words': float(np.mean(word_counts)),
        'std_words': float(np.std(word_counts)),
        'min_words': int(np.min(word_counts)),
        'max_words': int(np.max(word_counts))
    }


def hit_rate_at_k(retrieved_lists: List[List[int]], relevant_sets: List[Set[int]], k: int = 1) -> float:
    """
    Hit rate at k: fraction of queries with at least one relevant doc in top-k.
    
    Args:
        retrieved_lists: List of retrieved doc lists per query
        relevant_sets: List of relevant doc sets per query
        k: Cutoff
        
    Returns:
        Hit rate in [0, 1]
    """
    hits = 0
    
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        top_k = retrieved[:k]
        if any(doc_id in relevant for doc_id in top_k):
            hits += 1
    
    return hits / len(retrieved_lists) if retrieved_lists else 0.0