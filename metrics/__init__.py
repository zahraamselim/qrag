"""
Metrics package for model evaluation.

Provides atomic measurement functions for accuracy, faithfulness, 
efficiency, retrieval quality, and perplexity.
"""

from metrics.accuracy import (
    exact_match,
    token_f1,
    batch_f1,
    batch_exact_match,
    substring_match,
    best_f1,
    compute_rouge_scores
)

from metrics.faithfulness import (
    token_overlap,
    context_precision,
    context_sufficiency,
    answer_coverage,
    batch_faithfulness,
    hallucination_score
)

from metrics.efficiency import (
    measure_time,
    measure_latency,
    measure_throughput,
    measure_memory_usage,
    reset_memory_stats,
    compute_compression_ratio,
    tokens_per_second,
    ms_per_token,
    estimate_kv_cache_size,
    compute_speedup
)

from metrics.retrieval import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    average_precision,
    mean_average_precision,
    retrieval_score_statistics,
    context_length_statistics,
    hit_rate_at_k
)

from metrics.perplexity import (
    compute_perplexity,
    compute_loss,
    bits_per_byte,
    perplexity_degradation,
    expected_calibration_error
)

__all__ = [
    'exact_match',
    'token_f1',
    'batch_f1',
    'batch_exact_match',
    'substring_match',
    'best_f1',
    'compute_rouge_scores',
    'token_overlap',
    'context_precision',
    'context_sufficiency',
    'answer_coverage',
    'batch_faithfulness',
    'hallucination_score',
    'measure_time',
    'measure_latency',
    'measure_throughput',
    'measure_memory_usage',
    'reset_memory_stats',
    'compute_compression_ratio',
    'tokens_per_second',
    'ms_per_token',
    'estimate_kv_cache_size',
    'compute_speedup',
    'precision_at_k',
    'recall_at_k',
    'mean_reciprocal_rank',
    'ndcg_at_k',
    'average_precision',
    'mean_average_precision',
    'retrieval_score_statistics',
    'context_length_statistics',
    'hit_rate_at_k',
    'compute_perplexity',
    'compute_loss',
    'bits_per_byte',
    'perplexity_degradation',
    'expected_calibration_error'
]