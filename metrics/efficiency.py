"""
Efficiency metrics for model performance evaluation.

Includes latency, throughput, and memory measurements.
"""

import time
import torch
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager
import numpy as np


@contextmanager
def measure_time():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    times = {'start': start}
    try:
        yield times
    finally:
        times['end'] = time.perf_counter()
        times['elapsed'] = times['end'] - times['start']


def measure_latency(
    generate_fn: Callable,
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, float]:
    """
    Measure generation latency statistics.
    
    Args:
        generate_fn: Function that performs generation (no args)
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dict with mean, std, min, max latency in milliseconds
    """
    for _ in range(warmup_runs):
        generate_fn()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    latencies = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        generate_fn()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies))
    }


def measure_throughput(
    token_counts: List[int],
    latencies_ms: List[float]
) -> Dict[str, float]:
    """
    Calculate throughput from token counts and latencies.
    
    Args:
        token_counts: Number of tokens generated in each run
        latencies_ms: Latency for each run in milliseconds
        
    Returns:
        Dict with throughput statistics in tokens/second
    """
    latencies_s = [l / 1000 for l in latencies_ms]
    throughputs = [t / l for t, l in zip(token_counts, latencies_s) if l > 0]
    
    if not throughputs:
        return {'mean_tps': 0.0, 'std_tps': 0.0}
    
    return {
        'mean_tps': float(np.mean(throughputs)),
        'std_tps': float(np.std(throughputs)),
        'total_tokens': sum(token_counts),
        'total_time_s': sum(latencies_s)
    }


def measure_memory_usage(device: str = "cuda:0") -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Args:
        device: CUDA device identifier
        
    Returns:
        Dict with memory statistics in MB
    """
    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'peak_mb': 0.0
        }
    
    device_id = 0 if "cuda" in device else int(device.split(":")[-1])
    
    return {
        'allocated_mb': torch.cuda.memory_allocated(device_id) / (1024**2),
        'reserved_mb': torch.cuda.memory_reserved(device_id) / (1024**2),
        'peak_mb': torch.cuda.max_memory_allocated(device_id) / (1024**2)
    }


def reset_memory_stats(device: str = "cuda:0"):
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        device_id = 0 if "cuda" in device else int(device.split(":")[-1])
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.empty_cache()


def compute_compression_ratio(
    original_size_bytes: int,
    compressed_size_bytes: int
) -> Dict[str, float]:
    """
    Calculate compression metrics.
    
    Returns:
        Dict with compression ratio and space savings
    """
    if compressed_size_bytes == 0:
        return {'ratio': 0.0, 'savings_percent': 0.0}
    
    ratio = original_size_bytes / compressed_size_bytes
    savings = ((original_size_bytes - compressed_size_bytes) / original_size_bytes) * 100
    
    return {
        'ratio': float(ratio),
        'savings_percent': float(savings)
    }


def tokens_per_second(num_tokens: int, latency_ms: float) -> float:
    """Calculate tokens per second from latency."""
    if latency_ms <= 0:
        return 0.0
    return (num_tokens / latency_ms) * 1000


def ms_per_token(latency_ms: float, num_tokens: int) -> float:
    """Calculate milliseconds per token."""
    if num_tokens <= 0:
        return float('inf')
    return latency_ms / num_tokens


def estimate_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
    batch_size: int = 1,
    bytes_per_element: int = 2
) -> float:
    """
    Estimate KV cache memory for transformer models.
    
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        sequence_length: Sequence length
        batch_size: Batch size
        bytes_per_element: Bytes per element (2 for fp16, 4 for fp32)
        
    Returns:
        Estimated cache size in MB
    """
    cache_bytes = (
        2 *  # K and V
        num_layers *
        batch_size *
        num_heads *
        sequence_length *
        head_dim *
        bytes_per_element
    )
    
    return cache_bytes / (1024**2)


def compute_speedup(baseline_ms: float, optimized_ms: float) -> Dict[str, float]:
    """
    Calculate speedup metrics.
    
    Returns:
        Dict with speedup multiplier and percent improvement
    """
    if optimized_ms <= 0:
        return {'speedup': 0.0, 'improvement_percent': 0.0}
    
    speedup = baseline_ms / optimized_ms
    improvement = ((baseline_ms - optimized_ms) / baseline_ms) * 100
    
    return {
        'speedup': float(speedup),
        'improvement_percent': float(improvement)
    }