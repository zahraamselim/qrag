"""
Performance evaluator for time and space metrics.
"""

import time
import logging
import torch
import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from evaluators.base import BaseEvaluator
from metrics.efficiency import (
    measure_time,
    measure_memory_usage,
    reset_memory_stats,
    tokens_per_second,
    ms_per_token,
    estimate_kv_cache_size
)
from models.model_interface import GenerationConfig

logger = logging.getLogger(__name__)


class PerformanceEvaluator(BaseEvaluator):
    """
    Evaluate model performance: latency, throughput, memory, model size.
    """
    
    def __init__(
        self,
        model_interface,
        prompts: List[str] = None,
        num_warmup: int = 3,
        num_runs: int = 10,
        max_new_tokens: int = 128
    ):
        super().__init__(model_interface)
        self.prompts = prompts or self._default_prompts()
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.max_new_tokens = max_new_tokens
    
    @staticmethod
    def _default_prompts():
        return [
            "The capital of France is",
            "Artificial intelligence is defined as",
            "In machine learning, overfitting occurs when",
            "Quantum computing differs from classical computing because",
            "The theory of relativity states that",
            "Natural language processing enables",
            "Deep learning models are characterized by",
            "The Transformer architecture introduced",
            "Reinforcement learning agents learn by",
            "Neural networks consist of"
        ]
    
    def run(self) -> Dict[str, Any]:
        """Run full performance evaluation."""
        logger.info("Starting performance evaluation")
        
        reset_memory_stats(str(self.model.device))
        
        results = {}
        
        logger.info("Measuring latency and throughput")
        timing_results = self._measure_timing()
        results['timing'] = timing_results
        
        logger.info("Measuring memory usage")
        memory_results = self._measure_memory()
        results['memory'] = memory_results
        
        logger.info("Computing model size")
        size_results = self._measure_model_size()
        results['model_size'] = size_results
        
        self.results = self._convert_to_serializable(results)
        
        self._print_summary()
        
        return self.results
    
    def _measure_timing(self) -> Dict[str, Any]:
        """Measure latency and throughput."""
        config = GenerationConfig(max_new_tokens=10, do_sample=False)
        
        for i in range(self.num_warmup):
            try:
                self.model.generate(self.prompts[0], config)
            except Exception as e:
                logger.warning(f"Warmup {i} failed: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        config = GenerationConfig(max_new_tokens=self.max_new_tokens, do_sample=False)
        
        latencies = []
        token_counts = []
        ttfts = []
        
        for i in range(self.num_runs):
            prompt = self.prompts[i % len(self.prompts)]
            
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                output = self.model.generate(prompt, config)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latency = (time.perf_counter() - start) * 1000
                
                latencies.append(latency)
                token_counts.append(output.num_generated_tokens)
                
                if i == 0:
                    ttft_config = GenerationConfig(max_new_tokens=1, do_sample=False)
                    ttft_start = time.perf_counter()
                    self.model.generate(prompt, ttft_config)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    ttft = (time.perf_counter() - ttft_start) * 1000
                    ttfts.append(ttft)
                
            except Exception as e:
                logger.warning(f"Timing run {i} failed: {e}")
                continue
        
        if not latencies:
            return {}
        
        import numpy as np
        
        per_token_latencies = [
            lat / tok for lat, tok in zip(latencies, token_counts) if tok > 0
        ]
        
        throughputs = [
            tokens_per_second(tok, lat) 
            for tok, lat in zip(token_counts, latencies) if lat > 0
        ]
        
        return {
            'latency_ms_per_token': float(np.mean(per_token_latencies)),
            'latency_std': float(np.std(per_token_latencies)),
            'latency_min': float(np.min(per_token_latencies)),
            'latency_max': float(np.max(per_token_latencies)),
            'throughput_tokens_per_sec': float(np.mean(throughputs)),
            'throughput_std': float(np.std(throughputs)),
            'ttft_ms': float(np.mean(ttfts)) if ttfts else 0.0,
            'avg_tokens_generated': float(np.mean(token_counts)),
            'num_runs': len(latencies)
        }
    
    def _measure_memory(self) -> Dict[str, Any]:
        """Measure memory usage."""
        memory_stats = measure_memory_usage(str(self.model.device))
        
        try:
            config = self.model.model.config
            kv_cache = estimate_kv_cache_size(
                num_layers=config.num_hidden_layers,
                num_heads=config.num_attention_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                sequence_length=2048,
                batch_size=1
            )
        except Exception:
            kv_cache = 0.0
        
        return {
            'peak_mb': memory_stats['peak_mb'],
            'allocated_mb': memory_stats['allocated_mb'],
            'reserved_mb': memory_stats['reserved_mb'],
            'estimated_kv_cache_mb': kv_cache
        }
    
    def _measure_model_size(self) -> Dict[str, Any]:
        """Measure model size and parameter count."""
        try:
            param_size = sum(
                p.element_size() * p.numel()
                for p in self.model.model.parameters()
            )
            
            buffer_size = sum(
                b.element_size() * b.numel()
                for b in self.model.model.buffers()
            )
            
            total_bytes = param_size + buffer_size
            size_gb = total_bytes / (1024**3)
            
            total_params = sum(p.numel() for p in self.model.model.parameters())
            
            sample_param = next(self.model.model.parameters())
            bits_per_param = sample_param.element_size() * 8
            
            return {
                'size_gb': float(size_gb),
                'size_bytes': int(total_bytes),
                'total_params': int(total_params),
                'bits_per_param': float(bits_per_param)
            }
        
        except Exception as e:
            logger.warning(f"Failed to measure model size: {e}")
            return {}
    
    def _print_summary(self):
        """Print formatted summary."""
        timing = self.results.get('timing', {})
        memory = self.results.get('memory', {})
        size = self.results.get('model_size', {})
        
        logger.info("Performance Summary:")
        
        if timing:
            logger.info(f"Latency: {timing.get('latency_ms_per_token', 0):.3f} ms/token")
            logger.info(f"Throughput: {timing.get('throughput_tokens_per_sec', 0):.2f} tokens/s")
            logger.info(f"TTFT: {timing.get('ttft_ms', 0):.3f} ms")
        
        if memory:
            logger.info(f"Peak memory: {memory.get('peak_mb', 0):.2f} MB")
        
        if size:
            logger.info(f"Model size: {size.get('size_gb', 0):.3f} GB")
            logger.info(f"Parameters: {size.get('total_params', 0):,}")
    
    def save_results(self, output_path: Path):
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")