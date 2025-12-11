"""
Perplexity and language modeling metrics.
"""

import torch
import numpy as np
from typing import Dict, Optional


def compute_perplexity(losses: list, token_counts: list) -> float:
    """
    Compute perplexity from losses and token counts.
    
    Args:
        losses: List of cross-entropy losses
        token_counts: List of token counts per sample
        
    Returns:
        Perplexity score
    """
    if not losses or not token_counts:
        return float('inf')
    
    total_loss = sum(l * c for l, c in zip(losses, token_counts))
    total_tokens = sum(token_counts)
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute cross-entropy loss from logits and labels.
    
    Args:
        logits: Model logits [batch, seq_len, vocab]
        labels: Target labels [batch, seq_len]
        attention_mask: Optional mask for valid tokens
        
    Returns:
        Average loss value
    """
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1)
    
    loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
        reduction='none'
    )
    
    if attention_mask is not None:
        loss = loss * attention_mask
        num_tokens = attention_mask.sum().item()
    else:
        num_tokens = labels.numel()
    
    return (loss.sum() / num_tokens).item() if num_tokens > 0 else 0.0


def bits_per_byte(perplexity: float, avg_chars_per_token: float = 4.0) -> float:
    """
    Convert perplexity to bits per byte.
    
    Args:
        perplexity: Perplexity score
        avg_chars_per_token: Average characters per token
        
    Returns:
        Bits per byte
    """
    if perplexity <= 1:
        return 0.0
    
    bits_per_token = np.log2(perplexity)
    return bits_per_token / avg_chars_per_token


def perplexity_degradation(baseline_ppl: float, quantized_ppl: float) -> Dict[str, float]:
    """
    Compute perplexity degradation metrics.
    
    Args:
        baseline_ppl: Baseline model perplexity
        quantized_ppl: Quantized model perplexity
        
    Returns:
        Dict with degradation metrics
    """
    if baseline_ppl <= 0 or quantized_ppl <= 0:
        return {
            'ratio': 1.0,
            'delta': 0.0,
            'percent_increase': 0.0
        }
    
    ratio = quantized_ppl / baseline_ppl
    delta = quantized_ppl - baseline_ppl
    percent = ((quantized_ppl - baseline_ppl) / baseline_ppl) * 100
    
    return {
        'ratio': float(ratio),
        'delta': float(delta),
        'percent_increase': float(percent)
    }


def expected_calibration_error(
    confidences: list,
    accuracies: list,
    num_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error for model confidence.
    
    Args:
        confidences: List of model confidence scores
        accuracies: List of binary accuracy (1 or 0)
        num_bins: Number of bins for calibration
        
    Returns:
        ECE score
    """
    if not confidences or not accuracies:
        return 0.0
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = accuracies[in_bin].mean()
            bin_weight = in_bin.sum() / len(confidences)
            
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
    
    return float(ece)