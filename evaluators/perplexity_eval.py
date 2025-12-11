"""
Perplexity evaluator for language modeling quality assessment.
"""

import logging
import torch
from typing import Dict, Any
from datasets import load_dataset

from evaluators.base import BaseEvaluator
from metrics.perplexity import compute_perplexity

logger = logging.getLogger(__name__)


class PerplexityEvaluator(BaseEvaluator):
    """
    Evaluate model perplexity on standard datasets.
    
    Supports: WikiText-2, UltraChat, Alpaca (instruction-following)
    """
    
    def __init__(
        self,
        model_interface,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "test",
        max_samples: int = 100,
        max_length: int = 512,
        min_text_length: int = 100
    ):
        super().__init__(model_interface)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.max_samples = max_samples
        self.max_length = max_length
        self.min_text_length = min_text_length
    
    def run(self) -> Dict[str, Any]:
        """Run perplexity evaluation."""
        logger.info(f"Evaluating perplexity on {self.dataset_name}")
        if self.dataset_config:
            logger.info(f"Config: {self.dataset_config}")
        
        if self.dataset_config:
            dataset = load_dataset(self.dataset_name, self.dataset_config, split=self.split)
        else:
            dataset = load_dataset(self.dataset_name, split=self.split)
        
        text_field = self._detect_text_field(dataset)
        logger.info(f"Using text field: {text_field}")
        
        dataset = dataset.filter(lambda x: len(x.get(text_field, "")) > self.min_text_length)
        
        logger.info(f"Loaded {len(dataset)} samples")
        
        if len(dataset) == 0:
            raise ValueError("No valid samples after filtering")
        
        losses = []
        token_counts = []
        valid_samples = 0
        
        for i in range(min(self.max_samples, len(dataset))):
            text = self._extract_text(dataset[i], text_field)
            
            if not text or len(text) < self.min_text_length:
                continue
            
            try:
                inputs = self.model.encode(text, max_length=self.max_length)
                
                with torch.no_grad():
                    outputs = self.model.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                
                attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
                num_tokens = attention_mask.sum().item() - 1
                
                if num_tokens > 0:
                    losses.append(loss)
                    token_counts.append(num_tokens)
                    valid_samples += 1
                
                if (i + 1) % 20 == 0:
                    current_ppl = compute_perplexity(losses, token_counts)
                    logger.info(f"Progress: {i+1}/{min(self.max_samples, len(dataset))}, PPL: {current_ppl:.2f}")
            
            except Exception as e:
                logger.warning(f"Failed on sample {i}: {e}")
                continue
        
        if not losses:
            raise ValueError("No valid samples for perplexity calculation")
        
        perplexity = compute_perplexity(losses, token_counts)
        avg_loss = sum(l * c for l, c in zip(losses, token_counts)) / sum(token_counts)
        
        self.results = {
            "perplexity": float(perplexity),
            "loss": float(avg_loss),
            "dataset": f"{self.dataset_name}" + (f"/{self.dataset_config}" if self.dataset_config else ""),
            "split": self.split,
            "num_samples": valid_samples,
            "total_tokens": sum(token_counts),
            "max_length": self.max_length
        }
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Valid samples: {valid_samples}")
        
        return self.results
    
    def _detect_text_field(self, dataset) -> str:
        """Detect which field contains the text."""
        if len(dataset) == 0:
            return "text"
        
        sample = dataset[0]
        
        if "text" in sample:
            return "text"
        elif "messages" in sample:
            return "messages"
        elif "prompt" in sample and "completion" in sample:
            return "prompt_completion"
        elif "instruction" in sample:
            return "instruction"
        else:
            raise ValueError(f"Unknown dataset format. Available fields: {sample.keys()}")
    
    def _extract_text(self, sample: dict, text_field: str) -> str:
        """Extract text from sample based on field type."""
        if text_field == "text":
            return sample.get("text", "")
        
        elif text_field == "messages":
            messages = sample.get("messages", [])
            if isinstance(messages, list):
                texts = []
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        texts.append(msg["content"])
                return "\n".join(texts)
            return str(messages)
        
        elif text_field == "prompt_completion":
            prompt = sample.get("prompt", "")
            completion = sample.get("completion", "")
            return f"{prompt}\n{completion}"
        
        elif text_field == "instruction":
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")
            if input_text:
                return f"{instruction}\n{input_text}\n{output}"
            return f"{instruction}\n{output}"
        
        return ""