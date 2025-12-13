"""
Context length evaluator for measuring performance degradation with context size.
"""

import logging
import random
import gc
import json
import torch
import numpy as np
from typing import Dict, List, Any
from datasets import load_dataset
from scipy import stats

from evaluators.base import BaseEvaluator
from metrics.accuracy import token_f1, substring_match
from models.model_interface import GenerationConfig

logger = logging.getLogger(__name__)


class ContextLengthEvaluator(BaseEvaluator):
    """
    Evaluate how model performance degrades as context length increases.
    
    Tests:
    - Performance at different context lengths (512 to 16k+)
    - Answer position sensitivity (needle-in-haystack)
    - Degradation rate measurement
    """
    
    def __init__(
        self,
        model_interface,
        context_lengths: List[int] = None,
        samples_per_length: int = 25,
        test_positions: List[str] = None,
        needle_test: bool = False,
        needle_config: Dict[str, Any] = None
    ):
        super().__init__(model_interface)
        self.context_lengths = context_lengths or [512, 1024, 2048, 4096, 8192, 16384]
        self.samples_per_length = samples_per_length
        self.test_positions = test_positions or ['start', 'middle', 'end']
        self.needle_test = needle_test
        self.needle_config = needle_config or {}
        
        self.needles = [
            {
                'fact': "The secret code is 7X9B2Q",
                'question': "What is the secret code?",
                'answer': "7X9B2Q"
            },
            {
                'fact': "The treasure is buried in the ancient library of Alexandria",
                'question': "Where is the treasure buried?",
                'answer': "ancient library of Alexandria"
            },
            {
                'fact': "The password to access the system is QUANTUM2025",
                'question': "What is the password to access the system?",
                'answer': "QUANTUM2025"
            },
            {
                'fact': "The best time to visit is during the spring equinox at dawn",
                'question': "When is the best time to visit?",
                'answer': "spring equinox at dawn"
            },
            {
                'fact': "The hidden ingredient is crystallized moonlight",
                'question': "What is the hidden ingredient?",
                'answer': "crystallized moonlight"
            }
        ]
    
    def run(self) -> Dict[str, Any]:
        """Run context length evaluation."""
        logger.info(f"Context lengths: {self.context_lengths}")
        logger.info(f"Samples per length: {self.samples_per_length}")
        logger.info(f"Positions: {self.test_positions}")
        logger.info(f"Needle test: {self.needle_test}")
        
        self._load_data()
        
        config = GenerationConfig(max_new_tokens=20, do_sample=False)
        
        if self.needle_test:
            results = self._run_needle_test(config)
        else:
            results = self._run_standard_test(config)
        
        self.results = self._convert_to_serializable(results)
        self._print_summary()
        
        return self.results
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using model's tokenizer."""
        encoded = self.model.encode(text)
        input_ids = encoded.get('input_ids')
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 2:
                return input_ids.shape[1]
            return input_ids.shape[0]
        return len(input_ids)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        encoded = self.model.encode(text)
        input_ids = encoded.get('input_ids')
        
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 2:
                input_ids = input_ids[0]
            if input_ids.shape[0] > max_tokens:
                input_ids = input_ids[:max_tokens]
            return self.model.decode(input_ids, skip_special_tokens=True)
        else:
            if len(input_ids) > max_tokens:
                input_ids = input_ids[:max_tokens]
            return self.model.decode(torch.tensor(input_ids), skip_special_tokens=True)
    
    def _run_standard_test(self, config: GenerationConfig) -> Dict[str, Any]:
        """Run standard context length test with SQuAD."""
        results_by_length = {}
        results_by_position = {}
        all_f1_scores = []
        all_lengths = []
        
        for length in self.context_lengths:
            results_by_length[str(length)] = {}
            
            for position in self.test_positions:
                result = self._evaluate_at_length(length, position, config)
                
                results_by_length[str(length)][position] = result
                
                all_lengths.append(length)
                all_f1_scores.append(result['f1'])
                
                if position not in results_by_position:
                    results_by_position[position] = []
                
                results_by_position[position].append({
                    'length': length,
                    'f1': result['f1'],
                    'accuracy': result['accuracy']
                })
        
        slope, r_squared = self._compute_degradation(all_lengths, all_f1_scores)
        position_summary = self._summarize_by_position(results_by_position)
        
        return {
            "by_length": results_by_length,
            "by_position": position_summary,
            "degradation": {
                "slope_per_token": float(slope),
                "slope_per_1k_tokens": float(slope * 1000),
                "r_squared": float(r_squared),
                "interpretation": self._interpret_slope(slope * 1000)
            },
            "metadata": {
                "context_lengths": self.context_lengths,
                "samples_per_length": self.samples_per_length,
                "positions_tested": self.test_positions,
                "test_type": "standard"
            }
        }
    
    def _run_needle_test(self, config: GenerationConfig) -> Dict[str, Any]:
        """Run needle-in-haystack test."""
        needle_config = GenerationConfig(max_new_tokens=50, do_sample=False, temperature=0.0)
        
        results_by_length = {}
        
        position_percentiles = self.needle_config.get('position_percentiles', [10, 30, 50, 70, 90])
        num_needles = min(self.needle_config.get('num_needles', 5), len(self.needles))
        
        for length in self.context_lengths:
            logger.info(f"Testing needle at context length: {length}")
            
            results_by_position = {}
            
            for position_pct in position_percentiles:
                scores = []
                
                for needle_idx in range(num_needles):
                    try:
                        needle = self.needles[needle_idx]
                        
                        context = self._build_needle_context(
                            needle['fact'],
                            target_length=length,
                            position_percentile=position_pct
                        )
                        
                        actual_length = self._count_tokens(context)
                        
                        if actual_length < length * 0.8:
                            logger.warning(f"Context too short: {actual_length} < {length}")
                            continue
                        
                        prompt = f"{context}\n\nBased on the information above, {needle['question']}"
                        
                        output = self.model.generate(prompt, needle_config)
                        response = output.generated_text.strip().lower()
                        
                        expected = needle['answer'].lower()
                        found = expected in response or any(
                            word in response for word in expected.split() if len(word) > 3
                        )
                        
                        scores.append(1.0 if found else 0.0)
                        
                        if (needle_idx + 1) % 2 == 0:
                            self._cleanup()
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(f"OOM at length {length}, position {position_pct}%")
                            self._cleanup()
                            break
                        raise
                
                if scores:
                    results_by_position[f"{position_pct}pct"] = {
                        'accuracy': float(np.mean(scores)),
                        'num_samples': len(scores)
                    }
                    logger.info(f"  Position {position_pct}%: {np.mean(scores):.3f}")
            
            results_by_length[str(length)] = results_by_position
        
        return {
            "by_length": results_by_length,
            "metadata": {
                "context_lengths": self.context_lengths,
                "position_percentiles": position_percentiles,
                "num_needles": num_needles,
                "test_type": "needle_in_haystack"
            }
        }
    
    def _load_data(self):
        """Load SQuAD and WikiText datasets."""
        logger.info("Loading datasets")
        
        self.squad = load_dataset("squad_v2", split="validation[:500]")
        
        wiki = load_dataset("wikitext", "wikitext-103-v1", split="train[:2000]")
        self.wiki_passages = [
            item['text'].strip() 
            for item in wiki 
            if len(item['text'].strip()) > 100
        ][:800]
        
        logger.info(f"Loaded {len(self.squad)} questions, {len(self.wiki_passages)} filler passages")
    
    def _build_context_at_position(
        self,
        answer_context: str,
        question: str,
        target_length: int,
        position: str = 'middle'
    ) -> str:
        """Build context with answer at specified position."""
        answer_length = self._count_tokens(answer_context)
        
        if answer_length >= target_length:
            return self._truncate_to_tokens(answer_context, target_length)
        
        tokens_needed = target_length - answer_length
        
        if position == 'start':
            before_tokens, after_tokens = 0, tokens_needed
        elif position == 'end':
            before_tokens, after_tokens = tokens_needed, 0
        elif position == 'early':
            before_tokens = int(tokens_needed * 0.25)
            after_tokens = tokens_needed - before_tokens
        elif position == 'late':
            before_tokens = int(tokens_needed * 0.75)
            after_tokens = tokens_needed - before_tokens
        else:
            before_tokens = tokens_needed // 2
            after_tokens = tokens_needed - before_tokens
        
        before_filler = self._build_filler(before_tokens) if before_tokens > 0 else ""
        after_filler = self._build_filler(after_tokens) if after_tokens > 0 else ""
        
        parts = []
        if before_filler:
            parts.append(before_filler)
        parts.append(answer_context)
        if after_filler:
            parts.append(after_filler)
        
        combined = "\n\n".join(parts) + f"\n\nQuestion: {question}\nAnswer:"
        
        return self._truncate_to_tokens(combined, target_length)
    
    def _build_needle_context(
        self,
        needle_fact: str,
        target_length: int,
        position_percentile: int
    ) -> str:
        """Build context with needle at specified percentile position."""
        needle_length = self._count_tokens(needle_fact)
        
        if needle_length >= target_length:
            return self._truncate_to_tokens(needle_fact, target_length)
        
        tokens_needed = target_length - needle_length
        
        before_tokens = int(tokens_needed * position_percentile / 100)
        after_tokens = tokens_needed - before_tokens
        
        before_filler = self._build_filler(before_tokens) if before_tokens > 0 else ""
        after_filler = self._build_filler(after_tokens) if after_tokens > 0 else ""
        
        parts = []
        if before_filler:
            parts.append(before_filler)
        parts.append(needle_fact)
        if after_filler:
            parts.append(after_filler)
        
        combined = "\n\n".join(parts)
        
        return self._truncate_to_tokens(combined, target_length)
    
    def _build_filler(self, num_tokens: int) -> str:
        """Build filler text of approximately num_tokens length."""
        if num_tokens <= 0:
            return ""
        
        filler_parts = []
        current_tokens = 0
        
        available = random.sample(
            self.wiki_passages,
            min(len(self.wiki_passages), num_tokens // 50 + 10)
        )
        
        for passage in available:
            if current_tokens >= num_tokens:
                break
            
            passage_truncated = self._truncate_to_tokens(passage, 150)
            filler_parts.append(passage_truncated)
            current_tokens += self._count_tokens(passage_truncated)
        
        return "\n\n".join(filler_parts)
    
    def _evaluate_at_length(
        self,
        context_length: int,
        position: str,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """Evaluate at specific context length and position."""
        logger.info(f"Testing {context_length} tokens (answer at {position})")
        
        correct = 0
        f1_scores = []
        total = 0
        
        for sample in self.squad:
            if total >= self.samples_per_length:
                break
            
            question = sample.get('question', '')
            context = sample.get('context', '')
            answers = sample.get('answers', {}).get('text', [])
            
            if not question or not answers or not context:
                continue
            
            try:
                full_context = self._build_context_at_position(
                    context, question, context_length, position
                )
                
                actual_length = self._count_tokens(full_context)
                if actual_length < context_length * 0.85:
                    continue
                
                output = self.model.generate(full_context, config)
                
                response = output.generated_text.lower().strip()
                
                if substring_match(response, answers[0]):
                    correct += 1
                
                f1 = max(token_f1(output.generated_text, ans) for ans in answers)
                f1_scores.append(f1)
                total += 1
                
                if total % 2 == 0:
                    self._cleanup()
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at {context_length} tokens")
                    self._cleanup()
                    break
            except Exception:
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        
        logger.info(f"Results: Acc={accuracy:.3f}, F1={mean_f1:.3f}, Samples={total}")
        
        return {
            'accuracy': float(accuracy),
            'f1': mean_f1,
            'num_samples': total,
            'answer_position': position
        }
    
    def _compute_degradation(self, lengths: List[int], f1_scores: List[float]):
        """Compute degradation slope and R-squared."""
        if len(f1_scores) < 2:
            return 0.0, 0.0
        
        lengths_array = np.array(lengths)
        f1_array = np.array(f1_scores)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            lengths_array, f1_array
        )
        
        r_squared = r_value ** 2
        
        logger.info(f"Degradation: {slope * 1000:.4f} per 1K tokens, R²={r_squared:.3f}")
        
        return slope, r_squared
    
    def _summarize_by_position(self, results_by_position: Dict) -> Dict:
        """Summarize results by answer position."""
        summary = {}
        
        for position, data in results_by_position.items():
            f1_values = [d['f1'] for d in data]
            summary[position] = {
                'mean_f1': float(np.mean(f1_values)),
                'std_f1': float(np.std(f1_values)),
                'min_f1': float(np.min(f1_values)),
                'max_f1': float(np.max(f1_values))
            }
        
        return summary
    
    @staticmethod
    def _interpret_slope(slope_per_1k: float) -> str:
        """Interpret degradation slope."""
        abs_slope = abs(slope_per_1k)
        if abs_slope < 0.001:
            return "negligible"
        elif abs_slope < 0.01:
            return "minimal"
        elif abs_slope < 0.05:
            return "moderate"
        else:
            return "significant"
    
    def _cleanup(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _print_summary(self):
        """Print formatted summary."""
        logger.info("Context Length Summary:")
        
        if self.needle_test:
            for length, data in self.results['by_length'].items():
                logger.info(f"{length} tokens:")
                for position, metrics in data.items():
                    logger.info(f"  {position}: Acc={metrics['accuracy']:.3f}")
        else:
            for length, data in self.results['by_length'].items():
                logger.info(f"{length} tokens:")
                for position, metrics in data.items():
                    logger.info(f"  {position}: F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}")
            
            deg = self.results['degradation']
            logger.info(f"Degradation: {deg['slope_per_1k_tokens']:.4f} per 1K tokens ({deg['interpretation']})")