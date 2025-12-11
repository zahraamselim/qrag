"""
Robustness evaluator for RAG systems.

Tests model behavior under adversarial conditions:
- Noisy/irrelevant document injection
- Passage order sensitivity
- Varying retrieval counts
"""

import logging
import random
import numpy as np
from typing import List, Dict, Any
from copy import deepcopy

from evaluators.base import BaseEvaluator
from metrics.accuracy import token_f1, exact_match
from models.model_interface import GenerationConfig

logger = logging.getLogger(__name__)


class RobustnessEvaluator(BaseEvaluator):
    """
    Evaluate RAG robustness to adversarial conditions.
    
    Tests:
    1. Noisy document injection
    2. Passage order sensitivity
    3. Retrieval count variation (k=1,3,5,10)
    """
    
    def __init__(
        self,
        model_interface,
        rag_pipeline,
        num_samples: int = 50,
        noise_ratios: List[float] = None,
        retrieval_ks: List[int] = None
    ):
        super().__init__(model_interface)
        self.rag_pipeline = rag_pipeline
        self.num_samples = num_samples
        self.noise_ratios = noise_ratios or [0.0, 0.2, 0.5]
        self.retrieval_ks = retrieval_ks or [1, 3, 5, 10]
    
    def run(
        self,
        questions: List[str],
        ground_truth_answers: List[str]
    ) -> Dict[str, Any]:
        """
        Run robustness evaluation.
        
        Args:
            questions: Test questions
            ground_truth_answers: Ground truth answers
            
        Returns:
            Robustness metrics
        """
        logger.info(f"Running robustness evaluation on {len(questions)} questions")
        
        config = GenerationConfig(max_new_tokens=50, do_sample=False)
        
        results = {
            'noise_injection': self._test_noise_injection(
                questions[:self.num_samples],
                ground_truth_answers[:self.num_samples],
                config
            ),
            'order_sensitivity': self._test_order_sensitivity(
                questions[:self.num_samples],
                ground_truth_answers[:self.num_samples],
                config
            ),
            'retrieval_count': self._test_retrieval_count(
                questions[:self.num_samples],
                ground_truth_answers[:self.num_samples],
                config
            ),
            'metadata': {
                'num_samples': min(self.num_samples, len(questions)),
                'noise_ratios': self.noise_ratios,
                'retrieval_ks': self.retrieval_ks
            }
        }
        
        self.results = self._convert_to_serializable(results)
        
        self._print_summary()
        
        return self.results
    
    def _test_noise_injection(
        self,
        questions: List[str],
        answers: List[str],
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """Test robustness to noisy/irrelevant documents."""
        logger.info("Testing noise injection")
        
        results_by_noise = {}
        
        original_top_k = self.rag_pipeline.retriever.top_k
        
        for noise_ratio in self.noise_ratios:
            logger.info(f"  Noise ratio: {noise_ratio:.1%}")
            
            f1_scores = []
            em_scores = []
            
            for question, answer in zip(questions, answers):
                try:
                    retrieved = self.rag_pipeline.retrieve(question)
                    
                    if noise_ratio > 0:
                        num_noise = int(len(retrieved) * noise_ratio)
                        noise_docs = self._generate_noise_documents(num_noise)
                        
                        insertion_points = random.sample(range(len(retrieved) + 1), num_noise)
                        for i, doc in zip(sorted(insertion_points, reverse=True), noise_docs):
                            retrieved.insert(i, doc)
                    
                    pred = self.rag_pipeline.generate_answer(question, retrieved)
                    
                    f1_scores.append(token_f1(pred, answer))
                    em_scores.append(exact_match(pred, answer))
                
                except Exception as e:
                    logger.warning(f"Error: {e}")
                    continue
            
            results_by_noise[f"noise_{int(noise_ratio*100)}pct"] = {
                'f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
                'exact_match': float(np.mean(em_scores)) if em_scores else 0.0,
                'num_samples': len(f1_scores)
            }
        
        self.rag_pipeline.retriever.top_k = original_top_k
        
        degradation = self._compute_degradation(results_by_noise)
        
        return {
            'by_noise_level': results_by_noise,
            'degradation': degradation
        }
    
    def _test_order_sensitivity(
        self,
        questions: List[str],
        answers: List[str],
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """Test sensitivity to passage ordering."""
        logger.info("Testing order sensitivity")
        
        order_variations = []
        f1_stds = []
        
        for question, answer in zip(questions, answers):
            try:
                retrieved = self.rag_pipeline.retrieve(question)
                
                if len(retrieved) < 2:
                    continue
                
                predictions = []
                
                for trial in range(3):
                    shuffled = deepcopy(retrieved)
                    random.shuffle(shuffled)
                    
                    pred = self.rag_pipeline.generate_answer(question, shuffled)
                    predictions.append(pred)
                
                f1_scores = [token_f1(pred, answer) for pred in predictions]
                
                order_variations.append(np.std(f1_scores))
                f1_stds.append(np.std(f1_scores))
            
            except Exception as e:
                logger.warning(f"Error: {e}")
                continue
        
        return {
            'mean_f1_std': float(np.mean(f1_stds)) if f1_stds else 0.0,
            'max_f1_std': float(np.max(f1_stds)) if f1_stds else 0.0,
            'num_samples': len(f1_stds),
            'interpretation': 'low' if np.mean(f1_stds) < 0.05 else 'medium' if np.mean(f1_stds) < 0.1 else 'high'
        }
    
    def _test_retrieval_count(
        self,
        questions: List[str],
        answers: List[str],
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """Test performance with varying retrieval counts."""
        logger.info("Testing retrieval count variation")
        
        results_by_k = {}
        original_top_k = self.rag_pipeline.retriever.top_k
        
        for k in self.retrieval_ks:
            if k > original_top_k:
                continue
            
            logger.info(f"  k={k}")
            
            self.rag_pipeline.retriever.top_k = k
            
            f1_scores = []
            em_scores = []
            
            for question, answer in zip(questions, answers):
                try:
                    retrieved = self.rag_pipeline.retrieve(question)
                    pred = self.rag_pipeline.generate_answer(question, retrieved)
                    
                    f1_scores.append(token_f1(pred, answer))
                    em_scores.append(exact_match(pred, answer))
                
                except Exception as e:
                    logger.warning(f"Error: {e}")
                    continue
            
            results_by_k[f"k_{k}"] = {
                'f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
                'exact_match': float(np.mean(em_scores)) if em_scores else 0.0,
                'num_samples': len(f1_scores)
            }
        
        self.rag_pipeline.retriever.top_k = original_top_k
        
        optimal_k = self._find_optimal_k(results_by_k)
        
        return {
            'by_k': results_by_k,
            'optimal_k': optimal_k,
            'diminishing_returns': self._compute_diminishing_returns(results_by_k)
        }
    
    def _generate_noise_documents(self, num_docs: int) -> List[Dict[str, Any]]:
        """Generate irrelevant noise documents."""
        noise_templates = [
            "This document discusses unrelated topics such as weather patterns and climate change.",
            "Historical events from the 19th century are covered in detail here.",
            "Technical specifications for various electronic devices are listed.",
            "Cooking recipes and culinary techniques from around the world.",
            "Geographic information about various regions and landmarks."
        ]
        
        docs = []
        for i in range(num_docs):
            docs.append({
                'text': random.choice(noise_templates) + f" Document {i}.",
                'score': 0.1,
                'metadata': {'noise': True}
            })
        
        return docs
    
    def _compute_degradation(self, results_by_noise: Dict) -> Dict[str, float]:
        """Compute degradation metrics from noise injection."""
        clean_f1 = results_by_noise.get('noise_0pct', {}).get('f1', 0.0)
        
        degradations = []
        for key, value in results_by_noise.items():
            if key != 'noise_0pct':
                degradation = clean_f1 - value['f1']
                degradations.append(degradation)
        
        return {
            'mean_degradation': float(np.mean(degradations)) if degradations else 0.0,
            'max_degradation': float(np.max(degradations)) if degradations else 0.0
        }
    
    def _find_optimal_k(self, results_by_k: Dict) -> int:
        """Find optimal retrieval count."""
        f1_scores = {int(k.split('_')[1]): v['f1'] for k, v in results_by_k.items()}
        
        if not f1_scores:
            return 5
        
        optimal = max(f1_scores.items(), key=lambda x: x[1])
        return optimal[0]
    
    def _compute_diminishing_returns(self, results_by_k: Dict) -> Dict[str, Any]:
        """Analyze diminishing returns of increasing k."""
        f1_scores = sorted(
            [(int(k.split('_')[1]), v['f1']) for k, v in results_by_k.items()],
            key=lambda x: x[0]
        )
        
        if len(f1_scores) < 2:
            return {'plateau_k': None, 'marginal_gains': []}
        
        marginal_gains = []
        for i in range(1, len(f1_scores)):
            gain = f1_scores[i][1] - f1_scores[i-1][1]
            marginal_gains.append({
                'from_k': f1_scores[i-1][0],
                'to_k': f1_scores[i][0],
                'gain': float(gain)
            })
        
        plateau_threshold = 0.01
        plateau_k = None
        for i, gain_info in enumerate(marginal_gains):
            if gain_info['gain'] < plateau_threshold:
                plateau_k = gain_info['from_k']
                break
        
        return {
            'plateau_k': plateau_k,
            'marginal_gains': marginal_gains
        }
    
    def _print_summary(self):
        """Print formatted summary."""
        logger.info("Robustness Summary:")
        
        noise = self.results['noise_injection']
        logger.info(f"Noise Injection:")
        for level, metrics in noise['by_noise_level'].items():
            logger.info(f"  {level}: F1={metrics['f1']:.3f}, EM={metrics['exact_match']:.3f}")
        logger.info(f"  Mean degradation: {noise['degradation']['mean_degradation']:.3f}")
        
        order = self.results['order_sensitivity']
        logger.info(f"Order Sensitivity:")
        logger.info(f"  Mean F1 std: {order['mean_f1_std']:.3f} ({order['interpretation']})")
        
        retr = self.results['retrieval_count']
        logger.info(f"Retrieval Count:")
        logger.info(f"  Optimal k: {retr['optimal_k']}")
        if retr['diminishing_returns']['plateau_k']:
            logger.info(f"  Plateau at k: {retr['diminishing_returns']['plateau_k']}")