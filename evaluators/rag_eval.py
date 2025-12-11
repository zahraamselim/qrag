"""
RAG evaluator for comprehensive retrieval-augmented generation evaluation.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from evaluators.base import BaseEvaluator
from metrics.accuracy import token_f1, exact_match, compute_rouge_scores
from metrics.faithfulness import token_overlap, context_sufficiency, answer_coverage
from metrics.retrieval import retrieval_score_statistics, context_length_statistics

logger = logging.getLogger(__name__)


class RAGEvaluator(BaseEvaluator):
    """
    Comprehensive RAG evaluation.
    
    Measures:
    1. Retrieval quality: context sufficiency, precision, scores
    2. Answer quality: EM, F1, ROUGE, faithfulness
    3. Efficiency: retrieval and generation latency
    4. RAG vs no-RAG comparison
    """
    
    def __init__(self, model_interface, rag_pipeline, config: Dict[str, Any] = None):
        super().__init__(model_interface, config)
        self.rag_pipeline = rag_pipeline
    
    def run(
        self,
        questions: List[str],
        ground_truth_answers: List[str],
        documents: Optional[List[str]] = None,
        compare_no_rag: bool = True,
        save_detailed: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive RAG evaluation.
        
        Args:
            questions: List of test questions
            ground_truth_answers: List of ground truth answers
            documents: Optional documents to index
            compare_no_rag: Whether to compare with no-RAG baseline
            save_detailed: Save per-question details
            output_dir: Directory for detailed results
            
        Returns:
            Evaluation results dict
        """
        logger.info(f"Evaluating RAG on {len(questions)} questions")
        
        if documents:
            logger.info(f"Indexing {len(documents)} documents")
            self.rag_pipeline.index_documents(documents, show_progress=True)
        
        stats = self.rag_pipeline.get_stats()
        if stats['vector_store'].get('count', 0) == 0:
            raise ValueError("No documents indexed")
        
        logger.info(f"Vector store: {stats['vector_store']['count']} chunks")
        
        detailed_results = [] if save_detailed else None
        
        rag_predictions = []
        no_rag_predictions = []
        contexts = []
        retrieved_chunks_list = []
        retrieval_times = []
        rag_gen_times = []
        no_rag_gen_times = []
        
        for i, (question, ground_truth) in enumerate(zip(questions, ground_truth_answers)):
            try:
                retrieval_start = time.perf_counter()
                retrieved_chunks = self.rag_pipeline.retrieve(question)
                retrieval_time = (time.perf_counter() - retrieval_start) * 1000
                retrieval_times.append(retrieval_time)
                
                retrieved_chunks_list.append(retrieved_chunks)
                context_str = '\n\n'.join([chunk['text'] for chunk in retrieved_chunks])
                contexts.append(context_str)
                
                rag_gen_start = time.perf_counter()
                rag_answer = self.rag_pipeline.generate_answer(question, retrieved_chunks)
                rag_gen_time = (time.perf_counter() - rag_gen_start) * 1000
                rag_gen_times.append(rag_gen_time)
                rag_predictions.append(rag_answer)
                
                no_rag_answer = None
                no_rag_time = 0.0
                if compare_no_rag:
                    no_rag_start = time.perf_counter()
                    no_rag_answer = self.rag_pipeline.generator.generate_without_context(question)
                    no_rag_time = (time.perf_counter() - no_rag_start) * 1000
                    no_rag_gen_times.append(no_rag_time)
                    no_rag_predictions.append(no_rag_answer)
                
                if save_detailed:
                    detailed_results.append({
                        'question_id': i + 1,
                        'question': question,
                        'ground_truth': ground_truth,
                        'rag_answer': rag_answer,
                        'no_rag_answer': no_rag_answer,
                        'num_chunks_retrieved': len(retrieved_chunks),
                        'avg_retrieval_score': float(np.mean([c.get('score', 0.0) for c in retrieved_chunks])),
                        'context_length_chars': len(context_str),
                        'retrieval_time_ms': retrieval_time,
                        'rag_generation_time_ms': rag_gen_time,
                        'no_rag_generation_time_ms': no_rag_time
                    })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(questions)}")
            
            except Exception as e:
                logger.error(f"Error on question {i+1}: {e}")
                rag_predictions.append("")
                if compare_no_rag:
                    no_rag_predictions.append("")
                contexts.append("")
                retrieved_chunks_list.append([])
                retrieval_times.append(0.0)
                rag_gen_times.append(0.0)
                if compare_no_rag:
                    no_rag_gen_times.append(0.0)
        
        logger.info("Computing metrics")
        
        results = {}
        
        results['retrieval_quality'] = self._evaluate_retrieval_quality(
            questions, contexts, ground_truth_answers, retrieved_chunks_list
        )
        
        results['answer_quality'] = self._evaluate_answer_quality(
            rag_predictions, ground_truth_answers, contexts
        )
        
        if compare_no_rag and no_rag_predictions:
            results['no_rag_baseline'] = self._evaluate_answer_quality_simple(
                no_rag_predictions, ground_truth_answers
            )
            
            results['rag_improvement'] = self._compute_improvement(
                results['answer_quality'], results['no_rag_baseline']
            )
        
        results['efficiency'] = self._evaluate_efficiency(
            retrieval_times, rag_gen_times, no_rag_gen_times if compare_no_rag else None
        )
        
        results['metadata'] = {
            'num_questions': len(questions),
            'num_chunks_indexed': stats['vector_store'].get('count', 0),
            'avg_chunks_per_query': float(np.mean([len(c) for c in retrieved_chunks_list])),
            'retrieval_config': self.rag_pipeline.retriever.top_k
        }
        
        self.results = self._convert_to_serializable(results)
        
        if save_detailed and output_dir and detailed_results:
            self._save_detailed_results(detailed_results, output_dir)
        
        self._print_summary()
        
        return self.results
    
    def _evaluate_retrieval_quality(
        self,
        questions: List[str],
        contexts: List[str],
        ground_truths: List[str],
        retrieved_chunks: List[List[Dict]]
    ) -> Dict[str, Any]:
        """Evaluate retrieval quality."""
        sufficiency_scores = []
        coverage_scores = []
        retrieval_scores = []
        
        for question, context, answer, chunks in zip(questions, contexts, ground_truths, retrieved_chunks):
            if not context.strip():
                sufficiency_scores.append(0.0)
                coverage_scores.append(0.0)
                retrieval_scores.append(0.0)
                continue
            
            sufficiency_scores.append(context_sufficiency(answer, context))
            coverage_scores.append(answer_coverage(answer, context))
            
            if chunks:
                retrieval_scores.append(np.mean([c.get('score', 0.0) for c in chunks]))
            else:
                retrieval_scores.append(0.0)
        
        context_stats = context_length_statistics(contexts)
        score_stats = retrieval_score_statistics(retrieval_scores)
        
        return {
            'context_sufficiency': float(np.mean(sufficiency_scores)),
            'answer_coverage': float(np.mean(coverage_scores)),
            'avg_retrieval_score': score_stats['mean'],
            'avg_context_length_words': context_stats['mean_words'],
            'retrieval_score_std': score_stats['std']
        }
    
    def _evaluate_answer_quality(
        self,
        predictions: List[str],
        references: List[str],
        contexts: List[str]
    ) -> Dict[str, Any]:
        """Evaluate answer quality with faithfulness."""
        em_scores = [exact_match(pred, ref) for pred, ref in zip(predictions, references)]
        f1_scores = [token_f1(pred, ref) for pred, ref in zip(predictions, references)]
        faithfulness_scores = [token_overlap(pred, ctx) for pred, ctx in zip(predictions, contexts)]
        
        results = {
            'exact_match': float(np.mean(em_scores)),
            'f1': float(np.mean(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
            'faithfulness': float(np.mean(faithfulness_scores)),
            'avg_answer_length_words': float(np.mean([len(p.split()) for p in predictions]))
        }
        
        rouge = compute_rouge_scores(predictions, references)
        if rouge:
            results.update(rouge)
        
        return results
    
    def _evaluate_answer_quality_simple(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Any]:
        """Evaluate answer quality without faithfulness (for no-RAG baseline)."""
        em_scores = [exact_match(pred, ref) for pred, ref in zip(predictions, references)]
        f1_scores = [token_f1(pred, ref) for pred, ref in zip(predictions, references)]
        
        return {
            'exact_match': float(np.mean(em_scores)),
            'f1': float(np.mean(f1_scores)),
            'avg_answer_length_words': float(np.mean([len(p.split()) for p in predictions]))
        }
    
    def _compute_improvement(self, rag_metrics: Dict, baseline_metrics: Dict) -> Dict[str, Any]:
        """Compute RAG improvement over baseline."""
        f1_gain = rag_metrics['f1'] - baseline_metrics['f1']
        f1_gain_percent = (f1_gain / max(baseline_metrics['f1'], 0.01)) * 100
        em_gain = rag_metrics['exact_match'] - baseline_metrics['exact_match']
        
        return {
            'f1_gain': float(f1_gain),
            'f1_gain_percent': float(f1_gain_percent),
            'em_gain': float(em_gain)
        }
    
    def _evaluate_efficiency(
        self,
        retrieval_times: List[float],
        rag_gen_times: List[float],
        no_rag_gen_times: Optional[List[float]]
    ) -> Dict[str, Any]:
        """Evaluate efficiency metrics."""
        metrics = {}
        
        if retrieval_times:
            metrics['avg_retrieval_time_ms'] = float(np.mean(retrieval_times))
            metrics['retrieval_time_std_ms'] = float(np.std(retrieval_times))
        
        if rag_gen_times:
            metrics['avg_rag_generation_time_ms'] = float(np.mean(rag_gen_times))
            metrics['rag_generation_time_std_ms'] = float(np.std(rag_gen_times))
        
        if no_rag_gen_times:
            metrics['avg_no_rag_generation_time_ms'] = float(np.mean(no_rag_gen_times))
            
            rag_total = np.mean(retrieval_times) + np.mean(rag_gen_times)
            no_rag_total = np.mean(no_rag_gen_times)
            metrics['rag_overhead_ms'] = float(rag_total - no_rag_total)
            metrics['rag_overhead_percent'] = float((rag_total - no_rag_total) / no_rag_total * 100)
        
        return metrics
    
    def _save_detailed_results(self, detailed_results: List[Dict], output_dir: str):
        """Save detailed per-question results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_path = output_path / 'rag_evaluation_detailed.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': self.results,
                'per_question': detailed_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to {json_path}")
    
    def _print_summary(self):
        """Print evaluation summary."""
        retr = self.results['retrieval_quality']
        logger.info("Retrieval Quality:")
        logger.info(f"  Context sufficiency: {retr['context_sufficiency']:.3f}")
        logger.info(f"  Answer coverage: {retr['answer_coverage']:.3f}")
        logger.info(f"  Avg retrieval score: {retr['avg_retrieval_score']:.3f}")
        
        ans = self.results['answer_quality']
        logger.info("Answer Quality (RAG):")
        logger.info(f"  Exact match: {ans['exact_match']:.3f}")
        logger.info(f"  F1 score: {ans['f1']:.3f}")
        logger.info(f"  Faithfulness: {ans['faithfulness']:.3f}")
        
        if 'rag_improvement' in self.results:
            imp = self.results['rag_improvement']
            logger.info("RAG vs No-RAG:")
            logger.info(f"  F1 gain: {imp['f1_gain']:+.3f} ({imp['f1_gain_percent']:+.1f}%)")
        
        eff = self.results['efficiency']
        logger.info("Efficiency:")
        logger.info(f"  Retrieval: {eff.get('avg_retrieval_time_ms', 0):.1f}ms")
        logger.info(f"  Generation: {eff.get('avg_rag_generation_time_ms', 0):.1f}ms")