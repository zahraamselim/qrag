"""
Main evaluation script for running comprehensive benchmark suite.

Usage:
    python run_evaluation.py --model_path <path> --model_type <type> --output_dir <dir>
    
Example:
    python run_evaluation.py \
        --model_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --model_type hf \
        --output_dir results/baseline
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

from models import AWQModel, HuggingFaceModel, BitsAndBytesModel, ExLlamaModel
from evaluators import (
    PerformanceEvaluator,
    PerplexityEvaluator,
    ContextLengthEvaluator,
    RAGEvaluator
)
from rag import RAGPipeline
from rag.config import StandardCorpus
from dataset_loader import DatasetLoader
from utils import setup_logging, save_json, get_gpu_info, clear_memory

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comprehensive model evaluation suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model ID"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['awq', 'hf', 'huggingface', 'bitsandbytes', 'bnb', 'exllama', 'gptq'],
        required=True,
        help="Model type"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (cuda:0, cpu)"
    )
    
    parser.add_argument(
        "--skip_performance",
        action="store_true",
        help="Skip performance evaluation"
    )
    
    parser.add_argument(
        "--skip_perplexity",
        action="store_true",
        help="Skip perplexity evaluation"
    )
    
    parser.add_argument(
        "--skip_context",
        action="store_true",
        help="Skip context length evaluation"
    )
    
    parser.add_argument(
        "--skip_rag",
        action="store_true",
        help="Skip RAG evaluation"
    )
    
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs for performance benchmarks"
    )
    
    parser.add_argument(
        "--ppl_samples",
        type=int,
        default=50,
        help="Number of samples for perplexity evaluation"
    )
    
    parser.add_argument(
        "--ppl_dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Dataset for perplexity (ultrachat_200k or wikitext)"
    )
    
    parser.add_argument(
        "--context_samples",
        type=int,
        default=5,
        help="Samples per context length"
    )
    
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs='+',
        default=[512, 1024, 2048, 4096],
        help="Context lengths to test"
    )
    
    parser.add_argument(
        "--needle_test",
        action="store_true",
        help="Run needle-in-haystack test instead of standard context eval"
    )
    
    parser.add_argument(
        "--rag_corpus",
        type=str,
        default="squad_v2",
        choices=['squad_v2', 'hotpotqa'],
        help="Corpus for RAG evaluation"
    )
    
    parser.add_argument(
        "--rag_max_documents",
        type=int,
        default=500,
        help="Maximum documents to index for RAG"
    )
    
    parser.add_argument(
        "--rag_max_questions",
        type=int,
        default=100,
        help="Maximum questions for RAG evaluation"
    )
    
    parser.add_argument(
        "--rag_compare_no_rag",
        action="store_true",
        help="Compare RAG vs no-RAG baseline"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Logging level"
    )
    
    return parser.parse_args()


def load_model(model_path: str, model_type: str, device: str):
    """Load model based on type."""
    logger.info(f"Loading {model_type} model from {model_path}")
    
    if model_type == 'awq':
        model = AWQModel(model_path, device=device)
    elif model_type in ['hf', 'huggingface']:
        model = HuggingFaceModel(model_path, device=device)
    elif model_type in ['bitsandbytes', 'bnb']:
        model = BitsAndBytesModel(model_path, device=device, load_in_4bit=True)
    elif model_type in ['exllama', 'gptq']:
        model = ExLlamaModel(model_path, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        model.load()
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def run_performance_eval(model, args, output_dir):
    """Run performance evaluation."""
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE EVALUATION")
    logger.info("="*60)
    
    try:
        perf_eval = PerformanceEvaluator(
            model,
            num_runs=args.num_runs,
            max_new_tokens=128
        )
        
        perf_results = perf_eval.run()
        perf_eval.save_results(output_dir / "performance.json")
        
        clear_memory()
        return perf_results
    except Exception as e:
        logger.error(f"Performance evaluation failed: {e}", exc_info=True)
        return None


def run_perplexity_eval(model, args, output_dir):
    """Run perplexity evaluation."""
    logger.info("\n" + "="*60)
    logger.info("PERPLEXITY EVALUATION")
    logger.info("="*60)
    
    try:
        if args.ppl_dataset == "wikitext":
            dataset_name = "wikitext"
            dataset_config = "wikitext-2-raw-v1"
            split = "test"
        else:
            dataset_name = "HuggingFaceH4/ultrachat_200k"
            dataset_config = None
            split = "train_sft"
        
        ppl_eval = PerplexityEvaluator(
            model,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_samples=args.ppl_samples
        )
        
        ppl_results = ppl_eval.run()
        save_json(ppl_results, output_dir / "perplexity.json")
        
        clear_memory()
        return ppl_results
    except Exception as e:
        logger.error(f"Perplexity evaluation failed: {e}", exc_info=True)
        return None


def run_context_eval(model, args, output_dir):
    """Run context length evaluation."""
    logger.info("\n" + "="*60)
    logger.info("CONTEXT LENGTH EVALUATION")
    logger.info("="*60)
    
    try:
        needle_config = {}
        if args.needle_test:
            needle_config = {
                'position_percentiles': [10, 50, 90],
                'num_needles': 3
            }
        
        context_eval = ContextLengthEvaluator(
            model,
            context_lengths=args.context_lengths,
            samples_per_length=args.context_samples,
            test_positions=['start', 'middle', 'end'],
            needle_test=args.needle_test,
            needle_config=needle_config
        )
        
        context_results = context_eval.run()
        save_json(context_results, output_dir / "context.json")
        
        clear_memory()
        return context_results
    except Exception as e:
        logger.error(f"Context evaluation failed: {e}", exc_info=True)
        return None


def run_rag_eval(model, args, output_dir):
    """Run RAG evaluation."""
    logger.info("\n" + "="*60)
    logger.info("RAG EVALUATION")
    logger.info("="*60)
    
    try:
        rag_config = {
            'chunking': {
                'strategy': 'semantic',
                'chunk_size': 512,
                'chunk_overlap': 50,
                'min_chunk_size': 100
            },
            'embedding': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'batch_size': 32,
                'normalize': True,
                'device': args.device
            },
            'vector_store': {
                'collection_name': 'rag_eval',
                'persist_directory': None
            },
            'retrieval': {
                'top_k': 3,
                'similarity_threshold': 0.0,
                'rerank': False,
                'diversity_penalty': 0.0
            },
            'generation': {
                'max_new_tokens': 64,
                'temperature': 0.3,
                'top_p': 0.9,
                'do_sample': True,
                'repetition_penalty': 1.15,
                'use_chat_template': True
            }
        }
        
        logger.info(f"Setting up RAG pipeline with {args.rag_corpus} corpus")
        pipeline = RAGPipeline(rag_config)
        pipeline.setup(model)
        
        logger.info("Loading standard corpus...")
        documents = StandardCorpus.load_corpus(
            corpus_name=args.rag_corpus,
            max_documents=args.rag_max_documents
        )
        
        doc_texts = [doc['text'] for doc in documents]
        indexing_time = pipeline.index_documents(doc_texts, show_progress=True)
        
        logger.info("Loading evaluation questions...")
        dataset_loader = DatasetLoader()
        
        if args.rag_corpus == 'squad_v2':
            test_data = dataset_loader.load(
                'squad_v2',
                split='validation',
                max_samples=args.rag_max_questions
            )
        elif args.rag_corpus == 'hotpotqa':
            test_data = dataset_loader.load(
                'hotpotqa',
                split='validation',
                max_samples=args.rag_max_questions
            )
        else:
            raise ValueError(f"Unsupported corpus: {args.rag_corpus}")
        
        questions = [item['question'] for item in test_data]
        answers = [
            item['answer'] if isinstance(item['answer'], str) else item['answer'][0]
            for item in test_data
        ]
        
        logger.info(f"Evaluating on {len(questions)} questions...")
        
        evaluator = RAGEvaluator(model, pipeline, config={})
        
        rag_results = evaluator.run(
            questions=questions,
            ground_truth_answers=answers,
            compare_no_rag=args.rag_compare_no_rag,
            save_detailed=True,
            output_dir=str(output_dir / "rag_detailed")
        )
        
        rag_results['metadata']['indexing_time_seconds'] = indexing_time
        rag_results['metadata']['corpus'] = args.rag_corpus
        rag_results['metadata']['num_documents'] = len(documents)
        rag_results['rag_stats'] = pipeline.get_stats()
        
        save_json(rag_results, output_dir / "rag.json")
        
        clear_memory()
        return rag_results
    except Exception as e:
        logger.error(f"RAG evaluation failed: {e}", exc_info=True)
        return None


def run_evaluation_suite(args):
    """Run complete evaluation suite."""
    start_time = datetime.now()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(args.log_level, str(output_dir / "evaluation.log"))
    
    logger.info("="*60)
    logger.info("MODEL EVALUATION SUITE")
    logger.info("="*60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Type: {args.model_type}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)
    
    gpu_info = get_gpu_info()
    if gpu_info['available']:
        logger.info(f"GPU: {gpu_info['device_name']}")
        logger.info(f"VRAM: {gpu_info['memory_total_mb']:.0f} MB")
    else:
        logger.warning("No GPU available")
    
    model = load_model(args.model_path, args.model_type, args.device)
    
    results = {
        'model_path': args.model_path,
        'model_type': args.model_type,
        'model_info': model.get_model_info(),
        'start_time': start_time.isoformat(),
        'gpu_info': gpu_info,
        'evaluations_run': []
    }
    
    if not args.skip_performance:
        perf_results = run_performance_eval(model, args, output_dir)
        if perf_results:
            results['performance'] = perf_results
            results['evaluations_run'].append('performance')
    
    if not args.skip_perplexity:
        ppl_results = run_perplexity_eval(model, args, output_dir)
        if ppl_results:
            results['perplexity'] = ppl_results
            results['evaluations_run'].append('perplexity')
    
    if not args.skip_context:
        context_results = run_context_eval(model, args, output_dir)
        if context_results:
            results['context'] = context_results
            results['evaluations_run'].append('context')
    
    if not args.skip_rag:
        rag_results = run_rag_eval(model, args, output_dir)
        if rag_results:
            results['rag'] = rag_results
            results['evaluations_run'].append('rag')
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results['end_time'] = end_time.isoformat()
    results['duration_seconds'] = duration
    results['duration_minutes'] = duration / 60
    
    save_json(results, output_dir / "complete_results.json")
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Duration: {duration/60:.1f} minutes")
    logger.info(f"Evaluations run: {', '.join(results['evaluations_run'])}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)
    
    model.unload()
    clear_memory()
    
    return results


def main():
    args = parse_args()
    
    try:
        results = run_evaluation_suite(args)
        return 0
    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())