"""
Main evaluation script for running comprehensive benchmark suite.

Usage:
    python run_evaluation.py --model_path <path> --model_type <type> --output_dir <dir>
    
Example:
    python run_evaluation.py \
        --model_path "mistralai/Mistral-7B-v0.1" \
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
    ContextLengthEvaluator
)
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
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs for performance benchmarks"
    )
    
    parser.add_argument(
        "--ppl_samples",
        type=int,
        default=100,
        help="Number of samples for perplexity evaluation"
    )
    
    parser.add_argument(
        "--context_samples",
        type=int,
        default=25,
        help="Samples per context length"
    )
    
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs='+',
        default=[512, 1024, 2048],
        help="Context lengths to test (default: 512 1024 2048)"
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
    
    model.load()
    return model


def run_evaluation_suite(args):
    """Run complete evaluation suite."""
    start_time = datetime.now()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(args.log_level, str(output_dir / "evaluation.log"))
    
    logger.info("Model Evaluation Suite")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Type: {args.model_type}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {output_dir}")
    
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
        'gpu_info': gpu_info
    }
    
    if not args.skip_performance:
        logger.info("\nPerformance Evaluation")
        
        perf_eval = PerformanceEvaluator(
            model,
            num_runs=args.num_runs,
            max_new_tokens=128
        )
        
        perf_results = perf_eval.run()
        results['performance'] = perf_results
        
        perf_eval.save_results(output_dir / "performance.json")
        
        clear_memory()
    
    if not args.skip_perplexity:
        logger.info("\nPerplexity Evaluation")
        
        ppl_eval = PerplexityEvaluator(
            model,
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            max_samples=args.ppl_samples
        )
        
        ppl_results = ppl_eval.run()
        results['perplexity'] = ppl_results
        
        save_json(ppl_results, output_dir / "perplexity.json")
        
        clear_memory()
    
    if not args.skip_context:
        logger.info("\nContext Length Evaluation")
        
        context_eval = ContextLengthEvaluator(
            model,
            context_lengths=args.context_lengths,
            samples_per_length=args.context_samples,
            test_positions=['middle']
        )
        
        context_results = context_eval.run()
        results['context'] = context_results
        
        save_json(context_results, output_dir / "context.json")
        
        clear_memory()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results['end_time'] = end_time.isoformat()
    results['duration_seconds'] = duration
    results['duration_minutes'] = duration / 60
    
    save_json(results, output_dir / "complete_results.json")
    
    logger.info("\nEvaluation Complete")
    logger.info(f"Duration: {duration/60:.1f} minutes")
    logger.info(f"Results saved to: {output_dir}")
    
    model.unload()
    clear_memory()
    
    return results


def main():
    args = parse_args()
    
    try:
        results = run_evaluation_suite(args)
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())