"""
LM-Evaluation-Harness integration utilities.

Helpers for running and parsing lm-eval results.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class LMEvalRunner:
    """
    Wrapper for running lm-evaluation-harness tasks.
    
    Automates running standard benchmarks and parsing results.
    """
    
    TASK_GROUPS = {
        'reasoning': ['arc_easy', 'arc_challenge', 'winogrande', 'hellaswag'],
        'qa': ['nq_open', 'triviaqa'],
        'knowledge': ['mmlu'],
        'math': ['gsm8k'],
        'code': ['humaneval']
    }
    
    def __init__(self, model_path: str, model_type: str = "hf"):
        """
        Initialize LM-eval runner.
        
        Args:
            model_path: Path to model or HF model ID
            model_type: Model type (hf, awq, gptq, etc.)
        """
        self.model_path = model_path
        self.model_type = model_type
    
    def run_tasks(
        self,
        tasks: List[str],
        output_path: Path,
        num_fewshot: int = 0,
        batch_size: int = 1,
        device: str = "cuda:0"
    ) -> Dict[str, Any]:
        """
        Run lm-eval tasks.
        
        Args:
            tasks: List of task names or group names
            output_path: Where to save results
            num_fewshot: Number of few-shot examples
            batch_size: Batch size
            device: Device to use
            
        Returns:
            Parsed results dict
        """
        expanded_tasks = self._expand_task_groups(tasks)
        task_string = ",".join(expanded_tasks)
        
        logger.info(f"Running lm-eval tasks: {task_string}")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_args = self._get_model_args()
        
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", task_string,
            "--num_fewshot", str(num_fewshot),
            "--device", device,
            "--batch_size", str(batch_size),
            "--output_path", str(output_path)
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("LM-eval completed successfully")
            
            return self.parse_results(output_path)
        
        except subprocess.CalledProcessError as e:
            logger.error(f"LM-eval failed: {e.stderr}")
            raise
    
    def run_task_group(
        self,
        group: str,
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a predefined task group.
        
        Args:
            group: Group name (reasoning, qa, knowledge, math, code)
            output_path: Where to save results
            **kwargs: Additional arguments for run_tasks
            
        Returns:
            Parsed results
        """
        if group not in self.TASK_GROUPS:
            raise ValueError(f"Unknown task group: {group}")
        
        tasks = self.TASK_GROUPS[group]
        
        return self.run_tasks(tasks, output_path, **kwargs)
    
    def parse_results(self, results_dir: Path) -> Dict[str, Any]:
        """
        Parse lm-eval results from output directory.
        
        Args:
            results_dir: Directory containing results.json
            
        Returns:
            Parsed results dict
        """
        results_file = results_dir / "results.json"
        
        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            return {}
        
        with open(results_file, 'r') as f:
            raw_results = json.load(f)
        
        parsed = {
            'results': {},
            'config': raw_results.get('config', {}),
            'summary': {}
        }
        
        if 'results' in raw_results:
            for task, metrics in raw_results['results'].items():
                parsed['results'][task] = self._extract_key_metrics(metrics)
        
        parsed['summary'] = self._compute_summary(parsed['results'])
        
        return parsed
    
    def _get_model_args(self) -> str:
        """Get model_args string for lm-eval."""
        if self.model_type == "hf":
            return f"pretrained={self.model_path}"
        elif self.model_type == "awq":
            return f"pretrained={self.model_path}"
        elif self.model_type == "gptq":
            return f"pretrained={self.model_path},use_fast_tokenizer=False"
        else:
            return f"pretrained={self.model_path}"
    
    def _expand_task_groups(self, tasks: List[str]) -> List[str]:
        """Expand task group names to individual tasks."""
        expanded = []
        
        for task in tasks:
            if task in self.TASK_GROUPS:
                expanded.extend(self.TASK_GROUPS[task])
            else:
                expanded.append(task)
        
        return expanded
    
    def _extract_key_metrics(self, task_metrics: Dict) -> Dict[str, float]:
        """Extract key metrics from task results."""
        key_metrics = {}
        
        metric_priority = ['acc', 'acc_norm', 'em', 'f1', 'bleu', 'rouge']
        
        for metric in metric_priority:
            if metric in task_metrics:
                key_metrics[metric] = task_metrics[metric]
        
        for key, value in task_metrics.items():
            if key not in key_metrics and isinstance(value, (int, float)):
                key_metrics[key] = value
        
        return key_metrics
    
    def _compute_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute summary statistics across tasks."""
        all_accs = []
        
        for task, metrics in results.items():
            if 'acc' in metrics:
                all_accs.append(metrics['acc'])
            elif 'acc_norm' in metrics:
                all_accs.append(metrics['acc_norm'])
        
        summary = {}
        
        if all_accs:
            import numpy as np
            summary['mean_accuracy'] = float(np.mean(all_accs))
            summary['std_accuracy'] = float(np.std(all_accs))
            summary['num_tasks'] = len(all_accs)
        
        return summary


def run_lm_eval_suite(
    model_path: str,
    model_type: str,
    output_dir: Path,
    task_groups: List[str] = None
) -> Dict[str, Any]:
    """
    Run complete lm-eval suite on a model.
    
    Args:
        model_path: Model path or ID
        model_type: Model type
        output_dir: Output directory
        task_groups: Task groups to run (default: reasoning, qa)
        
    Returns:
        Combined results from all task groups
    """
    if task_groups is None:
        task_groups = ['reasoning', 'qa']
    
    runner = LMEvalRunner(model_path, model_type)
    
    all_results = {
        'model_path': model_path,
        'model_type': model_type,
        'task_groups': {}
    }
    
    for group in task_groups:
        logger.info(f"Running task group: {group}")
        
        group_output = output_dir / group
        results = runner.run_task_group(group, group_output)
        
        all_results['task_groups'][group] = results
    
    combined_file = output_dir / "lm_eval_combined.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Combined results saved to {combined_file}")
    
    return all_results


def compare_lm_eval_results(
    baseline_results: Dict,
    quantized_results: Dict
) -> Dict[str, Any]:
    """
    Compare lm-eval results between baseline and quantized models.
    
    Args:
        baseline_results: Baseline model results
        quantized_results: Quantized model results
        
    Returns:
        Comparison dict with degradation metrics
    """
    comparison = {
        'by_task': {},
        'summary': {}
    }
    
    baseline_tasks = baseline_results.get('results', {})
    quantized_tasks = quantized_results.get('results', {})
    
    for task in baseline_tasks:
        if task not in quantized_tasks:
            continue
        
        base_metrics = baseline_tasks[task]
        quant_metrics = quantized_tasks[task]
        
        task_comp = {}
        
        for metric in base_metrics:
            if metric in quant_metrics:
                base_val = base_metrics[metric]
                quant_val = quant_metrics[metric]
                
                degradation = quant_val - base_val
                degradation_pct = (degradation / base_val * 100) if base_val != 0 else 0
                
                task_comp[metric] = {
                    'baseline': base_val,
                    'quantized': quant_val,
                    'degradation': degradation,
                    'degradation_pct': degradation_pct
                }
        
        comparison['by_task'][task] = task_comp
    
    import numpy as np
    all_degradations = []
    
    for task_comp in comparison['by_task'].values():
        for metric_comp in task_comp.values():
            all_degradations.append(metric_comp['degradation_pct'])
    
    if all_degradations:
        comparison['summary'] = {
            'mean_degradation_pct': float(np.mean(all_degradations)),
            'max_degradation_pct': float(np.max(all_degradations)),
            'min_degradation_pct': float(np.min(all_degradations)),
            'num_comparisons': len(all_degradations)
        }
    
    return comparison