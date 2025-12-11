"""
Base evaluator interface for all evaluation tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Evaluators orchestrate experiments: load data, run models, apply metrics,
    aggregate results, and format output.
    """
    
    def __init__(self, model_interface, config: Dict[str, Any] = None):
        """
        Initialize evaluator.
        
        Args:
            model_interface: ModelInterface instance
            config: Optional configuration dict
        """
        self.model = model_interface
        self.config = config or {}
        self.results = {}
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Execute evaluation.
        
        Returns:
            Dict containing evaluation results
        """
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """Get stored results."""
        return self.results
    
    def clear_results(self):
        """Clear stored results."""
        self.results = {}
    
    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy/torch types to native Python types."""
        import numpy as np
        import torch
        
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, dict):
            return {key: BaseEvaluator._convert_to_serializable(value) 
                    for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BaseEvaluator._convert_to_serializable(item) for item in obj]
        return obj