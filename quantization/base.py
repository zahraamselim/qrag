"""
Base quantization interface.

Provides abstract interface for all quantization methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class QuantizationMethod(ABC):
    """
    Abstract base class for quantization methods.
    
    All quantization implementations must inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize quantization method.
        
        Args:
            config: Configuration dict with method-specific parameters
        """
        self.config = config or {}
        self.method_name = self.__class__.__name__
    
    @abstractmethod
    def quantize(
        self,
        model_path: str,
        output_path: str,
        calibration_data: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Quantize a model.
        
        Args:
            model_path: Path to original model or HF model ID
            output_path: Path to save quantized model
            calibration_data: Optional calibration dataset
            
        Returns:
            Dict with quantization metadata (time, size, etc.)
        """
        pass
    
    @abstractmethod
    def load_quantized(
        self,
        model_path: str,
        device: str = "cuda:0"
    ):
        """
        Load a quantized model.
        
        Args:
            model_path: Path to quantized model
            device: Device to load on
            
        Returns:
            Loaded model instance
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def validate_config(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if valid, False otherwise
        """
        return True
    
    @staticmethod
    def estimate_size_reduction(
        original_bits: int,
        quantized_bits: int,
        num_parameters: int
    ) -> Dict[str, float]:
        """
        Estimate size reduction from quantization.
        
        Args:
            original_bits: Bits per parameter in original model
            quantized_bits: Bits per parameter after quantization
            num_parameters: Number of parameters
            
        Returns:
            Dict with size estimates in GB
        """
        original_bytes = (num_parameters * original_bits) / 8
        quantized_bytes = (num_parameters * quantized_bits) / 8
        
        original_gb = original_bytes / (1024**3)
        quantized_gb = quantized_bytes / (1024**3)
        
        compression_ratio = original_gb / quantized_gb if quantized_gb > 0 else 0
        savings_percent = ((original_gb - quantized_gb) / original_gb * 100) if original_gb > 0 else 0
        
        return {
            'original_size_gb': original_gb,
            'quantized_size_gb': quantized_gb,
            'compression_ratio': compression_ratio,
            'savings_percent': savings_percent
        }