"""
Quantization methods package.

Provides interfaces for quantizing models with different methods.
"""

from quantization.base import QuantizationMethod
from quantization.awq_quantizer import AWQQuantizer

__all__ = ['QuantizationMethod', 'AWQQuantizer']