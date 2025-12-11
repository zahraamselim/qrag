"""
AWQ quantization implementation.

Activation-aware Weight Quantization for efficient 4-bit quantization.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from datasets import load_dataset

from quantization.base import QuantizationMethod

logger = logging.getLogger(__name__)


class AWQQuantizer(QuantizationMethod):
    """
    AWQ quantization method.
    
    Default config:
    - w_bit: 4 (weight bits)
    - q_group_size: 128
    - zero_point: True
    - version: GEMM
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'w_bit': 4,
            'q_group_size': 128,
            'zero_point': True,
            'version': 'GEMM',
            'calib_dataset': 'wikitext',
            'calib_config': 'wikitext-2-raw-v1',
            'calib_samples': 128,
            'calib_max_length': 2048
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
    
    def quantize(
        self,
        model_path: str,
        output_path: str,
        calibration_data: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Quantize model using AWQ.
        
        Args:
            model_path: HuggingFace model ID or path
            output_path: Path to save quantized model
            calibration_data: Optional list of calibration texts
            
        Returns:
            Quantization metadata
        """
        logger.info(f"Quantizing {model_path} with AWQ")
        logger.info(f"Config: {self.config}")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        if calibration_data is None:
            logger.info("Preparing calibration data from dataset")
            calibration_data = self._prepare_calibration_data(tokenizer)
        
        logger.info(f"Using {len(calibration_data)} calibration samples")
        
        logger.info("Loading model for quantization")
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        
        quant_config = {
            'zero_point': self.config['zero_point'],
            'q_group_size': self.config['q_group_size'],
            'w_bit': self.config['w_bit'],
            'version': self.config['version']
        }
        
        logger.info("Starting quantization")
        start_time = time.time()
        
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calibration_data
        )
        
        quantization_time = time.time() - start_time
        
        logger.info(f"Quantization completed in {quantization_time:.2f}s")
        
        logger.info(f"Saving quantized model to {output_path}")
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        import json
        with open(output_path / "quantization_metadata.json", "w") as f:
            json.dump({
                'method': 'AWQ',
                'quantization_time_seconds': quantization_time,
                'config': quant_config,
                'calibration_samples': len(calibration_data),
                'source_model': model_path
            }, f, indent=2)
        
        logger.info("Quantization complete")
        
        return {
            'method': 'AWQ',
            'output_path': str(output_path),
            'quantization_time': quantization_time,
            'config': quant_config
        }
    
    def load_quantized(self, model_path: str, device: str = "cuda:0"):
        """
        Load AWQ quantized model.
        
        Args:
            model_path: Path to quantized model
            device: Device to load on
            
        Returns:
            Loaded AWQ model
        """
        logger.info(f"Loading AWQ model from {model_path}")
        
        model = AutoAWQForCausalLM.from_quantized(
            model_path,
            fuse_layers=True
        )
        
        logger.info(f"Model loaded on {model.device}")
        
        return model
    
    def _prepare_calibration_data(self, tokenizer) -> List[str]:
        """Prepare calibration data from dataset."""
        dataset = load_dataset(
            self.config['calib_dataset'],
            self.config['calib_config'],
            split="train"
        )
        
        dataset = dataset.filter(lambda x: len(x["text"]) > 200)
        
        texts = []
        for i in range(min(self.config['calib_samples'], len(dataset))):
            texts.append(dataset[i]["text"])
        
        return texts
    
    def validate_config(self) -> bool:
        """Validate AWQ configuration."""
        required_keys = ['w_bit', 'q_group_size', 'zero_point', 'version']
        
        for key in required_keys:
            if key not in self.config:
                logger.error(f"Missing required config key: {key}")
                return False
        
        if self.config['w_bit'] not in [2, 3, 4, 8]:
            logger.error(f"Invalid w_bit: {self.config['w_bit']}")
            return False
        
        return True