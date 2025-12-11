"""
Common utility functions.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict
import torch
import numpy as np

logger = logging.getLogger(__name__)


def clear_memory():
    """Clear GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )


def save_json(data: Dict[str, Any], output_path: Path):
    """
    Save data to JSON with numpy/torch conversion.
    
    Args:
        data: Data to save
        output_path: Path to save to
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert(obj):
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
            return {key: convert(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj
    
    converted_data = convert(data)
    
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def load_json(input_path: Path) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        input_path: Path to load from
        
    Returns:
        Loaded data
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    return data


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information.
    
    Returns:
        Dict with GPU details
    """
    if not torch.cuda.is_available():
        return {'available': False}
    
    return {
        'available': True,
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'device_name': torch.cuda.get_device_name(0),
        'memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
        'memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
        'memory_total_mb': torch.cuda.get_device_properties(0).total_memory / (1024**2)
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_size(bytes_val: int) -> str:
    """
    Format bytes into human-readable size.
    
    Args:
        bytes_val: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f}PB"