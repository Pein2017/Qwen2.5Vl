"""
Training components for Qwen2.5VL BBU training.

Use direct imports for better clarity:
- from src.training.unified_trainer import UnifiedBBUTrainer
- from src.training.callbacks import BestCheckpointCallback
- from src.training.stability import StabilityMonitor
"""

# Keep minimal exports for backwards compatibility
from .trainer import BBUTrainer, create_trainer

__all__ = ["BBUTrainer", "create_trainer"]
