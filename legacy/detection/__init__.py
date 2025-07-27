"""
Detection Package

This package contains object detection components for BBU training:
- detection_loss: Hungarian matching and multi-task loss computation
- detection_head: DETR-style decoder with dual-stream processing
- detection_adapter: Vision and language adapters for detection
"""

from .detection_loss import DetectionLoss
from .detection_head import DetectionHead
from .detection_adapter import DetectionAdapter

__all__ = [
    "DetectionLoss",
    "DetectionHead", 
    "DetectionAdapter",
]