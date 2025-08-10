"""
Performance monitoring utilities for model initialization optimization.

This module provides tools to track and measure the performance improvements
from the token expansion optimization strategies.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

from .rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for model initialization."""

    total_init_time: float = 0.0
    vocab_extension_time: float = 0.0
    embedding_resize_time: float = 0.0
    coordinate_init_time: float = 0.0
    checkpoint_load_time: float = 0.0

    # Optimization flags
    used_intelligent_detection: bool = False
    used_safetensors: bool = False
    used_lazy_loading: bool = False
    used_distributed_optimization: bool = False

    # Memory metrics
    peak_memory_mb: Optional[float] = None
    vocab_size_before: int = 0
    vocab_size_after: int = 0

    # Additional metrics
    coordinate_tokens_count: int = 0
    rank: int = 0

    def __post_init__(self):
        """Calculate derived metrics."""
        self.vocab_expansion_ratio = (
            self.vocab_size_after / self.vocab_size_before
            if self.vocab_size_before > 0
            else 0.0
        )


class PerformanceMonitor:
    """Monitor and track performance metrics during model initialization."""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._start_times: Dict[str, float] = {}

    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time operations."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            setattr(self.metrics, f"{operation_name}_time", elapsed)
            logger.info(f"⏱️ {operation_name}: {elapsed:.2f}s")

    def start_timer(self, operation_name: str):
        """Start timing an operation."""
        self._start_times[operation_name] = time.time()

    def end_timer(self, operation_name: str):
        """End timing an operation and record the result."""
        if operation_name in self._start_times:
            elapsed = time.time() - self._start_times[operation_name]
            setattr(self.metrics, f"{operation_name}_time", elapsed)
            logger.info(f"⏱️ {operation_name}: {elapsed:.2f}s")
            del self._start_times[operation_name]

    def record_optimization_used(self, optimization: str, used: bool = True):
        """Record which optimizations were used."""
        setattr(self.metrics, f"used_{optimization}", used)
        if used:
            logger.info(f"🚀 Optimization enabled: {optimization}")

    def record_memory_usage(self):
        """Record current memory usage."""
        try:
            import torch

            if torch.cuda.is_available():
                self.metrics.peak_memory_mb = (
                    torch.cuda.max_memory_allocated() / 1024 / 1024
                )
        except Exception:
            pass

    def record_vocab_sizes(self, before: int, after: int):
        """Record vocabulary sizes before and after extension."""
        self.metrics.vocab_size_before = before
        self.metrics.vocab_size_after = after

    def record_coordinate_tokens(self, count: int):
        """Record number of coordinate tokens."""
        self.metrics.coordinate_tokens_count = count

    def record_rank(self, rank: int):
        """Record distributed training rank."""
        self.metrics.rank = rank

    def get_summary(self) -> str:
        """Get a formatted summary of performance metrics."""
        m = self.metrics

        summary = [
            "🚀 Model Initialization Performance Summary",
            "=" * 50,
            f"Total initialization time: {m.total_init_time:.2f}s",
            f"Checkpoint loading: {m.checkpoint_load_time:.2f}s",
            f"Vocabulary extension: {m.vocab_extension_time:.2f}s",
            f"Embedding resize: {m.embedding_resize_time:.2f}s",
            f"Coordinate initialization: {m.coordinate_init_time:.2f}s",
            "",
            "Optimizations Used:",
            f"  ✅ Intelligent detection: {m.used_intelligent_detection}",
            f"  ✅ SafeTensors format: {m.used_safetensors}",
            f"  ✅ Lazy loading: {m.used_lazy_loading}",
            f"  ✅ Distributed optimization: {m.used_distributed_optimization}",
            "",
            "Vocabulary Statistics:",
            f"  Before: {m.vocab_size_before:,} tokens",
            f"  After: {m.vocab_size_after:,} tokens",
            f"  Expansion ratio: {m.vocab_expansion_ratio:.2f}x",
            f"  Coordinate tokens: {m.coordinate_tokens_count:,}",
            "",
            f"Rank: {m.rank}",
        ]

        if m.peak_memory_mb:
            summary.append(f"Peak memory: {m.peak_memory_mb:.1f} MB")

        return "\n".join(summary)

    def log_summary(self):
        """Log the performance summary."""
        logger.info(self.get_summary())


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def reset_performance_monitor():
    """Reset the global performance monitor."""
    global _global_monitor
    _global_monitor = PerformanceMonitor()


@contextmanager
def monitor_initialization():
    """Context manager for monitoring complete model initialization."""
    monitor = get_performance_monitor()
    monitor.start_timer("total_init")
    try:
        yield monitor
    finally:
        monitor.end_timer("total_init")
        monitor.record_memory_usage()
        monitor.log_summary()
