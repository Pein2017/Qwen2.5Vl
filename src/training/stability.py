"""
Training stability utilities for Qwen2.5VL BBU training.

Provides NaN detection, gradient monitoring, and stability recovery mechanisms.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

from ..logging import get_logger


@dataclass
class StabilityMetrics:
    """Track training stability metrics."""

    # NaN tracking
    consecutive_nan_count: int = 0
    total_nan_count: int = 0
    nan_steps: List[int] = field(default_factory=list)

    # Zero loss tracking
    consecutive_zero_count: int = 0
    total_zero_count: int = 0
    zero_steps: List[int] = field(default_factory=list)

    # Gradient tracking
    gradient_norms: deque = field(default_factory=lambda: deque(maxlen=100))
    large_gradient_count: int = 0

    # Recovery tracking
    recovery_attempts: int = 0
    successful_recoveries: int = 0

    # Loss history for trend analysis
    recent_losses: deque = field(default_factory=lambda: deque(maxlen=50))

    def reset(self):
        """Reset all counters."""
        self.consecutive_nan_count = 0
        self.consecutive_zero_count = 0
        self.recovery_attempts = 0

    def add_loss(self, loss_value: float, step: int):
        """Add a loss value to tracking."""
        self.recent_losses.append((step, loss_value))

        if torch.isnan(torch.tensor(loss_value)):
            self.consecutive_nan_count += 1
            self.total_nan_count += 1
            self.nan_steps.append(step)
        else:
            self.consecutive_nan_count = 0

        if abs(loss_value) < 1e-8:
            self.consecutive_zero_count += 1
            self.total_zero_count += 1
            self.zero_steps.append(step)
        else:
            self.consecutive_zero_count = 0

    def add_gradient_norm(self, grad_norm: float):
        """Add gradient norm to tracking."""
        self.gradient_norms.append(grad_norm)
        if grad_norm > 10.0:  # Threshold for large gradients
            self.large_gradient_count += 1

    def get_recent_nan_ratio(self, window: int = 20) -> float:
        """Get ratio of NaN losses in recent window."""
        if len(self.recent_losses) < window:
            window = len(self.recent_losses)

        if window == 0:
            return 0.0

        recent_window = list(self.recent_losses)[-window:]
        nan_count = sum(
            1 for _, loss in recent_window if torch.isnan(torch.tensor(loss))
        )
        return nan_count / window

    def is_unstable(self, config) -> bool:
        """Check if training is becoming unstable."""
        # Check consecutive issues
        max_consecutive_zero = getattr(
            config, "max_consecutive_zero", 10
        )  # Default to 10 if not set

        if (
            self.consecutive_nan_count >= config.max_consecutive_nan
            or self.consecutive_zero_count >= max_consecutive_zero
        ):
            return True

        # Check recent NaN ratio
        if (
            self.get_recent_nan_ratio(config.nan_monitoring_window)
            > config.max_nan_ratio
        ):
            return True

        # Check gradient explosion
        if len(self.gradient_norms) >= 5 and all(
            norm > 10.0 for norm in list(self.gradient_norms)[-5:]
        ):
            return True

        return False


class StabilityMonitor:
    """
    Monitor and manage training stability.

    Features:
    - NaN/Inf detection and recovery
    - Gradient explosion detection
    - Automatic learning rate adjustment
    - Loss trend analysis
    - Training health diagnostics
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.metrics = StabilityMetrics()

        # Recovery mechanisms
        self.original_lr = getattr(config, "learning_rate", 1e-5)
        self.current_lr_reduction = 1.0
        self.original_grad_clip = getattr(config, "max_grad_norm", 1.0)

        # Timing for performance monitoring
        self.step_times = deque(maxlen=100)

    def check_loss_stability(self, loss: torch.Tensor, step: int) -> Dict[str, Any]:
        """
        Check if loss is stable and handle problems.

        Returns:
            Dict with status and recommendations
        """
        loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        # Add to metrics
        self.metrics.add_loss(loss_value, step)

        # Check for problems
        is_nan = torch.isnan(torch.tensor(loss_value))
        is_inf = torch.isinf(torch.tensor(loss_value))
        is_zero = abs(loss_value) < 1e-8
        is_large = loss_value > 100.0

        status = {
            "is_stable": True,
            "issues": [],
            "recommendations": [],
            "should_skip": False,
            "should_stop": False,
            "recovery_needed": False,
        }

        # Analyze issues
        if is_nan:
            status["is_stable"] = False
            status["issues"].append("NaN loss detected")

        if is_inf:
            status["is_stable"] = False
            status["issues"].append("Infinite loss detected")

        if is_zero:
            status["issues"].append("Zero loss detected")

        if is_large:
            status["issues"].append("Unusually large loss detected")

        # Determine actions
        if is_nan or is_inf:
            if self.metrics.consecutive_nan_count <= self.config.max_consecutive_nan:
                status["should_skip"] = True
                status["recovery_needed"] = True
                status["recommendations"].append("Skip step and attempt recovery")
            else:
                status["should_stop"] = True
                status["recommendations"].append(
                    "Stop training - too many consecutive NaN/Inf"
                )

        elif self.metrics.is_unstable(self.config):
            status["recovery_needed"] = True
            status["recommendations"].append("Apply stability recovery measures")

        return status

    def attempt_recovery(self, model, optimizer=None) -> bool:
        """
        Attempt to recover from training instability.

        Returns:
            True if recovery was attempted, False otherwise
        """
        self.metrics.recovery_attempts += 1

        if not self.config.nan_recovery_enabled:
            self.logger.warning("🚫 NaN recovery is disabled in config")
            return False

        self.logger.warning(
            f"🔧 Attempting stability recovery (attempt #{self.metrics.recovery_attempts})"
        )

        recovery_success = False

        # 1. Reset model gradients
        if model is not None:
            model.zero_grad()
            recovery_success = True
            self.logger.info("   ✅ Reset model gradients")

        # 2. Reduce learning rate
        if optimizer is not None and hasattr(optimizer, "param_groups"):
            reduction_factor = self.config.learning_rate_reduction_factor
            self.current_lr_reduction *= reduction_factor

            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                param_group["lr"] = old_lr * reduction_factor
                self.logger.info(
                    f"   📉 Reduced LR: {old_lr:.2e} → {param_group['lr']:.2e}"
                )

            recovery_success = True

        # 3. Reduce gradient clipping
        new_grad_clip = (
            self.original_grad_clip * self.config.gradient_clip_reduction_factor
        )
        self.logger.info(
            f"   ✂️ Reduced gradient clipping: {self.original_grad_clip} → {new_grad_clip}"
        )

        # 4. Clear optimizer state if available
        if optimizer is not None and hasattr(optimizer, "state"):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param in optimizer.state:
                        optimizer.state[param] = {}
            self.logger.info("   🧹 Cleared optimizer state")
            recovery_success = True

        if recovery_success:
            self.metrics.successful_recoveries += 1
            self.metrics.reset()  # Reset consecutive counters
            self.logger.info("✅ Recovery attempt completed")
        else:
            self.logger.error("❌ Recovery attempt failed")

        return recovery_success

    def monitor_gradients(self, model) -> Dict[str, float]:
        """
        Monitor gradient health.

        Returns:
            Dict with gradient statistics
        """
        total_norm = 0.0
        param_count = 0
        max_grad = 0.0
        min_grad = float("inf")

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                max_grad = max(max_grad, param_norm.item())
                min_grad = min(min_grad, param_norm.item())

        total_norm = total_norm ** (1.0 / 2)

        # Track gradient norm
        self.metrics.add_gradient_norm(total_norm)

        stats = {
            "total_norm": total_norm,
            "max_grad": max_grad,
            "min_grad": min_grad if min_grad != float("inf") else 0.0,
            "param_count": param_count,
            "is_exploding": total_norm > 10.0,
            "is_vanishing": total_norm < 1e-6,
        }

        return stats

    def log_step_timing(self, step_time: float):
        """Log step timing for performance monitoring."""
        self.step_times.append(step_time)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive stability diagnostics."""
        diagnostics = {
            "stability_metrics": {
                "total_nan_count": self.metrics.total_nan_count,
                "total_zero_count": self.metrics.total_zero_count,
                "consecutive_nan": self.metrics.consecutive_nan_count,
                "consecutive_zero": self.metrics.consecutive_zero_count,
                "recent_nan_ratio": self.metrics.get_recent_nan_ratio(),
                "recovery_attempts": self.metrics.recovery_attempts,
                "successful_recoveries": self.metrics.successful_recoveries,
                "large_gradient_count": self.metrics.large_gradient_count,
            },
            "performance_metrics": {
                "avg_step_time": sum(self.step_times) / len(self.step_times)
                if self.step_times
                else 0.0,
                "recent_step_time": list(self.step_times)[-10:]
                if self.step_times
                else [],
            },
            "training_health": {
                "is_stable": not self.metrics.is_unstable(self.config),
                "lr_reduction_factor": self.current_lr_reduction,
                "recent_losses": list(self.metrics.recent_losses)[-10:],
            },
        }

        return diagnostics

    def should_continue_training(self) -> Tuple[bool, str]:
        """
        Determine if training should continue based on stability.

        Returns:
            Tuple of (should_continue, reason)
        """
        # Check if too many recovery attempts
        if self.metrics.recovery_attempts > 10:
            return False, "Too many recovery attempts - training unstable"

        # Check if consecutive issues exceed limits
        if self.metrics.consecutive_nan_count > self.config.max_consecutive_nan * 2:
            return False, "Excessive consecutive NaN losses"

        # Check recent NaN ratio
        if self.metrics.get_recent_nan_ratio() > 0.8:
            return False, "High proportion of recent NaN losses"

        # Check if learning rate has been reduced too much
        if self.current_lr_reduction < 0.001:  # LR reduced by more than 1000x
            return False, "Learning rate reduced too aggressively"

        return True, "Training appears stable"


def create_stability_monitor(config, logger=None) -> StabilityMonitor:
    """Create a stability monitor with the given configuration."""
    return StabilityMonitor(config, logger)
