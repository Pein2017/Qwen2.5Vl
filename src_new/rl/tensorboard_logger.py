"""TensorBoard logging utilities for RL training."""

from typing import Any, Dict, List, Optional


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class TensorBoardLogger:
    """Structured TensorBoard logger for RL metrics."""

    def __init__(self, writer: Optional[Any]):
        self.writer = writer
        self.enabled = writer is not None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        if not self.enabled:
            return
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)

    def log_training_scalars(
        self,
        step: int,
        reward_mean: float,
        reward_std: float,
        raw_reward_mean: float,
        raw_reward_std: float,
        learning_rate: float,
        grad_norm: float,
        epoch: float,
        temperature: float,
        beta: float,
        eta_minutes: float,
    ) -> None:
        """Log core training scalars."""
        if not self.enabled:
            return

        self.writer.add_scalar("reward", reward_mean, step)
        self.writer.add_scalar("reward_std", reward_std, step)
        self.writer.add_scalar("raw_reward", raw_reward_mean, step)
        self.writer.add_scalar("raw_reward_std", raw_reward_std, step)
        self.writer.add_scalar("train/learning_rate", learning_rate, step)
        self.writer.add_scalar("train/grad_norm", grad_norm, step)
        self.writer.add_scalar("train/epoch", epoch, step)
        self.writer.add_scalar("temperature", temperature, step)
        self.writer.add_scalar("beta", beta, step)
        self.writer.add_scalar("train/eta_minutes", eta_minutes, step)
        self.writer.add_scalar("step", step, step)

    def log_completion_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log completion-related metrics (lengths, ratios, etc.)."""
        if not self.enabled:
            return

        completion_keys = [
            "completions/mean_len_tok",
            "gt/mean_len_tok",
            "completions/cap_hit_ratio",
            "completions/ratio_to_gt_mean",
            "completions/over_upper_ratio",
            "completions/under_lower_ratio",
            "completions/terminated_ratio",
            "completions/zero_len_ratio",
        ]

        for key in completion_keys:
            if key in metrics:
                self.writer.add_scalar(key, metrics[key], step)

    def log_dynamic_length_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log dynamic length cap metrics."""
        if not self.enabled:
            return

        dyn_keys = [
            "dynamic_length/enabled",
            "dynamic_length/mean_cap",
            "dynamic_length/min_cap",
            "dynamic_length/max_cap",
        ]

        for key in dyn_keys:
            if key in metrics:
                self.writer.add_scalar(key, metrics[key], step)

    def log_clip_ratios(self, metrics: Dict[str, float], step: int) -> None:
        """Log policy clipping ratios."""
        if not self.enabled:
            return

        clip_keys = [
            "clip_ratio/low_mean",
            "clip_ratio/high_mean",
            "clip_ratio/region_mean",
        ]

        for key in clip_keys:
            if key in metrics:
                self.writer.add_scalar(key, metrics[key], step)

    def log_advantage_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log advantage statistics."""
        if not self.enabled:
            return

        adv_keys = ["advantages/std", "advantages/max_abs"]
        for key in adv_keys:
            if key in metrics:
                self.writer.add_scalar(key, metrics[key], step)

    def log_per_reward_components(
        self,
        components: Dict[str, Dict[str, float]],
        step: int,
        prefix: str = "rewards",
    ) -> None:
        """Log per-reward component statistics.

        Args:
            components: Dict mapping reward names to {mean, std} dicts
            step: Global step
            prefix: Prefix for tags (e.g., 'rewards' or 'raw_rewards')
        """
        if not self.enabled:
            return

        for name, stats in components.items():
            self.writer.add_scalar(f"{prefix}/{name}/mean", stats["mean"], step)
            self.writer.add_scalar(f"{prefix}/{name}/std", stats["std"], step)

    def log_eval_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log evaluation metrics with 'eval/' prefix."""
        if not self.enabled:
            return

        for key, value in metrics.items():
            self.writer.add_scalar(f"eval/{key}", value, step)

    def log_loss(self, loss: float, step: int) -> None:
        """Log training loss."""
        if self.enabled:
            self.writer.add_scalar("train/loss", loss, step)

    def log_global_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log global cross-rank metrics."""
        if not self.enabled:
            return

        global_keys = [
            "rewards_global/mean",
            "rewards_global/std",
            "advantages_global/std",
            "advantages_global/max_abs",
        ]

        for key in global_keys:
            if key in metrics:
                self.writer.add_scalar(key, metrics[key], step)

    def log_all_from_dict(
        self,
        logs: Dict[str, Any],
        reward_names: List[str],
        step: int,
    ) -> None:
        """Log all metrics from a consolidated dict.

        This is a convenience method that calls all specialized log methods
        based on what's present in the logs dict.
        """
        if not self.enabled:
            return

        # Core training scalars
        if all(k in logs for k in ["reward", "raw_reward", "learning_rate"]):
            self.log_training_scalars(
                step=step,
                reward_mean=logs["reward"],
                reward_std=logs.get("reward_std", 0.0),
                raw_reward_mean=logs["raw_reward"],
                raw_reward_std=logs.get("raw_reward_std", 0.0),
                learning_rate=logs["learning_rate"],
                grad_norm=logs.get("grad_norm", 0.0),
                epoch=logs.get("epoch", 0.0),
                temperature=logs.get("temperature", 0.0),
                beta=logs.get("beta", 0.0),
                eta_minutes=logs.get("eta_minutes", 0.0),
            )

        # Loss
        if "loss" in logs:
            self.log_loss(logs["loss"], step)

        # Advantages
        self.log_advantage_metrics(logs, step)

        # Global cross-rank metrics
        self.log_global_metrics(logs, step)

        # Completion metrics
        self.log_completion_metrics(logs, step)

        # Dynamic length
        self.log_dynamic_length_metrics(logs, step)

        # Clipping ratios
        self.log_clip_ratios(logs, step)

        # Per-reward components (normalized)
        norm_components = {}
        for name in reward_names:
            mean_key = f"rewards/{name}/mean"
            std_key = f"rewards/{name}/std"
            if mean_key in logs or std_key in logs:
                norm_components[name] = {
                    "mean": logs.get(mean_key, 0.0),
                    "std": logs.get(std_key, 0.0),
                }
        if norm_components:
            self.log_per_reward_components(norm_components, step, prefix="rewards")

        # Per-reward components (raw)
        raw_components = {}
        for name in reward_names:
            mean_key = f"raw_rewards/{name}/mean"
            std_key = f"raw_rewards/{name}/std"
            if mean_key in logs or std_key in logs:
                raw_components[name] = {
                    "mean": logs.get(mean_key, 0.0),
                    "std": logs.get(std_key, 0.0),
                }
        if raw_components:
            self.log_per_reward_components(raw_components, step, prefix="raw_rewards")

    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.enabled:
            self.writer.close()


__all__ = ["TensorBoardLogger"]
