"""Console metrics collection for RL training."""

from typing import Any, Dict, List


class ConsoleMetricsCollector:
    """Collects and organizes metrics for console logging."""

    def __init__(self):
        pass

    def collect_all_metrics(
        self,
        *,
        global_step: int,
        epoch_progress: float,
        loss_value: float,
        reward_mean: float,
        reward_std: float,
        raw_reward_mean: float,
        raw_reward_std: float,
        current_lr: float,
        grad_norm: float,
        eta_minutes: float,
        term_ratio: float,
        logs: Dict[str, Any],
        generation_result: Dict[str, Any],
        reward_names: List[str],
    ) -> Dict[str, Any]:
        """Collect all metrics from logs and generation_result into unified dict.

        Args:
            global_step: Current training step
            epoch_progress: Training progress (0..1)
            loss_value: Loss value
            reward_mean: Mean normalized reward
            reward_std: Std of normalized rewards
            raw_reward_mean: Mean raw reward
            raw_reward_std: Std of raw rewards
            current_lr: Current learning rate
            grad_norm: Gradient norm
            eta_minutes: ETA in minutes
            term_ratio: Termination ratio
            logs: Full metrics dict from trainer
            generation_result: Generation result dict with rewards/advantages
            reward_names: List of reward function names

        Returns:
            Unified dict with all metrics for console logging
        """
        console_logs = {
            "step": global_step,
            "epoch": epoch_progress,
            "loss": loss_value,
            "reward": reward_mean,
            "reward_std": reward_std,
            "raw_reward": raw_reward_mean,
            "raw_reward_std": raw_reward_std,
            "learning_rate": current_lr,
            "grad_norm": grad_norm,
            "eta_minutes": eta_minutes,
            "temperature": float(generation_result.get("temperature", 0.0)),
            "beta": generation_result.get("beta", 0.0),
            "completions/terminated_ratio": term_ratio,
        }

        # Add advantage metrics
        self._add_if_present(
            console_logs, logs, ["advantages/std", "advantages/max_abs"]
        )

        # Add global cross-rank metrics (from generation_result)
        self._add_from_source(
            console_logs,
            generation_result,
            [
                "rewards_global/mean",
                "rewards_global/std",
                "advantages_global/std",
                "advantages_global/max_abs",
            ],
        )

        # Add length metrics
        self._add_if_present(
            console_logs,
            logs,
            [
                "completions/mean_len_tok",
                "gt/mean_len_tok",
                "completions/cap_hit_ratio",
                "completions/ratio_to_gt_mean",
                "completions/over_upper_ratio",
                "completions/under_lower_ratio",
            ],
        )

        # Add dynamic length metrics
        self._add_if_present(
            console_logs,
            logs,
            [
                "dynamic_length/enabled",
                "dynamic_length/mean_cap",
                "dynamic_length/min_cap",
                "dynamic_length/max_cap",
            ],
        )

        # Add clipping metrics
        self._add_if_present(
            console_logs,
            logs,
            ["clip_ratio/low_mean", "clip_ratio/high_mean", "clip_ratio/region_mean"],
        )

        # Add per-reward components (normalized and raw)
        for name in reward_names:
            self._add_if_present(
                console_logs,
                logs,
                [f"rewards/{name}/mean", f"rewards/{name}/std"],
            )
            self._add_if_present(
                console_logs,
                logs,
                [f"raw_rewards/{name}/mean", f"raw_rewards/{name}/std"],
            )

        # Add prompt batch telemetry
        pb_metrics = generation_result.get("prompt_batch_metrics", {})
        if pb_metrics:
            console_logs["prompt_batch_metrics"] = pb_metrics

        return console_logs

    @staticmethod
    def _add_if_present(
        target: Dict[str, Any], source: Dict[str, Any], keys: List[str]
    ) -> None:
        """Add keys from source to target if they exist."""
        for key in keys:
            if key in source:
                target[key] = source[key]

    @staticmethod
    def _add_from_source(
        target: Dict[str, Any], source: Dict[str, Any], keys: List[str]
    ) -> None:
        """Alias for _add_if_present for clarity."""
        ConsoleMetricsCollector._add_if_present(target, source, keys)


__all__ = ["ConsoleMetricsCollector"]
