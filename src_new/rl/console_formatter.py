"""Console logging formatter for RL training."""

from typing import Any, Dict, List, Optional


class ConsoleFormatter:
    """Formats training metrics for console output."""

    @staticmethod
    def format_training_summary(
        step: int,
        epoch: float,
        reward_mean: float,
        reward_std: float,
        raw_reward_mean: float,
        raw_reward_std: float,
        learning_rate: float,
        grad_norm: float,
        eta_minutes: float,
        logs: Dict[str, Any],
        reward_logger_summary: Optional[str] = None,
    ) -> str:
        """Format core training metrics as a console summary line.

        Args:
            step: Current global step
            epoch: Training epoch progress (0..1)
            reward_mean: Mean normalized reward
            reward_std: Std of normalized rewards
            raw_reward_mean: Mean unnormalized reward
            raw_reward_std: Std of unnormalized rewards
            learning_rate: Current learning rate
            grad_norm: Gradient norm
            eta_minutes: Estimated time to completion (minutes)
            logs: Full metrics dict (for extracting additional fields)
            reward_logger_summary: Pre-formatted reward component summary

        Returns:
            Formatted string like "[step=X epoch=Y reward=Z ...]"
        """
        parts = [
            f"step={step}",
            f"epoch={epoch:.3f}",
            f"reward={reward_mean:.4f}±{reward_std:.4f}",
            f"lr={learning_rate:.3e}",
            f"grad_norm={grad_norm:.3f}",
            f"eta={eta_minutes:.1f}min",
        ]
        # Only print raw_reward when available in logs (prevents 0.0000±0.0000 noise)
        if "raw_reward" in logs and "raw_reward_std" in logs:
            parts.insert(
                3, f"raw_reward={logs['raw_reward']:.4f}±{logs['raw_reward_std']:.4f}"
            )

        # Add advantages if available
        if "advantages/std" in logs:
            parts.append(f"adv_std={logs['advantages/std']:.4f}")
        if "advantages/max_abs" in logs:
            parts.append(f"adv_max={logs['advantages/max_abs']:.4f}")

        # Add termination ratio if available
        if "completions/terminated_ratio" in logs:
            parts.append(f"term_ratio={logs['completions/terminated_ratio']:.3f}")

        # Append reward component summary
        if reward_logger_summary:
            parts.append(reward_logger_summary)

        return "[" + " ".join(parts) + "]"

    @staticmethod
    def format_detailed_training_summary(
        logs: Dict[str, Any],
        reward_names: List[str],
    ) -> str:
        """Format comprehensive training metrics as multi-line grouped summary.

        Args:
            logs: Full metrics dict with all logged values
            reward_names: List of reward function names

        Returns:
            Multi-line formatted string with logical grouping
        """
        lines = []

        # Line 1: Core metrics
        core = [
            f"step={logs.get('step', 0)}",
            f"epoch={logs.get('epoch', 0.0):.3f}",
            f"loss={logs.get('loss', 0.0):.4f}",
            f"reward={logs.get('reward', 0.0):.4f}±{logs.get('reward_std', 0.0):.4f}",
            f"lr={logs.get('learning_rate', 0.0):.3e}",
            f"grad_norm={logs.get('grad_norm', 0.0):.3f}",
            f"eta={logs.get('eta_minutes', 0.0):.1f}min",
        ]
        if "raw_reward" in logs and "raw_reward_std" in logs:
            core.insert(
                4, f"raw_reward={logs['raw_reward']:.4f}±{logs['raw_reward_std']:.4f}"
            )
        lines.append("[CORE] " + " ".join(core))

        # Line 2: Advantages (local + global if present)
        adv_parts = []
        if "advantages/std" in logs:
            adv_parts.append(f"adv_std={logs['advantages/std']:.4f}")
        if "advantages/max_abs" in logs:
            adv_parts.append(f"adv_max={logs['advantages/max_abs']:.4f}")
        if "advantages_global/std" in logs:
            adv_parts.append(f"global_std={logs['advantages_global/std']:.4f}")
        if "advantages_global/max_abs" in logs:
            adv_parts.append(f"global_max={logs['advantages_global/max_abs']:.4f}")
        if adv_parts:
            lines.append("[ADVANTAGES] " + " ".join(adv_parts))

        # Line 3: Global rewards (cross-rank)
        global_parts = []
        if "rewards_global/mean" in logs:
            global_parts.append(f"mean={logs['rewards_global/mean']:.4f}")
        if "rewards_global/std" in logs:
            global_parts.append(f"std={logs['rewards_global/std']:.4f}")
        if global_parts:
            lines.append("[REWARDS_GLOBAL] " + " ".join(global_parts))

        # Line 4: Generation params
        gen_parts = [
            f"temperature={logs.get('temperature', 0.0):.4f}",
            f"beta={logs.get('beta', 0.0):.6f}",
        ]
        if "completions/terminated_ratio" in logs:
            gen_parts.append(f"term_ratio={logs['completions/terminated_ratio']:.3f}")
        lines.append("[GENERATION] " + " ".join(gen_parts))

        # Line 5: Token lengths
        len_parts = []
        if "completions/mean_len_tok" in logs:
            len_parts.append(f"comp_len={logs['completions/mean_len_tok']:.1f}")
        if "gt/mean_len_tok" in logs:
            len_parts.append(f"gt_len={logs['gt/mean_len_tok']:.1f}")
        if "completions/ratio_to_gt_mean" in logs:
            len_parts.append(f"ratio={logs['completions/ratio_to_gt_mean']:.3f}")
        if len_parts:
            lines.append("[LENGTHS] " + " ".join(len_parts))

        # Line 6: Length bounds
        bounds_parts = []
        if "completions/over_upper_ratio" in logs:
            bounds_parts.append(f"over={logs['completions/over_upper_ratio']:.3f}")
        if "completions/under_lower_ratio" in logs:
            bounds_parts.append(f"under={logs['completions/under_lower_ratio']:.3f}")
        # cap_hit removed
        if bounds_parts:
            lines.append("[BOUNDS] " + " ".join(bounds_parts))

        # Dynamic length caps removed

        # Line 8: Policy clipping
        clip_parts = []
        if "clip_ratio/low_mean" in logs:
            clip_parts.append(f"low={logs['clip_ratio/low_mean']:.3f}")
        if "clip_ratio/high_mean" in logs:
            clip_parts.append(f"high={logs['clip_ratio/high_mean']:.3f}")
        if "clip_ratio/region_mean" in logs:
            clip_parts.append(f"region={logs['clip_ratio/region_mean']:.3f}")
        if "completions/zero_len_ratio" in logs:
            clip_parts.append(f"zero_len={logs['completions/zero_len_ratio']:.3f}")
        if clip_parts:
            lines.append("[CLIPPING] " + " ".join(clip_parts))

        # Line 9: Normalized reward components
        norm_parts = []
        for name in reward_names:
            key = f"rewards/{name}/mean"
            if key in logs:
                norm_parts.append(f"{name}={logs[key]:.3f}")
        if norm_parts:
            lines.append("[REWARDS_NORM] " + " ".join(norm_parts))

        # Line 10: Raw reward components
        raw_parts = []
        for name in reward_names:
            key = f"raw_rewards/{name}/mean"
            if key in logs:
                raw_parts.append(f"{name}={logs[key]:.3f}")
        if raw_parts:
            lines.append("[REWARDS_RAW] " + " ".join(raw_parts))

        # Line 11: Prompt batch telemetry
        pb_metrics = logs.get("prompt_batch_metrics", {})
        if pb_metrics:
            pb_parts = [
                f"fill={pb_metrics.get('fill_ratio', 0.0):.2f}",
                f"reward_avg={pb_metrics.get('reward_average', 0.0):.4f}",
                f"traj={int(pb_metrics.get('trajectories_collected', 0))}",
            ]
            dropped = int(pb_metrics.get("dropped_prompts", 0))
            if dropped > 0:
                pb_parts.append(f"dropped={dropped}")
            lines.append("[PROMPT_BATCH] " + " ".join(pb_parts))

        return "\n".join(lines)

    @staticmethod
    def format_prompt_batch_summary(prompt_batch_metrics: Dict[str, Any]) -> List[str]:
        """Format prompt batch telemetry as console parts.

        Returns:
            List of formatted strings like ["fill=0.95", "reward_avg=1.23", ...]
        """
        parts = []
        if not prompt_batch_metrics:
            return parts

        fill_ratio = prompt_batch_metrics.get("fill_ratio", 0.0)
        reward_avg = prompt_batch_metrics.get("reward_average", 0.0)
        traj_count = int(prompt_batch_metrics.get("trajectories_collected", 0))
        dropped = int(prompt_batch_metrics.get("dropped_prompts", 0))

        parts.append(f"fill={fill_ratio:.2f}")
        parts.append(f"reward_avg={reward_avg:.4f}")
        parts.append(f"traj={traj_count}")
        if dropped > 0:
            parts.append(f"dropped={dropped}")

        return parts


__all__ = ["ConsoleFormatter"]
