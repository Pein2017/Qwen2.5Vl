"""Cross-rank advantage normalization for GRPO training."""

from typing import Any, Dict, List

import torch
from accelerate import Accelerator

from src_new.utils.rank_aware_logging import get_rank_aware_logger


_LOGGER = get_rank_aware_logger("rl.AdvantageNormalizer")


class AdvantageNormalizer:
    """Handles cross-rank advantage normalization with proper padding/alignment."""

    def __init__(
        self,
        accelerator: Accelerator,
        device: torch.device,
        scale_rewards: bool,
        max_advantage_magnitude: float | None,
    ):
        self.accelerator = accelerator
        self.device = device
        self.world_size = getattr(accelerator.state, "num_processes", 1)
        self.scale_rewards = scale_rewards
        self.max_advantage_magnitude = max_advantage_magnitude
        self._warned_rewards_mismatch = False

    def normalize_advantages_cross_rank(
        self,
        chunks: List[Dict[str, Any]],
        debug_timing: bool = False,
        global_step: int = 0,
    ) -> None:
        """Normalize advantages across all ranks using global statistics.

        Modifies chunks in-place, replacing per-chunk advantages with
        globally normalized values.

        Args:
            chunks: List of generation result dicts (one per prompt)
            debug_timing: Whether to log detailed timing info
            global_step: Current training step (for logging)
        """
        if self.world_size <= 1:
            # Single GPU: normalize per-chunk independently
            for chunk in chunks:
                self._normalize_local(chunk)
            return

        # Multi-GPU: normalize using global mean/std
        for chunk in chunks:
            rewards_local = chunk.get("rewards")
            if rewards_local is None:
                continue

            # Prepare local rewards
            rewards_local = (
                rewards_local.detach().to(self.device).flatten().contiguous()
            )
            rewards_len = int(rewards_local.numel())

            # Gather lengths from all ranks to determine max_len
            length_tensor = torch.tensor(
                [rewards_len], dtype=torch.int32, device=self.device
            )
            lengths_all = self.accelerator.gather(length_tensor)
            if lengths_all.device.type != "cpu":
                lengths_all = lengths_all.cpu()
            lengths_all = lengths_all.view(-1)

            max_len = (
                int(lengths_all.max().item())
                if lengths_all.numel() > 0
                else rewards_len
            )

            if debug_timing:
                _LOGGER.warning(
                    "Rank %d step %d reward lengths local=%d gathered=%s max=%d",
                    self.accelerator.state.process_index,
                    global_step,
                    rewards_len,
                    lengths_all.tolist() if lengths_all.numel() > 0 else [],
                    max_len,
                )

            # Warn once about length mismatches
            if rewards_len != max_len and not self._warned_rewards_mismatch:
                _LOGGER.warning(
                    "Rank %d rewards length %d mismatched (max_len=%d) at step %d; padding/trimming",
                    self.accelerator.state.process_index,
                    rewards_len,
                    max_len,
                    global_step,
                )
                self._warned_rewards_mismatch = True

            # Pad or trim to max_len
            if max_len > 0:
                if rewards_local.size(0) < max_len:
                    pad = torch.zeros(
                        max_len - rewards_local.size(0),
                        dtype=rewards_local.dtype,
                        device=self.device,
                    )
                    rewards_padded = torch.cat([rewards_local, pad], dim=0)
                else:
                    rewards_padded = rewards_local[:max_len]

                # Gather across ranks
                gathered_rewards = self.accelerator.gather(rewards_padded)
                if gathered_rewards.device.type != "cpu":
                    gathered_rewards = gathered_rewards.cpu()
                gathered_rewards = gathered_rewards.view(self.world_size, max_len)

                # Compute global statistics with masking
                lengths_cpu = lengths_all.view(self.world_size)
                arange_vec = torch.arange(max_len, device=lengths_cpu.device).unsqueeze(
                    0
                )
                mask = arange_vec < lengths_cpu.unsqueeze(1)

                if mask.any():
                    masked_vals = gathered_rewards[mask]
                    global_mean = masked_vals.mean()
                    global_std = masked_vals.std(unbiased=False)

                    # Attach global scalar summaries for logging (per prompt)
                    try:
                        gstd = torch.clamp(global_std, min=1e-4)
                        if self.scale_rewards:
                            global_adv = (masked_vals - global_mean) / gstd
                        else:
                            global_adv = masked_vals - global_mean

                        chunk["rewards_global/mean"] = float(global_mean.item())
                        chunk["rewards_global/std"] = float(global_std.item())
                        chunk["advantages_global/std"] = float(
                            global_adv.detach().std(unbiased=False).item()
                        )
                        chunk["advantages_global/max_abs"] = float(
                            global_adv.detach().abs().max().item()
                        )
                    except Exception:
                        pass
                else:
                    global_mean = torch.tensor(0.0)
                    global_std = torch.tensor(1.0)

                if debug_timing:
                    _LOGGER.warning(
                        "Rank %d step %d global_mean=%.4f global_std=%.4f valid=%d",
                        self.accelerator.state.process_index,
                        global_step,
                        float(global_mean),
                        float(global_std),
                        int(mask.sum().item()),
                    )
            else:
                global_mean = torch.tensor(0.0)
                global_std = torch.tensor(1.0)

            # Compute local advantages using global stats
            global_std = torch.clamp(global_std, min=1e-4)
            adv_local = rewards_local.detach().cpu() - global_mean

            if self.scale_rewards:
                adv_local = adv_local / global_std

            # Apply magnitude clipping if configured
            if (
                self.max_advantage_magnitude is not None
                and self.max_advantage_magnitude > 0
            ):
                mag = float(self.max_advantage_magnitude)
                adv_local = torch.clamp(adv_local, min=-mag, max=mag)

            # Store normalized advantages back into chunk
            chunk["advantages"] = adv_local.to(self.device)

    def _normalize_local(self, chunk: Dict[str, Any]) -> None:
        """Normalize advantages using local statistics only (single-GPU fallback)."""
        rewards = chunk.get("rewards")
        if rewards is None:
            return

        rewards_flat = rewards.detach().flatten()
        mean = rewards_flat.mean()
        std = torch.clamp(rewards_flat.std(unbiased=False), min=1e-4)

        if self.scale_rewards:
            advantages = (rewards_flat - mean) / std
        else:
            advantages = rewards_flat - mean

        # Apply magnitude clipping if configured
        if (
            self.max_advantage_magnitude is not None
            and self.max_advantage_magnitude > 0
        ):
            mag = float(self.max_advantage_magnitude)
            advantages = torch.clamp(advantages, min=-mag, max=mag)

        chunk["advantages"] = advantages


__all__ = ["AdvantageNormalizer"]
