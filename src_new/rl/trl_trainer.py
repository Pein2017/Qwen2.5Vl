#!/usr/bin/env python3
"""
TRL GRPO wrapper for Qwen2.5-VL dense captioning.

Delegates multimodal prompt preparation and reward computation to the existing
buffer stack. Requires the optional `trl` package; we fall back to a clear
ImportError when it is unavailable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from accelerate.utils import gather_object

from src_new.config.rl_config_v2 import RLConfig
from src_new.rl import buffer as rl_buffer
from src_new.rl.data.collator import PromptOnlyCollator


try:  # pragma: no cover - optional dependency
    from trl import GRPOTrainer as _BaseGRPOTrainer
except ImportError as exc:  # pragma: no cover - enforce availability
    raise ImportError(
        "GRPOVLMTrainer requires the optional `trl` package. "
        "Install it (pip install trl) or run with `--trainer manual`."
    ) from exc


class GRPOVLMTrainer(_BaseGRPOTrainer):
    """TRL trainer wrapper tailored for Qwen2.5-VL dense captioning."""

    prompt_collator: Optional[PromptOnlyCollator]
    rl_config: Optional[RLConfig]

    def __init__(
        self,
        *args: Any,
        prompt_collator: Optional[PromptOnlyCollator] = None,
        rl_config: Optional[RLConfig] = None,
        use_native_generation: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.rl_config = rl_config
        self.prompt_collator = prompt_collator
        # No custom reward standardizer; rely on TRL's scale_rewards
        # Allow switching to TRL-native generation without changing strict config:
        # env wins if provided; param overrides env when explicitly set
        env_native = os.getenv("USE_TRL_NATIVE_GENERATION", "0").strip() == "1"
        self._use_native_generation: bool = (
            bool(use_native_generation)
            if use_native_generation is not None
            else env_native
        )
        super().__init__(*args, **kwargs)
        if self.prompt_collator is not None:
            self.data_collator = self.prompt_collator
        # (removed) custom per-component standardizer
        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_token_id is None or pad_token_id < 0:
            raise ValueError("processing_class must expose a valid pad_token_id")
        self._pad_token_id = int(pad_token_id)

        if self.beta != 0.0:
            raise NotImplementedError(
                "KL regularisation (beta>0) is not yet supported in GRPOVLMTrainer."
            )

    @staticmethod
    def _ensure_1d_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        if value is None:
            raise ValueError("Expected tensor value, got None")
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
        else:
            tensor = torch.tensor(value)
        tensor = tensor.to(dtype=dtype)
        if tensor.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got shape={tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def _maybe_tensor(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu()
        return torch.tensor(value)

    @staticmethod
    def _explode_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a collated batch dict into a per-sample list of dicts."""
        if "input_ids" not in batch:
            raise ValueError("Batch dictionary missing 'input_ids'")
        size = 0
        value = batch["input_ids"]
        if torch.is_tensor(value):
            size = int(value.size(0))
        elif isinstance(value, list):
            size = len(value)
        else:
            raise ValueError("Unsupported input_ids type in batch")

        records: List[Dict[str, Any]] = []
        for idx in range(size):
            sample: Dict[str, Any] = {}
            for key, entry in batch.items():
                if torch.is_tensor(entry):
                    sample[key] = entry[idx]
                elif isinstance(entry, list):
                    sample[key] = entry[idx]
                else:
                    sample[key] = entry
            records.append(sample)
        return records

    def _generate_and_score_completions(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Optional: delegate to TRL's native generation/scoring for simplest flow
        if getattr(self, "_use_native_generation", False):
            # Defer entirely to the base class (official path)
            return super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        if isinstance(inputs, dict):
            inputs = self._explode_batch(inputs)
        if not inputs:
            raise ValueError("Generation batch must be non-empty")

        processed: List[Dict[str, Any]] = []
        for feature in inputs:
            sample = {
                "prompt": feature.get("prompt", ""),
                "input_ids": self._ensure_1d_tensor(
                    feature.get("input_ids"), dtype=torch.long
                ),
                "attention_mask": self._ensure_1d_tensor(
                    feature.get("attention_mask"), dtype=torch.long
                ),
                "pixel_values": self._maybe_tensor(feature.get("pixel_values")),
                "image_grid_thw": self._maybe_tensor(feature.get("image_grid_thw")),
                "meta": feature.get("meta"),
            }
            processed.append(sample)

        device = self.accelerator.device
        reward_weights = self.reward_weights.to(device)
        rl_cfg = self.rl_config
        result = rl_buffer.generate_and_score(
            model=self.model,
            tokenizer=self.processing_class,
            inputs=processed,
            reward_fns=self.reward_funcs,
            reward_weights=reward_weights,
            reward_names=self.reward_func_names,
            reward_standardizer=None,
            pad_token_id=self._pad_token_id,
            sample_k=int(self.num_generations),
            max_new_tokens=int(self.max_completion_length),
            min_new_tokens=int(rl_cfg.generation.min_new_tokens)
            if rl_cfg is not None
            else None,
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            repetition_penalty=self.repetition_penalty,
            mask_truncated_completions=(
                bool(rl_cfg.grpo.mask_truncated_completions)
                if rl_cfg is not None
                else bool(self.mask_truncated_completions)
            ),
            scale_rewards=bool(rl_cfg.grpo.scale_rewards)
            if rl_cfg is not None
            else bool(self.scale_rewards),
            max_advantage_magnitude=None,
            reward_clip_sigma=rl_cfg.rewards.config.clip_sigma
            if rl_cfg is not None
            else None,
        )

        prompt_ids = result["prompt_ids"].to(device)
        prompt_mask = result["prompt_mask"].to(device)
        completion_ids = result["completion_ids"].to(device)
        completion_mask = result["completion_mask"].to(device)
        advantages = result["advantages"].to(device)
        old_per_token_logps = result["generation_logps"].to(device)

        # Ensure tensors match TRL's fixed max_completion_length along the sequence dim
        target_len = int(self.max_completion_length)
        if target_len is not None and target_len > 0:

            def _pad_right(
                t: torch.Tensor, target: int, pad_value: int | float
            ) -> torch.Tensor:
                if t is None or not torch.is_tensor(t):
                    return t
                if t.dim() != 2:
                    return t
                bsz, cur = int(t.size(0)), int(t.size(1))
                if cur == target:
                    return t
                if cur < target:
                    pad = torch.full(
                        (bsz, target - cur), pad_value, dtype=t.dtype, device=t.device
                    )
                    return torch.cat([t, pad], dim=1)
                # Truncate if somehow longer (shouldn't happen as we cap generation)
                return t[:, :target]

            # Pad/truncate to target_len with appropriate pad values
            completion_ids = _pad_right(completion_ids, target_len, self._pad_token_id)
            completion_mask = _pad_right(completion_mask, target_len, 0)
            old_per_token_logps = _pad_right(old_per_token_logps, target_len, 0.0)

        mode = "train" if self.model.training else "eval"
        rewards = result["rewards"].to(device).detach()
        self._metrics[mode]["reward"].append(
            float(self.accelerator.gather(rewards).float().mean().item())
        )
        self._metrics[mode]["reward_std"].append(
            float(self.accelerator.gather(rewards).float().std(unbiased=False).item())
        )

        lengths = result.get("completion_lengths")
        if torch.is_tensor(lengths):
            gathered_lengths = self.accelerator.gather(lengths.to(device).detach())
            self._metrics[mode]["completions/mean_length"].append(
                float(gathered_lengths.float().mean().item())
            )

        prompts = result.get("prompts", [])
        completions = result.get("completions", [])
        self._textual_logs["prompt"].extend(gather_object(prompts))
        self._textual_logs["completion"].extend(gather_object(completions))
        self._textual_logs["advantages"].extend(
            self.accelerator.gather(advantages.detach()).tolist()
        )

        # Rely on TRL's own logging integrations; no custom env-driven dumps

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": None,
        }

    def _collate_prompts(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate raw prompt samples using the shared RL collator when available."""
        if self.prompt_collator is None:
            return {"input_ids": [feat["input_ids"] for feat in features]}
        return self.prompt_collator(features)

    # ---- Saving: write only model weights as safetensors, no optimizer state ----
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = True):  # type: ignore[override]
        import os

        from safetensors.torch import save_file

        out_dir = output_dir or str(self.args.output_dir)
        if self.accelerator.is_main_process:
            os.makedirs(out_dir, exist_ok=True)

        # Ensure all ranks reach this point before gathering state dict
        self.accelerator.wait_for_everyone()

        # Prefer the underlying HF model if wrapped
        to_save = getattr(self.model, "base_model", self.model)

        # Use Accelerate to gather a full, consolidated state_dict (handles ZeRO-3)
        state_dict = self.accelerator.get_state_dict(to_save)

        # Write safetensors directly to avoid shared-tensor duplication from wrappers
        if self.accelerator.is_main_process:
            # Save config alongside weights for compatibility
            try:
                to_save.config.save_pretrained(out_dir)  # type: ignore[attr-defined]
            except Exception:
                pass
            save_path = os.path.join(out_dir, "model.safetensors")
            save_file(state_dict, save_path)
        # Ensure the save is visible before proceeding
        self.accelerator.wait_for_everyone()


__all__ = ["GRPOVLMTrainer"]
