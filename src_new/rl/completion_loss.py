"""Per-completion loss computation for GRPO training."""

from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from torch import nn

from src_new.rl import logprobs, losses
from src_new.rl.diagnostics.trust_region import compute_trust_region_diagnostic


class CompletionLossComputer:
    """Handles per-completion GRPO loss computation with streaming backward."""

    def __init__(
        self,
        accelerator: Accelerator,
        device: torch.device,
        epsilon_low: float,
        epsilon_high: float,
        loss_type: str,
    ):
        self.accelerator = accelerator
        self.device = device
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.loss_type = loss_type

    def compute_streaming_loss(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module],
        generation_result: Dict[str, Any],
        current_beta: float,
    ) -> Dict[str, Any]:
        """Compute loss for all completions with streaming backward.

        Returns:
            Dict with:
                - loss_value: Average loss across completions
                - non_finite: Whether any loss was non-finite
                - clip_stats: Token-level clipping statistics
        """
        prompt_ids_all = generation_result["prompt_ids"].to(self.device)
        prompt_mask_all = generation_result["prompt_mask"].to(self.device)
        completion_ids_all = generation_result["completion_ids"].to(self.device)
        completion_mask_all = generation_result["completion_mask"].to(self.device)
        advantages_all = generation_result["advantages"].to(self.device)

        pixel_values = generation_result.get("pixel_values")
        image_grid_thw = generation_result.get("image_grid_thw")
        images_per_sample_full = generation_result.get("images_per_sample")

        # Prepare single-image count vector for logprob slicing
        if images_per_sample_full is not None:
            try:
                num_images = int(images_per_sample_full[0].item())
            except Exception:
                num_images = (
                    int(images_per_sample_full.item())
                    if images_per_sample_full.numel() == 1
                    else 0
                )
            images_per_sample_single = torch.tensor(
                [num_images], dtype=torch.long, device=self.device
            )
        else:
            images_per_sample_single = None

        # Prepare single prompt row
        if prompt_ids_all.dim() == 2:
            prompt_ids_row = prompt_ids_all[0]
            prompt_mask_row = prompt_mask_all[0]
        else:
            prompt_ids_row = prompt_ids_all
            prompt_mask_row = prompt_mask_all

        num_completions = int(completion_ids_all.size(0))
        loss_value_for_log = 0.0
        non_finite = False

        # Clipping diagnostics counters
        clip_low_tokens = 0
        clip_high_tokens = 0
        clip_region_tokens = 0
        clip_total_tokens = 0

        with self.accelerator.autocast():
            for k in range(num_completions):
                comp_ids_row = completion_ids_all[k]
                comp_mask_row = completion_mask_all[k]
                effective_len = int(comp_mask_row.long().sum().item())

                if effective_len <= 0:
                    continue

                comp_ids_eff = comp_ids_row[:effective_len]
                comp_mask_eff = comp_mask_row[:effective_len].unsqueeze(0)

                # Build 1-sample inputs
                input_ids_k = torch.cat(
                    [prompt_ids_row, comp_ids_eff], dim=0
                ).unsqueeze(0)
                attention_mask_k = torch.cat(
                    [
                        prompt_mask_row,
                        torch.ones_like(comp_ids_eff, dtype=prompt_mask_row.dtype),
                    ],
                    dim=0,
                ).unsqueeze(0)

                # Compute current policy logprobs
                per_token_logps_k = logprobs.get_per_token_logps(
                    model=model,
                    input_ids=input_ids_k,
                    attention_mask=attention_mask_k,
                    logits_to_keep=effective_len,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    images_per_sample=images_per_sample_single,
                    temperature=generation_result.get("temperature", 1.0),
                )

                # Use stored generation logprobs for proper GRPO ratio
                old_logps_k = self._get_generation_logprobs(
                    generation_result, k, effective_len, per_token_logps_k
                )

                # Accumulate token-level clipping diagnostics
                adv_k = advantages_all[k : k + 1]
                clip_stats = self._compute_clip_stats(
                    per_token_logps_k, old_logps_k, adv_k, comp_mask_eff
                )
                clip_low_tokens += clip_stats["low"]
                clip_high_tokens += clip_stats["high"]
                clip_region_tokens += clip_stats["region"]
                clip_total_tokens += clip_stats["total"]

                # Compute KL divergence if using reference model
                per_token_kl_k = None
                if current_beta > 0.0 and ref_model is not None:
                    per_token_kl_k = self._compute_reference_kl(
                        ref_model,
                        input_ids_k,
                        attention_mask_k,
                        effective_len,
                        pixel_values,
                        image_grid_thw,
                        images_per_sample_single,
                        generation_result.get("temperature", 1.0),
                        per_token_logps_k,
                    )

                # Compute GRPO loss for this completion
                loss_k = losses.compute_grpo_loss(
                    per_token_logps_k,
                    old_logps_k,
                    adv_k,
                    comp_mask_eff,
                    epsilon_low=self.epsilon_low,
                    epsilon_high=self.epsilon_high,
                    loss_type=self.loss_type,
                    beta=current_beta,
                    per_token_kl=per_token_kl_k,
                )

                if not torch.isfinite(loss_k):
                    non_finite = True
                    break

                # Stream backward per completion; average across K
                scaled_part = loss_k / float(max(num_completions, 1))
                self.accelerator.backward(scaled_part)

                try:
                    loss_value_for_log += float(loss_k.detach().float().item()) / float(
                        max(num_completions, 1)
                    )
                except Exception:
                    pass

                # Free per-completion temporaries
                self._cleanup_completion_tensors(locals())

        # Compute clipping ratios
        clip_ratios = {}
        if clip_total_tokens > 0:
            clip_ratios["policy_clip_low_ratio"] = float(clip_low_tokens) / float(
                clip_total_tokens
            )
            clip_ratios["policy_clip_high_ratio"] = float(clip_high_tokens) / float(
                clip_total_tokens
            )
            clip_ratios["policy_clip_region_ratio"] = float(clip_region_tokens) / float(
                clip_total_tokens
            )
        else:
            clip_ratios["policy_clip_low_ratio"] = 0.0
            clip_ratios["policy_clip_high_ratio"] = 0.0
            clip_ratios["policy_clip_region_ratio"] = 0.0

        return {
            "loss_value": loss_value_for_log,
            "non_finite": non_finite,
            "clip_stats": clip_ratios,
            "generation_result": generation_result,  # Pass through for diagnostic
        }

    def compute_trust_region_diagnostic_from_result(
        self,
        generation_result: Dict[str, Any],
        global_step: int,
    ):
        """
        Compute trust region diagnostic from generation result.
        
        Call this after compute_streaming_loss() to extract ratio statistics.
        
        Args:
            generation_result: The generation result dict with generation_logps
            global_step: Current training step
            
        Returns:
            TrustRegionDiagnostic instance or None if no ratios available
        """
        stored_gen_logps = generation_result.get("generation_logps")
        
        if stored_gen_logps is None:
            # No generation logps - cannot compute proper diagnostic
            return None
            
        # We need to recompute ratios to get statistics
        # For now, return a placeholder - trainer should compute this
        # after it has both current and generation logprobs
        return None

    def _get_generation_logprobs(
        self,
        generation_result: Dict[str, Any],
        k: int,
        effective_len: int,
        per_token_logps_k: torch.Tensor,
    ) -> torch.Tensor:
        """Extract and align generation-time logprobs for completion k."""
        stored_gen_logps = generation_result.get("generation_logps")

        if stored_gen_logps is not None and k < stored_gen_logps.size(0):
            old_logps_k = stored_gen_logps[k, :effective_len].to(self.device)

            # Ensure shapes match
            if old_logps_k.dim() == 1:
                old_logps_k = old_logps_k.unsqueeze(0)

            if old_logps_k.size(-1) != per_token_logps_k.size(-1):
                # Pad or trim to match current logprobs length
                if old_logps_k.size(-1) < per_token_logps_k.size(-1):
                    pad_len = per_token_logps_k.size(-1) - old_logps_k.size(-1)
                    old_logps_k = torch.cat(
                        [
                            old_logps_k,
                            torch.zeros(
                                old_logps_k.size(0), pad_len, device=self.device
                            ),
                        ],
                        dim=-1,
                    )
                else:
                    old_logps_k = old_logps_k[:, : per_token_logps_k.size(-1)]

            return old_logps_k
        else:
            # Fallback: use current policy (degenerates to ratio~1.0)
            return per_token_logps_k.detach()

    def _compute_clip_stats(
        self,
        per_token_logps_k: torch.Tensor,
        old_logps_k: torch.Tensor,
        adv_k: torch.Tensor,
        comp_mask_eff: torch.Tensor,
    ) -> Dict[str, int]:
        """Compute token-level clipping statistics."""
        coef_1 = torch.exp(per_token_logps_k - old_logps_k)
        adv_is_neg = (adv_k < 0).view(1, 1).expand_as(coef_1)
        adv_is_pos = (adv_k > 0).view(1, 1).expand_as(coef_1)

        is_low_clipped = (coef_1 < (1.0 - self.epsilon_low)) & adv_is_neg
        is_high_clipped = (coef_1 > (1.0 + self.epsilon_high)) & adv_is_pos
        token_mask_bool = comp_mask_eff > 0

        return {
            "low": int((is_low_clipped & token_mask_bool).sum().item()),
            "high": int((is_high_clipped & token_mask_bool).sum().item()),
            "region": int(
                ((is_low_clipped | is_high_clipped) & token_mask_bool).sum().item()
            ),
            "total": int(token_mask_bool.sum().item()),
        }

    def _compute_reference_kl(
        self,
        ref_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        effective_len: int,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        images_per_sample: Optional[torch.Tensor],
        temperature: float,
        per_token_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token KL divergence with reference model."""
        with torch.no_grad():
            ref_logps = logprobs.get_per_token_logps(
                model=ref_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=effective_len,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                images_per_sample=images_per_sample,
                temperature=temperature,
                detach=True,
            )

        return losses.compute_kl(per_token_logps, ref_logps)

    @staticmethod
    def _cleanup_completion_tensors(local_vars: Dict[str, Any]) -> None:
        """Free per-completion temporary tensors."""
        try:
            del (
                local_vars["input_ids_k"],
                local_vars["attention_mask_k"],
                local_vars["per_token_logps_k"],
                local_vars["old_logps_k"],
            )
            if "ref_logps" in local_vars:
                del local_vars["ref_logps"]
            logprobs.clear_gpu_memory()
        except Exception:
            pass


__all__ = ["CompletionLossComputer"]
