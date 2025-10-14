from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from src_new_json.utils.debug_logging import get_rank_aware_logger


logger = get_rank_aware_logger(__name__)

# Internal phase-scoped defaults (used when caller passes only `phase`)
#
# Model components (Qwen2.5-VL):
# - Vision backbone: `model.visual.patch_embed`, `model.visual.blocks.*`
# - Aligner (patch-merger MLP): `model.visual.merger` (bridge from vision to LLM)
# - Language model (LLM): `model.language_model.embed_tokens`, `model.language_model.layers.*`, `lm_head`
#
# Keys (unified):
# - llm_top_k_block: number of last LLM decoder blocks to unfreeze (affects `model.language_model.layers.*`).
#   * 0 means keep all LLM blocks frozen.
# - vision_top_k_block: in phase_3 only, restrict vision unfreeze to the last K vision blocks.
#   * 0 means default vision behavior (no block restriction; combined with `freeze_patch_embed`).
# - freeze_patch_embed: when True, keep `model.visual.patch_embed` frozen in phase_3 for stability.
#
# Phases (balanced defaults for both 3B and 7B):
# - phase_1: train only the aligner (`visual.merger`); LLM and vision backbone frozen.
# - phase_2: phase_1 plus unfreeze the last K (=6) LLM layers; vision backbone remains frozen; aligner trainable.
# - phase_3: unfreeze all by default; optionally limit to last K vision blocks first (vision_top_k_block > 0) while keeping patch_embed
#   frozen if freeze_patch_embed is True.
PHASE_DEFAULTS = {
    "phase_1": {
        "llm_top_k_block": 0,
        "vision_top_k_block": 0,
        "freeze_patch_embed": True,
    },
    "phase_2": {
        "llm_top_k_block": 6,
        "vision_top_k_block": 0,
        "freeze_patch_embed": True,
    },
    "phase_3": {
        "llm_top_k_block": 0,
        "vision_top_k_block": 0,
        "freeze_patch_embed": True,
    },
}


@dataclass
class FreezeSummary:
    phase: str
    num_trainable_params: int
    top_k_llm_layers: int
    top_k_vision_blocks: int
    patch_embed_frozen: bool


class PhaseFreezeManager:
    """Apply per-phase training freezes for separate-run scheduling (JSON mode)."""

    def __init__(self) -> None:
        self._mask_handles: List[Any] = []

    # --------- Public API ---------
    @staticmethod
    def infer_phase_from_run_name(run_name: str) -> Optional[str]:
        if not run_name:
            return None
        name = run_name.lower()
        if "phase_1" in name:
            return "phase_1"
        if "phase_2" in name:
            return "phase_2"
        if "phase_3" in name:
            return "phase_3"
        return None

    def apply_phase(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        phase: str,
        *,
        llm_top_k_block: Optional[int] = None,
        vision_top_k_block: Optional[int] = None,
        freeze_patch_embed: Optional[bool] = None,
        trainable_token_strings: Optional[List[str]] = None,
    ) -> FreezeSummary:
        if phase not in ("phase_1", "phase_2", "phase_3"):
            raise ValueError(f"Unknown phase: {phase}")

        if phase not in PHASE_DEFAULTS:
            raise ValueError(f"Unknown phase defaults for: {phase}")
        defaults = PHASE_DEFAULTS[phase]
        eff_llm_top_k_block = (
            int(defaults["llm_top_k_block"]) if llm_top_k_block is None else int(llm_top_k_block)
        )
        eff_vision_top_k_blocks = (
            int(defaults["vision_top_k_block"]) if vision_top_k_block is None else int(vision_top_k_block)
        )
        eff_freeze_patch_embed = (
            bool(defaults["freeze_patch_embed"]) if freeze_patch_embed is None else bool(freeze_patch_embed)
        )

        self.clear()

        # 1) Freeze everything
        for _, p in model.named_parameters():
            p.requires_grad = False

        # 2) Always unfreeze visual.merger (MLP aligner)
        for name, p in model.named_parameters():
            if "visual.merger" in name:
                p.requires_grad = True

        # 3) Optional: restrict training to specific token IDs
        if trainable_token_strings:
            try:
                vocab = tokenizer.get_vocab()
                missing = [t for t in trainable_token_strings if t not in vocab]
                if missing:
                    raise ValueError(
                        f"Requested trainable tokens not in tokenizer vocab: {missing}"
                    )
                allowed_token_ids = [vocab[t] for t in trainable_token_strings]
                emb, lm_head = self._find_embedding_and_lm_head(model)
                if emb is not None and lm_head is not None:
                    emb.requires_grad = True
                    lm_head.requires_grad = True
                    self._apply_token_id_grad_masks(emb, lm_head, allowed_token_ids)
                else:
                    logger.warning(
                        "[PhaseFreeze] Could not access embeddings/LM head for token-slice masking"
                    )
            except Exception as e:
                logger.warning(f"[PhaseFreeze] trainable_token_strings handling failed: {e}")

        # 4) Unfreeze last K LLM layers for phase_2 and all for phase_3
        if phase in ("phase_2", "phase_3"):
            try:
                layers = list(model.language_model.layers)
                if eff_llm_top_k_block > 0:
                    for b in range(1, eff_llm_top_k_block + 1):
                        for p in layers[-b].parameters():
                            p.requires_grad = True
                else:
                    for p in model.language_model.parameters():
                        p.requires_grad = True
            except Exception as e:
                logger.warning(f"[PhaseFreeze] Failed to unfreeze LLM layers: {e}")

        # 5) Unfreeze vision blocks in phase_3 (optional top-k)
        if phase == "phase_3":
            try:
                blocks = list(model.visual.blocks)
                if eff_vision_top_k_blocks > 0:
                    for b in range(1, eff_vision_top_k_blocks + 1):
                        for p in blocks[-b].parameters():
                            p.requires_grad = True
                else:
                    for p in model.visual.parameters():
                        p.requires_grad = True
                if eff_freeze_patch_embed:
                    for p in model.visual.patch_embed.parameters():
                        p.requires_grad = False
            except Exception as e:
                logger.warning(f"[PhaseFreeze] Failed to unfreeze vision blocks: {e}")

        # Summary
        num_trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
        return FreezeSummary(
            phase=phase,
            num_trainable_params=num_trainable,
            top_k_llm_layers=eff_llm_top_k_block,
            top_k_vision_blocks=eff_vision_top_k_blocks,
            patch_embed_frozen=bool(eff_freeze_patch_embed),
        )

    # --------- Internals ---------
    def clear(self) -> None:
        for h in self._mask_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._mask_handles.clear()

    @staticmethod
    def _find_embedding_and_lm_head(model: torch.nn.Module) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
        emb = getattr(getattr(model, "language_model", None), "embed_tokens", None)
        lm_head = getattr(model, "lm_head", None)
        return emb, lm_head

    def _apply_token_id_grad_masks(
        self, emb: torch.nn.Module, lm_head: torch.nn.Module, allowed_token_ids: List[int]
    ) -> None:
        try:

            # Build boolean masks over vocab dimension
            vocab_size = emb.weight.shape[0]
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=emb.weight.device)
            for tid in allowed_token_ids:
                if 0 <= int(tid) < vocab_size:
                    mask[int(tid)] = True

            def mask_gradients_param(param):
                def hook(grad):
                    if grad is not None:
                        return grad * mask.view(-1, *([1] * (grad.dim() - 1))).to(grad.dtype)
                    return grad

                return param.register_hook(hook)

            self._mask_handles.append(mask_gradients_param(emb.weight))
            self._mask_handles.append(mask_gradients_param(lm_head.weight))
        except Exception as e:
            logger.warning(f"[PhaseFreeze] Failed to apply token-id grad masks: {e}")
