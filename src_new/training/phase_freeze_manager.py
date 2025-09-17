from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from src_new.processing.special_tokens import get_coord_token_range
from src_new.utils.debug_logging import get_rank_aware_logger


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
# - coord_slice_only: when True and coord-token range is available, apply grad masks to only the coordinate-token
#   rows of embeddings (`embed_tokens.weight`) and output head (`lm_head.weight`) during phase_1/phase_2.
# - freeze_patch_embed: when True, keep `model.visual.patch_embed` frozen in phase_3 for stability.
#
# Phases (balanced defaults for both 3B and 7B):
# - phase_1: train only the aligner (`visual.merger`) + optional coord-slice on embeddings/LM head; LLM and vision backbone frozen.
# - phase_2: phase_1 plus unfreeze the last K (=6) LLM layers; vision backbone remains frozen; aligner trainable.
# - phase_3: unfreeze all by default; keep `visual.patch_embed` frozen unless overridden; optionally limit to last K vision blocks.
PHASE_DEFAULTS = {
    "phase_1": {
        "llm_top_k_block": 0,
        "vision_top_k_block": 0,
        "coord_slice_only": True,
        "freeze_patch_embed": True,
    },
    "phase_2": {
        "llm_top_k_block": 6,
        "vision_top_k_block": 0,
        "coord_slice_only": True,
        "freeze_patch_embed": True,
    },
    "phase_3": {
        "llm_top_k_block": 0,
        "vision_top_k_block": 0,
        "coord_slice_only": True,
        "freeze_patch_embed": True,
    },
}


@dataclass
class FreezeSummary:
    phase: str
    num_trainable_params: int
    coord_slice_enabled: bool
    coord_range: Optional[Tuple[int, int]]
    top_k_llm_layers: int
    top_k_vision_blocks: int
    patch_embed_frozen: bool


class PhaseFreezeManager:
    """Apply per-phase training freezes for separate-run scheduling.

    Phases:
      - phase1: unfreeze visual.merger and (if available) only coordinate-token rows of
                embed_tokens.weight and lm_head.weight (via grad masks). LLM layers and
                vision backbone stay frozen.
      - phase2: phase1 plus unfreeze the last K LLM decoder layers (llm_top_k_block).
                Vision backbone remains frozen except visual.merger.
      - phase3: unfreeze all parameters by default. Optionally, unfreeze only the last
                K vision blocks first (vision_top_k_block > 0) while keeping patch_embed
                frozen if freeze_patch_embed is True.

    This manager supports both standard-LLM mode (no coordinate tokens present) and
    coordinate-token mode. In standard mode, coord-slice masking is skipped.
    """

    def __init__(self) -> None:
        self._mask_handles: List[Any] = []

    # --------- Public API ---------
    @staticmethod
    def infer_phase_from_run_name(run_name: str) -> Optional[str]:
        if not run_name:
            return None
        name = run_name.lower()
        # Strict pattern: only accept explicit "phase_*" markers
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
        coord_slice_only: Optional[bool] = None,
        freeze_patch_embed: Optional[bool] = None,
        trainable_token_strings: Optional[List[str]] = None,
    ) -> FreezeSummary:
        """Apply per-phase freeze policy with optional overrides.

        Args:
            model: The Qwen2.5-VL conditional generation model (or wrapper exposing the same parameter names).
            tokenizer: Tokenizer for detecting coordinate-token range (if present).
            phase: One of {"phase_1", "phase_2", "phase_3"}.
            llm_top_k_block: Controls LLM unfreeze: 0=frozen, -1=all LLM blocks, k=last k decoder blocks.
            vision_top_k_block: Controls vision unfreeze: 0=frozen, -1=all vision blocks, k=last k blocks.
            coord_slice_only: Optional override for enabling coord-slice masking on embeddings/LM head in phase_1/2.
            freeze_patch_embed: Optional override to keep `visual.patch_embed` frozen when not unfreezing all vision.
            trainable_token_strings: Optional list of exact token strings to restrict training to those embedding/LM-head rows only.

        Notes:
            - The aligner (patch-merger MLP) at `model.visual.merger` is always kept trainable in all phases.
            - LLM "layers" refer to full decoder blocks under `model.language_model.layers.*` (attention + MLP + norms).
            - Vision "blocks" refer to transformer blocks under `model.visual.blocks.*`.
        """
        if phase not in ("phase_1", "phase_2", "phase_3"):
            raise ValueError(f"Unknown phase: {phase}")

        # Resolve effective settings directly from arguments (no phase-derived defaults)
        eff_top_k_layers = int(llm_top_k_block) if llm_top_k_block is not None else 0
        eff_vision_top_k_blocks = int(vision_top_k_block) if vision_top_k_block is not None else 0
        eff_coord_slice_only = bool(coord_slice_only) if coord_slice_only is not None else False
        eff_freeze_patch_embed = bool(freeze_patch_embed) if freeze_patch_embed is not None else True

        # Reset any prior hooks
        self.clear()

        # Determine coordinate token range from tokenizer
        coord_rng = None
        coord_slice_enabled = False
        try:
            rng = get_coord_token_range(tokenizer)
            if rng is not None and int(rng.end_exclusive) > int(rng.start_id):
                coord_rng = (int(rng.start_id), int(rng.end_exclusive))
        except Exception as e:
            logger.warning(f"[PhaseFreeze] Failed to derive coord token range: {e}")

        # 1) Freeze everything
        for _, p in model.named_parameters():
            p.requires_grad = False

        # 2) Always unfreeze visual.merger (MLP aligner)
        for name, p in model.named_parameters():
            if "visual.merger" in name:
                p.requires_grad = True

        # 2b) Always unfreeze LM head so it can be tuned across phases
        for name, p in model.named_parameters():
            if name.endswith("lm_head.weight"):
                p.requires_grad = True

        # Debug: summarize trainable modules after applying phase policy
        try:
            vision_trainable = 0
            vision_total = 0
            llm_trainable = 0
            llm_total = 0
            merger_trainable = False
            lm_head_trainable = False

            for name, p in model.named_parameters():
                if ".visual.blocks." in name:
                    vision_total += 1
                    if p.requires_grad:
                        vision_trainable += 1
                if (".language_model.layers." in name) or (".model.layers." in name):
                    llm_total += 1
                    if p.requires_grad:
                        llm_trainable += 1
                if "visual.merger" in name and p.requires_grad:
                    merger_trainable = True
                if name.endswith("lm_head.weight") and p.requires_grad:
                    lm_head_trainable = True

            phase_logger = None
            try:
                from src_new.utils.rank_aware_logging import get_rank_aware_logger as _get
                phase_logger = _get("training.phase_freeze_manager")
            except Exception:
                import logging as _logging
                phase_logger = _logging.getLogger("training.phase_freeze_manager")

            phase_logger.info(
                f"🧊 Phase freeze summary — vision_trainable_params={vision_trainable}/{vision_total}, "
                f"llm_trainable_params={llm_trainable}/{llm_total}, merger_trainable={merger_trainable}, "
                f"lm_head_trainable={lm_head_trainable}"
            )
        except Exception:
            pass

        # 3) Optional: restrict training to specific token IDs (e.g., line_start/line_end)
        token_slice_enabled = False
        allowed_token_ids: List[int] = []
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
                    token_slice_enabled = True
                else:
                    logger.warning(
                        "[PhaseFreeze] Could not access embeddings/LM head for token-slice masking"
                    )
            except Exception as e:
                logger.warning(f"[PhaseFreeze] trainable_token_strings handling failed: {e}")

        # 3b) Optionally unfreeze coord-token slices of embeddings/LM head in phase_1/2
        if (
            coord_rng is not None
            and eff_coord_slice_only
            and phase in ("phase_1", "phase_2")
            and not token_slice_enabled
        ):
            emb, lm_head = self._find_embedding_and_lm_head(model)
            if emb is not None and lm_head is not None:
                emb.requires_grad = True
                lm_head.requires_grad = True
                self._apply_coord_slice_grad_masks(
                    emb, lm_head, coord_rng[0], coord_rng[1]
                )
                coord_slice_enabled = True
            else:
                logger.warning(
                    "[PhaseFreeze] Could not access embeddings/LM head for coord-slice masking"
                )

        # 4) Unified unfreezing controlled solely by top-k settings
        # LLM control: 0=frozen, -1=all, k=last-k
        if eff_top_k_layers == -1:
            # Unfreeze all detected LLM layers
            self._unfreeze_last_k_llm_layers(model, k=10**9)
            # Ensure merger remains trainable
            for name, p in model.named_parameters():
                if "visual.merger" in name:
                    p.requires_grad = True
        elif eff_top_k_layers > 0:
            self._unfreeze_last_k_llm_layers(model, eff_top_k_layers)

        # Vision control: 0=frozen, -1=all, k=last-k
        if eff_vision_top_k_blocks == -1:
            # Unfreeze all vision blocks and (optionally) patch_embed
            self._unfreeze_last_k_vision_blocks(model, k=10**9)
            for name, p in model.named_parameters():
                if "visual.patch_embed" in name:
                    p.requires_grad = True
        elif eff_vision_top_k_blocks > 0:
            self._unfreeze_last_k_vision_blocks(model, eff_vision_top_k_blocks)
            # Keep patch_embed frozen unless explicitly overridden
            if eff_freeze_patch_embed:
                for name, p in model.named_parameters():
                    if "visual.patch_embed" in name:
                        p.requires_grad = False

        # Respect token-slice preference: if token_slice_enabled and we didn't unfreeze-all LLM, embeddings/LM head are masked
        if token_slice_enabled and eff_top_k_layers != -1:
            # No additional action needed; mask hooks are already installed
            pass

        # 5) Summarize
        num_trainable = 0
        for _, p in model.named_parameters():
            if p.requires_grad:
                try:
                    num_trainable += p.numel()
                except Exception:
                    pass

        summary = FreezeSummary(
            phase=phase,
            num_trainable_params=num_trainable,
            coord_slice_enabled=coord_slice_enabled,
            coord_range=coord_rng,
            top_k_llm_layers=int(eff_top_k_layers or 0),
            top_k_vision_blocks=int(eff_vision_top_k_blocks or 0),
            patch_embed_frozen=bool(eff_freeze_patch_embed),
        )

        try:
            logger.info(
                "[PhaseFreeze] Applied phase=%s | trainable_params≈%s | coord_slice=%s | coord_range=%s | topK_llm=%s | topK_vision=%s | patch_embed_frozen=%s",
                summary.phase,
                str(summary.num_trainable_params),
                str(summary.coord_slice_enabled),
                str(summary.coord_range),
                str(summary.top_k_llm_layers),
                str(summary.top_k_vision_blocks),
                str(summary.patch_embed_frozen),
            )
        except Exception:
            pass

        return summary

    def clear(self) -> None:
        for h in self._mask_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._mask_handles.clear()

    # --------- Internals ---------
    def _find_embedding_and_lm_head(
        self, model: torch.nn.Module
    ) -> Tuple[Optional[torch.nn.Parameter], Optional[torch.nn.Parameter]]:
        # For Qwen2.5-VL, the container is under `.model`, not `.base_model`.
        base = getattr(model, "model", getattr(model, "base_model", model))

        emb_weight: Optional[torch.nn.Parameter] = None
        lm_head_weight: Optional[torch.nn.Parameter] = None

        # Try standard HF API for embeddings
        try:
            if hasattr(base, "get_input_embeddings"):
                emb = base.get_input_embeddings()
                if (
                    emb is not None
                    and hasattr(emb, "weight")
                    and isinstance(emb.weight, torch.nn.Parameter)
                ):
                    emb_weight = emb.weight
        except Exception:
            pass
        # Fallback: scan for embed_tokens.weight
        if emb_weight is None:
            try:
                for n, p in model.named_parameters():
                    if n.endswith("embed_tokens.weight") and isinstance(
                        p, torch.nn.Parameter
                    ):
                        emb_weight = p
                        break
            except Exception:
                pass

        # Locate lm_head at top-level first (Qwen2.5-VL stores it on the outer module)
        try:
            if (
                hasattr(model, "lm_head")
                and hasattr(model.lm_head, "weight")
                and isinstance(model.lm_head.weight, torch.nn.Parameter)
            ):
                lm_head_weight = model.lm_head.weight
        except Exception:
            pass
        # Fallback to base.lm_head
        if lm_head_weight is None:
            try:
                if (
                    hasattr(base, "lm_head")
                    and hasattr(base.lm_head, "weight")
                    and isinstance(base.lm_head.weight, torch.nn.Parameter)
                ):
                    lm_head_weight = base.lm_head.weight
            except Exception:
                pass
        # Final fallback: search by name
        if lm_head_weight is None:
            try:
                for n, p in model.named_parameters():
                    if n.endswith("lm_head.weight") and isinstance(
                        p, torch.nn.Parameter
                    ):
                        lm_head_weight = p
                        break
            except Exception:
                pass

        return emb_weight, lm_head_weight

    def _apply_coord_slice_grad_masks(
        self,
        embed_param: torch.nn.Parameter,
        lm_head_param: torch.nn.Parameter,
        coord_start: int,
        coord_end_exclusive: int,
    ) -> None:
        device = embed_param.device
        vocab_rows = (
            embed_param.shape[0] if embed_param.dim() == 2 else embed_param.shape[1]
        )
        row_mask = torch.zeros(vocab_rows, dtype=torch.bool, device=device)
        row_mask[coord_start:coord_end_exclusive] = True

        def _mask_embed_grad(g: torch.Tensor) -> torch.Tensor:
            if g is None:
                return g
            return g * row_mask.unsqueeze(1).to(device=g.device, dtype=g.dtype)

        def _mask_lm_head_grad(g: torch.Tensor) -> torch.Tensor:
            if g is None or g.dim() != 2:
                return g
            if lm_head_param.shape[0] == vocab_rows:
                return g * row_mask.unsqueeze(1).to(device=g.device, dtype=g.dtype)
            elif lm_head_param.shape[1] == vocab_rows:
                return g * row_mask.to(device=g.device, dtype=g.dtype)
            return g

        self._mask_handles.append(embed_param.register_hook(_mask_embed_grad))
        self._mask_handles.append(lm_head_param.register_hook(_mask_lm_head_grad))

    def _apply_token_id_grad_masks(
        self,
        embed_param: torch.nn.Parameter,
        lm_head_param: torch.nn.Parameter,
        allowed_token_ids: List[int],
    ) -> None:
        device = embed_param.device
        vocab_rows = (
            embed_param.shape[0] if embed_param.dim() == 2 else embed_param.shape[1]
        )
        row_mask = torch.zeros(vocab_rows, dtype=torch.bool, device=device)
        for tid in allowed_token_ids:
            if 0 <= int(tid) < int(vocab_rows):
                row_mask[int(tid)] = True

        def _mask_embed_grad(g: torch.Tensor) -> torch.Tensor:
            if g is None:
                return g
            return g * row_mask.unsqueeze(1).to(device=g.device, dtype=g.dtype)

        def _mask_lm_head_grad(g: torch.Tensor) -> torch.Tensor:
            if g is None or g.dim() != 2:
                return g
            if lm_head_param.shape[0] == vocab_rows:
                return g * row_mask.unsqueeze(1).to(device=g.device, dtype=g.dtype)
            elif lm_head_param.shape[1] == vocab_rows:
                return g * row_mask.to(device=g.device, dtype=g.dtype)
            return g

        self._mask_handles.append(embed_param.register_hook(_mask_embed_grad))
        self._mask_handles.append(lm_head_param.register_hook(_mask_lm_head_grad))

    def _unfreeze_last_k_llm_layers(self, model: torch.nn.Module, k: int) -> None:
        # Support both Qwen2.5-VL ("model.language_model.layers.") and legacy ("model.layers.") patterns
        layer_markers: List[str] = [
            "model.language_model.layers.",
            "language_model.layers.",
            "model.layers.",
            "transformer.layers.",
        ]

        def _extract_idx(param_name: str, marker: str) -> Optional[int]:
            if marker not in param_name:
                return None
            try:
                after = param_name.split(marker, 1)[1]
                idx_str = after.split(".", 1)[0]
                return int(idx_str)
            except Exception:
                return None

        # Compute max index per marker
        max_idx_by_marker: dict[str, int] = {}
        for name, _ in model.named_parameters():
            for marker in layer_markers:
                idx = _extract_idx(name, marker)
                if idx is not None:
                    prev = max_idx_by_marker.get(marker, -1)
                    if idx > prev:
                        max_idx_by_marker[marker] = idx

        if not max_idx_by_marker:
            logger.warning(
                "[PhaseFreeze] Could not detect LLM decoder layers for top-K unfreeze (no known markers)"
            )
            return

        thresholds: dict[str, int] = {}
        for marker, max_idx in max_idx_by_marker.items():
            thresholds[marker] = max(0, (max_idx + 1) - k)

        # Apply requires_grad=True for last-k layers across all detected markers
        for name, p in model.named_parameters():
            for marker, threshold in thresholds.items():
                idx = _extract_idx(name, marker)
                if idx is not None and idx >= threshold:
                    p.requires_grad = True
                    break

    def _freeze_all_vision_blocks(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if "visual.blocks." in name:
                p.requires_grad = False

    def _unfreeze_last_k_vision_blocks(self, model: torch.nn.Module, k: int) -> None:
        # Find max vision block index
        max_idx = -1
        for name, _ in model.named_parameters():
            if "visual.blocks." in name:
                try:
                    after = name.split("visual.blocks.", 1)[1]
                    idx_str = after.split(".", 1)[0]
                    idx = int(idx_str)
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    continue
        if max_idx < 0:
            logger.warning(
                "[PhaseFreeze] Could not detect vision blocks for top-K unfreeze"
            )
            return

        threshold = max(0, (max_idx + 1) - k)
        for name, p in model.named_parameters():
            if "visual.blocks." in name:
                try:
                    after = name.split("visual.blocks.", 1)[1]
                    idx_str = after.split(".", 1)[0]
                    idx = int(idx_str)
                    if idx >= threshold:
                        p.requires_grad = True
                except Exception:
                    continue
