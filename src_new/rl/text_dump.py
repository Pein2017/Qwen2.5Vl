"""Utilities to append raw generation and GT text to a JSONL file.

This module is intentionally isolated from the training logic. It operates on the
GenerationBuffer and performs lightweight decoding and JSONL appends on rank 0.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import torch

from src_new.rl.utils import resolve_im_end_id_strict


_LOGGER = logging.getLogger("rl.text_dump")


def _ensure_parent_dir(path: str) -> None:
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _build_gt_text(meta: Dict[str, Any]) -> tuple[str | None, int | None]:
    """Build ground-truth text (and its tokenizer length) from meta objects if present.

    Returns (gt_text, gt_len_tokenizer). Either may be None when unavailable.
    """
    if not isinstance(meta, dict):
        return None, None
    try:
        if meta.get("objects"):
            from src_new.processing.coordinate_converter import (
                CoordinateTokenConverter as _Conv,
            )

            conv = _Conv()
            objs = meta.get("objects") or []
            out_lines: list[str] = []
            for idx, obj in enumerate(objs):
                # Use 'desc' directly as the caption (it contains the full hierarchical description)
                desc = obj.get("desc", "")
                if not isinstance(desc, str) or not desc.strip():
                    _LOGGER.warning(
                        f"Object {idx} missing non-empty 'desc' for GT caption, skipping"
                    )
                    continue
                caption = desc.strip()
                ref_text = conv.format_object_ref_for_user(caption)
                geom_text = conv.format_geometry_for_user(obj)
                out_lines.append(f"{ref_text}{geom_text}")
            gt_text = "\n".join(out_lines)
            return gt_text, None  # length computed later with tokenizer
    except Exception as _e:
        _LOGGER.debug("Failed to build GT text from meta: %s", _e)
    return None, None


def append_generation_text_samples(
    tokenizer: Any,
    gen_buffer: Any,
    output_file: str,
    step: int,
) -> None:
    """Append raw generated completions and GT text to a JSONL file.

    - Decodes completions with skip_special_tokens=False
    - Includes simple flags and lengths for quick analysis
    - Best-effort; failures are logged and ignored
    """
    if tokenizer is None or gen_buffer is None:
        return

    try:
        eos_id = resolve_im_end_id_strict(tokenizer)
    except Exception:
        eos_id = None

    try:
        P = len(gen_buffer.completion_ids_list)
        if P == 0:
            return
    except Exception:
        return

    _ensure_parent_dir(output_file)

    try:
        with open(output_file, "a", encoding="utf-8") as f:
            for prompt_idx in range(P):
                try:
                    meta = gen_buffer.meta_list[prompt_idx]
                except Exception:
                    meta = {}

                # Build GT text once per prompt
                gt_text, gt_len_tok = _build_gt_text(meta)
                if gt_text is not None and gt_len_tok is None:
                    try:
                        ids = tokenizer(
                            gt_text,
                            add_special_tokens=False,
                            return_attention_mask=False,
                        ).get("input_ids", [])
                        gt_len_tok = _safe_int(
                            len(ids)
                            if isinstance(ids, list)
                            else getattr(ids, "shape", [0])[0]
                        )
                    except Exception:
                        gt_len_tok = None

                try:
                    comp_ids_mat = gen_buffer.completion_ids_list[prompt_idx]
                    comp_mask_mat = gen_buffer.completion_mask_list[prompt_idx]
                    K = int(comp_ids_mat.size(0))
                except Exception:
                    continue

                for completion_idx in range(K):
                    try:
                        comp_ids = comp_ids_mat[completion_idx]
                        comp_mask = comp_mask_mat[completion_idx]
                        if isinstance(comp_ids, torch.Tensor):
                            comp_ids_t = comp_ids.detach().cpu().long()
                        else:
                            comp_ids_t = torch.tensor(comp_ids, dtype=torch.long)
                        if isinstance(comp_mask, torch.Tensor):
                            comp_mask_t = comp_mask.detach().cpu().long()
                        else:
                            comp_mask_t = torch.tensor(comp_mask, dtype=torch.long)

                        # Trim decode to effective length according to mask
                        try:
                            eff_len = int(comp_mask_t.long().sum().item())
                        except Exception:
                            eff_len = int(comp_ids_t.numel())
                        prediction_text_raw = tokenizer.decode(
                            comp_ids_t[:eff_len], skip_special_tokens=False
                        )

                        terminated_with_eos = False
                        if eos_id is not None:
                            try:
                                terminated_with_eos = bool(
                                    (comp_ids_t == int(eos_id)).any().item()
                                )
                            except Exception:
                                terminated_with_eos = False

                        # Truncated means masked zeros WITHOUT an EOS (cap/overflow),
                        # since zeros after EOS are expected padding
                        truncated = False
                        try:
                            has_zeros = bool((comp_mask_t == 0).any().item())
                            truncated = bool(has_zeros and (not terminated_with_eos))
                        except Exception:
                            truncated = False

                        row = {
                            "step": int(step),
                            "prompt_idx": int(prompt_idx),
                            "completion_idx": int(completion_idx),
                            "prediction_text_raw": prediction_text_raw,
                            "ground_truth_text": gt_text,
                            "gen_len_tokenizer": int(comp_ids_t.numel()),
                            "gt_len_tokenizer": (
                                None if gt_len_tok is None else int(gt_len_tok)
                            ),
                            "terminated_with_eos": bool(terminated_with_eos),
                            "truncated": bool(truncated),
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    except Exception as _e:
                        _LOGGER.debug(
                            "Failed to append text sample at prompt %d completion %d: %s",
                            prompt_idx,
                            completion_idx,
                            _e,
                        )
                        continue
    except Exception as e:
        _LOGGER.warning("Failed to write text samples to %s: %s", output_file, e)


__all__ = ["append_generation_text_samples"]
