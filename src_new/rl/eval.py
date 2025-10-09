#!/usr/bin/env python3
"""
Evaluation harness for dense-caption RL:
- Loads components from RL config
- Builds generation-ready inputs via ConversationBuilder (SFT parity)
- Generates outputs and computes reward-based metrics
"""

from __future__ import annotations

import json
from typing import Any, Dict

import torch

from src_new.processing.special_tokens import (
    IM_END,
    require_core_special_tokens,
    require_geometry_tokens,
)
from src_new.rl.data.dataset import RLDenseJSONLDataset
from src_new.rl.eval_utils import dump_samples, objects_to_boxes
from src_new.rl.generation import generate_completions, prepare_generate_inputs
from src_new.rl.prompting.conversation import (
    RLConversationContext,
)
from src_new.rl.rewards.registry import REGISTRY, combine
from src_new.rl.utils import create_builder, trim_at_first_im_end
from src_new.utils.rank_aware_logging import get_rank_aware_logger


_LOGGER = get_rank_aware_logger("rl.eval")


def _prepare_gen_kwargs(tok, cfg: Dict[str, Any]) -> Dict[str, Any]:
    eos_id = getattr(tok, "convert_tokens_to_ids", lambda x: None)(IM_END)
    kwargs = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
        "do_sample": True,
        "temperature": float(cfg.get("temperature", 0.9)),
        "repetition_penalty": float(cfg.get("repetition_penalty", 1.0)),
        "top_p": float(cfg.get("top_p", 0.9)),
        "use_cache": True,
    }
    if isinstance(eos_id, int) and eos_id >= 0:
        kwargs["eos_token_id"] = eos_id
    return kwargs


def _generate(
    model, tokenizer, inputs: Dict[str, Any], gen_kwargs: Dict[str, Any]
) -> str:
    # Sanitize to device and shapes using shared helper
    try:
        batch = prepare_generate_inputs(model, inputs)
    except Exception:
        # Fallback to minimal path if unusual shapes
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        if batch.get("input_ids") is not None and batch["input_ids"].dim() == 1:
            batch["input_ids"] = batch["input_ids"].unsqueeze(0)
        if (
            batch.get("attention_mask") is not None
            and batch["attention_mask"].dim() == 1
        ):
            batch["attention_mask"] = batch["attention_mask"].unsqueeze(0)
    input_len = int(batch["input_ids"].shape[1])
    seq = generate_completions(model, tokenizer, batch, **gen_kwargs)
    out_ids = seq[0][input_len:] if seq.dim() == 2 else seq[input_len:]
    text = tokenizer.decode(out_ids, skip_special_tokens=False)
    return trim_at_first_im_end(text, tokenizer)


def compute_absolute_metrics(
    text: str, meta: Dict[str, Any] | None
) -> Dict[str, float]:
    """Compute objective metrics on a single sample."""
    try:
        from src_new.rl.rewards.detection_rewards import (
            giou_bbox as _giou,
        )
        from src_new.rl.rewards.detection_rewards import (
            parse_dense_caption as _parse_pred,
        )
        from src_new.rl.rewards.format_rewards import (
            check_banned_vocab as _vocab,
        )
        from src_new.rl.rewards.format_rewards import (
            check_coords_counts as _cc,
        )
        from src_new.rl.rewards.format_rewards import (
            check_wrappers as _w,
        )
        from src_new.rl.rewards.format_rewards import (
            duplicate_penalty as _dup,
        )
        from src_new.rl.rewards.format_rewards import (
            pairing_ratio as _pair,
        )
        from src_new.rl.rewards.format_rewards import (
            separators_score as _sep,
        )
    except Exception:
        return {}
    pred_objs = []
    try:
        pred_objs = _parse_pred(text)
    except Exception:
        pred_objs = []
    gt_objs = meta.get("objects", []) if isinstance(meta, dict) else []

    def _boxes(objs):
        out = []
        for o in objs or []:
            if isinstance(o, dict):
                if (
                    "bbox_2d" in o
                    and isinstance(o["bbox_2d"], list)
                    and len(o["bbox_2d"]) >= 4
                ):
                    x1, y1, x2, y2 = map(int, o["bbox_2d"][:4])
                    out.append((x1, y1, x2, y2))
                elif (
                    "quad" in o and isinstance(o["quad"], list) and len(o["quad"]) >= 8
                ):
                    q = o["quad"][:8]
                    xs, ys = q[0::2], q[1::2]
                    out.append((min(xs), min(ys), max(xs), max(ys)))
        return out

    pred_b, gt_b = objects_to_boxes(pred_objs), objects_to_boxes(gt_objs)
    giou_vals = []
    if pred_b and gt_b:
        used = set()
        for pb in pred_b:
            best, best_v = None, -2.0
            for j, gb in enumerate(gt_b):
                if j in used:
                    continue
                v = _giou(pb, gb)
                if v > best_v:
                    best_v, best = v, j
            if best is not None:
                used.add(best)
                giou_vals.append(best_v)
    giou_mean = float(sum(giou_vals) / len(giou_vals)) if giou_vals else 0.0
    giou_sorted = sorted(giou_vals) if giou_vals else []
    giou_p90 = (
        float(giou_sorted[int(0.9 * (len(giou_sorted) - 1))]) if giou_sorted else 0.0
    )

    strict_ok = (
        1.0 if (_w(text) > 0.5 and _cc(text) >= 1.0 and _sep(text) >= 0.8) else 0.0
    )
    pr = float(_pair(text, meta=meta))
    dup_rate = float(1.0 - _dup(text))
    banned_rate = 1.0 - float(_vocab(text))

    # length proxies
    import re as _re

    nums = len(_re.findall(r"-?\d+", text))
    digits = sum(ch.isdigit() for ch in text)
    total = max(1, len(text))
    numeric_tail_fraction = max(
        float(nums) / float(max(1, len(text.split()))), float(digits) / float(total)
    )

    return {
        "object_count_mae": float(
            abs(len(pred_objs) - (len(gt_objs) if isinstance(gt_objs, list) else 0))
        ),
        "giou_mean": giou_mean,
        "giou_p90": giou_p90,
        "strict_parse_rate": strict_ok,
        "pairing_ratio": pr,
        "duplicate_rate": dup_rate,
        "banned_vocab_rate": banned_rate,
        "numeric_tail_fraction": float(numeric_tail_fraction),
    }


def evaluate(config_path: str) -> Dict[str, Any]:
    # Local import to avoid circular import during module initialization
    from src_new.rl.runner import _load_yaml, build_components

    cfg = _load_yaml(config_path)
    comps = build_components(config_path)
    tok, proc, model = comps["tokenizer"], comps["processor"], comps["model"]

    # Validate required tokens (fail fast)
    require_core_special_tokens(tok)
    require_geometry_tokens(tok, require_line=True)

    # Load required dataset paths strictly from YAML
    jsonl_path = cfg.get("val_data_path")
    data_root = cfg.get("data_root")
    if not jsonl_path or not isinstance(jsonl_path, str):
        raise ValueError("Missing required 'val_data_path' in config for evaluation")
    if not data_root or not isinstance(data_root, str):
        raise ValueError("Missing required 'data_root' in config for evaluation")

    builder = create_builder(proc)
    ctx = RLConversationContext(builder=builder, data_root=data_root)

    rewards_w: Dict[str, float] = cfg.get("rewards", {})
    observe_rewards_raw = cfg.get("observe_rewards", [])
    if not isinstance(observe_rewards_raw, list):
        observe_rewards_raw = []

    # Build dynamic reward keys from config
    active_keys = [k for k, w in rewards_w.items() if float(w) != 0.0 and k in REGISTRY]
    observe_keys = [
        k
        for k in observe_rewards_raw
        if isinstance(k, str) and k in REGISTRY and k not in active_keys
    ]
    eval_keys = sorted(set(active_keys) | set(observe_keys))

    ds = RLDenseJSONLDataset(jsonl_path, ctx)
    gen_kwargs = _prepare_gen_kwargs(tok, cfg)
    # Parser for sample dumps
    try:
        from src_new.rl.rewards.detection_rewards import (
            parse_dense_caption as _parse_pred,
        )
    except Exception:
        _parse_pred = None

    n = 0
    sums = {key: 0.0 for key in eval_keys}
    sums.update(
        {
            "parse_reward": 0.0,
            "reward": 0.0,
        }
    )

    def _safe_call(name: str, text: str, meta: Dict[str, Any] | None) -> float:
        fn = REGISTRY.get(name)
        if fn is None:
            return 0.0
        try:
            return float(fn(text, meta=meta))
        except TypeError:
            return float(fn(text))
        except Exception:
            return 0.0

    # Multi-rank: one sample per rank per pass
    try:
        from accelerate import Accelerator

        accelerator = Accelerator()
        rank = getattr(accelerator.state, "process_index", 0)
        world = getattr(accelerator.state, "num_processes", 1)
    except Exception:
        accelerator = None
        rank, world = 0, 1

    # Iterate over dataset with stride = world, starting at rank
    total = 0
    saved_rows: list[Dict[str, Any]] = []
    save_limit_cfg = int(cfg.get("evaluation", {}).get("save_samples", 0) or 0)
    for idx, batch in enumerate(ds):
        if (idx % max(world, 1)) != rank:
            continue
        text = _generate(model, tok, batch, gen_kwargs)
        meta = batch.get("meta") if isinstance(batch, dict) else None
        r = combine(text, rewards_w, meta=meta)
        n += 1
        sums["reward"] += r
        sums["parse_reward"] += 1.0 if r > 0 else 0.0
        for key in eval_keys:
            sums[key] += _safe_call(key, text, meta)
        # Absolute metrics (rank-local accumulation; reduce later)
        abs_metrics = compute_absolute_metrics(text, meta or {})
        for k, v in abs_metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        total += 1
        # Save sample rows on main process only (avoid cross-rank string gather)
        if (
            save_limit_cfg > 0
            and (accelerator is None or accelerator.is_main_process)
            and len(saved_rows) < save_limit_cfg
        ):
            try:
                pred_objs = _parse_pred(text) if _parse_pred is not None else []
            except Exception:
                pred_objs = []
            saved_rows.append(
                {
                    "image": batch.get("image") if isinstance(batch, dict) else None,
                    "prediction_text_raw": text,
                    "prediction": pred_objs,
                    "ground_truth": meta.get("objects", [])
                    if isinstance(meta, dict)
                    else [],
                    "width": meta.get("width") if isinstance(meta, dict) else None,
                    "height": meta.get("height") if isinstance(meta, dict) else None,
                }
            )
        # Exactly one generation per rank if asked via env
        if bool(int(cfg.get("evaluation", {}).get("per_rank_samples", 1))):
            break

    # Cross-rank gather and average on rank 0
    if accelerator is not None and world > 1:
        import torch

        keys = list(sums.keys())
        vec = torch.tensor([float(sums[k]) for k in keys], dtype=torch.float32)
        cnt = torch.tensor([float(n)], dtype=torch.float32)
        vec_all = accelerator.gather(vec)
        cnt_all = accelerator.gather(cnt)
        if accelerator.is_main_process:
            sums = {k: float(vec_all[:, i].sum().item()) for i, k in enumerate(keys)}
            n = int(cnt_all.sum().item())
        else:
            n = int(cnt_all.sum().item())

    def _avg(x: float) -> float:
        return float(x / max(n, 1))

    report = {
        "samples": n,
        "valid_parse_rate": _avg(sums["parse_reward"]),
        "avg_reward": _avg(sums["reward"]),
    }

    # Add dynamic reward metrics and absolute metrics
    for key in eval_keys:
        report[f"{key}_mean"] = _avg(sums[key])
    for k in list(sums.keys()):
        if k in {"reward", "parse_reward"}:
            continue
        if k in eval_keys:
            continue
        if k.endswith("_mean"):
            continue
        # absolute metrics
        report[f"{k}_mean"] = _avg(sums[k])

    output_file = cfg.get("eval_output_file")
    if isinstance(output_file, str) and output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    # Dump sample rows to JSONL on main process
    if saved_rows and (accelerator is None or accelerator.is_main_process):
        dump_samples(
            saved_rows,
            output_file=output_file if isinstance(output_file, str) else None,
            output_dir=(cfg.get("output", {}) or {}).get("output_dir"),
            step=None,
            limit=save_limit_cfg if save_limit_cfg > 0 else None,
        )
    return report


__all__ = ["evaluate", "compute_absolute_metrics"]
