#!/usr/bin/env python3
"""
Evaluation harness for dense-caption RL:
- Loads components from RL config
- Builds generation-ready inputs via ConversationBuilder (SFT parity)
- Generates outputs and computes reward-based metrics
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

import torch

from src_new.processing.special_tokens import (
    IM_END,
    require_core_special_tokens,
    require_geometry_tokens,
)
from src_new.rl.data.dataset import RLDenseJSONLDataset
from src_new.rl.prompting.conversation import (
    RLConversationContext,
)
from src_new.rl.rewards.format_rewards import (
    check_ascii_separators,
    check_banned_vocab,
    check_coords_counts,
    check_wrappers,
)
from src_new.rl.rewards.registry import REGISTRY, combine
from src_new.rl.runner import _load_yaml, build_components
from src_new.rl.utils import create_builder
from src_new.utils.rank_aware_logging import get_rank_aware_logger


_LOGGER = get_rank_aware_logger("rl.eval")


def _prepare_gen_kwargs(tok, cfg: Dict[str, Any]) -> Dict[str, Any]:
    eos_id = getattr(tok, "convert_tokens_to_ids", lambda x: None)(IM_END)
    kwargs = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
        "do_sample": True,
        "temperature": float(cfg.get("temperature", 0.9)),
        "repetition_penalty": float(cfg.get("repetition_penalty", 1.0)),
        "use_cache": True,
    }
    if isinstance(eos_id, int) and eos_id >= 0:
        kwargs["eos_token_id"] = eos_id
    return kwargs


def _generate(
    model, tokenizer, inputs: Dict[str, Any], gen_kwargs: Dict[str, Any]
) -> str:
    # Move to device and filter allowed keys
    device = next(model.parameters()).device
    allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    payload: Dict[str, Any] = {}
    for k, v in inputs.items():
        if k in allowed and torch.is_tensor(v):
            payload[k] = v.to(device)
    if payload["input_ids"].dim() == 1:
        payload["input_ids"] = payload["input_ids"].unsqueeze(0)
    if payload["attention_mask"].dim() == 1:
        payload["attention_mask"] = payload["attention_mask"].unsqueeze(0)
    input_len = int(payload["input_ids"].shape[1])
    with torch.no_grad():
        seq = model.generate(**payload, **gen_kwargs)
    if isinstance(seq, torch.Tensor) and seq.dim() == 2:
        out_ids = seq[0][input_len:]
    elif isinstance(seq, torch.Tensor) and seq.dim() == 1:
        out_ids = seq[input_len:]
    else:
        out_ids = getattr(seq, "sequences", None)
        if out_ids is None:
            raise RuntimeError("Unexpected generate output structure")
        out_ids = out_ids[0][input_len:]
    text = tokenizer.decode(out_ids, skip_special_tokens=False)
    # Trim at first <|im_end|>
    end_tok = "<|im_end|>"
    idx = text.find(end_tok)
    if idx != -1:
        text = text[:idx]
    return text.strip()


def evaluate(config_path: str) -> Dict[str, Any]:
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
    ds = RLDenseJSONLDataset(jsonl_path, ctx)
    gen_kwargs = _prepare_gen_kwargs(tok, cfg)

    n = 0
    sums = {
        "parse_reward": 0.0,
        "wrappers": 0.0,
        "coords": 0.0,
        "separators": 0.0,
        "vocab": 0.0,
        "reward": 0.0,
        # detection metrics
        "bbox_giou": 0.0,
        "quad_l1": 0.0,
        "line_l1": 0.0,
        "ordering": 0.0,
        "coverage": 0.0,
        "geometry_sanity": 0.0,
    }

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

    for batch in ds:
        text = _generate(model, tok, batch, gen_kwargs)
        meta = batch.get("meta") if isinstance(batch, dict) else None
        # Component formatting metrics
        w = check_wrappers(text)
        c = check_coords_counts(text)
        s = check_ascii_separators(text)
        v = check_banned_vocab(text)
        r = combine(text, rewards_w, meta=meta)
        n += 1
        sums["wrappers"] += w
        sums["coords"] += c
        sums["separators"] += s
        sums["vocab"] += v
        sums["reward"] += r
        # parse success approximated by presence of any geometry wrapper in combine
        sums["parse_reward"] += 1.0 if r > 0 else 0.0
        # Detection metrics
        sums["bbox_giou"] += _safe_call("bbox_giou", text, meta)
        sums["quad_l1"] += _safe_call("quad_l1", text, meta)
        sums["line_l1"] += _safe_call("line_l1", text, meta)
        sums["ordering"] += _safe_call("ordering", text, meta)
        sums["coverage"] += _safe_call("coverage", text, meta)
        sums["geometry_sanity"] += _safe_call("geometry_sanity", text, meta)

    def _avg(x: float) -> float:
        return float(x / max(n, 1))

    report = {
        "samples": n,
        "valid_parse_rate": _avg(sums["parse_reward"]),
        "wrapper_ok_rate": _avg(sums["wrappers"]),
        "coords_ok_rate": _avg(sums["coords"]),
        "ascii_sep_rate": _avg(sums["separators"]),
        "banned_vocab_pass_rate": _avg(sums["vocab"]),
        "avg_reward": _avg(sums["reward"]),
        # detection summary (means)
        "bbox_giou_mean": _avg(sums["bbox_giou"]),
        "quad_l1_mean": _avg(sums["quad_l1"]),
        "line_l1_mean": _avg(sums["line_l1"]),
        "ordering_ok_rate": _avg(sums["ordering"]),
        "coverage_mean": _avg(sums["coverage"]),
        "geometry_sanity_rate": _avg(sums["geometry_sanity"]),
    }

    output_file = cfg.get("eval_output_file")
    if isinstance(output_file, str) and output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dense-caption outputs with reward metrics"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to RL YAML config"
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    _LOGGER.setLevel(logging.INFO)

    rpt = evaluate(args.config)
    print(json.dumps(rpt, ensure_ascii=False))


if __name__ == "__main__":
    main()
