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
from typing import Any, Dict, List, Optional

import torch

from src_rl.runner import build_components, _load_yaml  # type: ignore
from src_new.processing.conversation.builder import ConversationBuilder
from src_rl.prompting.conversation import RLConversationContext, build_simple_generation_inputs
from src_rl.data.dataset import RLDenseJSONLDataset
from src_rl.rewards.registry import combine
from src_rl.rewards.format_rewards import (
    check_wrappers,
    check_coords_counts,
    check_ascii_separators,
    check_banned_vocab,
)

_LOGGER = logging.getLogger("src_rl.eval")


def _create_builder(processor, cfg: Dict[str, Any]) -> ConversationBuilder:
    # Required explicit keys for builder
    max_coord_value = cfg.get("max_coord_value")
    coordinate_tokens_enabled = bool(cfg.get("coordinate_tokens_enabled", False))
    if max_coord_value is None:
        raise ValueError("RL YAML missing required 'max_coord_value' for ConversationBuilder")
    return ConversationBuilder(
        processor=processor,
        max_coord_value=int(max_coord_value),
        coordinate_tokens_enabled=coordinate_tokens_enabled,
    )


def _prepare_gen_kwargs(tok, cfg: Dict[str, Any]) -> Dict[str, Any]:
    eos_id = getattr(tok, "convert_tokens_to_ids", lambda x: None)("<|im_end|>")
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


def _generate(model, tokenizer, inputs: Dict[str, Any], gen_kwargs: Dict[str, Any]) -> str:
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


def evaluate(jsonl_path: str, config_path: str, data_root: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    comps = build_components(config_path)
    tok, proc, model = comps["tokenizer"], comps["processor"], comps["model"]

    builder = _create_builder(proc, cfg)
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
    }

    for batch in ds:
        text = _generate(model, tok, batch, gen_kwargs)
        # Component metrics
        w = check_wrappers(text)
        c = check_coords_counts(text)
        s = check_ascii_separators(text)
        v = check_banned_vocab(text)
        r = combine(text, rewards_w)
        n += 1
        sums["wrappers"] += w
        sums["coords"] += c
        sums["separators"] += s
        sums["vocab"] += v
        sums["reward"] += r
        # parse success approximated by presence of any geometry wrapper in combine
        sums["parse_reward"] += 1.0 if r > 0 else 0.0

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
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense-caption outputs with reward metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to RL YAML config")
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSONL data file")
    parser.add_argument("--data_root", type=str, required=True, help="Data root for image resolution")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    _LOGGER.setLevel(logging.INFO)

    rpt = evaluate(args.input_file, args.config, args.data_root, args.output_file)
    print(json.dumps(rpt, ensure_ascii=False))


if __name__ == "__main__":
    main()
