#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from src_post.rewards import REGISTRY


@dataclass(frozen=True)
class RLRunnerConfig:
    # Required paths (explicit)
    checkpoint: str
    processor: str
    output_dir: str
    # Dataset (one of the two must be provided)
    train_data_dir: Optional[str]
    eval_data_dir: Optional[str]
    # Optional outputs
    results_jsonl: Optional[str]
    metrics_jsonl: Optional[str]
    # Mission
    mission: Optional[str]
    # Runtime
    device: str
    seed: int
    limit_groups: int
    skip_save_checkpoints: bool
    # Logging
    tb_log_dir: Optional[str]
    run_name: Optional[str]
    # Generation
    temperature: float
    top_p: float
    max_new_tokens_stage_a: int
    max_new_tokens_stage_b: int
    mask_geometry_tokens: bool
    mask_coordinate_tokens: bool
    sanitize_stage_a: bool
    # GRPO sampling
    K_B: int
    K_A: int
    adv_clip: float
    length_norm: bool
    # Training
    train: bool
    num_updates: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    epochs: int
    drop_last: bool
    # Selective unfreeze + per-group LRs
    top_k_llm_layers: int
    top_k_vision_blocks: int
    freeze_vision_patch_embed: bool
    aligner_lr: Optional[float]
    llm_lr: Optional[float]
    vision_lr: Optional[float]
    # Legacy/compat knobs (kept to avoid breakage; not primary path)
    train_lm_head: bool
    train_last_n_layers: int
    lr_lm_head: Optional[float]
    lr_last_layers: Optional[float]
    # KL to reference policy
    use_ref_kl: bool
    ref_checkpoint: Optional[str]
    lambda_kl_stage_b: float
    lambda_kl_stage_a: float
    # Stage‑A training mode
    train_stage_a_mode: str  # 'off' | 'conditional' | 'joint'
    stage_a_weight: float
    K_set: int
    max_images_tf: int
    # Rewards
    reward_fns: List[str]
    reward_weights: List[float]
    # Diagnostics
    enable_phase_a_diagnostics: bool


def _read_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = p.read_text(encoding="utf-8")
    # Prefer YAML
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    # Fallback JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse config file {path}: {e}")
    return {}


def _as_float_opt(x: Any) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        return float(x)
    except Exception:
        return None


def load_and_validate_config(path: str) -> RLRunnerConfig:
    cfg: Dict[str, Any] = _read_config_file(path)
    # Back-compat alias
    if not cfg.get("processor") and cfg.get("processor_path"):
        cfg["processor"] = cfg.get("processor_path")

    # Core required keys
    checkpoint = str(cfg.get("checkpoint") or "").strip()
    processor = str(cfg.get("processor") or "").strip()
    output_dir = str(cfg.get("output_dir") or "").strip()
    if not checkpoint or not processor or not output_dir:
        raise ValueError(
            "Missing required keys: checkpoint, processor, output_dir. Provide absolute paths."
        )

    # Dataset: require at least one
    train_data_dir = cfg.get("train_data_dir")
    eval_data_dir = cfg.get("eval_data_dir")
    if not train_data_dir and not eval_data_dir:
        raise ValueError(
            "You must provide one of 'train_data_dir' or 'eval_data_dir'."
        )

    # Rewards: parse and validate
    rf = cfg.get("reward_fns", "label_match")
    if isinstance(rf, str):
        reward_fns = [s.strip() for s in rf.split(",") if s and str(s).strip()]
    else:
        reward_fns = [str(s).strip() for s in (rf or []) if str(s).strip()]
    rw = cfg.get("reward_weights", "1.0")
    if isinstance(rw, str):
        reward_weights = [float(s) for s in rw.split(",") if str(s).strip()]
    else:
        reward_weights = [float(s) for s in (rw or [])]
    if len(reward_fns) != len(reward_weights):
        raise ValueError(
            f"reward_weights length ({len(reward_weights)}) must match reward_fns length ({len(reward_fns)})."
        )
    unknown = [n for n in reward_fns if n not in REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown reward names: {unknown}. Valid: {sorted(list(REGISTRY.keys()))}."
        )

    # Stage-A mode
    mode = str(cfg.get("train_stage_a_mode", "conditional")).strip().lower()
    if mode not in {"off", "conditional", "joint"}:
        raise ValueError(
            f"train_stage_a_mode must be one of ['off','conditional','joint']; got '{mode}'."
        )

    # KL
    use_ref_kl = bool(cfg.get("use_ref_kl", True))
    ref_checkpoint = cfg.get("ref_checkpoint")
    if use_ref_kl and not ref_checkpoint:
        # Allow falling back to checkpoint only if explicitly set in config; else require ref
        raise ValueError("When use_ref_kl=true, you must set 'ref_checkpoint'.")

    # Numeric validations (fail-fast)
    K_B = int(cfg.get("K_B", 3))
    K_A = int(cfg.get("K_A", 3))
    K_set = int(cfg.get("K_set", 1))
    if K_B < 1 or K_A < 1:
        raise ValueError(f"K_B and K_A must be >=1; got K_B={K_B}, K_A={K_A}.")
    if mode == "joint" and K_set < 1:
        raise ValueError(f"When train_stage_a_mode=joint, K_set must be >=1; got {K_set}.")

    # Training core knobs
    train = bool(cfg.get("train", True))
    epochs = int(cfg.get("epochs", 0))
    batch_size = int(cfg.get("batch_size", 0))
    if train:
        if epochs < 1:
            raise ValueError(f"epochs must be >=1 when train=true; got {epochs}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >=1 when train=true; got {batch_size}.")

    # Results/metrics paths: derive if missing (non-core)
    results_jsonl = cfg.get("results_jsonl")
    metrics_jsonl = cfg.get("metrics_jsonl")
    if not results_jsonl:
        results_jsonl = str(Path(output_dir) / "results.jsonl")
    if not metrics_jsonl:
        metrics_jsonl = str(Path(output_dir) / "metrics.jsonl")

    # Limit groups normalization: (None|0|-1) => -1
    limit_groups_raw = cfg.get("limit_groups", -1)
    try:
        lg = int(limit_groups_raw if limit_groups_raw is not None else -1)
    except Exception:
        lg = -1
    if lg <= 0:
        lg = -1

    return RLRunnerConfig(
        checkpoint=checkpoint,
        processor=str(processor),
        output_dir=str(output_dir),
        train_data_dir=str(train_data_dir) if train_data_dir else None,
        eval_data_dir=str(eval_data_dir) if eval_data_dir else None,
        results_jsonl=str(results_jsonl),
        metrics_jsonl=str(metrics_jsonl),
        mission=(str(cfg.get("mission")) if cfg.get("mission") else None),
        device=str(cfg.get("device") or ("cuda")),
        seed=int(cfg.get("seed", 42)),
        limit_groups=int(lg),
        skip_save_checkpoints=bool(cfg.get("skip_save_checkpoints", False)),
        tb_log_dir=(str(cfg.get("tb_log_dir")) if cfg.get("tb_log_dir") else None),
        run_name=(str(cfg.get("run_name")) if cfg.get("run_name") else None),
        temperature=float(cfg.get("temperature", 0.5)),
        top_p=float(cfg.get("top_p", 0.9)),
        max_new_tokens_stage_a=int(cfg.get("max_new_tokens_stage_a", 32)),
        max_new_tokens_stage_b=int(cfg.get("max_new_tokens_stage_b", 128)),
        mask_geometry_tokens=bool(cfg.get("mask_geometry_tokens", False)),
        mask_coordinate_tokens=bool(cfg.get("mask_coordinate_tokens", False)),
        sanitize_stage_a=bool(cfg.get("sanitize_stage_a", False)),
        K_B=K_B,
        K_A=K_A,
        adv_clip=float(cfg.get("adv_clip", 1.5)),
        length_norm=bool(cfg.get("length_norm", True)),
        train=train,
        num_updates=int(cfg.get("num_updates", 0)),
        batch_size=batch_size,
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        epochs=epochs,
        drop_last=bool(cfg.get("drop_last", True)),
        top_k_llm_layers=int(cfg.get("top_k_llm_layers", int(cfg.get("train_last_n_layers", 0) or 0))),
        top_k_vision_blocks=int(cfg.get("top_k_vision_blocks", 0)),
        freeze_vision_patch_embed=bool(cfg.get("freeze_vision_patch_embed", True)),
        aligner_lr=_as_float_opt(cfg.get("aligner_lr") or cfg.get("merger_lr") or cfg.get("lr_aligner")),
        llm_lr=_as_float_opt(cfg.get("llm_lr") or cfg.get("lr_last_layers")),
        vision_lr=_as_float_opt(cfg.get("vision_lr")),
        train_lm_head=bool(cfg.get("train_lm_head", True)),
        train_last_n_layers=int(cfg.get("train_last_n_layers", 0)),
        lr_lm_head=_as_float_opt(cfg.get("lr_lm_head")),
        lr_last_layers=_as_float_opt(cfg.get("lr_last_layers")),
        use_ref_kl=use_ref_kl,
        ref_checkpoint=(str(ref_checkpoint) if ref_checkpoint else None),
        lambda_kl_stage_b=float(cfg.get("lambda_kl_stage_b", 0.02)),
        lambda_kl_stage_a=float(cfg.get("lambda_kl_stage_a", 0.02)),
        train_stage_a_mode=mode,
        stage_a_weight=float(cfg.get("stage_a_weight", 1.0)),
        K_set=K_set,
        max_images_tf=int(cfg.get("max_images_tf", 1)),
        reward_fns=reward_fns,
        reward_weights=reward_weights,
        enable_phase_a_diagnostics=bool(cfg.get("enable_phase_a_diagnostics", False)),
    )
