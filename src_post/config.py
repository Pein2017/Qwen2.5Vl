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
    top_p: float  # recommended Stage-A baseline: temperature=0.01, top_p=0.5
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
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    epochs: int
    drop_last: bool
    # Selective unfreeze + per-group LRs
    llm_top_k_block: int
    vision_top_k_block: int
    freeze_patch_embed: bool
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
    # New: Stage‑B control and prompt bias
    train_stage_b: bool
    stage_b_weight: float
    freeze_stage_b_steps: int
    use_mission_checklist: bool
    # New: group reward strategy
    group_reward_mode: str  # 'margin_only' | 'label_match' | 'combined'
    # New: pairwise fallback
    pairwise_credit_enabled: bool
    pairwise_pairs_per_group: int
    pairwise_delta_threshold: float
    # New: uncertainty gate
    use_uncertainty_gate: bool
    uncertainty_gate_min_entropy: float
    # New: optimizer/metrics extras
    grad_accum_steps: int
    decision_ce_ema_beta: float
    # New: toggles
    train_aligner: bool


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

# -------- Required key helpers (fail-fast for core knobs) --------

def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required config key: '{key}'")
    return cfg[key]


def _req_str(cfg: Dict[str, Any], key: str) -> str:
    v = _require(cfg, key)
    s = str(v).strip()
    if not s:
        raise ValueError(f"Config key '{key}' must be a non-empty string")
    return s


def _req_int(cfg: Dict[str, Any], key: str) -> int:
    v = _require(cfg, key)
    try:
        return int(v)
    except Exception:
        raise ValueError(f"Config key '{key}' must be an integer; got {v!r}")


def _req_float(cfg: Dict[str, Any], key: str) -> float:
    v = _require(cfg, key)
    try:
        return float(v)
    except Exception:
        raise ValueError(f"Config key '{key}' must be a float; got {v!r}")


def _req_bool(cfg: Dict[str, Any], key: str) -> bool:
    v = _require(cfg, key)
    if isinstance(v, bool):
        return v
    # Accept common string/int booleans explicitly
    if isinstance(v, str):
        vs = v.strip().lower()
        if vs in {"true", "1", "yes"}:
            return True
        if vs in {"false", "0", "no"}:
            return False
    if isinstance(v, int):
        if v in (0, 1):
            return bool(v)
    raise ValueError(f"Config key '{key}' must be a boolean; got {v!r}")


def load_and_validate_config(path: str) -> RLRunnerConfig:
    cfg: Dict[str, Any] = _read_config_file(path)
    # Back-compat alias
    if not cfg.get("processor") and cfg.get("processor_path"):
        cfg["processor"] = cfg.get("processor_path")

    # Aggregate missing required keys (fail-fast with a single error)
    required_keys: List[str] = [
        # Paths
        "checkpoint", "processor", "output_dir",
        # Runtime
        "device", "seed",
        # Generation
        "temperature", "top_p", "max_new_tokens_stage_a", "max_new_tokens_stage_b",
        "mask_geometry_tokens", "mask_coordinate_tokens", "sanitize_stage_a",
        # Sampling/GRPO
        "K_B", "K_A", "K_set", "adv_clip", "length_norm",
        # Training
        "train", "epochs", "batch_size", "learning_rate", "weight_decay", "max_grad_norm", "drop_last",
        # Selective unfreeze (must be explicit)
        "llm_top_k_block", "vision_top_k_block", "freeze_patch_embed", "train_aligner",
        # KL
        "use_ref_kl", "lambda_kl_stage_b", "lambda_kl_stage_a",
        # Stage-A/Stage-B controls
        "train_stage_a_mode", "stage_a_weight", "group_reward_mode",
        "train_stage_b", "stage_b_weight", "freeze_stage_b_steps", "use_mission_checklist",
        # Rewards
        "reward_fns", "reward_weights",
        # Pairwise/uncertainty/diagnostics
        "pairwise_credit_enabled", "pairwise_pairs_per_group", "pairwise_delta_threshold",
        "use_uncertainty_gate", "uncertainty_gate_min_entropy",
        "enable_phase_a_diagnostics",
        # Stage-A TF budget
        "max_images_tf",
        # New extras
        "grad_accum_steps", "decision_ce_ema_beta",
    ]
    missing_keys: List[str] = [k for k in required_keys if k not in cfg]
    # Dataset composite requirement
    has_train_dir = bool(cfg.get("train_data_dir"))
    has_eval_dir = bool(cfg.get("eval_data_dir"))
    dataset_ok = has_train_dir or has_eval_dir
    # use_ref_kl-dependent requirement
    if cfg.get("use_ref_kl") in (True, "true", "True", 1, "1", "yes", "Yes"):
        if not cfg.get("ref_checkpoint"):
            missing_keys.append("ref_checkpoint (required when use_ref_kl=true)")
    if not dataset_ok:
        missing_keys.append("train_data_dir|eval_data_dir (provide at least one)")
    if missing_keys:
        missing_list = ", ".join(sorted(set(missing_keys)))
        raise ValueError(
            f"Missing required config keys: {missing_list}. Please set them explicitly in your YAML."
        )

    # Core required keys
    checkpoint = _req_str(cfg, "checkpoint")
    processor = _req_str(cfg, "processor")
    output_dir = _req_str(cfg, "output_dir")

    # Dataset: require at least one
    train_data_dir = cfg.get("train_data_dir")
    eval_data_dir = cfg.get("eval_data_dir")
    if not train_data_dir and not eval_data_dir:
        raise ValueError(
            "You must provide one of 'train_data_dir' or 'eval_data_dir'."
        )

    # Rewards: parse and validate (require explicit)
    rf = _require(cfg, "reward_fns")
    if isinstance(rf, str):
        reward_fns = [s.strip() for s in rf.split(",") if s and str(s).strip()]
    elif isinstance(rf, list):
        reward_fns = [str(s).strip() for s in rf if str(s).strip()]
    else:
        raise ValueError("'reward_fns' must be a comma-separated string or list of names")

    rwv = _require(cfg, "reward_weights")
    if isinstance(rwv, str):
        reward_weights = [float(s) for s in rwv.split(",") if str(s).strip()]
    elif isinstance(rwv, list):
        reward_weights = [float(s) for s in rwv]
    else:
        raise ValueError("'reward_weights' must be a comma-separated string or list of floats")

    if len(reward_fns) != len(reward_weights):
        raise ValueError(
            f"reward_weights length ({len(reward_weights)}) must match reward_fns length ({len(reward_fns)})."
        )
    unknown = [n for n in reward_fns if n not in REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown reward names: {unknown}. Valid: {sorted(list(REGISTRY.keys()))}."
        )

    # Stage-A mode (explicit)
    mode = str(_req_str(cfg, "train_stage_a_mode")).strip().lower()
    if mode not in {"off", "conditional", "joint"}:
        raise ValueError(
            f"train_stage_a_mode must be one of ['off','conditional','joint']; got '{mode}'."
        )

    # New enums (explicit)
    group_reward_mode = str(_req_str(cfg, "group_reward_mode")).strip().lower()
    if group_reward_mode not in {"margin_only", "label_match", "combined"}:
        raise ValueError(
            f"group_reward_mode must be one of ['margin_only','label_match','combined']; got '{group_reward_mode}'."
        )

    # KL
    use_ref_kl = _req_bool(cfg, "use_ref_kl")
    ref_checkpoint = cfg.get("ref_checkpoint")
    if use_ref_kl and not ref_checkpoint:
        raise ValueError("When use_ref_kl=true, you must set 'ref_checkpoint'.")

    # Mission validity when provided
    mission = str(cfg.get("mission")) if cfg.get("mission") else None
    if mission is not None:
        try:
            from src_post.prompting.conversation import MISSION_HINTS
            if mission not in MISSION_HINTS:
                valid = ", ".join(MISSION_HINTS.keys())
                raise ValueError(f"Unknown mission '{mission}'. Valid missions: [{valid}].")
        except Exception as e:
            # Re-raise with actionable hint
            raise ValueError(f"Mission validation failed: {e}")

    # Numeric validations (fail-fast)
    K_B = _req_int(cfg, "K_B")
    K_A = _req_int(cfg, "K_A")
    K_set = _req_int(cfg, "K_set")
    if K_B < 1 or K_A < 1:
        raise ValueError(f"K_B and K_A must be >=1; got K_B={K_B}, K_A={K_A}.")
    if mode == "joint" and K_set < 1:
        raise ValueError(f"When train_stage_a_mode=joint, K_set must be >=1; got {K_set}.")

    # Stage‑B scheduling and pairwise (explicit)
    freeze_stage_b_steps = _req_int(cfg, "freeze_stage_b_steps")
    if freeze_stage_b_steps < 0:
        raise ValueError("freeze_stage_b_steps must be >= 0")
    stage_b_weight = _req_float(cfg, "stage_b_weight")
    if stage_b_weight < 0.0:
        raise ValueError("stage_b_weight must be >= 0")
    pairwise_pairs_per_group = _req_int(cfg, "pairwise_pairs_per_group")
    if pairwise_pairs_per_group < 0:
        raise ValueError("pairwise_pairs_per_group must be >= 0")
    pairwise_delta_threshold = _req_float(cfg, "pairwise_delta_threshold")

    # Selective unfreeze/LR coupling (fail-fast)
    llm_k = int(cfg.get("llm_top_k_block", 0))
    vis_k = int(cfg.get("vision_top_k_block", 0))
    train_aligner_flag = bool(cfg.get("train_aligner", True))
    if train_aligner_flag and _as_float_opt(cfg.get("aligner_lr") or cfg.get("merger_lr") or cfg.get("lr_aligner")) is None:
        raise ValueError("train_aligner=true requires explicit 'aligner_lr'.")
    if llm_k > 0 and _as_float_opt(cfg.get("llm_lr") or cfg.get("lr_last_layers")) is None:
        raise ValueError("llm_top_k_block>0 requires explicit 'llm_lr'.")
    if vis_k > 0 and _as_float_opt(cfg.get("vision_lr")) is None:
        raise ValueError("vision_top_k_block>0 requires explicit 'vision_lr'.")

    # New: grad accumulation and EMA beta
    grad_accum_steps = _req_int(cfg, "grad_accum_steps")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    decision_ce_ema_beta = _req_float(cfg, "decision_ce_ema_beta")
    if not (0.0 < decision_ce_ema_beta < 1.0):
        raise ValueError("decision_ce_ema_beta must be in (0,1)")

    # Training core knobs (explicit)
    train = _req_bool(cfg, "train")
    epochs = _req_int(cfg, "epochs")
    batch_size = _req_int(cfg, "batch_size")
    if train:
        if epochs < 1:
            raise ValueError(f"epochs must be >=1 when train=true; got {epochs}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >=1 when train=true; got {batch_size}.")

    # Path validations (fail-fast)
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not Path(processor).exists():
        raise FileNotFoundError(f"processor not found: {processor}")
    data_root = train_data_dir or eval_data_dir
    if data_root and not Path(data_root).exists():
        raise FileNotFoundError(f"dataset path not found: {data_root}")

    # Results/metrics paths: derive if missing (non-core)
    output_dir = _req_str(cfg, "output_dir")
    results_jsonl = cfg.get("results_jsonl") or str(Path(output_dir) / "results.jsonl")
    metrics_jsonl = cfg.get("metrics_jsonl") or str(Path(output_dir) / "metrics.jsonl")

    # Limit groups normalization: (None|0|-1) => -1
    limit_groups_raw = cfg.get("limit_groups", -1)
    try:
        lg = int(limit_groups_raw if limit_groups_raw is not None else -1)
    except Exception:
        lg = -1
    if lg <= 0:
        lg = -1

    # Generation (explicit)
    temperature = _req_float(cfg, "temperature")
    top_p = _req_float(cfg, "top_p")
    max_new_tokens_stage_a = _req_int(cfg, "max_new_tokens_stage_a")
    max_new_tokens_stage_b = _req_int(cfg, "max_new_tokens_stage_b")
    mask_geometry_tokens = _req_bool(cfg, "mask_geometry_tokens")
    mask_coordinate_tokens = _req_bool(cfg, "mask_coordinate_tokens")
    sanitize_stage_a = _req_bool(cfg, "sanitize_stage_a")

    # Other core training knobs (explicit)
    adv_clip = _req_float(cfg, "adv_clip")
    length_norm = _req_bool(cfg, "length_norm")
    learning_rate = _req_float(cfg, "learning_rate")
    weight_decay = _req_float(cfg, "weight_decay")
    max_grad_norm = _req_float(cfg, "max_grad_norm")
    drop_last = _req_bool(cfg, "drop_last")

    return RLRunnerConfig(
        checkpoint=checkpoint,
        processor=str(processor),
        output_dir=str(output_dir),
        train_data_dir=str(train_data_dir) if train_data_dir else None,
        eval_data_dir=str(eval_data_dir) if eval_data_dir else None,
        results_jsonl=str(results_jsonl),
        metrics_jsonl=str(metrics_jsonl),
        mission=(str(cfg.get("mission")) if cfg.get("mission") else None),
        device=_req_str(cfg, "device"),
        seed=_req_int(cfg, "seed"),
        limit_groups=int(lg),
        skip_save_checkpoints=bool(cfg.get("skip_save_checkpoints", False)),
        tb_log_dir=(str(cfg.get("tb_log_dir")) if cfg.get("tb_log_dir") else None),
        run_name=(str(cfg.get("run_name")) if cfg.get("run_name") else None),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens_stage_a=max_new_tokens_stage_a,
        max_new_tokens_stage_b=max_new_tokens_stage_b,
        mask_geometry_tokens=mask_geometry_tokens,
        mask_coordinate_tokens=mask_coordinate_tokens,
        sanitize_stage_a=sanitize_stage_a,
        K_B=K_B,
        K_A=K_A,
        adv_clip=adv_clip,
        length_norm=length_norm,
        train=train,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        drop_last=drop_last,
        llm_top_k_block=_req_int(cfg, "llm_top_k_block"),
        vision_top_k_block=_req_int(cfg, "vision_top_k_block"),
        freeze_patch_embed=_req_bool(cfg, "freeze_patch_embed"),
        aligner_lr=_as_float_opt(cfg.get("aligner_lr") or cfg.get("merger_lr") or cfg.get("lr_aligner")),
        llm_lr=_as_float_opt(cfg.get("llm_lr") or cfg.get("lr_last_layers")),
        vision_lr=_as_float_opt(cfg.get("vision_lr")),
        train_lm_head=bool(cfg.get("train_lm_head", True)),
        train_last_n_layers=int(cfg.get("train_last_n_layers", 0)),
        lr_lm_head=_as_float_opt(cfg.get("lr_lm_head")),
        lr_last_layers=_as_float_opt(cfg.get("lr_last_layers")),
        use_ref_kl=use_ref_kl,
        ref_checkpoint=(str(ref_checkpoint) if ref_checkpoint else None),
        lambda_kl_stage_b=_req_float(cfg, "lambda_kl_stage_b"),
        lambda_kl_stage_a=_req_float(cfg, "lambda_kl_stage_a"),
        train_stage_a_mode=mode,
        stage_a_weight=_req_float(cfg, "stage_a_weight"),
        K_set=K_set,
        max_images_tf=_req_int(cfg, "max_images_tf"),
        reward_fns=reward_fns,
        reward_weights=reward_weights,
        enable_phase_a_diagnostics=_req_bool(cfg, "enable_phase_a_diagnostics"),
        # New knobs
        train_stage_b=_req_bool(cfg, "train_stage_b"),
        stage_b_weight=stage_b_weight,
        freeze_stage_b_steps=freeze_stage_b_steps,
        use_mission_checklist=_req_bool(cfg, "use_mission_checklist"),
        group_reward_mode=group_reward_mode,
        pairwise_credit_enabled=_req_bool(cfg, "pairwise_credit_enabled"),
        pairwise_pairs_per_group=pairwise_pairs_per_group,
        pairwise_delta_threshold=pairwise_delta_threshold,
        use_uncertainty_gate=_req_bool(cfg, "use_uncertainty_gate"),
        uncertainty_gate_min_entropy=_req_float(cfg, "uncertainty_gate_min_entropy"),
        grad_accum_steps=grad_accum_steps,
        decision_ce_ema_beta=decision_ce_ema_beta,
        train_aligner=_req_bool(cfg, "train_aligner"),
    )
