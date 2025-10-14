#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loader for src_post.

Currently mirrors the legacy implementation while exposing a modular API
so that future refactors can compose additional validation layers and
sectioned configs.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from src_post.rewards import REGISTRY

from .config import RLRunnerConfig
from .translation import translate_legacy_config


class ConfigValidationError(ValueError):
    """Raised when the configuration payload fails validation."""


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _configs_dir(root: Path) -> Path:
    return root / "configs"


def _resolve_config_path(
    name: str,
    configs_dir: Path,
    *,
    base_dir: Optional[Path] = None,
) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_paths: List[Path] = []

    def _append_with_suffix(base: Path, rel: Path, suffix: str) -> None:
        search_paths.append(base / Path(f"{rel}{suffix}"))

    if base_dir is not None:
        potential = base_dir / candidate
        if potential.exists():
            return potential
        if not candidate.suffix:
            _append_with_suffix(base_dir, candidate, ".yaml")
            _append_with_suffix(base_dir, candidate, ".yml")

    if candidate.suffix:
        search_paths.append(configs_dir / candidate)
    else:
        _append_with_suffix(configs_dir, candidate, ".yaml")
        _append_with_suffix(configs_dir, candidate, ".yml")
        search_paths.append(configs_dir / candidate)

    for path in search_paths:
        if path.exists():
            return path

    raise FileNotFoundError(f"Unable to resolve configuration path '{name}'")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load layered configs.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigValidationError(
            f"Configuration file must contain a mapping: {path}"
        )
    data = copy.deepcopy(data)
    # NOTE: do not pop 'extends' here; it is consumed by _gather_layers to support inheritance
    return data


def _gather_layers(
    path: Path,
    configs_dir: Path,
    *,
    visited: Optional[Dict[Path, Dict[str, Any]]] = None,
    stack: Optional[List[Path]] = None,
) -> List[Tuple[Path, Dict[str, Any]]]:
    visited = visited or {}
    stack = stack or []

    if path in stack:
        cycle = " -> ".join(str(p) for p in stack + [path])
        raise ConfigValidationError(f"Detected cyclic extends chain: {cycle}")

    if path in visited:
        return []

    raw = _load_yaml(path)

    extends = raw.get("extends", [])
    if extends and not isinstance(extends, list):
        raise ConfigValidationError(f"'extends' must be a list when provided ({path})")

    stack.append(path)
    layers: List[Tuple[Path, Dict[str, Any]]] = []
    base_dir = path.parent

    for ref in extends or []:
        if not isinstance(ref, str):
            raise ConfigValidationError(
                f"'extends' entries must be strings (file paths). Invalid entry in {path}: {ref!r}"
            )
        resolved = _resolve_config_path(ref, configs_dir, base_dir=base_dir)
        layers.extend(
            _gather_layers(resolved, configs_dir, visited=visited, stack=stack)
        )

    stack.pop()

    data = copy.deepcopy(raw)
    data.pop("extends", None)
    visited[path] = data
    layers.append((path, data))
    return layers


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_layered_config(path: str) -> Dict[str, Any]:
    root = _project_root()
    configs_dir = _configs_dir(root)
    target_path = Path(path)
    if not target_path.exists():
        target_path = _resolve_config_path(path, configs_dir)
    layers = _gather_layers(target_path, configs_dir)
    merged: Dict[str, Any] = {}
    for _, layer in layers:
        merged = _deep_merge(merged, layer)
    return merged


def _as_float_opt(x: Any) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise ConfigValidationError(f"Missing required config key: '{key}'")
    return cfg[key]


def _req_str(cfg: Dict[str, Any], key: str) -> str:
    v = _require(cfg, key)
    s = str(v).strip()
    if not s:
        raise ConfigValidationError(f"Config key '{key}' must be a non-empty string")
    return s


def _req_int(cfg: Dict[str, Any], key: str) -> int:
    v = _require(cfg, key)
    try:
        return int(v)
    except Exception:
        raise ConfigValidationError(f"Config key '{key}' must be an integer; got {v!r}")


def _req_float(cfg: Dict[str, Any], key: str) -> float:
    v = _require(cfg, key)
    try:
        return float(v)
    except Exception:
        raise ConfigValidationError(f"Config key '{key}' must be a float; got {v!r}")


def _req_bool(cfg: Dict[str, Any], key: str) -> bool:
    v = _require(cfg, key)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vs = v.strip().lower()
        if vs in {"true", "1", "yes"}:
            return True
        if vs in {"false", "0", "no"}:
            return False
    if isinstance(v, int):
        if v in (0, 1):
            return bool(v)
    raise ConfigValidationError(f"Config key '{key}' must be a boolean; got {v!r}")


def load_raw_config(path: str) -> Dict[str, Any]:
    """Load raw configuration data (layered extends supported)."""

    if yaml is None:
        logging.getLogger("src_post.config").warning(
            "PyYAML not available; falling back to JSON-only config parsing."
        )
        return _read_config_file(path)
    try:
        return load_layered_config(path)
    except FileNotFoundError:
        return _read_config_file(path)


def _read_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = p.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse config file {path}: {e}")
    return {}


def load_and_validate_config(path: str) -> RLRunnerConfig:
    """Load a configuration file and return a validated RLRunnerConfig."""

    cfg: Dict[str, Any] = load_raw_config(path)
    if not cfg:
        raise ConfigValidationError(f"Config file '{path}' is empty or invalid.")

    # Back-compat alias
    warnings = translate_legacy_config(cfg)
    for msg in warnings:
        logging.getLogger("src_post.config").warning(msg)

    required_keys: List[str] = [
        # Paths
        "checkpoint",
        "processor",
        "output_dir",
        # Runtime
        "device",
        "seed",
        # Generation
        "temperature",
        "top_p",
        "max_new_tokens_stage_a",
        "max_new_tokens_stage_b",
        "mask_geometry_tokens",
        "mask_coordinate_tokens",
        "sanitize_stage_a",
        # Sampling/GRPO
        "K_B",
        "K_A",
        "K_set",
        "adv_clip",
        "length_norm",
        # Training
        "train",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "drop_last",
        # Selective unfreeze
        "llm_top_k_block",
        "vision_top_k_block",
        "freeze_patch_embed",
        "train_aligner",
        # KL
        "use_ref_kl",
        "lambda_kl_stage_b",
        "lambda_kl_stage_a",
        # Stage-A/Stage-B controls
        "train_stage_a_mode",
        "stage_a_weight",
        "group_reward_mode",
        "train_stage_b",
        "stage_b_weight",
        "freeze_stage_b_steps",
        "use_mission_checklist",
        # Rewards
        "reward_fns",
        "reward_weights",
        # Pairwise/uncertainty/diagnostics
        "pairwise_credit_enabled",
        "pairwise_pairs_per_group",
        "pairwise_delta_threshold",
        "use_uncertainty_gate",
        "uncertainty_gate_min_entropy",
        "enable_phase_a_diagnostics",
        # Stage-A TF budget
        "max_images_tf",
        # Optimizer extras
        "grad_accum_steps",
        "decision_ce_ema_beta",
    ]

    missing_keys: List[str] = [k for k in required_keys if k not in cfg]
    has_train_dir = bool(cfg.get("train_data_dir"))
    has_eval_dir = bool(cfg.get("eval_data_dir"))
    if not (has_train_dir or has_eval_dir):
        missing_keys.append("train_data_dir|eval_data_dir (provide at least one)")

    if cfg.get("use_ref_kl") in (True, "true", "True", 1, "1", "yes", "Yes"):
        if not cfg.get("ref_checkpoint"):
            missing_keys.append("ref_checkpoint (required when use_ref_kl=true)")

    if missing_keys:
        missing_list = ", ".join(sorted(set(missing_keys)))
        raise ConfigValidationError(
            f"Missing required config keys: {missing_list}. Please set them explicitly in your YAML."
        )

    checkpoint = _req_str(cfg, "checkpoint")
    processor = _req_str(cfg, "processor")
    output_dir = _req_str(cfg, "output_dir")

    train_data_dir = cfg.get("train_data_dir")
    eval_data_dir = cfg.get("eval_data_dir")
    if not train_data_dir and not eval_data_dir:
        raise ConfigValidationError(
            "You must provide one of 'train_data_dir' or 'eval_data_dir'."
        )

    rf = _require(cfg, "reward_fns")
    if isinstance(rf, str):
        reward_fns = [s.strip() for s in rf.split(",") if s and str(s).strip()]
    elif isinstance(rf, list):
        reward_fns = [str(s).strip() for s in rf if str(s).strip()]
    else:
        raise ConfigValidationError(
            "'reward_fns' must be a comma-separated string or list of names"
        )

    rwv = _require(cfg, "reward_weights")
    if isinstance(rwv, str):
        reward_weights = [float(s) for s in rwv.split(",") if str(s).strip()]
    elif isinstance(rwv, list):
        reward_weights = [float(s) for s in rwv]
    else:
        raise ConfigValidationError(
            "'reward_weights' must be a comma-separated string or list of floats"
        )

    if len(reward_fns) != len(reward_weights):
        raise ConfigValidationError(
            f"reward_weights length ({len(reward_weights)}) must match reward_fns length ({len(reward_fns)})."
        )

    unknown = [n for n in reward_fns if n not in REGISTRY]
    if unknown:
        raise ConfigValidationError(
            f"Unknown reward names: {unknown}. Valid: {sorted(list(REGISTRY.keys()))}."
        )

    mode = str(_req_str(cfg, "train_stage_a_mode")).strip().lower()
    if mode not in {"off", "conditional", "joint"}:
        raise ConfigValidationError(
            f"train_stage_a_mode must be one of ['off','conditional','joint']; got '{mode}'."
        )

    group_reward_mode = str(_req_str(cfg, "group_reward_mode")).strip().lower()
    if group_reward_mode not in {"margin_only", "combined"}:
        raise ConfigValidationError(
            f"group_reward_mode must be one of ['margin_only','combined']; got '{group_reward_mode}'."
        )

    use_ref_kl = _req_bool(cfg, "use_ref_kl")
    ref_checkpoint = cfg.get("ref_checkpoint")
    if use_ref_kl and not ref_checkpoint:
        raise ConfigValidationError(
            "When use_ref_kl=true, you must set 'ref_checkpoint'."
        )

    mission = str(cfg.get("mission")) if cfg.get("mission") else None
    if mission is not None:
        try:
            from src_post.prompting.schema import get_mission_checks

            checks = get_mission_checks(mission)
            if not checks:
                raise ValueError(
                    f"Unknown or unsupported mission '{mission}' per group_annotation/table.json"
                )
        except Exception as e:
            raise ConfigValidationError(f"Mission validation failed: {e}")

    K_B = _req_int(cfg, "K_B")
    K_A = _req_int(cfg, "K_A")
    K_set = _req_int(cfg, "K_set")
    if K_B < 1 or K_A < 1:
        raise ConfigValidationError(
            f"K_B and K_A must be >=1; got K_B={K_B}, K_A={K_A}."
        )
    if mode == "joint" and K_set < 1:
        raise ConfigValidationError(
            f"When train_stage_a_mode=joint, K_set must be >=1; got {K_set}."
        )

    freeze_stage_b_steps = _req_int(cfg, "freeze_stage_b_steps")
    if freeze_stage_b_steps < 0:
        raise ConfigValidationError("freeze_stage_b_steps must be >= 0")
    stage_b_weight = _req_float(cfg, "stage_b_weight")
    if stage_b_weight < 0.0:
        raise ConfigValidationError("stage_b_weight must be >= 0")
    pairwise_pairs_per_group = _req_int(cfg, "pairwise_pairs_per_group")
    if pairwise_pairs_per_group < 0:
        raise ConfigValidationError("pairwise_pairs_per_group must be >= 0")
    pairwise_delta_threshold = _req_float(cfg, "pairwise_delta_threshold")

    llm_k = int(cfg.get("llm_top_k_block", 0))
    vis_k = int(cfg.get("vision_top_k_block", 0))
    train_aligner_flag = bool(cfg.get("train_aligner", True))
    if (
        train_aligner_flag
        and _as_float_opt(
            cfg.get("aligner_lr") or cfg.get("merger_lr") or cfg.get("lr_aligner")
        )
        is None
    ):
        raise ConfigValidationError(
            "train_aligner=true requires explicit 'aligner_lr'."
        )
    if (
        llm_k > 0
        and _as_float_opt(cfg.get("llm_lr") or cfg.get("lr_last_layers")) is None
    ):
        raise ConfigValidationError("llm_top_k_block>0 requires explicit 'llm_lr'.")
    if vis_k > 0 and _as_float_opt(cfg.get("vision_lr")) is None:
        raise ConfigValidationError(
            "vision_top_k_block>0 requires explicit 'vision_lr'."
        )

    grad_accum_steps = _req_int(cfg, "grad_accum_steps")
    if grad_accum_steps < 1:
        raise ConfigValidationError("grad_accum_steps must be >= 1")
    decision_ce_ema_beta = _req_float(cfg, "decision_ce_ema_beta")
    if not (0.0 < decision_ce_ema_beta < 1.0):
        raise ConfigValidationError("decision_ce_ema_beta must be in (0,1)")

    train_flag = _req_bool(cfg, "train")
    epochs = _req_int(cfg, "epochs")
    batch_size = _req_int(cfg, "batch_size")
    if train_flag:
        if epochs < 1:
            raise ConfigValidationError(
                f"epochs must be >=1 when train=true; got {epochs}."
            )
        if batch_size < 1:
            raise ConfigValidationError(
                f"batch_size must be >=1 when train=true; got {batch_size}."
            )

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not Path(processor).exists():
        raise FileNotFoundError(f"processor not found: {processor}")
    data_root = train_data_dir or eval_data_dir
    if data_root and not Path(data_root).exists():
        raise FileNotFoundError(f"dataset path not found: {data_root}")

    output_dir = _req_str(cfg, "output_dir")
    results_jsonl = cfg.get("results_jsonl") or str(Path(output_dir) / "results.jsonl")
    metrics_jsonl = cfg.get("metrics_jsonl") or str(Path(output_dir) / "metrics.jsonl")
    balance_pass_fail = bool(cfg.get("balance_pass_fail", False))
    try:
        log_step = int(cfg.get("log_step", 1) or 1)
    except Exception:
        log_step = 1
    if log_step < 1:
        log_step = 1

    stage_a_bias_fail_enabled = bool(cfg.get("stage_a_bias_fail_enabled", False))
    stage_a_bias_fail_value = float(cfg.get("stage_a_bias_fail_value", 0.0) or 0.0)
    stage_a_bias_pass_enabled = bool(cfg.get("stage_a_bias_pass_enabled", False))
    stage_a_bias_pass_value = float(cfg.get("stage_a_bias_pass_value", 0.0) or 0.0)
    save_step = int(cfg.get("save_step", 0) or 0)
    save_limit = int(cfg.get("save_limit", 3) or 0)
    try:
        log_step = int(cfg.get("log_step", 10))
    except Exception:
        log_step = 10

    limit_groups_raw = cfg.get("limit_groups", -1)
    try:
        lg = int(limit_groups_raw if limit_groups_raw is not None else -1)
    except Exception:
        lg = -1
    if lg <= 0:
        lg = -1

    temperature = _req_float(cfg, "temperature")
    top_p = _req_float(cfg, "top_p")
    max_new_tokens_stage_a = _req_int(cfg, "max_new_tokens_stage_a")
    max_new_tokens_stage_b = _req_int(cfg, "max_new_tokens_stage_b")
    temperature_stage_a = _as_float_opt(cfg.get("temperature_stage_a"))
    temperature_stage_b = _as_float_opt(cfg.get("temperature_stage_b"))
    top_p_stage_a = _as_float_opt(cfg.get("top_p_stage_a"))
    top_p_stage_b = _as_float_opt(cfg.get("top_p_stage_b"))
    repetition_penalty_stage_a = _as_float_opt(cfg.get("repetition_penalty_stage_a"))
    repetition_penalty_stage_b = _as_float_opt(cfg.get("repetition_penalty_stage_b"))
    min_new_tokens_stage_a = cfg.get("min_new_tokens_stage_a")
    min_new_tokens_stage_b = cfg.get("min_new_tokens_stage_b")
    no_repeat_ngram_size_stage_a = cfg.get("no_repeat_ngram_size_stage_a")
    no_repeat_ngram_size_stage_b = cfg.get("no_repeat_ngram_size_stage_b")
    mask_geometry_tokens = _req_bool(cfg, "mask_geometry_tokens")
    mask_coordinate_tokens = _req_bool(cfg, "mask_coordinate_tokens")
    sanitize_stage_a = _req_bool(cfg, "sanitize_stage_a")
    adv_clip = _req_float(cfg, "adv_clip")
    length_norm = _req_bool(cfg, "length_norm")
    learning_rate = _req_float(cfg, "learning_rate")
    weight_decay = _req_float(cfg, "weight_decay")
    max_grad_norm = _req_float(cfg, "max_grad_norm")
    drop_last = _req_bool(cfg, "drop_last")
    aligner_lr = _as_float_opt(
        cfg.get("aligner_lr") or cfg.get("merger_lr") or cfg.get("lr_aligner")
    )
    llm_lr = _as_float_opt(cfg.get("llm_lr") or cfg.get("lr_last_layers"))
    vision_lr = _as_float_opt(cfg.get("vision_lr"))
    train_lm_head = bool(cfg.get("train_lm_head", True))
    train_last_n_layers = int(cfg.get("train_last_n_layers", 0))
    lr_lm_head = _as_float_opt(cfg.get("lr_lm_head"))
    lr_last_layers = _as_float_opt(cfg.get("lr_last_layers"))

    enable_clipped_grpo = bool(cfg.get("enable_clipped_grpo", False))
    epsilon_low = _as_float_opt(cfg.get("epsilon_low"))
    epsilon_high = _as_float_opt(cfg.get("epsilon_high"))
    loss_type_stage_b = cfg.get("loss_type_stage_b")
    enable_entropy_mask_stage_b = bool(cfg.get("enable_entropy_mask_stage_b", False))
    entropy_top_quantile_stage_b = _as_float_opt(
        cfg.get("entropy_top_quantile_stage_b")
    )
    entropy_min_threshold_stage_b = _as_float_opt(
        cfg.get("entropy_min_threshold_stage_b")
    )
    max_resample_times = int(cfg.get("max_resample_times", 0) or 0)
    stage_a_top_m = int(cfg.get("stage_a_top_m", 0) or 0)
    uncertainty_decay_factor = float(cfg.get("uncertainty_decay_factor", 1.0) or 1.0)
    pairwise_select = str(cfg.get("pairwise_select", "heuristic") or "heuristic")
    log_all_candidates = bool(cfg.get("log_all_candidates", False))
    soft_overlong_penalty_enabled = bool(
        cfg.get("soft_overlong_penalty_enabled", False)
    )
    soft_overlong_penalty_weight = _as_float_opt(
        cfg.get("soft_overlong_penalty_weight")
    )
    stage_a_bias_fail_value = stage_a_bias_fail_value
    stage_a_bias_pass_value = stage_a_bias_pass_value

    save_step = save_step
    save_limit = save_limit

    return RLRunnerConfig(
        checkpoint=checkpoint,
        processor=str(processor),
        output_dir=str(output_dir),
        train_data_dir=str(train_data_dir) if train_data_dir else None,
        eval_data_dir=str(eval_data_dir) if eval_data_dir else None,
        balance_pass_fail=balance_pass_fail,
        results_jsonl=str(results_jsonl),
        metrics_jsonl=str(metrics_jsonl),
        mission=(str(cfg.get("mission")) if cfg.get("mission") else None),
        device=_req_str(cfg, "device"),
        seed=_req_int(cfg, "seed"),
        limit_groups=int(lg),
        skip_save_checkpoints=bool(cfg.get("skip_save_checkpoints", False)),
        tb_log_dir=(str(cfg.get("tb_log_dir")) if cfg.get("tb_log_dir") else None),
        run_name=(str(cfg.get("run_name")) if cfg.get("run_name") else None),
        log_step=int(log_step),
        temperature=temperature,
        top_p=top_p,
        max_new_tokens_stage_a=max_new_tokens_stage_a,
        max_new_tokens_stage_b=max_new_tokens_stage_b,
        temperature_stage_a=temperature_stage_a,
        temperature_stage_b=temperature_stage_b,
        top_p_stage_a=top_p_stage_a,
        top_p_stage_b=top_p_stage_b,
        repetition_penalty_stage_a=repetition_penalty_stage_a,
        repetition_penalty_stage_b=repetition_penalty_stage_b,
        min_new_tokens_stage_a=min_new_tokens_stage_a,
        min_new_tokens_stage_b=min_new_tokens_stage_b,
        no_repeat_ngram_size_stage_a=no_repeat_ngram_size_stage_a,
        no_repeat_ngram_size_stage_b=no_repeat_ngram_size_stage_b,
        mask_geometry_tokens=mask_geometry_tokens,
        mask_coordinate_tokens=mask_coordinate_tokens,
        sanitize_stage_a=sanitize_stage_a,
        K_B=K_B,
        K_A=K_A,
        adv_clip=adv_clip,
        length_norm=length_norm,
        train=train_flag,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        drop_last=drop_last,
        llm_top_k_block=_req_int(cfg, "llm_top_k_block"),
        vision_top_k_block=_req_int(cfg, "vision_top_k_block"),
        freeze_patch_embed=_req_bool(cfg, "freeze_patch_embed"),
        aligner_lr=aligner_lr,
        llm_lr=llm_lr,
        vision_lr=vision_lr,
        train_lm_head=train_lm_head,
        train_last_n_layers=train_last_n_layers,
        lr_lm_head=lr_lm_head,
        lr_last_layers=lr_last_layers,
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
        enable_clipped_grpo=enable_clipped_grpo,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        loss_type_stage_b=loss_type_stage_b,
        enable_entropy_mask_stage_b=enable_entropy_mask_stage_b,
        entropy_top_quantile_stage_b=entropy_top_quantile_stage_b,
        entropy_min_threshold_stage_b=entropy_min_threshold_stage_b,
        max_resample_times=max_resample_times,
        stage_a_top_m=stage_a_top_m,
        uncertainty_decay_factor=uncertainty_decay_factor,
        pairwise_select=pairwise_select,
        log_all_candidates=log_all_candidates,
        soft_overlong_penalty_enabled=soft_overlong_penalty_enabled,
        soft_overlong_penalty_weight=soft_overlong_penalty_weight,
        stage_a_bias_fail_enabled=stage_a_bias_fail_enabled,
        stage_a_bias_fail_value=stage_a_bias_fail_value,
        stage_a_bias_pass_enabled=stage_a_bias_pass_enabled,
        stage_a_bias_pass_value=stage_a_bias_pass_value,
        save_step=save_step,
        save_limit=save_limit,
        # Prompt enhancement toggles (optional; default False/None)
        enable_stage_b_diversity_prompt=bool(
            cfg.get("enable_stage_b_diversity_prompt", False)
        ),
        enable_stage_b_uncertainty_prompt=bool(
            cfg.get("enable_stage_b_uncertainty_prompt", False)
        ),
        enable_stage_a_negative_cues_prompt=bool(
            cfg.get("enable_stage_a_negative_cues_prompt", False)
        ),
        stage_a_negative_examples=(
            [str(x) for x in cfg.get("stage_a_negative_examples", [])]
            if isinstance(cfg.get("stage_a_negative_examples"), list)
            else None
        ),
    )


def load_config(path: str) -> RLRunnerConfig:
    """Alias for compatibility with planned API naming."""

    return load_and_validate_config(path)
