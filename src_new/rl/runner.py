#!/usr/bin/env python3
"""Unified RL loader (HF-first parity) for Qwen2.5-VL GRPO runs."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import torch

from src_new.config.rl_config import EnhancedRLConfig
from src_new.processing.special_tokens import (
    require_core_special_tokens,
    require_geometry_tokens,
)
from src_new.rl.data.dataset import RLDenseJSONLDataset
from src_new.rl.grpo_trainer import BBUGRPOTrainer
from src_new.rl.prompting.conversation import RLConversationContext
from src_new.rl.rewards.registry import REGISTRY
from src_new.rl.utils import create_builder
from src_new.training.phase_freeze_manager import PhaseFreezeManager
from src_new.utils.hf_components import HFComponents, build_hf_components
from src_new.utils.rank_aware_logging import get_rank_aware_logger


_LOGGER = get_rank_aware_logger("rl.runner")


@dataclass(frozen=True)
class RLLoaderConfig:
    """Minimal loader subset extracted from YAML for component bootstrap."""

    model_path: str
    image_max_pixels: Optional[int]
    attn_implementation: str
    bf16: bool

    @staticmethod
    def from_yaml_dict(cfg: Dict[str, Any]) -> "RLLoaderConfig":
        if not isinstance(cfg, dict):
            raise ValueError("Config must be a mapping (YAML dict)")

        model_path = cfg.get("model_path")
        if not model_path or not isinstance(model_path, str):
            raise ValueError(
                "Missing required 'model_path' (absolute path to SFT checkpoint)"
            )

        # Require explicit model config section
        model_section = cfg.get("model")
        if not isinstance(model_section, dict):
            raise ValueError("Missing required 'model' section in config")

        image_max_pixels = model_section.get("image_max_pixels")
        if image_max_pixels is None:
            raise ValueError(
                "Missing required 'model.image_max_pixels' - must be explicitly set"
            )
        image_max_pixels = int(image_max_pixels)

        attn = model_section.get("attn_implementation")
        if not attn:
            raise ValueError(
                "Missing required 'model.attn_implementation' - must be explicitly set"
            )
        if not isinstance(attn, str):
            raise ValueError("model.attn_implementation must be a string")
        attn = attn.strip().lower()
        if attn not in {"eager", "flash_attention_2", "sdpa"}:
            raise ValueError(
                "model.attn_implementation must be one of {'eager','flash_attention_2','sdpa'}"
            )

        bf16 = cfg.get("bf16")
        if bf16 is None:
            raise ValueError(
                "Missing required 'bf16' - must be explicitly set to true for RL"
            )
        if not bf16:
            raise ValueError(
                "bf16 is mandatory for RL runs. Set root-level 'bf16: true' in the YAML."
            )

        # Require explicit loss weights section
        loss_section = cfg.get("loss")
        if not isinstance(loss_section, dict):
            raise ValueError("Missing required 'loss' section in config")
        required_loss_keys = [
            "caption_loss_weight",
            "grounding_loss_weight",
            "formatting_loss_weight",
        ]
        for key in required_loss_keys:
            if key not in loss_section:
                raise ValueError(
                    f"Missing required 'loss.{key}' - must be explicitly set"
                )

        # Require explicit hierarchical sections
        training = cfg.get("training") or {}
        optimizer = cfg.get("optimizer") or {}
        logging_cfg = cfg.get("logging") or {}
        checkpointing = cfg.get("checkpointing") or {}
        runtime = cfg.get("runtime") or {}
        grpo = cfg.get("grpo") or {}

        # Training keys: require only max_steps and warmup_steps (optimizer_step_batch_size removed)
        for key in ["max_steps", "warmup_steps"]:
            if key not in training:
                raise ValueError(
                    f"Missing required 'training.{key}' - must be explicitly set"
                )

        # Optimizer keys: require explicit group LRs and weight_decay
        lr_section = optimizer.get("learning_rates")
        if not isinstance(lr_section, dict):
            raise ValueError("Missing required 'optimizer.learning_rates' section")
        for key in ["llm", "vision", "merger"]:
            if key not in lr_section:
                raise ValueError(
                    f"Missing required 'optimizer.learning_rates.{key}' - must be explicitly set"
                )
        if "weight_decay" not in optimizer:
            raise ValueError(
                "Missing required 'optimizer.weight_decay' - must be explicitly set"
            )

        # Logging/checkpointing/runtime keys
        if "logging_steps" not in logging_cfg:
            raise ValueError(
                "Missing required 'logging.logging_steps' - must be explicitly set"
            )
        if "save_steps" not in checkpointing:
            raise ValueError(
                "Missing required 'checkpointing.save_steps' - must be explicitly set"
            )
        if "seed" not in runtime:
            raise ValueError("Missing required 'runtime.seed' - must be explicitly set")

        # GRPO keys
        for key in [
            "sample_k",
            "max_new_tokens",
            "temperature",
            "top_p",
            "repetition_penalty",
        ]:
            if key not in grpo:
                raise ValueError(
                    f"Missing required 'grpo.{key}' - must be explicitly set"
                )

        return RLLoaderConfig(
            model_path=model_path,
            image_max_pixels=image_max_pixels,
            attn_implementation=attn,
            bf16=bf16,
        )


def _load_yaml(path: str) -> Dict[str, Any]:
    import copy
    from pathlib import Path

    import yaml  # local import to avoid import at module scope when unused

    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _resolve_extends_path(extends_path: str, base_path: Path) -> Path:
        """Resolve extends path relative to the current config file."""
        if extends_path.startswith("/"):
            return Path(extends_path)
        return base_path.parent / extends_path

    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping")

    # Handle extends directive
    extends = data.get("extends", [])
    if extends:
        if not isinstance(extends, list):
            raise ValueError(f"extends must be a list when provided: {extends!r}")

        merged_data = {}
        for extends_path in extends:
            if not isinstance(extends_path, str):
                raise ValueError(f"extends entries must be strings: {extends_path!r}")

            resolved_path = _resolve_extends_path(extends_path, path_obj)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Extended config not found: {resolved_path}")

            base_data = _load_yaml(str(resolved_path))
            merged_data = _deep_merge(merged_data, base_data)

        # Remove extends from current data and merge with base
        current_data = copy.deepcopy(data)
        current_data.pop("extends", None)
        merged_data = _deep_merge(merged_data, current_data)
        return merged_data

    return data


def _prefer_device_map() -> Dict[str, str]:
    if torch.cuda.is_available():
        try:
            import os as _os

            local_rank_str = _os.getenv("LOCAL_RANK")
            if local_rank_str is not None and str(local_rank_str).isdigit():
                return {"": f"cuda:{int(local_rank_str)}"}
            rank_str = _os.getenv("RANK")
            if rank_str is not None and str(rank_str).isdigit():
                rank_val = int(rank_str)
                num = torch.cuda.device_count() or 1
                return {"": f"cuda:{rank_val % num}"}
        except Exception:
            pass
        return {"": "cuda:0"}
    return {"": "cpu"}


def _resolve_bf16(requested: bool, *, log: bool = True) -> bool:
    """Enable bf16 only when the runtime can actually support it."""

    def _warn(msg: str, *args: Any) -> None:
        if log:
            _LOGGER.warning(msg, *args)

    if not requested:
        return False
    if not torch.cuda.is_available():
        _warn("bf16 requested but CUDA is unavailable; falling back to float32")
        return False
    try:
        if hasattr(torch.cuda, "is_bf16_supported"):
            if torch.cuda.is_bf16_supported():
                return True
            _warn(
                "bf16 requested but CUDA device lacks bfloat16 support; falling back to float32"
            )
            return False
    except Exception as exc:  # pragma: no cover - defensive guard
        _warn("bf16 capability probe failed (%s); falling back to float32", exc)
        return False
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception as exc:  # pragma: no cover - defensive guard
        _warn("Unable to query CUDA capability (%s); falling back to float32", exc)
        return False
    if major >= 8:
        return True
    _warn(
        "bf16 requested but CUDA capability %d is insufficient; falling back to float32",
        major,
    )
    return False


def _create_model_config(
    raw: Dict[str, Any], loader_cfg: RLLoaderConfig, *, torch_dtype_name: str
) -> Any:
    # Require explicit loss configuration for LossManager compatibility
    loss_config = raw.get("loss")
    if not isinstance(loss_config, dict):
        raise ValueError("Missing required 'loss' section in config")

    required_loss_keys = [
        "teacher_loss_weight",
        "student_loss_weight",
        "caption_loss_weight",
        "grounding_loss_weight",
        "formatting_loss_weight",
    ]
    for key in required_loss_keys:
        if key not in loss_config:
            raise ValueError(f"Missing required 'loss.{key}' - must be explicitly set")

    teacher_loss_weight = float(loss_config["teacher_loss_weight"])
    student_loss_weight = float(loss_config["student_loss_weight"])
    caption_loss_weight = float(loss_config["caption_loss_weight"])
    grounding_loss_weight = float(loss_config["grounding_loss_weight"])
    formatting_loss_weight = float(loss_config["formatting_loss_weight"])

    return type(
        "RLShimConfig",
        (),
        {
            "attn_implementation": loader_cfg.attn_implementation,
            "torch_dtype": torch_dtype_name,
            "use_cache": True,
            # Loss weights for LossManager compatibility
            "teacher_loss_weight": teacher_loss_weight,
            "student_loss_weight": student_loss_weight,
            "caption_loss_weight": caption_loss_weight,
            "grounding_loss_weight": grounding_loss_weight,
            "formatting_loss_weight": formatting_loss_weight,
        },
    )()


def build_components(config_path: str) -> Dict[str, Any]:
    raw_cfg = _load_yaml(config_path)
    loader_cfg = RLLoaderConfig.from_yaml_dict(raw_cfg)

    level = logging.INFO
    logging.getLogger().setLevel(level)
    _LOGGER.setLevel(level)

    device_map = _prefer_device_map()

    torch_dtype_name = "bfloat16" if loader_cfg.bf16 else "float32"
    model_config = _create_model_config(
        raw_cfg, loader_cfg, torch_dtype_name=torch_dtype_name
    )

    components: HFComponents = build_hf_components(
        model_path=loader_cfg.model_path,
        model_config=model_config,
        attn_implementation=loader_cfg.attn_implementation,
        image_max_pixels=loader_cfg.image_max_pixels,
        bf16=loader_cfg.bf16,
        force_eager_attention=False,
        device_map=device_map,
    )

    param = next(components.model.parameters())
    _LOGGER.info(
        "Device: %s | dtype: %s | tokenizer_vocab=%d | model_vocab=%d",
        param.device,
        param.dtype,
        len(components.tokenizer.get_vocab()),
        components.model.get_input_embeddings().num_embeddings,
    )

    return {
        "tokenizer": components.tokenizer,
        "processor": components.processor,
        "model": components.model,
        "hf_bundle": components,
        "raw_config": raw_cfg,
    }


def build_datasets(cfg_path: str) -> Dict[str, Any]:
    bundles = build_components(cfg_path)
    raw_cfg = bundles["raw_config"]
    enhanced_cfg = EnhancedRLConfig.from_yaml_dict(raw_cfg)

    train_path = raw_cfg.get("train_data_path")
    val_path = raw_cfg.get("val_data_path")
    data_root = raw_cfg.get("data_root")
    if not (train_path and val_path and data_root):
        raise ValueError(
            "RL YAML must include train_data_path, val_data_path, and data_root"
        )

    # Validate tokenizer has required core/geometry tokens (fail fast)
    tok = bundles["tokenizer"]
    require_core_special_tokens(tok)
    # For RL dense, we require object_ref + box/quad/line wrappers
    require_geometry_tokens(tok, require_line=True)

    builder = create_builder(bundles["processor"])
    ctx = RLConversationContext(
        builder=builder,
        data_root=data_root,
    )

    train_ds = RLDenseJSONLDataset(train_path, ctx)
    val_ds = RLDenseJSONLDataset(val_path, ctx)

    return {
        "train": train_ds,
        "val": val_ds,
        "tokenizer": bundles["tokenizer"],
        "model": bundles["model"],
        "processor": bundles["processor"],
        "cfg": raw_cfg,
        "enhanced_cfg": enhanced_cfg,
    }


def train(config_path: str) -> None:
    """Orchestrate a GRPO training run using the manual BBU trainer."""

    # Using Accelerate: avoid manual process group init and device selection

    bundles = build_datasets(config_path)
    cfg = bundles["cfg"]
    enhanced_cfg: EnhancedRLConfig = bundles["enhanced_cfg"]
    enhanced_cfg.setup_logging()

    # Require explicit rewards configuration
    rewards_section = cfg.get("rewards")
    if not isinstance(rewards_section, dict):
        raise ValueError("Missing required 'rewards' section in config")
    if len(rewards_section) == 0:
        raise ValueError(
            "'rewards' section cannot be empty - must explicitly set reward weights"
        )

    weights: Dict[str, float] = rewards_section
    active_keys: List[str] = [
        k for k, w in weights.items() if float(w) != 0.0 and k in REGISTRY
    ]
    if len(active_keys) == 0:
        raise ValueError(
            "RL rewards must include at least one non-zero component present in the registry. Available: "
            + str(list(REGISTRY.keys()))
        )

    # Support observe_rewards: compute/log metrics even when weight=0
    observe_rewards_raw = cfg.get("observe_rewards", [])
    if not isinstance(observe_rewards_raw, list):
        observe_rewards_raw = []
    observe_keys: List[str] = []
    for key in observe_rewards_raw:
        if isinstance(key, str) and key in REGISTRY and key not in active_keys:
            observe_keys.append(key)

    # Combine active + observe (unique, sorted for deterministic order)
    all_keys = sorted(set(active_keys) | set(observe_keys))
    reward_names: List[str] = all_keys

    reward_funcs: List[Callable[..., List[float]]] = []
    reward_weights: List[float] = []

    # Optional rewards_config for thresholds/hyperparams
    rewards_config = cfg.get("rewards_config", {})
    if not isinstance(rewards_config, dict):
        rewards_config = {}

    for key in all_keys:
        base_fn = REGISTRY[key]
        sig = inspect.signature(base_fn)
        expects_meta = "meta" in sig.parameters

        # Check if function expects threshold/config params (e.g., grounding_acc)
        param_names = set(sig.parameters.keys())
        threshold_params = param_names & {"tau_iou", "tau_quad", "tau_line"}
        # Pass-through hyperparameters for length_vs_gt
        lvgt_params = param_names & {"lower", "upper", "gamma", "tail_numeric_weight", "alpha", "estimator"}

        def _wrap(
            fn: Callable[..., float],
            *,
            wants_meta: bool,
            threshold_keys: Set[str],
            lvgt_keys: Set[str],
            cfg_dict: Dict[str, Any],
        ) -> Callable[..., List[float]]:
            def _inner(
                prompts: List[Any], completions: List[str], **kwargs
            ) -> List[float]:
                metas = kwargs.get("meta") if wants_meta else None
                out: List[float] = []
                for idx, text in enumerate(completions):
                    call_kwargs: Dict[str, Any] = {}
                    if wants_meta:
                        if isinstance(metas, list) and idx < len(metas):
                            call_kwargs["meta"] = metas[idx]
                        else:
                            call_kwargs["meta"] = None
                    # Inject thresholds from rewards_config if function expects them
                    for t_key in threshold_keys:
                        if t_key in cfg_dict:
                            call_kwargs[t_key] = cfg_dict[t_key]
                    # Inject length_vs_gt params if present
                    if "length_vs_gt" in cfg_dict and lvgt_keys:
                        lvgt_cfg = cfg_dict["length_vs_gt"]
                        if isinstance(lvgt_cfg, dict):
                            for p in lvgt_keys:
                                if p in lvgt_cfg:
                                    call_kwargs[p] = lvgt_cfg[p]
                    try:
                        out.append(float(fn(text, **call_kwargs)))
                    except TypeError:
                        out.append(float(fn(text)))
                return out

            return _inner

        reward_funcs.append(
            _wrap(
                base_fn,
                wants_meta=expects_meta,
                threshold_keys=threshold_params,
                lvgt_keys=lvgt_params,
                cfg_dict=rewards_config,
            )
        )
        # Weight: original if in active_keys, else 0.0 (observe-only)
        reward_weights.append(float(weights.get(key, 0.0)))

    train_ds = bundles["train"]
    val_ds = bundles["val"]

    # Use the DetectionModel wrapper directly so that fail-fast multimodal validations remain active
    hf_model = bundles["model"]  # type: ignore[assignment]

    # Require explicit layer_config for phase freezing
    layer_config_section = cfg.get("layer_config")
    if not isinstance(layer_config_section, dict):
        raise ValueError("Missing required 'layer_config' section in config")

    # Apply SFT-style phase freezing based on explicit RL layer_config
    lc = enhanced_cfg.layer_config
    pfm = PhaseFreezeManager()
    summary = pfm.apply_phase(
        model=hf_model,
        tokenizer=bundles["tokenizer"],
        phase="phase_3",
        llm_top_k_block=int(lc.llm_trainable_top_k_blocks),
        vision_top_k_block=int(lc.vision_trainable_top_k_blocks),
        freeze_patch_embed=bool(lc.vision_freeze_patch_embed),
    )
    _LOGGER.info(
        "Applied phase freeze: phase=%s top_k_llm=%d top_k_vision=%d patch_embed_frozen=%s",
        summary.phase,
        summary.top_k_llm_layers,
        summary.top_k_vision_blocks,
        str(summary.patch_embed_frozen),
    )

    # No manual DDP wrapping when using Accelerate

    output_dir = cfg.get("output_dir") or enhanced_cfg.output_dir
    if not output_dir:
        raise ValueError("Missing required 'output_dir' for manual trainer")
    os.makedirs(output_dir, exist_ok=True)

    manual_trainer = BBUGRPOTrainer(
        model=hf_model,
        tokenizer=bundles["tokenizer"],
        processor=bundles.get("processor"),
        train_dataset=train_ds,
        val_dataset=val_ds,
        reward_functions=reward_funcs,
        reward_names=reward_names,
        reward_weights=reward_weights,
        enhanced_cfg=enhanced_cfg,
        raw_config=cfg,
        output_dir=str(output_dir),
    )
    # Attach original YAML path for checkpoint reproducibility (no CLI overrides)
    try:
        manual_trainer._checkpoint_saver.args.original_config_path = config_path
    except Exception:
        pass
    try:
        # Ensure training mode after DDP wrapping
        if hasattr(hf_model, "train"):
            hf_model.train()
        manual_trainer.train()
    finally:
        manual_trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL RL Loader (HF-first parity)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to rl dense_grpo.yaml"
    )
    parser.add_argument(
        "--mode", type=str, default="load", choices=["load", "train"], help="Run mode"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args.config)
        return

    comps = build_components(args.config)
    bundle: HFComponents = comps["hf_bundle"]
    param = next(bundle.model.parameters())
    out = {
        "ok": True,
        "device": str(param.device),
        "dtype": str(param.dtype),
        "vocab_model": int(bundle.model.get_input_embeddings().num_embeddings),
        "vocab_tokenizer": int(len(bundle.tokenizer.get_vocab())),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
