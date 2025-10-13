#!/usr/bin/env python3
"""
Unified RL loader (HF-first parity) for Qwen2.5-VL GRPO runs.

This runner now uses TRL's GRPO trainer exclusively. The manual trainer
implementation has been deprecated and removed.

For evaluation: Use the evaluation utilities in src_new/rl/eval.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
from typing import Any, Callable, Dict, List, Set

import torch

from src_new.config.rl_config_v2 import ConfigValidationError, RLConfig
from src_new.processing.special_tokens import (
    require_core_special_tokens,
    require_geometry_tokens,
)
from src_new.rl.data.collator import PromptOnlyCollator
from src_new.rl.data.dataset import RLDenseJSONLDataset

# Manual trainer removed - using TRL only
from src_new.rl.prompting.conversation import RLConversationContext
from src_new.rl.rewards.registry import REGISTRY
from src_new.rl.trl_trainer import GRPOVLMTrainer
from src_new.rl.utils import create_builder
from src_new.training.phase_freeze_manager import PhaseFreezeManager
from src_new.utils.hf_components import HFComponents, build_hf_components
from src_new.utils.rank_aware_logging import get_rank_aware_logger


_LOGGER = get_rank_aware_logger("rl.runner")


# RLLoaderConfig removed - using RLConfig v2 directly


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


# _create_model_config removed - inline in build_components with typed config


def build_components(config_path: str) -> Dict[str, Any]:
    """Build components using strict v2 config."""
    raw_cfg = _load_yaml(config_path)

    try:
        rl_config = RLConfig.from_yaml_dict(raw_cfg)
    except ConfigValidationError as e:
        _LOGGER.error("Configuration validation failed:\n%s", e)
        raise

    level = logging.INFO
    logging.getLogger().setLevel(level)
    _LOGGER.setLevel(level)

    device_map = _prefer_device_map()

    # Use typed config
    bf16 = rl_config.training.bf16
    torch_dtype_name = "bfloat16" if bf16 else "float32"

    # Build model config from typed config
    model_config = type(
        "RLShimConfig",
        (),
        {
            "attn_implementation": rl_config.model.attn_implementation,
            "torch_dtype": torch_dtype_name,
            "use_cache": rl_config.model.use_cache,
            # Loss weights for LossManager compatibility
            "teacher_loss_weight": rl_config.loss.teacher_loss_weight,
            "student_loss_weight": rl_config.loss.student_loss_weight,
            "caption_loss_weight": rl_config.loss.caption_loss_weight,
            "grounding_loss_weight": rl_config.loss.grounding_loss_weight,
            "formatting_loss_weight": rl_config.loss.formatting_loss_weight,
        },
    )()

    components: HFComponents = build_hf_components(
        model_path=rl_config.paths.model_path,
        model_config=model_config,
        attn_implementation=rl_config.model.attn_implementation,
        image_max_pixels=rl_config.model.image_max_pixels,
        bf16=bf16,
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
        "config": rl_config,  # Return typed config instead of raw_config
    }


def build_datasets(cfg_path: str) -> Dict[str, Any]:
    bundles = build_components(cfg_path)
    rl_config = bundles["config"]

    # Use typed config - no more raw_cfg
    train_path = rl_config.paths.train_data_path
    val_path = rl_config.paths.val_data_path
    data_root = rl_config.paths.data_root

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

    collator = PromptOnlyCollator(tokenizer=bundles["tokenizer"])

    return {
        "train": train_ds,
        "val": val_ds,
        "collator": collator,
        "tokenizer": bundles["tokenizer"],
        "model": bundles["model"],
        "processor": bundles["processor"],
        "config": rl_config,  # Return typed config
    }


def train(config_path: str, *, trainer_type: str = "trl") -> None:
    """Orchestrate a GRPO training run (TRL trainer only)."""

    # Using Accelerate: avoid manual process group init and device selection

    bundles = build_datasets(config_path)
    rl_config: RLConfig = bundles["config"]

    # Setup logging using typed config
    level = getattr(logging, rl_config.logging.log_level.upper())
    logging.getLogger().setLevel(level)

    # Use typed config for sampling parameters
    prompt_batch_size = rl_config.sampling.prompt_batch_size
    sample_k = rl_config.sampling.sample_k

    # Compute world size from environment (set by torch.distributed/accelerate)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _LOGGER.info(
        "Prompt batching configured | prompts=%d | sample_k=%d | world_size=%d (TRL global E%%K rule)",
        prompt_batch_size,
        sample_k,
        world_size,
    )

    # Use typed config for rewards
    weights: Dict[str, float] = rl_config.rewards.weights
    active_keys: List[str] = [
        k for k, w in weights.items() if float(w) != 0.0 and k in REGISTRY
    ]
    if len(active_keys) == 0:
        raise ValueError(
            "RL rewards must include at least one non-zero component present in the registry. Available: "
            + str(list(REGISTRY.keys()))
        )

    # Support observe_rewards: compute/log metrics even when weight=0
    observe_keys: List[str] = []
    for key in rl_config.rewards.observe_only:
        if key in REGISTRY and key not in active_keys:
            observe_keys.append(key)

    # Combine active + observe (unique, sorted for deterministic order)
    all_keys = sorted(set(active_keys) | set(observe_keys))

    reward_funcs: List[Callable[..., List[float]]] = []
    reward_weights: List[float] = []

    # Use typed config for rewards hyperparams
    rewards_config_dict = {
        "length_vs_gt": {
            "use_ratio": rl_config.rewards.config.length_vs_gt.use_ratio,
            "sigma_ratio": rl_config.rewards.config.length_vs_gt.sigma_ratio,
            "sigma_tokens": rl_config.rewards.config.length_vs_gt.sigma_tokens,
            "min_reward": rl_config.rewards.config.length_vs_gt.min_reward,
        },
    }
    # Generic per-reward param blocks from rewards_config (if present)
    # Example: rewards_config.line_giou.buffer_frac -> injected into reward_line_giou
    try:
        extra_cfg = {}
        if getattr(rl_config.rewards.config, "duplicate_penalty", None) is not None:
            extra_cfg["duplicate_penalty"] = dict(
                rl_config.rewards.config.duplicate_penalty
            )
        if getattr(rl_config.rewards.config, "pattern_penalty", None) is not None:
            extra_cfg["pattern_penalty"] = dict(
                rl_config.rewards.config.pattern_penalty
            )
        if getattr(rl_config.rewards.config, "assignment_f1", None) is not None:
            extra_cfg["assignment_f1"] = dict(rl_config.rewards.config.assignment_f1)
        # Merge into rewards_config_dict for wrapper to consume
        rewards_config_dict.update(extra_cfg)
    except Exception:
        pass

    for key in all_keys:
        base_fn = REGISTRY[key]
        sig = inspect.signature(base_fn)
        expects_meta = "meta" in sig.parameters

        # Check if function expects threshold/config params (e.g., grounding_acc)
        param_names = set(sig.parameters.keys())
        threshold_params = param_names & {"tau_iou", "tau_quad", "tau_line"}
        # Pass-through hyperparameters for length_vs_gt
        lvgt_params = param_names & {
            "lower",
            "upper",
            "gamma",
            "tail_numeric_weight",
            "alpha",
            "estimator",
        }
        # Per-reward named params, e.g., line_giou(buffer_frac)
        per_reward_params = {}
        if key in rewards_config_dict and isinstance(rewards_config_dict[key], dict):
            per_reward_params = set(rewards_config_dict[key].keys()) & param_names

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
                    # Inject per-reward named params if present
                    if key in cfg_dict and per_reward_params:
                        named_cfg = cfg_dict.get(key, {})
                        if isinstance(named_cfg, dict):
                            for p in per_reward_params:
                                if p in named_cfg:
                                    call_kwargs[p] = named_cfg[p]
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
                cfg_dict=rewards_config_dict,
            )
        )
        # Weight: original if in active_keys, else 0.0 (observe-only)
        reward_weights.append(float(weights.get(key, 0.0)))

    train_ds = bundles["train"]
    val_ds = bundles["val"]

    # Use the DetectionModel wrapper directly so that fail-fast multimodal validations remain active
    hf_model = bundles["model"]  # type: ignore[assignment]

    if len(train_ds) < prompt_batch_size:
        raise ValueError(
            "prompt_batch.prompt_batch_size exceeds available prompts in dataset. "
            f"Configured {prompt_batch_size}, dataset has {len(train_ds)} prompts."
        )

    # Apply SFT-style phase freezing using typed config
    lc = rl_config.layer_freezing
    pfm = PhaseFreezeManager()
    summary = pfm.apply_phase(
        model=hf_model,
        tokenizer=bundles["tokenizer"],
        phase="phase_3",
        llm_top_k_block=lc.llm.trainable_top_k_blocks,
        vision_top_k_block=lc.vision_tower.trainable_top_k_blocks,
        freeze_patch_embed=lc.vision_tower.freeze_patch_embed,
    )
    _LOGGER.info(
        "Applied phase freeze: phase=%s top_k_llm=%d top_k_vision=%d patch_embed_frozen=%s",
        summary.phase,
        summary.top_k_llm_layers,
        summary.top_k_vision_blocks,
        str(summary.patch_embed_frozen),
    )

    # No manual DDP wrapping when using Accelerate

    # Use typed config for output directories
    # Concatenate run_name to output_dir for better checkpoint organization
    base_output_dir = rl_config.paths.output_dir
    run_name = rl_config.experiment.run_name
    output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    _LOGGER.info("Output directory (with run_name): %s", output_dir)

    # Ensure TB logs are written under {tb_dir}/{run_name}
    tb_root = rl_config.paths.tb_dir
    tb_dir = os.path.join(tb_root, run_name)
    os.makedirs(tb_dir, exist_ok=True)

    # Only TRL trainer is supported
    try:
        from trl.trainer.grpo_config import GRPOConfig as HFGRPOConfig
    except ImportError as exc:  # pragma: no cover - required dependency
        raise RuntimeError(
            "TRL trainer is required but the `trl` package is not installed. "
            "Please install it to use the GRPO trainer."
        ) from exc

    train_ds = bundles["train"]
    val_ds = bundles["val"]
    hf_model = bundles["model"]

    dataset_size = rl_config.training.dataset_size
    if dataset_size == -1:
        dataset_size = len(train_ds)
    steps_per_epoch = max(1, dataset_size // prompt_batch_size)
    max_steps = rl_config.training.num_train_epochs * steps_per_epoch

    prefetch_factor = rl_config.training.prefetch_factor
    if prefetch_factor is not None and prefetch_factor <= 0:
        prefetch_factor = None

    evaluation_enabled = rl_config.evaluation.enabled
    evaluation_strategy = "steps" if evaluation_enabled else "no"
    eval_steps = (
        rl_config.evaluation.eval_every_steps
        if evaluation_enabled and rl_config.evaluation.eval_every_steps > 0
        else None
    )

    per_device_base = rl_config.training.per_device_train_batch_size
    base_grad_accum = rl_config.training.gradient_accumulation_steps
    steps_per_generation = rl_config.grpo.steps_per_generation

    per_device_bs = per_device_base
    # TRL rule: let E = P (world size) × B (per_device_bs) × G (grad_accum). Require E % K == 0.
    # Keep per_device_bs unchanged (memory safety); adjust gradient_accumulation_steps to satisfy the rule.
    effective_global = max(world_size, 1) * per_device_bs * base_grad_accum
    if effective_global % sample_k != 0:
        gcd_val = math.gcd(effective_global, sample_k)
        multiplier = sample_k // gcd_val
        trl_grad_accum = base_grad_accum * multiplier
        effective_global = max(world_size, 1) * per_device_bs * trl_grad_accum
        _LOGGER.info(
            "Adjusted gradient_accumulation_steps for TRL: base=%d -> adjusted=%d so that (P*B*G)=%d is divisible by sample_k=%d",
            base_grad_accum,
            trl_grad_accum,
            effective_global,
            sample_k,
        )
    else:
        trl_grad_accum = base_grad_accum

    trl_args = HFGRPOConfig(
        output_dir=output_dir,
        run_name=rl_config.experiment.run_name,
        seed=rl_config.experiment.seed,
        do_train=True,
        do_eval=evaluation_enabled,
        num_train_epochs=rl_config.training.num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=trl_grad_accum,
        learning_rate=rl_config.optimizer.learning_rates.llm,
        weight_decay=rl_config.optimizer.weight_decay,
        max_grad_norm=rl_config.optimizer.max_grad_norm,
        lr_scheduler_type=rl_config.training.lr_scheduler_type,
        warmup_ratio=rl_config.training.warmup_ratio,
        logging_strategy="steps",
        logging_steps=rl_config.logging.logging_steps,
        logging_dir=tb_dir,
        save_strategy="steps",
        save_steps=rl_config.checkpointing.save_steps,
        save_total_limit=rl_config.checkpointing.save_total_limit,
        report_to=["tensorboard"],
        bf16=rl_config.training.bf16,
        fp16=rl_config.training.fp16,
        remove_unused_columns=False,
        dataloader_num_workers=rl_config.training.dataloader_num_workers,
        dataloader_prefetch_factor=prefetch_factor,
        dataloader_pin_memory=rl_config.training.pin_memory,
        dataloader_drop_last=False,
        disable_tqdm=False,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        num_generations=rl_config.sampling.sample_k,
        steps_per_generation=steps_per_generation,
        num_iterations=1,
        max_prompt_length=None,
        max_completion_length=rl_config.generation.max_new_tokens,
        temperature=rl_config.generation.temperature,
        top_p=rl_config.generation.top_p,
        repetition_penalty=rl_config.generation.repetition_penalty,
        mask_truncated_completions=rl_config.grpo.mask_truncated_completions,
        scale_rewards=rl_config.grpo.scale_rewards,
        reward_weights=reward_weights,
        beta=0.0,
        # TRL alignment||
        gradient_checkpointing=False,
        epsilon=rl_config.grpo.epsilon_low,
        epsilon_high=rl_config.grpo.epsilon_high,
        loss_type=rl_config.grpo.loss_type,
    )
    if rl_config.optimizer.adam_beta1 is not None:
        trl_args.adam_beta1 = rl_config.optimizer.adam_beta1
    if rl_config.optimizer.adam_beta2 is not None:
        trl_args.adam_beta2 = rl_config.optimizer.adam_beta2
    if rl_config.optimizer.adam_epsilon is not None:
        trl_args.adam_epsilon = rl_config.optimizer.adam_epsilon

    trl_args.push_to_hub = False
    trl_args.load_best_model_at_end = False
    trl_args.logging_first_step = True

    eval_dataset = val_ds if evaluation_enabled else None

    def _identity_collator(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return features

    # Force per-device batch size to 1 to avoid OOM on single GPU; satisfy TRL divisibility via accumulation only
    trl_args.per_device_train_batch_size = 1
    trl_args.per_device_eval_batch_size = 1

    trl_trainer = GRPOVLMTrainer(
        model=hf_model,
        reward_funcs=reward_funcs,
        args=trl_args,
        train_dataset=train_ds,
        eval_dataset=eval_dataset,
        processing_class=bundles["tokenizer"],
        prompt_collator=_identity_collator,
        rl_config=rl_config,
    )
    _LOGGER.info("Starting TRL GRPO training (max_steps=%d)", max_steps)
    trl_trainer.train()
    trl_trainer.save_model(output_dir)
    return


def main() -> None:
    """
    Main entry point for GRPO training/evaluation using TRL trainer.
    """
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL RL Loader (HF-first parity) - TRL trainer only"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to rl dense_grpo.yaml"
    )
    parser.add_argument(
        "--mode", type=str, default="load", choices=["load", "train"], help="Run mode"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="trl",
        choices=["trl"],
        help="Trainer implementation to run (TRL only - manual trainer is deprecated)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args.config, trainer_type=args.trainer)
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
