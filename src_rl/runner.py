#!/usr/bin/env python3
"""
RL loader (HF-first parity) for Qwen2.5-VL GRPO runs.

- Loads tokenizer/processor/model exactly like src_new training/inference
- Enforces chat_template inheritance, left padding, eos=<|im_end|>, pad=eos
- Forces eager attention and prefers bf16 when available
- Prints device/dtype/vocab parity on success

This file intentionally contains only loader/bootstrap logic.
Training loop (TRL/GRPO) will live in trainer.py and will import this loader.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen2_5_VLProcessor,
    AutoVideoProcessor,
)

from src_new.models.patches import apply_comprehensive_qwen25_fixes
from src_new.models.wrapper import DetectionModel
from src_rl.rewards.registry import combine, REGISTRY
from src_rl.data.dataset import RLDenseJSONLDataset
from src_rl.prompting.conversation import RLConversationContext
from src_new.processing.conversation.builder import ConversationBuilder
from trl.trainer.grpo_config import GRPOConfig

from src_rl.trainer import VisionGRPOTrainer


_LOGGER = logging.getLogger("src_rl.loader")


@dataclass(frozen=True)
class RLLoaderConfig:
    model_path: str
    image_max_pixels: Optional[int]
    attn_implementation: str
    bf16: bool

    @staticmethod
    def from_yaml_dict(cfg: Dict[str, Any]) -> "RLLoaderConfig":
        # Required
        if not isinstance(cfg, dict):
            raise ValueError("Config must be a mapping (YAML dict)")
        model_path = cfg.get("model_path")
        if not model_path or not isinstance(model_path, str):
            raise ValueError("Missing required 'model_path' (absolute path to SFT checkpoint)")
        # Optional, explicit
        image_section = cfg.get("image") or {}
        image_max_pixels = image_section.get("max_pixels") if isinstance(image_section, dict) else None
        attn = cfg.get("attn_implementation")
        if not isinstance(attn, str) or attn.strip().lower() not in {"eager", "flash_attention_2", "sdpa"}:
            raise ValueError("attn_implementation must be one of {'eager','flash_attention_2','sdpa'}")
        bf16 = bool(cfg.get("bf16", False))
        return RLLoaderConfig(
            model_path=model_path,
            image_max_pixels=int(image_max_pixels) if image_max_pixels is not None else None,
            attn_implementation=attn.strip().lower(),
            bf16=bf16,
        )


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml  # local import to avoid hard dependency where unused

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping")
    return data


def _prefer_device_map() -> Dict[str, str]:
    if torch.cuda.is_available():
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
            _warn("bf16 requested but CUDA device lacks bfloat16 support; falling back to float32")
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
    _warn("bf16 requested but CUDA capability %d is insufficient; falling back to float32", major)
    return False


def build_components(config_path: str) -> Dict[str, Any]:
    """
    Build tokenizer, processor, and model with strict HF-first parity.

    Returns a dict with keys: tokenizer, processor, model
    """
    # Load config
    raw = _load_yaml(config_path)
    cfg = RLLoaderConfig.from_yaml_dict(raw)

    log_level_name = os.getenv("SRC_RL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level_name, logging.INFO)
    logging.getLogger().setLevel(level)
    _LOGGER.setLevel(level)

    _LOGGER.info("Applying Qwen2.5-VL patches...")
    apply_comprehensive_qwen25_fixes()

    # Tokenizer
    _LOGGER.info("Loading tokenizer: %s", cfg.model_path)
    tok = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("Fast tokenizer required (use_fast=True)")
    test = tok("sanity", add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    if test.get("offset_mapping") is None:
        raise RuntimeError("Fast tokenizer did not return offset_mapping; ensure tokenizer.json is valid")

    # Image processor
    _LOGGER.info("Loading image processor: %s", cfg.model_path)
    img_proc = Qwen2VLImageProcessor.from_pretrained(cfg.model_path, trust_remote_code=False, do_resize=False)
    if cfg.image_max_pixels is not None:
        if not hasattr(img_proc, "max_pixels"):
            raise ValueError("Qwen2VLImageProcessor missing 'max_pixels' attribute")
        img_proc.max_pixels = int(cfg.image_max_pixels)
        _LOGGER.info("Set image_processor.max_pixels=%d", int(cfg.image_max_pixels))

    # Video processor (best-effort)
    try:
        proc_from_ckpt = Qwen2_5_VLProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
        vid_proc = getattr(proc_from_ckpt, "video_processor", None)
        if vid_proc is None:
            try:
                vid_proc = AutoVideoProcessor.from_pretrained(cfg.model_path)
            except Exception:
                from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor  # type: ignore
                vid_proc = Qwen2VLVideoProcessor()
    except Exception:
        try:
            vid_proc = AutoVideoProcessor.from_pretrained(cfg.model_path)
        except Exception:
            from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor  # type: ignore
            vid_proc = Qwen2VLVideoProcessor()

    # Unified processor
    unified = Qwen2_5_VLProcessor(image_processor=img_proc, tokenizer=tok, video_processor=vid_proc)

    # Chat template parity
    chat_template = getattr(tok, "chat_template", None)
    if not chat_template:
        raise ValueError(
            "Tokenizer is missing chat_template; ensure checkpoint tokenizer.json contains a valid chat_template"
        )
    unified.chat_template = chat_template

    # Runtime tokenizer parity: pad=eos; left padding
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    if hasattr(tok, "padding_side"):
        tok.padding_side = "left"

    # Model
    bf16_enabled = _resolve_bf16(cfg.bf16)
    dtype = torch.bfloat16 if bf16_enabled else torch.float32
    # Device placement: when DeepSpeed is enabled, let Trainer handle placement (no device_map)
    ds_enabled = os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"
    device_map = None if ds_enabled else _prefer_device_map()

    _LOGGER.info("Loading DetectionModel.from_pretrained_fast...")
    # DetectionModel wrapper expects a thin config with required fields
    torch_dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float32"
    coordinate_tokens_enabled = bool((raw.get("coordinate_tokens_enabled", False)))
    shim_cfg = type(
        "Shim",
        (),
        {
            "attn_implementation": cfg.attn_implementation,
            "torch_dtype": torch_dtype_name,
            "coordinate_tokens_enabled": coordinate_tokens_enabled,
            "max_coord_value": int(raw.get("max_coord_value", 8192)),
            "new_geometry_tokens": [],
            "coordinate_init_mode": "ms_mean",
            "use_cache": False,
        },
    )()
    model = DetectionModel.from_pretrained_fast(
        model_path=cfg.model_path,
        config=shim_cfg,
        tokenizer=tok,
        torch_dtype=dtype,
        attn_implementation="eager",  # force eager for stability
        device_map=device_map,
        trust_remote_code=False,
        use_cache=True,
        low_cpu_mem_usage=True,
    )
    # Provide warnings_issued dict expected by TRL GRPOTrainer
    try:
        if not hasattr(model, "warnings_issued"):
            setattr(model, "warnings_issued", {})
    except Exception:
        pass

    # Align model generation config pad/eos
    try:
        if getattr(tok, "pad_token_id", None) is not None:
            model.config.pad_token_id = tok.pad_token_id
        if getattr(tok, "eos_token_id", None) is not None:
            model.config.eos_token_id = tok.eos_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = model.config.pad_token_id
            model.generation_config.eos_token_id = model.config.eos_token_id
    except Exception:
        pass

    model.eval()

    # Log parity
    vocab_size_tok = len(tok.get_vocab())
    vocab_size_model = model.get_input_embeddings().num_embeddings
    _LOGGER.info("Device: %s", next(model.parameters()).device)
    _LOGGER.info("Dtype: %s", next(model.parameters()).dtype)
    _LOGGER.info("Tokenizer vocab: %d | Model vocab: %d", vocab_size_tok, vocab_size_model)

    return {"tokenizer": tok, "processor": unified, "model": model}


def build_reward_function(weights: Dict[str, float]) -> Callable[[List[str]], List[float]]:
    """Return a reward function that maps a list of generated strings to rewards."""
    def _fn(texts: List[str]) -> List[float]:
        out: List[float] = []
        for t in texts:
            try:
                out.append(float(combine(t, weights)))
            except Exception:
                out.append(0.0)
        return out
    return _fn


def _create_builder(processor, cfg: Dict[str, Any]) -> ConversationBuilder:
    max_coord_value = cfg.get("max_coord_value")
    coordinate_tokens_enabled = bool(cfg.get("coordinate_tokens_enabled", False))
    if max_coord_value is None:
        raise ValueError("RL YAML missing required 'max_coord_value' for ConversationBuilder")
    return ConversationBuilder(
        processor=processor,
        max_coord_value=int(max_coord_value),
        coordinate_tokens_enabled=coordinate_tokens_enabled,
    )


def build_datasets(cfg_path: str) -> Dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    comps = build_components(cfg_path)
    tok, proc, model = comps["tokenizer"], comps["processor"], comps["model"]

    train_path = cfg.get("train_data_path")
    val_path = cfg.get("val_data_path")
    data_root = cfg.get("data_root")
    if not (train_path and val_path and data_root):
        raise ValueError("RL YAML must include train_data_path, val_data_path, and data_root")

    builder = _create_builder(proc, cfg)
    ctx = RLConversationContext(
        builder=builder,
        data_root=data_root,
        max_coord_value=int(cfg.get("max_coord_value", 0)) if cfg.get("max_coord_value") is not None else None,
    )

    train_ds = RLDenseJSONLDataset(train_path, ctx)
    val_ds = RLDenseJSONLDataset(val_path, ctx)
    return {"train": train_ds, "val": val_ds, "tokenizer": tok, "model": model, "processor": proc, "cfg": cfg}


def _map_yaml_to_grpo_config(cfg: Dict[str, Any]) -> GRPOConfig:
    out_dir = cfg.get("output_dir")
    if not out_dir:
        raise ValueError("RL YAML missing 'output_dir'")
    os.makedirs(out_dir, exist_ok=True)

    # DeepSpeed toggles (mimic src_new scripts):
    ds_enabled = os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower() == "true"
    ds_config = os.getenv("BBU_DEEPSPEED_CONFIG", "scripts/zero2.json") if ds_enabled else None

    requested_bf16 = bool(cfg.get("bf16", True))
    effective_bf16 = _resolve_bf16(requested_bf16, log=False)

    args = GRPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("update_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 5e-6)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        num_train_epochs=1.0,  # we drive by max_steps
        max_steps=int(cfg.get("max_steps", 2000)),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=2,
        report_to=["tensorboard"],
        seed=int(cfg.get("seed", 42)),
        bf16=effective_bf16,
        remove_unused_columns=False,
        # GRPO-specific
        num_generations=int(cfg.get("sample_k", 4)),
        max_completion_length=int(cfg.get("max_new_tokens", 256)),
        temperature=float(cfg.get("temperature", 0.9)),
        top_p=float(cfg.get("top_p", 1.0)),
        repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
        # DeepSpeed (Zero-2) integration
        deepspeed=ds_config,
    )
    return args


def train(config_path: str) -> None:
    """Orchestrate a GRPO training run using TRL's GRPOTrainer."""
    bundles = build_datasets(config_path)
    cfg = bundles["cfg"]

    # Reward function wrappers adapted to TRL expected signature (per-component for logging)
    weights: Dict[str, float] = cfg.get("rewards", {})
    active_keys: List[str] = [k for k, w in weights.items() if float(w) != 0.0 and k in REGISTRY]
    if len(active_keys) == 0:
        raise ValueError("RL YAML 'rewards' must include at least one non-zero component present in the registry")

    reward_funcs: List[Callable[..., List[float]]] = []
    reward_weights: List[float] = []
    max_coord_value = cfg.get("max_coord_value")

    for key in active_keys:
        base_fn = REGISTRY[key]
        sig = inspect.signature(base_fn)
        expects_meta = "meta" in sig.parameters
        expects_max_coord = "max_coord_value" in sig.parameters

        def _wrap(fn: Callable[..., float], *, wants_meta: bool, wants_max_coord: bool) -> Callable[..., List[float]]:
            def _inner(prompts: List[Any], completions: List[str], **kwargs) -> List[float]:
                metas = kwargs.get("meta") if wants_meta else None
                out: List[float] = []
                for idx, text in enumerate(completions):
                    call_kwargs: Dict[str, Any] = {}
                    if wants_meta:
                        if isinstance(metas, list) and idx < len(metas):
                            call_kwargs["meta"] = metas[idx]
                        else:
                            call_kwargs["meta"] = None
                    if wants_max_coord and max_coord_value is not None:
                        call_kwargs["max_coord_value"] = max_coord_value
                    try:
                        out.append(float(fn(text, **call_kwargs)))
                    except TypeError:
                        out.append(float(fn(text)))
                return out

            return _inner

        reward_funcs.append(
            _wrap(base_fn, wants_meta=expects_meta, wants_max_coord=expects_max_coord)
        )
        reward_weights.append(float(weights[key]))

    # Build args and trainer
    grpo_args = _map_yaml_to_grpo_config(cfg)
    # Attach reward weights in the same order as reward_funcs
    grpo_args.reward_weights = reward_weights  # type: ignore[attr-defined]

    train_ds = bundles["train"]
    val_ds = bundles["val"]

    hf_model = getattr(bundles["model"], "base_model", bundles["model"])  # type: ignore[assignment]
    trainer = VisionGRPOTrainer(
        model=hf_model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=bundles["tokenizer"],
    )
    trainer.data_collator = lambda items: items

    resume = cfg.get("resume_from_checkpoint")
    if isinstance(resume, str) and os.path.isdir(resume):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    # Save final model/tokenizer/processor
    bundles["model"].save_pretrained(grpo_args.output_dir)  # type: ignore[attr-defined]
    bundles["tokenizer"].save_pretrained(grpo_args.output_dir)  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL RL Loader (HF-first parity)")
    parser.add_argument("--config", type=str, required=True, help="Path to rl dense_grpo.yaml")
    parser.add_argument("--mode", type=str, default="load", choices=["load", "train"], help="Run mode")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.config)
        return

    comps = build_components(args.config)
    # Minimal success print as JSON line for callers
    out = {
        "ok": True,
        "device": str(next(comps["model"].parameters()).device),
        "dtype": str(next(comps["model"].parameters()).dtype),
        "vocab_model": int(comps["model"].get_input_embeddings().num_embeddings),
        "vocab_tokenizer": int(len(comps["tokenizer"].get_vocab())),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
