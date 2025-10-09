"""Shared HuggingFace component loader for Qwen2.5-VL pipelines.

This utility centralises tokenizer/processor/model loading so that SFT inference
and RL post-training share one authoritative codepath. It applies the required
Qwen2.5 patches once, enforces chat-template inheritance, and respects runtime
capability checks such as bf16 probing and eager-attention overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoVideoProcessor,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor,
)

from src_new.models.patches import apply_comprehensive_qwen25_fixes
from src_new.models.wrapper import DetectionModel
from src_new.utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger("utils.hf_components")


@dataclass(frozen=True)
class HFComponents:
    """Bundle of loaded HuggingFace objects used across pipelines."""

    tokenizer: Any
    image_processor: Qwen2VLImageProcessor
    video_processor: Any
    processor: Qwen2_5_VLProcessor
    model: DetectionModel
    torch_dtype: torch.dtype
    device_map: Optional[Dict[str, str]]
    bf16_enabled: bool


def _ensure_hf_env() -> None:
    """Apply environment defaults to keep parity with training scripts."""

    os.environ.setdefault("HF_MODULES_CACHE", "./model_cache")
    os.environ.setdefault("HF_HOME", "./model_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _prefer_device_map() -> Dict[str, str]:
    if torch.cuda.is_available():
        return {"": "cuda:0"}
    return {"": "cpu"}


def _load_video_processor(model_path: str) -> Any:
    """Best-effort video processor loader to keep parity with training."""

    try:
        proc_from_ckpt = Qwen2_5_VLProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        vid_proc = getattr(proc_from_ckpt, "video_processor", None)
        if vid_proc is not None:
            return vid_proc
    except Exception:
        pass
    try:
        return AutoVideoProcessor.from_pretrained(model_path)
    except Exception:
        # Fall back to transformers' default implementation
        from transformers.models.qwen2_vl.video_processing_qwen2_vl import (  # type: ignore
            Qwen2VLVideoProcessor,
        )

        return Qwen2VLVideoProcessor()


def build_hf_components(
    model_path: str,
    *,
    model_config: Any,
    attn_implementation: str,
    image_max_pixels: Optional[int] = None,
    bf16: bool = False,
    force_eager_attention: bool = False,
    device_map: Optional[Dict[str, str]] = None,
    tokenizer_padding_side: str = "left",
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> HFComponents:
    """Load tokenizer, processor, and DetectionModel with strict HF-first parity.

    Args:
        model_path: Checkpoint directory containing tokenizer, processor, and weights.
        model_config: Configuration object passed to DetectionModel (must expose
            coordinate/token settings expected by the wrapper).
        attn_implementation: Requested attention backend ("eager", "flash_attention_2", ...).
        image_max_pixels: Optional cap applied to the image processor.
        bf16: Whether to request bf16 weights (downgrades automatically when unsupported).
        force_eager_attention: Override to force eager attention regardless of cfg value.
        device_map: Optional device map passed to `from_pretrained_fast`; defaults to single-device.
        tokenizer_padding_side: Padding side to assign to the tokenizer (defaults to "left").
        tokenizer_kwargs: Extra kwargs forwarded to `AutoTokenizer.from_pretrained`.
        model_kwargs: Extra kwargs forwarded to `DetectionModel.from_pretrained_fast`.

    Returns:
        HFComponents bundle with tokenizer, processor, and loaded DetectionModel.
    """

    _ensure_hf_env()

    apply_comprehensive_qwen25_fixes()

    tok_kwargs = {"trust_remote_code": True, "use_fast": True}
    if tokenizer_kwargs:
        tok_kwargs.update(tokenizer_kwargs)

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Fast tokenizer required (use_fast=True)")
    probe = tokenizer(
        "sanity",
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    if probe.get("offset_mapping") is None:
        raise RuntimeError(
            "Fast tokenizer did not return offset_mapping; ensure tokenizer.json is valid"
        )

    logger.info("Loading image processor from %s", model_path)
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        model_path, trust_remote_code=False, do_resize=False
    )
    if image_max_pixels is not None:
        if not hasattr(image_processor, "max_pixels"):
            raise ValueError("Qwen2VLImageProcessor missing 'max_pixels' attribute")
        image_processor.max_pixels = int(image_max_pixels)
        logger.info("Set image_processor.max_pixels=%d", int(image_max_pixels))

    video_processor = _load_video_processor(model_path)

    processor = Qwen2_5_VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
    )

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise ValueError(
            "Tokenizer is missing chat_template; ensure tokenizer.json contains a valid chat_template"
        )
    processor.chat_template = chat_template

    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = tokenizer_padding_side

    torch_dtype = torch.bfloat16 if bf16 else torch.float32
    effective_attn = "eager" if force_eager_attention else attn_implementation
    if device_map is None:
        device_map = _prefer_device_map()

    model_extra_kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": effective_attn,
        "device_map": device_map,
        "trust_remote_code": False,
        "use_cache": True,
        "low_cpu_mem_usage": True,
    }
    if model_kwargs:
        model_extra_kwargs.update(model_kwargs)

    logger.info("Loading DetectionModel.from_pretrained_fast...")
    model = DetectionModel.from_pretrained_fast(
        model_path=model_path,
        config=model_config,
        tokenizer=tokenizer,
        **model_extra_kwargs,
    )

    try:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore[attr-defined]
        if getattr(tokenizer, "eos_token_id", None) is not None:
            model.config.eos_token_id = tokenizer.eos_token_id  # type: ignore[attr-defined]
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = model.config.pad_token_id  # type: ignore[attr-defined]
            model.generation_config.eos_token_id = model.config.eos_token_id  # type: ignore[attr-defined]
    except Exception as exc:
        logger.debug("Tokenizer parity adjustments skipped: %s", exc)

    model.eval()

    return HFComponents(
        tokenizer=tokenizer,
        image_processor=image_processor,
        video_processor=video_processor,
        processor=processor,
        model=model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        bf16_enabled=bf16,
    )


__all__ = ["HFComponents", "build_hf_components"]
