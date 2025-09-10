#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a Qwen2.5-VL checkpoint by adding only line geometry tokens (no coordinate tokens).

- Loads the base checkpoint (no coord tokens)
- Extends tokenizer vocabulary and model embeddings using src_new.processing.token_processor
- Saves to an output directory like:
  ./model_cache/Qwen/Qwen2.5-VL-3B-Instruct-line_tokens

Usage:
  source ~/.bashrc && conda activate ms
  python ./scripts/migrate_to_expanded_cache.py
"""

import os

# Configuration - Define all parameters here
BASE_MODEL_PATH = "./model_cache/Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = "./model_cache/Qwen/Qwen2.5-VL-7B-Instruct-line_tokens"
MAX_COORD_VALUE = 1024
DTYPE = "bfloat16"  # Options: "float16", "bfloat16", "float32"
FORCE_OVERWRITE = True  # Set to True to overwrite existing output directory
COORDINATE_INIT_MODE = "fourier_ramp"  # Required: "ms_mean" or "fourier_ramp"
# Prefer a known-good fast tokenizer as reference (3B) to avoid fast conversion issues in some 7B bases
REFERENCE_TOKENIZER_PATH = "./model_cache/Qwen/Qwen2.5-VL-3B-Instruct"

import json
import shutil
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
)

from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor


# Apply Qwen compatibility patches early
try:
    from src_new.models.patches import apply_comprehensive_qwen25_fixes

    apply_comprehensive_qwen25_fixes()
except Exception:
    pass

# Import the model class after patches
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from src_new.processing.token_processor import TokenConfig, TokenProcessor


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def ensure_abs(path: str, name: str) -> Path:
    return Path(path)


def validate_base_checkpoint(base_path: Path) -> None:
    if not base_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_path}")
    # Basic sanity: tokenizer.json should exist
    tok_json = base_path / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(
            f"Missing tokenizer.json under base model path: {tok_json}"
        )
    # Strict FAST tokenizer validation: ensure fast tokenizer loads and returns offsets
    tok = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(
            "Fast tokenizer is required (use_fast=True), but a non-fast tokenizer was loaded. "
            "Ensure tokenizer.json is present and valid."
        )
    enc = tok(
        "sanity", return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt"
    )
    try:
        om = enc.get("offset_mapping")
    except Exception:
        om = None
    if om is None:
        raise RuntimeError(
            "Fast tokenizer did not return offset_mapping for a sanity string. "
            "Verify that your checkpoint provides a valid tokenizer.json and fast implementation."
        )


def maybe_prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --force to overwrite."
            )
        # Clean up and recreate
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def extend_and_save(
    base_model_path: Path, output_dir: Path, max_coord_value: int, dtype: torch.dtype
) -> Tuple[str, int]:
    # Load tokenizer (strict fast path from BASE MODEL ONLY)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Fast tokenizer required but not loaded from base_model_path")

    # Enforce padding deterministically
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is None:
            raise ValueError(
                "Tokenizer missing both pad_token and eos_token; cannot set padding deterministically"
            )
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    # Configure and extend
    token_config = TokenConfig(
        coordinate_tokens_enabled=False,
        max_coord_value=int(max_coord_value),
        coordinate_init_mode=COORDINATE_INIT_MODE,
    )
    processor = TokenProcessor(token_config)

    vocab_before = len(tokenizer.get_vocab())
    tokenizer = processor.extend_tokenizer_vocabulary(tokenizer)
    model = processor.extend_model_embeddings(model, tokenizer)
    vocab_after = len(tokenizer.get_vocab())

    # Ensure config reflects the resized embedding matrix size for larger models
    embed_rows = int(model.get_input_embeddings().weight.shape[0])
    if getattr(model.config, "vocab_size", None) != embed_rows:
        model.config.vocab_size = embed_rows
    if hasattr(model.config, "text_config") and getattr(model.config.text_config, "vocab_size", None) != embed_rows:
        model.config.text_config.vocab_size = embed_rows

    print(f"vocab_before: {vocab_before}")
    print(f"vocab_after: {vocab_after}")
    # Save model (SafeTensors) and tokenizer
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # Save image processor and combined processor with chat template
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        str(base_model_path), trust_remote_code=True
    )
    # Deterministic video processor
    video_processor = Qwen2VLVideoProcessor()
    combined_processor = Qwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=getattr(tokenizer, "chat_template", None),
    )
    combined_processor.save_pretrained(str(output_dir))

    # Copy generation_config.json if present (fallback to copy if copy2 fails on xattrs)
    gen_cfg = base_model_path / "generation_config.json"
    if gen_cfg.exists():
        shutil.copyfile(str(gen_cfg), str(output_dir / "generation_config.json"))

    # Write coordinate_config.json for auditability
    # Dynamically locate coordinate token ID range from tokenizer vocab
    vocab_map = tokenizer.get_vocab()
    coord_ids = [tid for tok, tid in vocab_map.items() if tok.startswith("<|coord_")]
    coord_start = min(coord_ids) if coord_ids else None
    coord_end_inclusive = max(coord_ids) if coord_ids else None

    coord_cfg = {
        "coordinate_tokens_enabled": False,
        "max_coord_value": int(max_coord_value),
        "coordinate_token_id_start": coord_start,
        "coordinate_token_id_end_inclusive": coord_end_inclusive,
        "vocab_size_after": vocab_after,
        "line_tokens_added": True,
    }
    with open(output_dir / "coordinate_config.json", "w", encoding="utf-8") as f:
        json.dump(coord_cfg, f, ensure_ascii=False, indent=2)

    # Strict FAST reload validation to guarantee use_fast=True works from the exported directory
    tok_fast = AutoTokenizer.from_pretrained(
        str(output_dir), trust_remote_code=True, use_fast=True
    )
    fast_vocab = tok_fast.get_vocab()
    if len(fast_vocab) != int(vocab_after):
        raise RuntimeError(
            f"Fast reload vocab mismatch: saved_after={vocab_after} vs reloaded={len(fast_vocab)}"
        )
    required_geom = [
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|line_start|>",
        "<|line_end|>",
    ]
    missing = [t for t in required_geom if t not in fast_vocab]
    if missing:
        raise RuntimeError(
            f"Geometry tokens missing after fast reload: {missing}"
        )
    # Coordinate tokens count check (>= in case base already had some)
    if token_config.coordinate_tokens_enabled:
        coord_tokens = [t for t in fast_vocab.keys() if isinstance(t, str) and t.startswith("<|coord_")]
        if len(coord_tokens) < int(max_coord_value) + 1:
            raise RuntimeError(
                f"Expected at least {int(max_coord_value)+1} coordinate tokens after fast reload, found {len(coord_tokens)}"
            )
    print("✅ Fast tokenizer reload validation passed (use_fast=True)")

    # Validate saved config reports increased vocab size relative to base
    saved_cfg = AutoConfig.from_pretrained(str(output_dir), trust_remote_code=False)
    if not hasattr(saved_cfg, "vocab_size") or int(saved_cfg.vocab_size) < int(vocab_after):
        raise RuntimeError(
            f"Expanded checkpoint appears invalid: saved vocab_size={getattr(saved_cfg, 'vocab_size', None)} vs expected={vocab_after}"
        )

    return str(output_dir), vocab_after


def main() -> None:
    base_model_path = ensure_abs(BASE_MODEL_PATH, "base_model_path")
    output_dir = ensure_abs(OUTPUT_DIR, "output_dir")
    dtype = resolve_dtype(DTYPE)

    if base_model_path == output_dir:
        raise ValueError("output_dir must differ from base_model_path")

    validate_base_checkpoint(base_model_path)
    maybe_prepare_output_dir(output_dir, FORCE_OVERWRITE)

    out_path, vocab_after = extend_and_save(
        base_model_path, output_dir, MAX_COORD_VALUE, dtype
    )

    print(
        f"✅ Expanded checkpoint exported: {out_path} (vocab_size={vocab_after})",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
