#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export an expanded Qwen2.5-VL checkpoint with coordinate tokens and resized embeddings.

- Loads the base checkpoint (no coord tokens)
- Extends tokenizer vocabulary and model embeddings using src_new.processing.token_processor
- Saves to an output directory like:
  /data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024

Usage (example):
  source ~/.bashrc && conda activate ms
  python /data3/Qwen2.5-VL-main/scripts/migrate_to_expanded_cache.py \
    --base_model_path /data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir /data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024 \
    --max_coord_value 1024
"""

import argparse
import json
import os
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


BASE_MODEL_DEFAULT = "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Qwen2.5-VL base checkpoint to expanded coordinate-token checkpoint"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=BASE_MODEL_DEFAULT,
        help="Absolute path to base Qwen2.5-VL checkpoint (no coord tokens)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Absolute path to save the expanded checkpoint",
    )
    parser.add_argument(
        "--max_coord_value",
        type=int,
        required=True,
        help="Maximum coordinate value; adds <|coord_0|>.. <|coord_MAX|>",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for loading during migration",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output_dir if present",
    )
    return parser.parse_args()


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def ensure_abs(path: str, name: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    return p


def validate_base_checkpoint(base_path: Path) -> None:
    if not base_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_path}")
    # Basic sanity: tokenizer.json should exist
    tok_json = base_path / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(
            f"Missing tokenizer.json under base model path: {tok_json}"
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
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    # Configure and extend
    token_config = TokenConfig(
        coordinate_tokens_enabled=True,
        max_coord_value=int(max_coord_value),
        # Optional: allow override via env COORD_INIT_MODE without changing script CLI
        coordinate_init_mode=(
            (os.environ.get("COORD_INIT_MODE") or None) if "os" in globals() else None
        ),
    )
    processor = TokenProcessor(token_config)

    vocab_before = len(tokenizer.get_vocab())
    tokenizer = processor.extend_tokenizer_vocabulary(tokenizer)
    model = processor.extend_model_embeddings(model, tokenizer)
    vocab_after = len(tokenizer.get_vocab())

    print(f"vocab_before: {vocab_before}")
    print(f"vocab_after: {vocab_after}")
    # Save model (SafeTensors) and tokenizer
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # Save image processor and combined processor with chat template
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        str(base_model_path), trust_remote_code=True
    )
    combined_processor = Qwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=getattr(tokenizer, "chat_template", None),
    )
    combined_processor.save_pretrained(str(output_dir))

    # Copy generation_config.json if present
    gen_cfg = base_model_path / "generation_config.json"
    if gen_cfg.exists():
        shutil.copy2(str(gen_cfg), str(output_dir / "generation_config.json"))

    # Write coordinate_config.json for auditability
    coord_cfg = {
        "coordinate_tokens_enabled": True,
        "max_coord_value": int(max_coord_value),
        "coordinate_token_id_start": 151667,
        "coordinate_token_id_end_inclusive": 151667 + int(max_coord_value),
        "vocab_size_after": vocab_after,
    }
    with open(output_dir / "coordinate_config.json", "w", encoding="utf-8") as f:
        json.dump(coord_cfg, f, ensure_ascii=False, indent=2)

    # Validate saved config reports increased vocab size
    saved_cfg = AutoConfig.from_pretrained(str(output_dir), trust_remote_code=False)
    if not hasattr(saved_cfg, "vocab_size") or int(saved_cfg.vocab_size) <= 151665:
        raise RuntimeError(
            f"Expanded checkpoint appears invalid: vocab_size={getattr(saved_cfg, 'vocab_size', None)}"
        )

    return str(output_dir), vocab_after


def main() -> None:
    args = parse_args()
    base_model_path = ensure_abs(args.base_model_path, "base_model_path")
    output_dir = ensure_abs(args.output_dir, "output_dir")
    dtype = resolve_dtype(args.dtype)

    if base_model_path == output_dir:
        raise ValueError("output_dir must differ from base_model_path")

    validate_base_checkpoint(base_model_path)
    maybe_prepare_output_dir(output_dir, args.force)

    out_path, vocab_after = extend_and_save(
        base_model_path, output_dir, args.max_coord_value, dtype
    )

    print(
        f"✅ Expanded checkpoint exported: {out_path} (vocab_size={vocab_after})",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
