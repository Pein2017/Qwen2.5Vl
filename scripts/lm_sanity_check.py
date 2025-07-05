#!/usr/bin/env python3
"""Minimal sanity check for language head of a Qwen2.5-VL checkpoint.

Usage:
    python scripts/lm_sanity_check.py --model_path /path/to/checkpoint [--prompt "自定义问题？"]

It loads the model **without** any image inputs, feeds a short system/user
conversation, and prints the generated assistant reply so we can verify the
language model is still coherent.
"""

import sys

# Allow "python scripts/..." to be run from repo root or elsewhere
PROJECT_ROOT = "/data4/Qwen2.5-VL-main"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoProcessor

# -----------------------------------------------------------------------------
# Disable heavy logging stack before we import any project modules that rely on
# src.logger_utils.  We replace all *_logger helpers with a stub that drops
# messages, thereby avoiding the need for an initialised DirectConfig.
# -----------------------------------------------------------------------------
import src.logger_utils as _logger_utils


class _NoOpLogger:  # pylint: disable=too-few-public-methods
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass

    def critical(self, *_args, **_kwargs):
        pass


# Patch only the logger used inside src.models.patches
_logger_utils.get_patches_logger = lambda *_, **__: _NoOpLogger()  # type: ignore

# Now we can safely import and apply the patches.
from src.models.patches import (
    apply_comprehensive_qwen25_fixes,
    verify_qwen25_patches,
)

if not apply_comprehensive_qwen25_fixes():
    raise RuntimeError("Failed to apply Qwen2.5-VL patches – cannot proceed.")

verify_qwen25_patches()

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------


def _load_processor(model_dir: str):
    """Load slow tokenizer + image processor in offline mode."""

    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,  # prevent BPE conversion issues
    )

    # Ensure pad/eos are valid (same trick as training)
    eos_token = "<|endoftext|>"
    processor.tokenizer.pad_token = eos_token
    processor.tokenizer.eos_token = eos_token
    processor.tokenizer.pad_token_id = processor.tokenizer.get_vocab().get(eos_token, 0)
    processor.tokenizer.eos_token_id = processor.tokenizer.pad_token_id

    return processor


def run_sanity_check(
    model_path: str, prompt: Optional[str] = None, device: str = "auto"
) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    processor = _load_processor(str(model_dir))

    # Import AFTER patches are in place so the patched functions are used
    from transformers import Qwen2_5_VLForConditionalGeneration

    # If the user passes "single" we force placement on cuda:0 (or cpu).
    if device == "single":
        device_map = None  # load then .to
    else:
        device_map = device  # could be "auto" or explicit map

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        local_files_only=True,
    )

    # If requested single-device, move entire model after load
    if device == "single":
        target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(target)
        device_str = str(target)
    else:
        device_str = str(next(model.parameters()).device)

    print(f"✅ Model loaded on {device_str}")

    # Clear any cached rope_deltas to avoid cross-device residue from previous runs
    model.rope_deltas = None

    model.eval()

    if prompt is None:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>"
            "<|im_start|>user\n请用一句话描述 Transformers 库的作用。\n<|im_end|>"
            "<|im_start|>assistant\n"
        )

    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {
        k: v.to(next(model.parameters()).device)
        for k, v in inputs.items()
        if torch.is_tensor(v)
    }

    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        generated = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    decoded = processor.decode(
        generated[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("==== PROMPT ====")
    print(prompt.replace("<|im_start|>", "\n<|im_start|>"))
    print("\n==== MODEL OUTPUT ====")
    print(decoded)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick language-model sanity check for Qwen2.5-VL."
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to fine-tuned checkpoint directory"
    )
    parser.add_argument(
        "--prompt",
        help="Custom system/user prompt (already wrapped with special tokens)",
    )
    parser.add_argument(
        "--device", default="auto", help="device_map argument for from_pretrained"
    )
    args = parser.parse_args()

    run_sanity_check(args.model_path, args.prompt, args.device)
