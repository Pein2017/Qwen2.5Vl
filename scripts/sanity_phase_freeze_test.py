#!/usr/bin/env python3
import argparse
import sys
from typing import List, Optional, Set

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser("Phase freeze sanity check (no checkpoint load)")
    p.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/phase_2/standard.yaml)")
    p.add_argument("--num_layers", type=int, default=24, help="Number of dummy LLM layers to simulate")
    p.add_argument("--num_vision_blocks", type=int, default=24, help="Number of dummy vision blocks to simulate")
    return p.parse_args()


class DummyBlock(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)


class DummyLanguageModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([DummyBlock() for _ in range(num_layers)])

    def forward(self, x):
        for blk in self.layers:
            x = blk.linear(x)
        return x


class DummyVision(nn.Module):
    def __init__(self, num_blocks: int):
        super().__init__()
        self.merger = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([DummyBlock() for _ in range(num_blocks)])
        # Add a parameter to mimic patch_embed
        self.register_parameter("patch_embed", nn.Parameter(torch.zeros(1)))


class DummyHFModel(nn.Module):
    """Mimic Qwen2_5_VLForConditionalGeneration parameter naming."""

    def __init__(self, num_layers: int, num_vision_blocks: int):
        super().__init__()
        model = nn.Module()
        model.language_model = DummyLanguageModel(num_layers)
        model.visual = DummyVision(num_vision_blocks)
        self.model = model
        self.lm_head = nn.Linear(8, 16, bias=False)


class DummyTokenizer:
    def __init__(self, tokens: List[str]):
        # Build a minimal vocab map
        self._vocab = {tok: i for i, tok in enumerate(tokens)}

    def get_vocab(self):
        return self._vocab


def extract_trainable_layer_indices(model: nn.Module) -> Set[int]:
    markers = [
        "model.language_model.layers.",  # Qwen2.5-VL
        "language_model.layers.",
        "model.layers.",  # legacy
        "transformer.layers.",  # legacy
    ]

    def _extract(param_name: str) -> Optional[int]:
        for m in markers:
            if m in param_name:
                try:
                    after = param_name.split(m, 1)[1]
                    idx_str = after.split(".", 1)[0]
                    return int(idx_str)
                except Exception:
                    return None
        return None

    idxs: Set[int] = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            i = _extract(n)
            if i is not None:
                idxs.add(i)
    return idxs


def main():
    args = parse_args()

    # Load config (no checkpoint or HF model)
    from src_new.config.config import load_config

    cfg = load_config(args.config)

    # Build dummy tokenizer covering trainable_token_strings if any
    trainable_tokens = []
    try:
        tts = getattr(cfg, "trainable_token_strings", None)
        if isinstance(tts, list):
            trainable_tokens.extend([str(t) for t in tts])
    except Exception:
        pass
    # Add a minimal set of tokens to avoid empty vocab
    if not trainable_tokens:
        trainable_tokens = ["<|bos|>", "<|eos|>", "<|pad|>"]
    tokenizer = DummyTokenizer(trainable_tokens)

    # Build dummy model (lightweight, correct param names)
    num_layers = int(args.num_layers)
    num_v_blocks = int(args.num_vision_blocks)
    dummy = DummyHFModel(num_layers, num_v_blocks)

    # Apply phase freezes
    try:
        from src_new.training.phase_freeze_manager import PhaseFreezeManager
    except Exception as e:
        print(f"Failed to import PhaseFreezeManager: {e}")
        sys.exit(2)

    pfm = PhaseFreezeManager()
    summary = pfm.apply_phase(
        model=dummy,
        tokenizer=tokenizer,
        phase=str(getattr(cfg, "phase_name", "off") or "off"),
        llm_top_k_block=getattr(cfg, "llm_top_k_block", None),
        vision_top_k_block=getattr(cfg, "vision_top_k_block", None),
        freeze_patch_embed=getattr(cfg, "freeze_patch_embed", None),
        trainable_token_strings=getattr(cfg, "trainable_token_strings", None),
    )

    # Compute trainable layer indices
    trainable_idxs = sorted(extract_trainable_layer_indices(dummy))

    # Report
    print("=== Phase Freeze Sanity Report ===")
    print(f"config: {args.config}")
    print(f"phase_name: {getattr(cfg, 'phase_name', None)}")
    print(f"llm_top_k_block: {getattr(cfg, 'llm_top_k_block', None)}")
    print(f"vision_top_k_block: {getattr(cfg, 'vision_top_k_block', None)}")
    print(f"freeze_patch_embed: {getattr(cfg, 'freeze_patch_embed', None)}")
    print(f"trainable_token_strings: {getattr(cfg, 'trainable_token_strings', None)}")
    print("--- Summary from manager ---")
    print(f"trainable_params (approx): {summary.num_trainable_params}")
    print(f"top_k_llm_layers: {summary.top_k_llm_layers}")
    print(f"top_k_vision_blocks: {summary.top_k_vision_blocks}")
    print(f"patch_embed_frozen: {summary.patch_embed_frozen}")
    print("--- Derived checks ---")
    print(f"trainable LLM layer indices: {trainable_idxs}")
    # Quick sanity: in phase_2 with K>0, expect last K indices to be trainable
    if str(getattr(cfg, "phase_name", "off")).lower() == "phase_2":
        k = int(getattr(cfg, "llm_top_k_block", 0) or 0)
        if k > 0:
            expected = list(range(max(0, num_layers - k), num_layers))
            ok = all(i in trainable_idxs for i in expected)
            print(f"expected last-K trainable: {expected} -> {'OK' if ok else 'MISMATCH'}")
        elif k == -1:
            expected = list(range(0, num_layers))
            ok = all(i in trainable_idxs for i in expected)
            print(f"expected all trainable: len={len(expected)} -> {'OK' if ok else 'MISMATCH'}")
        else:
            print("phase_2 but llm_top_k_block not >0 or -1; nothing to check")


if __name__ == "__main__":
    main()
