#!/usr/bin/env python3
import os

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


MODEL_PATH = "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024-fourier"


def main():
    # Use project caches
    os.environ["HF_HOME"] = "/data3/Qwen2.5-VL-main/model_cache"
    os.environ["HF_MODULES_CACHE"] = "/data3/Qwen2.5-VL-main/model_cache"

    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )

    cfg = model.config
    print("=== Qwen2.5-VL-3B Config Summary ===")
    print(f"hidden_size: {getattr(cfg, 'hidden_size', 'N/A')}")
    print(f"num_hidden_layers: {getattr(cfg, 'num_hidden_layers', 'N/A')}")
    print(f"num_attention_heads: {getattr(cfg, 'num_attention_heads', 'N/A')}")
    print(f"vocab_size: {getattr(cfg, 'vocab_size', 'N/A')}")
    print(f"vision_config: {getattr(cfg, 'vision_config', 'N/A')}")
    print()

    # Decoder layers overview
    core = getattr(model, "model", model)
    layers = getattr(core, "layers", None)
    if layers is not None:
        print(f"Decoder layers: {len(layers)}")
        # First and last layer param shapes
        first_shape = next(
            ((n, tuple(p.shape)) for n, p in layers[0].named_parameters()), None
        )
        last_shape = next(
            ((n, tuple(p.shape)) for n, p in layers[-1].named_parameters()), None
        )
        print("First layer param:", first_shape)
        print("Last layer param:", last_shape)
    else:
        print("Could not resolve model.layers")

    # Visual stack + merger presence
    visual = getattr(model, "visual", None)
    print("\nHas visual module:", visual is not None)
    if visual is not None:
        merger = getattr(visual, "merger", None)
        print("Has visual.merger:", merger is not None)

    # Embeddings / lm_head shapes
    emb = model.get_input_embeddings()
    print("\nEmbeddings weight:", tuple(emb.weight.shape))
    print("LM head weight:", tuple(model.lm_head.weight.shape))

    # Coordinate token window (if tokenizer is extended)
    vocab = tokenizer.get_vocab()
    has_coord = any(t.startswith("<|coord_") for t in vocab)
    print("\nCoordinate tokens present:", has_coord)
    if has_coord:
        ids = [vocab[t] for t in vocab if t.startswith("<|coord_")]
        print("Coordinate token ID range:", min(ids), "to", max(ids))


if __name__ == "__main__":
    main()
