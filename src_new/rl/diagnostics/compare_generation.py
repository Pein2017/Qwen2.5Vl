#!/usr/bin/env python3

"""
python -m src_new.rl.diagnostics.compare_generation   --model_path /data3/Qwen2.5-VL-main/outputs/7B-all_tokens/phase_3/9-23-phase_3-all_tokens-lower_grounding_weight-last_vision_4-with_text_only-resume/best-800-eval_loss0.1004   --data_root /data3/Qwen2.5-VL-main/data/ds_v2_full   --input_file /data3/Qwen2.5-VL-main/data/ds_v2_full/train.jsonl   --sample_idx 0   --max_new_tokens 2048   --temperature 0.7   --do_sample   --top_p 0.9   --repetition_penalty 1.05"""


import argparse
import json
import os
from typing import Any, Dict, Optional

import torch

from src_new.inference import InferenceEngine
from src_new.rl.generation import sample_k
from src_new.rl.prompting.conversation import (
    RLConversationContext,
    build_simple_generation_inputs,
)
from src_new.rl.utils import resolve_im_end_id


def _read_jsonl(path: str, idx: int) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Index {idx} out of range for {path}")


def _decode_new_tokens(tokenizer, sequences: torch.Tensor, prompt_len: int) -> str:
    seq = sequences.squeeze(0)
    new_tokens = seq[prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def _has_eos(seq: torch.Tensor, eos_id: Optional[int]) -> bool:
    if eos_id is None:
        return False
    seq = seq.view(-1)
    return (seq == eos_id).any().item() if seq.numel() > 0 else False


def compare(
    model_path: str,
    data_root: str,
    input_file: str,
    sample_idx: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float,
    repetition_penalty: float,
    device: str = "cuda:0",
):
    # Resolve config path from checkpoint
    config_path = os.path.join(model_path, "training_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"training_config.yaml not found under {model_path}")

    # Use InferenceEngine to load tokenizer/model/processor with auto-config behavior
    engine = InferenceEngine(
        config_path=config_path,
        model_path=model_path,
        data_root=data_root,
        config_auto_loaded=True,
    )
    tokenizer = engine.tokenizer
    model = engine.model
    model.eval()

    eos_id = resolve_im_end_id(tokenizer)

    # Build generation inputs using RL builder (parity with GRPO)
    ctx = RLConversationContext(
        data_root=data_root,
        builder=engine.conversation_processor,
    )

    sample = _read_jsonl(input_file, sample_idx)
    tensors = build_simple_generation_inputs(sample, ctx)

    input_ids = tensors["input_ids"].to(model.device)
    attention_mask = tensors["attention_mask"].to(model.device)
    pixel_values = tensors.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(model.device)
    image_grid_thw = tensors.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(model.device)

    gen_inputs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if pixel_values is not None:
        gen_inputs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        gen_inputs["image_grid_thw"] = image_grid_thw

    prompt_len = input_ids.size(1)

    # Inference-style generation (direct model.generate)
    infer_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    if eos_id is not None:
        infer_kwargs["eos_token_id"] = int(eos_id)

    with torch.no_grad():
        out_inf = model.generate(**gen_inputs, **infer_kwargs)
        if hasattr(out_inf, "sequences"):
            seq_inf = out_inf.sequences
        else:
            seq_inf = out_inf

    text_inf = _decode_new_tokens(tokenizer, seq_inf, prompt_len)
    eos_inf = _has_eos(seq_inf[0], eos_id)

    # RL-style generation (sample_k with k=1)
    batch = gen_inputs
    sequences, _gen_logps_list = sample_k(
        model=model,
        tokenizer=tokenizer,
        batch=batch,
        k=1,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        generation_config=None,
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
    )
    # sample_k returns (sequences_stacked[K,B,L], gen_logps_list)
    # We passed K=1 and B=1
    seq_rl = sequences[0]
    text_rl = _decode_new_tokens(tokenizer, seq_rl, prompt_len)
    eos_rl = _has_eos(seq_rl[0], eos_id)

    # Report
    print("=== Generation Parity Diagnostic ===")
    print(f"Model: {model_path}")
    print(f"Data root: {data_root}")
    print(f"Input file: {input_file} (idx={sample_idx})")
    print(
        f"Args: max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}, top_p={top_p}, repetition_penalty={repetition_penalty}"
    )
    print("")
    print("[Inference-style]")
    print(f"  EOS present: {eos_inf}")
    print(f"  New tokens: {seq_inf.size(1) - prompt_len}")
    print(
        f"  Preview: '{text_inf[:300].replace('\n', ' ')}{'...' if len(text_inf) > 300 else ''}'"
    )
    print("")
    print("[RL sample_k]")
    print(f"  EOS present: {eos_rl}")
    print(f"  New tokens: {seq_rl.size(1) - prompt_len}")
    print(
        f"  Preview: '{text_rl[:300].replace('\n', ' ')}{'...' if len(text_rl) > 300 else ''}'"
    )
    print("")

    if eos_inf and not eos_rl:
        print(
            "DIAGNOSIS: Inference generates EOS but RL does not. Check RL generation kwargs and dynamic length/Masking."
        )
    elif not eos_inf and eos_rl:
        print(
            "DIAGNOSIS: RL generates EOS but inference does not. Align inference kwargs."
        )
    elif not eos_inf and not eos_rl:
        print("DIAGNOSIS: Neither path generates EOS with current kwargs.")

    else:
        print("DIAGNOSIS: Both paths generate EOS.")


def main():
    p = argparse.ArgumentParser("Compare inference vs RL generation behavior")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    compare(
        model_path=args.model_path,
        data_root=args.data_root,
        input_file=args.input_file,
        sample_idx=args.sample_idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=bool(args.do_sample),
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
    )


if __name__ == "__main__":
    main()
