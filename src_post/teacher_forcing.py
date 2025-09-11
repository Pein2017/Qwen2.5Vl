#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.nn import functional as F

from transformers import Qwen2VLProcessor

from src_new.models.wrapper import DetectionModel


def maybe_autocast(enabled: bool):
    """Context manager for CUDA bf16 autocast when enabled."""
    if not enabled:
        return torch.autocast(enabled=False, device_type="cuda", dtype=torch.bfloat16)
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def compute_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-position log-probabilities for the given targets.

    Args:
        logits: [B, L, V] next-token logits.
        target_ids: [B, L] token ids of next-token targets.

    Returns:
        [B, L] log-probabilities for each target position.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


def tf_sum_logprob_over_response(
    model: DetectionModel,
    enc: Dict[str, torch.Tensor],
    response_ids: List[int],
    length_norm: bool,
) -> torch.Tensor:
    """Teacher-force over [prompt + response] and sum log-probs for response tokens.

    Args:
        model: wrapped policy.
        enc: encoded prompt (must contain `input_ids` and optional attention/image fields).
        response_ids: candidate response token ids.
        length_norm: if True, normalize by response length.
    """
    device = next(model.parameters()).device
    prompt_ids: torch.Tensor = enc["input_ids"].to(device)
    resp_ids = torch.tensor([response_ids], dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
    # Attention mask
    if "attention_mask" in enc and isinstance(enc["attention_mask"], torch.Tensor):
        attn = enc["attention_mask"].to(device)
        resp_mask_attn = torch.ones((attn.size(0), resp_ids.size(1)), dtype=attn.dtype, device=device)
        full_attn = torch.cat([attn, resp_mask_attn], dim=1)
    else:
        full_attn = None
    kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.to(device)) for k, v in enc.items() if k != "input_ids"}
    if full_attn is not None:
        kwargs["attention_mask"] = full_attn
    with maybe_autocast(enabled=torch.cuda.is_available()):
        outputs = model(input_ids=full_ids, **kwargs)
    logits = outputs.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    prompt_len = int(prompt_ids.size(1))
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, prompt_len - 1 :] = True
    logprobs = compute_logprobs(logits, targets)
    resp_logprobs = logprobs[mask]
    if resp_logprobs.numel() == 0:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device)
    s = resp_logprobs.sum()
    if length_norm:
        s = s / max(1, int(resp_logprobs.numel()))
    return s


def tf_sum_logprob_and_logits_over_response(
    model: DetectionModel,
    enc: Dict[str, torch.Tensor],
    response_ids: List[int],
    length_norm: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Teacher-forcing returning (sum_logp, logits, prompt_len)."""
    device = next(model.parameters()).device
    prompt_ids: torch.Tensor = enc["input_ids"].to(device)
    resp_ids = torch.tensor([response_ids], dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
    if "attention_mask" in enc and isinstance(enc["attention_mask"], torch.Tensor):
        attn = enc["attention_mask"].to(device)
        resp_mask_attn = torch.ones((attn.size(0), resp_ids.size(1)), dtype=attn.dtype, device=device)
        full_attn = torch.cat([attn, resp_mask_attn], dim=1)
    else:
        full_attn = None
    kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.to(device)) for k, v in enc.items() if k != "input_ids"}
    if full_attn is not None:
        kwargs["attention_mask"] = full_attn
    with maybe_autocast(enabled=torch.cuda.is_available()):
        outputs = model(input_ids=full_ids, **kwargs)
    logits = outputs.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    prompt_len = int(prompt_ids.size(1))
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, prompt_len - 1 :] = True
    logprobs = compute_logprobs(logits, targets)
    resp_logprobs = logprobs[mask]
    if resp_logprobs.numel() == 0:
        return torch.tensor(0.0, dtype=logprobs.dtype, device=logprobs.device), logits, prompt_len
    s = resp_logprobs.sum()
    if length_norm:
        s = s / max(1, int(resp_logprobs.numel()))
    return s, logits, prompt_len


def kl_to_ref_over_response(
    model: DetectionModel,
    ref_model: DetectionModel,
    enc: Dict[str, torch.Tensor],
    response_ids: List[int],
) -> torch.Tensor:
    """Compute KL(current || ref) over response positions using separate forwards."""
    device = next(model.parameters()).device
    prompt_ids: torch.Tensor = enc["input_ids"].to(device)
    resp_ids = torch.tensor([response_ids], dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
    if "attention_mask" in enc and isinstance(enc["attention_mask"], torch.Tensor):
        attn = enc["attention_mask"].to(device)
        resp_mask_attn = torch.ones((attn.size(0), resp_ids.size(1)), dtype=attn.dtype, device=device)
        full_attn = torch.cat([attn, resp_mask_attn], dim=1)
    else:
        full_attn = None
    kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.to(device)) for k, v in enc.items() if k != "input_ids"}
    if full_attn is not None:
        kwargs["attention_mask"] = full_attn
    with torch.no_grad():
        with maybe_autocast(enabled=torch.cuda.is_available()):
            out_ref = ref_model(input_ids=full_ids, **kwargs)
    with maybe_autocast(enabled=torch.cuda.is_available()):
        out_cur = model(input_ids=full_ids, **kwargs)
    logits_cur = out_cur.logits[:, :-1, :]
    logits_ref = out_ref.logits[:, :-1, :]
    prompt_len = int(prompt_ids.size(1))
    cur = F.log_softmax(logits_cur, dim=-1)
    ref = F.log_softmax(logits_ref, dim=-1)
    p = torch.exp(cur)
    B, Lm1, V = cur.size()
    mask = torch.zeros((B, Lm1), dtype=torch.bool, device=cur.device)
    mask[:, prompt_len - 1 :] = True
    cur = cur[mask]
    ref = ref[mask]
    p = p[mask]
    if p.numel() == 0:
        return torch.tensor(0.0, dtype=logits_cur.dtype, device=logits_cur.device)
    kl = torch.sum(p * (cur - ref), dim=-1).mean()
    return kl


def kl_to_ref_with_cur_logits(
    ref_model: DetectionModel,
    enc: Dict[str, torch.Tensor],
    response_ids: List[int],
    cur_logits: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute KL(current || ref) using cached current logits to save a forward."""
    device = cur_logits.device
    prompt_ids: torch.Tensor = enc["input_ids"].to(device)
    resp_ids = torch.tensor([response_ids], dtype=prompt_ids.dtype, device=device)
    full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
    if "attention_mask" in enc and isinstance(enc["attention_mask"], torch.Tensor):
        attn = enc["attention_mask"].to(device)
        resp_mask_attn = torch.ones((attn.size(0), resp_ids.size(1)), dtype=attn.dtype, device=device)
        full_attn = torch.cat([attn, resp_mask_attn], dim=1)
    else:
        full_attn = None
    kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.to(device)) for k, v in enc.items() if k != "input_ids"}
    if full_attn is not None:
        kwargs["attention_mask"] = full_attn
    with torch.no_grad():
        with maybe_autocast(enabled=torch.cuda.is_available()):
            out_ref = ref_model(input_ids=full_ids, **kwargs)
    logits_ref = out_ref.logits[:, :-1, :]
    cur = F.log_softmax(cur_logits.float(), dim=-1)
    ref = F.log_softmax(logits_ref.float(), dim=-1)
    p = torch.exp(cur)
    B, Lm1, V = cur.size()
    mask = torch.zeros((B, Lm1), dtype=torch.bool, device=cur.device)
    mask[:, prompt_len - 1 :] = True
    cur = cur[mask]
    ref = ref[mask]
    p = p[mask]
    if p.numel() == 0:
        return torch.tensor(0.0, dtype=cur_logits.dtype, device=cur_logits.device)
    kl = torch.sum(p * (cur - ref), dim=-1).mean()
    return kl


def tf_decision_probs(
    model: DetectionModel,
    enc: Dict[str, torch.Tensor],
    processor: Qwen2VLProcessor,
    length_norm: bool = True,
) -> Tuple[float, float]:
    """Compute probabilities for "总评: 通过" vs "总评: 不通过" via teacher-forcing.

    Returns (p_pass, p_fail).
    """
    pass_ids = processor.tokenizer.encode("总评: 通过", add_special_tokens=False)
    fail_ids = processor.tokenizer.encode("总评: 不通过", add_special_tokens=False)
    if not isinstance(pass_ids, list) or len(pass_ids) == 0:
        pass_ids = [int(processor.tokenizer.eos_token_id)]
    if not isinstance(fail_ids, list) or len(fail_ids) == 0:
        fail_ids = [int(processor.tokenizer.eos_token_id)]
    logp_pass = tf_sum_logprob_over_response(model, enc, pass_ids, length_norm)
    logp_fail = tf_sum_logprob_over_response(model, enc, fail_ids, length_norm)
    logs = torch.stack([logp_pass, logp_fail], dim=0).float()
    probs = torch.softmax(logs, dim=0)
    return float(probs[0].item()), float(probs[1].item())
