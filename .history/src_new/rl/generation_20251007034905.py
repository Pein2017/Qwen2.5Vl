"""Generation helpers for manual GRPO training.

These utilities are pure functions so they can be re-used by trainers,
evaluation harnesses, and unit tests. They intentionally avoid depending on
TRL so the manual trainer can stay lightweight.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from src_new.rl import logprobs as rl_logprobs
from src_new.rl.utils import resolve_im_end_id


_ALLOWED_GENERATION_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
}


def prepare_generate_inputs(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize inputs for ``model.generate``.

    The conversation builder returns tensors with optional singleton batch
    dimensions. We normalize them to 2-D shape and move them to the model's
    device. Non-tensor entries are skipped silently to make the helper tolerant
    to meta fields that might be present on the batch dict.
    """

    if not isinstance(batch, dict):
        raise ValueError("batch must be a dict of tensors")
    if "input_ids" not in batch or "attention_mask" not in batch:
        raise ValueError(
            "batch missing required keys: 'input_ids' and 'attention_mask'"
        )

    from src_new.rl.utils import get_model_device  # late import to avoid cycles

    device = get_model_device(model)
    gen_inputs: Dict[str, Any] = {}
    for key, value in batch.items():
        if key not in _ALLOWED_GENERATION_KEYS:
            continue
        if torch.is_tensor(value):
            tensor = value.to(device)
            if tensor.dim() == 1 and key in {"input_ids", "attention_mask"}:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() > 2 and key in {"input_ids", "attention_mask"}:
                raise ValueError(
                    f"{key} must be 1D or 2D tensor; got shape={tuple(tensor.shape)}"
                )
            gen_inputs[key] = tensor
    # Validate required keys after normalization
    for key in ("input_ids", "attention_mask"):
        if key not in gen_inputs:
            raise ValueError(f"Missing required tensor '{key}' after normalization")
    return gen_inputs


def generate_completions(
    model: nn.Module,
    tokenizer: Any,
    batch: Dict[str, Any],
    *,
    generation_config: Optional[Any] = None,
    eos_token_id: Optional[int] = None,
    **gen_kwargs: Any,
) -> torch.Tensor:
    """Generate assistant tokens for the given batch.

    ``generation_config`` takes precedence; otherwise ``gen_kwargs`` are forwarded
    directly to ``model.generate``. ``eos_token_id`` defaults to the tokenizer's
    ``<|im_end|>`` identifier when available.
    """

    gen_inputs = prepare_generate_inputs(model, batch)
    resolved = resolve_im_end_id(tokenizer)
    eos_id = resolved if resolved is not None else eos_token_id

    # Unwrap DDP if present to access .generate() method
    gen_model = getattr(model, "module", model)

    if generation_config is not None:
        if (
            eos_id is not None
            and getattr(generation_config, "eos_token_id", None) is None
        ):
            try:
                generation_config.eos_token_id = eos_id
            except Exception:
                pass
        with torch.no_grad():
            outputs = gen_model.generate(
                **gen_inputs, generation_config=generation_config
            )
    else:
        if eos_id is not None and "eos_token_id" not in gen_kwargs:
            gen_kwargs["eos_token_id"] = eos_id
        with torch.no_grad():
            outputs = gen_model.generate(**gen_inputs, **gen_kwargs)

    if isinstance(outputs, torch.Tensor):
        return outputs

    sequences = getattr(outputs, "sequences", None)
    if isinstance(sequences, torch.Tensor):
        return sequences

    raise RuntimeError(
        "Unexpected return from model.generate; expected Tensor or sequences field"
    )


def sample_k(
    model: nn.Module,
    tokenizer: Any,
    batch: Dict[str, Any],
    *,
    k: int,
    max_new_tokens: int,
    temperature: float = 0.9,
    repetition_penalty: float = 1.0,
    generation_config: Optional[Any] = None,
    generators: Optional[list[torch.Generator]] = None,
    **extra_kwargs: Any,
) -> torch.Tensor:
    """Generate ``k`` iid completions for the same prompt batch."""

    if int(k) <= 0:
        raise ValueError("k must be >= 1 for sampling")

    generations = []
    lengths = []
    for _ in range(int(k)):
        gen_idx = _
        if generators is not None and gen_idx < len(generators):
            extra_kwargs["generator"] = generators[gen_idx]
        seq = generate_completions(
            model,
            tokenizer,
            batch,
            generation_config=generation_config,
            do_sample=True,
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            **extra_kwargs,
        )
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        # Move to CPU immediately to keep GPU peak constant across k
        # Without this, peak memory scales with k since all K sequences accumulate on GPU
        seq = seq.to("cpu", non_blocking=True)
        rl_logprobs.clear_gpu_memory()
        generations.append(seq)
        lengths.append(int(seq.size(1)))
    # Right-pad to the maximum length across K samples so stacking succeeds
    max_len = max(lengths)
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    try:
        pad_id = int(pad_id) if pad_id is not None and int(pad_id) >= 0 else 0
    except Exception:
        pad_id = 0
    padded = []
    for seq in generations:
        if int(seq.size(1)) == max_len:
            padded.append(seq)
            continue
        bsz, cur_len = int(seq.size(0)), int(seq.size(1))
        pad = torch.full(
            (bsz, max_len - cur_len),
            fill_value=pad_id,
            dtype=seq.dtype,
            device=seq.device,
        )
        padded.append(torch.cat([seq, pad], dim=1))
    return torch.stack(padded, dim=0)


__all__ = [
    "prepare_generate_inputs",
    "generate_completions",
    "sample_k",
]
