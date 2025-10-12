"""Generation helpers for manual GRPO training.

These utilities are pure functions so they can be re-used by trainers,
evaluation harnesses, and unit tests. They intentionally avoid depending on
TRL so the manual trainer can stay lightweight.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from torch import nn

from src_new.rl import logprobs as rl_logprobs
from src_new.rl.utils import resolve_im_end_id_strict


_GEN_LOGGER = logging.getLogger("rl.generation")


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
    resolved = resolve_im_end_id_strict(tokenizer)
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
    generation_config: Optional[Any] = None,
    generators: Optional[list[torch.Generator]] = None,
    **extra_kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Generate ``k`` iid completions for the same prompt_batch_size chunk and return per-step log-probs.

    Returns:
        sequences_stacked: Tensor with shape [K, B, L_max] (right-padded)
        gen_logps_list: list of length K with 1D tensors of per-token log-probs (length=T_k per completion)
    """

    if int(k) <= 0:
        raise ValueError("k must be >= 1 for sampling")

    generations = []
    lengths = []
    gen_logps_list: list[torch.Tensor] = []

    # Prepare inputs once to get prompt length on device
    gen_inputs = prepare_generate_inputs(model, batch)

    # Enforce single-row prompts for memory efficiency
    batch_size = gen_inputs["input_ids"].shape[0]
    if batch_size != 1:
        raise ValueError(
            f"sample_k requires single-row prompts (batch_size=1), got batch_size={batch_size}. "
            f"This ensures memory-efficient single-sample generation and log-prob computation."
        )

    prompt_len = int(gen_inputs["input_ids"].shape[1])

    # Unwrap DDP to call generate
    gen_model = getattr(model, "module", model)

    # Resolve EOS id once and ensure it is honored during generation
    eos_id = resolve_im_end_id_strict(tokenizer)

    for _ in range(int(k)):
        gen_idx = _
        if generators is not None and gen_idx < len(generators):
            extra_kwargs["generator"] = generators[gen_idx]

        # Ensure EOS is set either on generation_config or as a kwarg
        if generation_config is not None:
            if (
                eos_id is not None
                and getattr(generation_config, "eos_token_id", None) is None
            ):
                try:
                    generation_config.eos_token_id = int(eos_id)
                except Exception:
                    pass
            gen_args = {"generation_config": generation_config}
        else:
            gen_args = {}
            if eos_id is not None and "eos_token_id" not in extra_kwargs:
                extra_kwargs["eos_token_id"] = int(eos_id)

        _GEN_LOGGER.info(
            "generation.sample_k starting generate: k_idx=%d, temp=%.3f, max_new=%d",
            gen_idx,
            float(temperature),
            int(max_new_tokens),
        )
        with torch.no_grad():
            outputs = gen_model.generate(
                **gen_inputs,
                do_sample=True,
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                **gen_args,
                **extra_kwargs,
            )
        _GEN_LOGGER.info(
            "generation.sample_k finished generate: k_idx=%d, seq_len=%d",
            gen_idx,
            int(getattr(outputs, "sequences", gen_inputs["input_ids"]).shape[-1]),
        )

        # If EOS is present in sequences, trim all sequences at first EOS for consistency
        try:
            sequences = getattr(outputs, "sequences", None)
            if isinstance(sequences, torch.Tensor) and eos_id is not None:
                eos = int(eos_id)
                # Right-pad tokens only after the first EOS that appears in the COMPLETION region
                pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
                for b in range(sequences.size(0)):
                    row = sequences[b]
                    # Search for EOS strictly within the completion slice
                    comp_slice = row[prompt_len:]
                    idxs = (comp_slice == eos).nonzero(as_tuple=True)
                    if idxs[0].numel() > 0:
                        # Convert relative position within completion to absolute index
                        first = int(prompt_len + int(idxs[0][0].item()))
                        if first + 1 < row.size(0):
                            row[first + 1 :] = pad_id
        except Exception:
            pass

        # Log effective generation kwargs once per call (on first completion)
        if gen_idx == 0:
            try:
                eff_kwargs = {
                    "do_sample": True,
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "top_p": float(extra_kwargs.get("top_p"))
                    if "top_p" in extra_kwargs
                    else None,
                    "min_new_tokens": int(extra_kwargs.get("min_new_tokens"))
                    if "min_new_tokens" in extra_kwargs
                    else None,
                    "eos_token_id": int(eos_id) if eos_id is not None else None,
                }
                _GEN_LOGGER.info(
                    "generation.sample_k effective kwargs: %s",
                    {k: v for k, v in eff_kwargs.items() if v is not None},
                )
            except Exception:
                pass

        # sequences: [B, prompt_len + T]; scores: list[T] of [B, vocab]
        sequences = outputs.sequences
        scores = list(outputs.scores) if outputs.scores is not None else []

        # Compute per-step log-probs for completion tokens only
        # For B==1 (we pass single-sample batches here)
        if len(scores) > 0:
            import torch.nn.functional as F

            step_logps = []
            for t, step_scores in enumerate(scores):
                # step_scores: [B, vocab]
                logp = F.log_softmax(step_scores, dim=-1)
                # token chosen at this step is sequences[:, prompt_len + t]
                token_t = sequences[:, prompt_len + t].unsqueeze(-1)
                logp_t = logp.gather(dim=-1, index=token_t).squeeze(-1)  # [B]
                # For B==1, take item tensor; keep as 1D tensor for consistency
                step_logps.append(logp_t.detach().to("cpu"))
            gen_logps = torch.stack(step_logps, dim=-1).squeeze(0)  # [T]
        else:
            gen_logps = torch.zeros(0, dtype=torch.float32)
        gen_logps_list.append(gen_logps)

        # Collect sequences and move to CPU to control peak memory
        seq = sequences
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        seq = seq.to("cpu", non_blocking=True)
        rl_logprobs.clear_gpu_memory()
        generations.append(seq)
        lengths.append(int(seq.size(1)))

    # Right-pad sequences to the maximum length across K samples so stacking succeeds
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

    return torch.stack(padded, dim=0), gen_logps_list


def build_generation_args(
    *,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: Optional[float] = None,
    min_new_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> dict:
    """Assemble standard generate kwargs (excluding eos and required args).

    Keeps behavior identical to callers by only supplying optional knobs.
    """
    args: dict = {}
    # top_p
    if top_p is not None:
        try:
            args["top_p"] = float(top_p)
        except Exception:
            pass
    # repetition penalty
    if repetition_penalty is not None:
        try:
            args["repetition_penalty"] = float(repetition_penalty)
        except Exception:
            pass
    # min_new_tokens
    if min_new_tokens is not None:
        try:
            args["min_new_tokens"] = int(min_new_tokens)
        except Exception:
            pass
    # Note: max_new_tokens and temperature are passed as explicit params in callers
    # to avoid accidental precedence issues; we leave them to the call site.
    return args


__all__ = [
    "prepare_generate_inputs",
    "generate_completions",
    "sample_k",
]
