"""Generation buffer utilities for manual GRPO training."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from src_new.rl import generation
from src_new.rl import logprobs as rl_logprobs
from src_new.rl import validators as rl_validators
from src_new.rl.rewards.standardizer import RewardStandardizer
from src_new.rl.utils import resolve_im_end_id


_LOGGER = logging.getLogger("rl.buffer")


def generate_and_score(
    *,
    model: Any,
    tokenizer: Any,
    inputs: Sequence[Dict[str, Any]],
    reward_fns: Sequence[Any],
    reward_weights: torch.Tensor,
    reward_names: Sequence[str],
    reward_standardizer: Optional[RewardStandardizer],
    pad_token_id: int,
    sample_k: int,
    max_new_tokens: int,
    min_new_tokens: Optional[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    mask_truncated_completions: bool,
    scale_rewards: bool,
    max_advantage_magnitude: Optional[float],
    reward_clip_sigma: Optional[float] = None,
    # Dynamic length options
    dyn_enabled: bool = True,
    dyn_alpha: float = 1.1,
    dyn_eos_margin: int = 16,
    dyn_min_cap: int = 32,
    dyn_max_cap: Optional[int] = None,
    dyn_mask_overflow_only: bool = False,
) -> Dict[str, Any]:
    if not inputs:
        raise ValueError("inputs must be a non-empty sequence")
    device = reward_weights.device
    eos_token_id = resolve_im_end_id(tokenizer)

    prompts: List[str] = []
    metas: List[Any] = []
    input_ids_list: List[torch.Tensor] = []
    attention_masks_list: List[torch.Tensor] = []
    pixel_values_list: List[Optional[torch.Tensor]] = []
    thw_list: List[Optional[torch.Tensor]] = []

    for sample in inputs:
        prompts.append(sample.get("prompt", ""))
        metas.append(sample.get("meta"))

        ids = sample.get("input_ids")
        mask = sample.get("attention_mask")
        if ids is None or mask is None:
            raise ValueError("Dataset sample missing input_ids or attention_mask")
        ids_t = ids.clone().detach() if torch.is_tensor(ids) else torch.tensor(ids)
        mask_t = mask.clone().detach() if torch.is_tensor(mask) else torch.tensor(mask)
        if ids_t.dim() != 1 or mask_t.dim() != 1:
            raise ValueError(
                "input_ids and attention_mask must be 1D tensors per sample"
            )
        input_ids_list.append(ids_t.long())
        attention_masks_list.append(mask_t.long())

        pv = sample.get("pixel_values")
        if pv is not None:
            pv_t = pv.clone().detach() if torch.is_tensor(pv) else torch.tensor(pv)
            pixel_values_list.append(pv_t)
        else:
            pixel_values_list.append(None)

        thw = sample.get("image_grid_thw")
        from src_new.rl.tensor_utils import normalize_thw  # late import

        thw_list.append(normalize_thw(thw) if thw is not None else None)

    prompt_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    prompt_mask = pad_sequence(attention_masks_list, batch_first=True, padding_value=0)

    completions_raw: List[torch.Tensor] = []
    completion_masks: List[torch.Tensor] = []
    completion_lengths: List[int] = []
    completion_has_eos: List[int] = []
    completion_truncated: List[int] = []
    prompt_index_repeat: List[int] = []
    completions_per_sample: List[List[str]] = []

    gen_kwargs = {"top_p": float(top_p)}
    if min_new_tokens is not None:
        gen_kwargs["min_new_tokens"] = int(min_new_tokens)

    dynamic_caps: List[int] = []

    for sample_idx, (ids_t, mask_t, pv_t, thw_t) in enumerate(
        zip(input_ids_list, attention_masks_list, pixel_values_list, thw_list)
    ):
        batch_dict: Dict[str, Any] = {
            "input_ids": ids_t,
            "attention_mask": mask_t,
        }
        if pv_t is not None:
            batch_dict["pixel_values"] = pv_t
        if thw_t is not None:
            batch_dict["image_grid_thw"] = thw_t

        # Default cap from arguments; can be overridden per-sample below
        per_sample_cap = int(max_new_tokens)

        # Compute dynamic cap only when enabled
        meta = metas[sample_idx] if sample_idx < len(metas) else None
        if dyn_enabled:
            try:
                if isinstance(meta, dict):
                    from src_new.processing.coordinate_converter import (
                        CoordinateTokenConverter as _Conv,
                    )

                    conv = _Conv()
                    objs = meta.get("objects") or []
                    gt_text = conv.convert_objects_to_tokens(objs)
                    # Tokenizer-aligned count
                    gt_ids = tokenizer(
                        gt_text, add_special_tokens=False, return_attention_mask=False
                    ).get("input_ids", [])
                    gt_len = (
                        int(len(gt_ids))
                        if isinstance(gt_ids, list)
                        else int(gt_ids.shape[0])
                    )
                    max_cap_val = (
                        int(dyn_max_cap)
                        if dyn_max_cap is not None
                        else int(max_new_tokens)
                    )
                    est = int(round(float(dyn_alpha) * gt_len + float(dyn_eos_margin)))
                    per_sample_cap = max(int(dyn_min_cap), min(int(max_cap_val), est))
                    dynamic_caps.append(per_sample_cap)
            except Exception:
                dynamic_caps.append(per_sample_cap)

        seqs = generation.sample_k(
            model=model,
            tokenizer=tokenizer,
            batch=batch_dict,
            k=sample_k,
            max_new_tokens=int(per_sample_cap),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            **gen_kwargs,
        )
        prompt_len = ids_t.size(0)
        completions_text_sample: List[str] = []

        for g in range(sample_k):
            seq = seqs[g]
            # seq is [B, total_len]; take the single batch row then slice tokens
            if seq.dim() == 2:
                seq_row = seq[0]
            else:
                seq_row = seq.view(-1)
            completion = seq_row[prompt_len:]
            if completion.numel() == 0:
                if eos_token_id is not None:
                    completion = torch.tensor([eos_token_id], dtype=torch.long)
                else:
                    completion = torch.tensor([pad_token_id], dtype=torch.long)

            # Force completion tokens to CPU for consistent concatenation below
            if completion.device.type != "cpu":
                completion = completion.to("cpu")

            mask_vec = torch.ones_like(completion, dtype=torch.long)
            if eos_token_id is not None:
                eos_positions = (completion == eos_token_id).nonzero(as_tuple=True)
                if eos_positions[0].numel() > 0:
                    first_eos = int(eos_positions[0][0].item())
                    mask_vec[first_eos + 1 :] = 0 if mask_truncated_completions else 1
                    completion_has_eos.append(1)
                    completion_truncated.append(
                        1
                        if mask_truncated_completions
                        and mask_vec[first_eos + 1 :].numel() > 0
                        else 0
                    )
                else:
                    completion_has_eos.append(0)
                    if mask_truncated_completions:
                        mask_vec[:] = 0
                        completion_truncated.append(1)
                    else:
                        completion_truncated.append(0)
            else:
                completion_has_eos.append(0)
                completion_truncated.append(0)

            # Optional: mask only overflow tokens beyond per-sample cap if requested
            try:
                if (
                    dyn_mask_overflow_only
                    and per_sample_cap > 0
                    and completion.numel() > per_sample_cap
                ):
                    overflow = completion.numel() - int(per_sample_cap)
                    if overflow > 0:
                        mask_vec[-overflow:] = 0
            except Exception:
                pass

            completion_masks.append(mask_vec)
            completions_raw.append(completion)
            completion_lengths.append(int(mask_vec.long().sum().item()))
            prompt_index_repeat.append(sample_idx)
            completions_text_sample.append(
                tokenizer.decode(completion, skip_special_tokens=False)
            )
        completions_per_sample.append(completions_text_sample)
        # Proactively free per-sample GPU memory to prevent accumulation across samples
        # Each sample's generation tensors are now on CPU, but we still clear any GPU cache
        try:
            del seqs
            rl_logprobs.clear_gpu_memory()
        except Exception:
            pass

    # ===== COMPUTE GENERATION LOGPROBS (Critical for GRPO trust region) =====
    # Store log-probs from the policy that GENERATED the completions.
    # This enables proper ratio = π_current / π_generation in GRPO loss.
    # Without this, ratio is always 1.0 and trust region clipping is ineffective.

    generation_logps_list: List[torch.Tensor] = []

    for comp_idx, (comp_ids, comp_mask, p_idx) in enumerate(
        zip(completions_raw, completion_masks, prompt_index_repeat)
    ):
        effective_len = int(comp_mask.long().sum().item())
        if effective_len == 0:
            # No valid tokens - store dummy zero tensor
            generation_logps_list.append(
                torch.zeros(1, dtype=torch.float32, device=device)
            )
            continue

        # Ensure prompt/completion are on CPU before concatenation
        comp_ids_eff = comp_ids[:effective_len].to("cpu")
        prompt_ids_single = input_ids_list[p_idx].to("cpu")

        # Concatenate prompt + completion
        full_ids = torch.cat([prompt_ids_single, comp_ids_eff], dim=0).unsqueeze(0)
        full_mask = torch.ones_like(full_ids, dtype=torch.long)

        # Build vision inputs for this sample
        pv_single = pixel_values_list[p_idx] if p_idx < len(pixel_values_list) else None
        thw_single = thw_list[p_idx] if p_idx < len(thw_list) else None
        images_count = (
            torch.tensor([thw_single.size(0)], dtype=torch.long)
            if thw_single is not None
            else None
        )

        # Compute logprobs with the GENERATION policy (current model state)
        with torch.no_grad():
            try:
                gen_logps = rl_logprobs.get_per_token_logps(
                    model=model,
                    input_ids=full_ids.to(device),
                    attention_mask=full_mask.to(device),
                    logits_to_keep=effective_len,
                    pixel_values=pv_single.to(device)
                    if pv_single is not None
                    else None,
                    image_grid_thw=thw_single.to(device)
                    if thw_single is not None
                    else None,
                    images_per_sample=images_count.to(device)
                    if images_count is not None
                    else None,
                    temperature=float(temperature),
                    detach=True,
                )
                generation_logps_list.append(gen_logps.squeeze(0).cpu())
            except Exception:
                # Fallback on error: store zeros (will fall back to old behavior)
                generation_logps_list.append(
                    torch.zeros(effective_len, dtype=torch.float32)
                )

        # Free memory after each logprob computation
        rl_logprobs.clear_gpu_memory()

    # Pad generation logprobs to same length
    max_logp_len = (
        max(lp.size(0) for lp in generation_logps_list) if generation_logps_list else 1
    )
    padded_generation_logps: List[torch.Tensor] = []
    for lp in generation_logps_list:
        if lp.size(0) < max_logp_len:
            pad = torch.zeros(
                max_logp_len - lp.size(0), dtype=lp.dtype, device=lp.device
            )
            padded_generation_logps.append(torch.cat([lp, pad], dim=0))
        else:
            padded_generation_logps.append(lp)
    generation_logps_tensor = (
        torch.stack(padded_generation_logps, dim=0)
        if padded_generation_logps
        else torch.zeros(len(completions_raw), 1, dtype=torch.float32)
    )

    prompt_ids = prompt_ids.repeat_interleave(sample_k, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(sample_k, dim=0)

    padded_completion_ids = pad_sequence(
        completions_raw, batch_first=True, padding_value=pad_token_id
    )
    padded_completion_mask = pad_sequence(
        completion_masks, batch_first=True, padding_value=0
    )

    completions_text = [text for sample in completions_per_sample for text in sample]
    prompts_repeated = [prompts[idx] for idx in prompt_index_repeat]
    meta_repeated = [metas[idx] for idx in prompt_index_repeat]

    if len(reward_fns) != len(reward_names):
        raise ValueError("reward_names must align with reward_fns")

    per_func_values: List[torch.Tensor] = []
    clip_sigma = float(reward_clip_sigma) if reward_clip_sigma is not None else 5.0
    for fn, reward_name in zip(reward_fns, reward_names):
        try:
            values = fn(
                prompts=prompts_repeated,
                completions=completions_text,
                meta=meta_repeated,
            )
        except TypeError:
            values = fn(prompts=prompts_repeated, completions=completions_text)
        values = [float(v) if v is not None else float("nan") for v in values]
        values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
        values_tensor = torch.nan_to_num(values_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        values_tensor = torch.clamp(values_tensor, -clip_sigma, clip_sigma)
        if reward_standardizer is not None:
            values_tensor = reward_standardizer.update_and_standardize(
                reward_name, values_tensor
            )
        per_func_values.append(values_tensor)

    rewards_per_func = torch.stack(per_func_values, dim=1) if per_func_values else None
    if rewards_per_func is None:
        rewards = torch.zeros(len(prompts_repeated), device=device)
    else:
        weighted = rewards_per_func * reward_weights.view(1, -1)
        rewards = torch.nan_to_num(weighted, nan=0.0).sum(dim=1)

    total_samples = len(inputs)
    grouped = rewards.view(total_samples, sample_k)
    mean_grouped = grouped.mean(dim=1, keepdim=True)
    std_grouped = grouped.std(dim=1, keepdim=True, unbiased=False)
    std_grouped = torch.where(
        std_grouped > 0, std_grouped, torch.full_like(std_grouped, 1e-4)
    )

    mean_repeated = mean_grouped.repeat_interleave(sample_k, dim=1).reshape(-1)
    std_repeated = std_grouped.repeat_interleave(sample_k, dim=1).reshape(-1)

    advantages = rewards - mean_repeated
    if scale_rewards:
        advantages = advantages / (std_repeated + 1e-4)
    if max_advantage_magnitude is not None and max_advantage_magnitude > 0:
        advantages = torch.clamp(
            advantages,
            min=-float(max_advantage_magnitude),
            max=float(max_advantage_magnitude),
        )

    # Vision tensor packing
    pixel_chunks: List[torch.Tensor] = []
    grid_chunks: List[torch.Tensor] = []
    images_per_sample: List[int] = []
    for pv_t, thw_t in zip(pixel_values_list, thw_list):
        num_images = int(thw_t.size(0)) if thw_t is not None else 0
        for _ in range(sample_k):
            images_per_sample.append(num_images)
            if thw_t is not None:
                grid_chunks.append(thw_t.clone().detach())
            if pv_t is not None and num_images > 0:
                pixel_chunks.append(pv_t.clone().detach())

    pixel_values = torch.cat(pixel_chunks, dim=0) if pixel_chunks else None
    image_grid_thw = torch.cat(grid_chunks, dim=0) if grid_chunks else None
    images_per_sample_tensor = (
        torch.tensor(images_per_sample, dtype=torch.long) if images_per_sample else None
    )

    # Validate packed vision alignment to catch drift early
    try:
        rl_validators.assert_patches_match_thw(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            images_per_sample=images_per_sample_tensor,
        )
    except (
        Exception
    ) as _e:  # keep non-fatal in buffer; trainer does strict checks later
        _LOGGER.debug("Vision alignment validation warning: %s", _e)

    result: Dict[str, Any] = {
        "prompt_ids": prompt_ids.long(),
        "prompt_mask": prompt_mask.long(),
        "completion_ids": padded_completion_ids.long(),
        "completion_mask": padded_completion_mask.long(),
        "advantages": advantages.to(device),
        "rewards": rewards.to(device),
        "reward_names": list(reward_names),
        "prompts": prompts_repeated,
        "completions": completions_text,
        "meta": meta_repeated,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "images_per_sample": images_per_sample_tensor,
        "completion_lengths": torch.tensor(
            completion_lengths, dtype=torch.long, device=device
        ),
        "terminated_with_eos": torch.tensor(
            completion_has_eos, dtype=torch.long, device=device
        ),
        "truncated_flags": torch.tensor(
            completion_truncated, dtype=torch.long, device=device
        ),
        "generation_logps": generation_logps_tensor.to(
            device
        ),  # Log-probs from generation policy
    }
    # Aggregate dynamic cap statistics for logging
    try:
        if dynamic_caps:
            caps_t = torch.tensor(dynamic_caps, dtype=torch.float32, device=device)
            result["dynamic_length/mean_cap"] = float(caps_t.mean().item())
            result["dynamic_length/min_cap"] = float(caps_t.min().item())
            result["dynamic_length/max_cap"] = float(caps_t.max().item())
    except Exception:
        pass
    if rewards_per_func is not None:
        result["rewards_per_func"] = rewards_per_func.to(device)
    return result


def split_buffer(
    buffer: Dict[str, Any], steps_per_generation: int
) -> List[Dict[str, Any]]:
    if steps_per_generation <= 1:
        return [buffer]

    prompt_ids = buffer.get("prompt_ids")
    if prompt_ids is None or not torch.is_tensor(prompt_ids):
        raise ValueError("buffer missing tensor prompt_ids for splitting")

    total = prompt_ids.size(0)
    if total % steps_per_generation != 0:
        raise ValueError(
            f"Cannot split buffer: total samples {total} not divisible by steps_per_generation {steps_per_generation}"
        )
    chunk_size = total // steps_per_generation

    images_per_sample = buffer.get("images_per_sample")
    image_grid_thw = buffer.get("image_grid_thw")
    pixel_values = buffer.get("pixel_values")

    image_cumsum = (
        images_per_sample.cumsum(0)
        if isinstance(images_per_sample, torch.Tensor)
        else None
    )
    patch_cumsum = None
    if (
        image_grid_thw is not None
        and torch.is_tensor(image_grid_thw)
        and image_grid_thw.numel() > 0
    ):
        patch_counts = (
            image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        ).long()
        patch_cumsum = patch_counts.cumsum(0)

    tensor_keys = [
        "prompt_ids",
        "prompt_mask",
        "completion_ids",
        "completion_mask",
        "advantages",
        "rewards",
        "rewards_per_func",
        "completion_lengths",
        "terminated_with_eos",
        "truncated_flags",
        "generation_logps",
    ]
    list_keys = ["prompts", "completions", "meta"]
    scalar_keys = ["temperature"]

    chunks: List[Dict[str, Any]] = []
    for idx in range(steps_per_generation):
        start = idx * chunk_size
        end = start + chunk_size
        chunk_dict: Dict[str, Any] = {}

        for key in tensor_keys:
            value = buffer.get(key)
            if value is None or not torch.is_tensor(value):
                continue
            if value.dim() == 0 or value.size(0) != total:
                continue
            chunk_dict[key] = value[start:end]

        for key in list_keys:
            value = buffer.get(key)
            if isinstance(value, list) and len(value) == total:
                chunk_dict[key] = value[start:end]

        if isinstance(images_per_sample, torch.Tensor):
            chunk_images = images_per_sample[start:end]
            chunk_dict["images_per_sample"] = chunk_images

            if image_cumsum is not None:
                img_start = int(image_cumsum[start - 1].item()) if start > 0 else 0
                img_end = int(image_cumsum[end - 1].item()) if end > 0 else img_start

                if image_grid_thw is not None and img_end > img_start:
                    chunk_dict["image_grid_thw"] = image_grid_thw[img_start:img_end]
                else:
                    chunk_dict["image_grid_thw"] = None

                if (
                    pixel_values is not None
                    and torch.is_tensor(pixel_values)
                    and patch_cumsum is not None
                    and img_end > img_start
                ):
                    patch_start = (
                        int(patch_cumsum[img_start - 1].item()) if img_start > 0 else 0
                    )
                    patch_end = int(patch_cumsum[img_end - 1].item())
                    chunk_dict["pixel_values"] = pixel_values[patch_start:patch_end]
                elif pixel_values is not None and torch.is_tensor(pixel_values):
                    chunk_dict["pixel_values"] = pixel_values.narrow(0, 0, 0)
                else:
                    chunk_dict["pixel_values"] = None
            else:
                chunk_dict["image_grid_thw"] = None
                chunk_dict["pixel_values"] = None
        else:
            chunk_dict["images_per_sample"] = None
            chunk_dict["image_grid_thw"] = None
            chunk_dict["pixel_values"] = None

        if "reward_names" in buffer:
            chunk_dict["reward_names"] = buffer["reward_names"]

        for key in scalar_keys:
            if key in buffer and key not in chunk_dict:
                chunk_dict[key] = buffer[key]

        chunks.append(chunk_dict)

    return chunks


__all__ = ["generate_and_score", "split_buffer"]
