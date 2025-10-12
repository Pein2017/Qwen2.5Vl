"""Generation buffer utilities for manual GRPO training."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from src_new.rl import generation
from src_new.rl import logprobs as rl_logprobs
from src_new.rl import validators as rl_validators
from src_new.rl.generation import build_generation_args
from src_new.rl.rewards.sanitizer import sanitize_tail_geometry_block
from src_new.rl.rewards.standardizer import RewardStandardizer
from src_new.rl.types import GenerationResult
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
    repetition_penalty: Optional[float] = None,
    mask_truncated_completions: bool,
    scale_rewards: bool,
    max_advantage_magnitude: Optional[float],
    reward_clip_sigma: Optional[float] = None,
    # Dynamic length options
    dyn_enabled: bool = True,
    dyn_alpha: float = 1.1,
    dyn_eos_margin: int = 16,
    dyn_max_cap: Optional[int] = None,
    dyn_estimator: str = "tokenizer",
    dyn_hard_cap: bool = True,
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
    # Enriched per-completion meta aligned with completions_raw
    meta_per_completion: List[Dict[str, Any]] = []

    # Token-length aggregates
    gen_len_tok_list: List[int] = []
    gt_len_tok_list: List[int] = []
    cap_hit_flags: List[int] = []
    # Generation-policy log-probs per completion (to be filled from generate())
    generation_logps_list: List[torch.Tensor] = []
    # Duplication diagnostics per sample (exact string equality)
    dup_exact_ratios: List[float] = []

    gen_kwargs = build_generation_args(
        tokenizer=tokenizer,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p) if top_p is not None else None,
        min_new_tokens=int(min_new_tokens) if min_new_tokens is not None else None,
        repetition_penalty=float(repetition_penalty)
        if repetition_penalty is not None
        else None,
    )

    dynamic_caps: List[int] = []
    gt_length_computed_count = 0  # Track successful GT length computations

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
        gt_len_tok: Optional[int] = None

        # ALWAYS compute GT length for reward functions (length_vs_gt needs this)
        # This computation is independent of whether dynamic_length is enabled
        meta = metas[sample_idx] if sample_idx < len(metas) else None
        if isinstance(meta, dict) and meta.get("objects"):
            try:
                from src_new.processing.coordinate_converter import (
                    CoordinateTokenConverter as _Conv,
                )

                conv = _Conv()
                objs = meta.get("objects") or []
                gt_text = conv.convert_objects_to_tokens(objs)
                # Tokenizer-aligned count for GT
                gt_ids = tokenizer(
                    gt_text, add_special_tokens=False, return_attention_mask=False
                ).get("input_ids", [])
                gt_len_tok = (
                    int(len(gt_ids))
                    if isinstance(gt_ids, list)
                    else int(gt_ids.shape[0])
                )
                gt_length_computed_count += 1
            except Exception as e:
                # Log warning if GT length computation fails (important for length_vs_gt reward)
                _LOGGER.debug(
                    "Failed to compute GT length for sample %d: %s", sample_idx, e
                )
                pass

        # Compute dynamic cap when enabled (uses gt_len_tok if available)
        if dyn_enabled:
            if str(dyn_estimator).lower() != "tokenizer":
                _LOGGER.warning(
                    "Unsupported dynamic_length.estimator=%s; falling back to tokenizer",
                    dyn_estimator,
                )
            try:
                if gt_len_tok is not None:
                    max_cap_val = (
                        int(dyn_max_cap)
                        if dyn_max_cap is not None
                        else int(max_new_tokens)
                    )
                    est_val = int(
                        round(
                            float(dyn_alpha) * int(gt_len_tok) + float(dyn_eos_margin)
                        )
                    )
                    per_sample_cap = min(int(max_cap_val), est_val)
                    dynamic_caps.append(per_sample_cap)
                else:
                    # No GT length available; use default cap
                    dynamic_caps.append(per_sample_cap)
            except Exception:
                dynamic_caps.append(per_sample_cap)

        # Choose generate cap based on hard_cap
        gen_cap_to_use = (
            int(per_sample_cap) if dyn_enabled and dyn_hard_cap else int(max_new_tokens)
        )

        seqs, gen_logps_steps = generation.sample_k(
            model=model,
            tokenizer=tokenizer,
            batch=batch_dict,
            k=sample_k,
            max_new_tokens=gen_cap_to_use,
            temperature=float(temperature),
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

            # Build mask over completion tokens; ALWAYS mask padding after EOS
            # mask_vec includes the EOS token itself (mask up to and including EOS, zero after)
            mask_vec = torch.ones_like(completion, dtype=torch.long)
            had_eos = 0
            if eos_token_id is not None:
                eos_positions = (completion == eos_token_id).nonzero(as_tuple=True)
                if eos_positions[0].numel() > 0:
                    first_eos = int(eos_positions[0][0].item())
                    # Mask everything AFTER the EOS (EOS itself is kept in the mask)
                    if first_eos + 1 < completion.numel():
                        mask_vec[first_eos + 1 :] = 0
                    had_eos = 1
            completion_has_eos.append(int(had_eos))

            # (length handling moved below after optional truncation masking)

            # When hard_cap is disabled, mask overflow tokens beyond per-sample cap (only if no EOS)
            try:
                if (
                    dyn_enabled
                    and (not dyn_hard_cap)
                    and per_sample_cap > 0
                    and completion.numel() > per_sample_cap
                    and int(had_eos) == 0
                ):
                    overflow = completion.numel() - int(per_sample_cap)
                    if overflow > 0:
                        mask_vec[-overflow:] = 0
            except Exception:
                pass

            # Cap-hit flag when hard cap is active (count only when no EOS)
            try:
                if (
                    dyn_enabled
                    and dyn_hard_cap
                    and per_sample_cap > 0
                    and int(had_eos) == 0
                ):
                    _cur_eff_len = int(mask_vec.long().sum().item())
                    cap_hit_flags.append(
                        1 if _cur_eff_len >= int(gen_cap_to_use) else 0
                    )
                else:
                    cap_hit_flags.append(0)
            except Exception:
                cap_hit_flags.append(0)

            # Truncation flag: masked zeros without EOS (hit max_new_tokens without terminating)
            try:
                has_zeros = bool((mask_vec == 0).any().item())
                is_truncated = bool(has_zeros and (int(had_eos) == 0))
                completion_truncated.append(1 if is_truncated else 0)
            except Exception:
                completion_truncated.append(0)

            # NEW: Skip non-EOS completions if configured by zeroing their masks
            try:
                if bool(mask_truncated_completions) and int(had_eos) == 0:
                    # Keep one token to avoid effective_len=0, which can desync ranks
                    if mask_vec.numel() > 0:
                        mask_vec[:] = 0
                        mask_vec[0:1] = 1
            except Exception:
                pass

            # Effective generated length = sum of mask (includes EOS if present)
            eff_len = int(mask_vec.long().sum().item())
            gen_len_tok_list.append(eff_len)
            if gt_len_tok is not None:
                gt_len_tok_list.append(int(gt_len_tok))
            else:
                # keep lists aligned
                gt_len_tok_list.append(0)

            completion_masks.append(mask_vec)
            completions_raw.append(completion)
            completion_lengths.append(eff_len)
            prompt_index_repeat.append(sample_idx)
            completions_text_sample.append(
                tokenizer.decode(completion[:eff_len], skip_special_tokens=False)
            )

            # Store generation-policy log-probs from generate(); align to effective length/mask
            try:
                eff_len = int(mask_vec.long().sum().item())
                gen_logps_k = gen_logps_steps[g]
                if isinstance(gen_logps_k, torch.Tensor):
                    generation_logps_list.append(
                        gen_logps_k[:eff_len].detach().to("cpu")
                    )
                else:
                    generation_logps_list.append(
                        torch.zeros(eff_len, dtype=torch.float32)
                    )
            except Exception:
                generation_logps_list.append(
                    torch.zeros(int(mask_vec.long().sum().item()), dtype=torch.float32)
                )

            # Enrich meta for this completion with tokenizer lengths if available
            # gt_len_tok is computed earlier (always when objects present)
            # gen_len_tok is computed just above (always)
            base_meta = metas[sample_idx] if sample_idx < len(metas) else {}
            try:
                m = dict(base_meta) if isinstance(base_meta, dict) else {}
            except Exception:
                m = {}
            # Always include GT length when available (critical for length_vs_gt reward)
            if gt_len_tok is not None:
                m["gt_len_tokenizer"] = int(gt_len_tok)
            # Always include generation length (use effective masked length)
            m["gen_len_tokenizer"] = int(eff_len)
            meta_per_completion.append(m)

        completions_per_sample.append(completions_text_sample)
        # Duplication detection for this prompt (exact string match)
        try:
            if len(completions_text_sample) > 1:
                uniq = len(set(completions_text_sample))
                n = float(len(completions_text_sample))
                dup_ratio = 1.0 - (float(uniq) / n)
                dup_exact_ratios.append(float(dup_ratio))
                if uniq == 1:
                    _LOGGER.warning(
                        "Identical completions detected for sample_idx=%d (K=%d). Consider increasing temperature/top_p or repetition_penalty.",
                        int(sample_idx),
                        int(n),
                    )
        except Exception:
            pass
        # Proactively free per-sample GPU memory to prevent accumulation across samples
        try:
            del seqs
            rl_logprobs.clear_gpu_memory()
        except Exception:
            pass

    # ===== GENERATION LOGPROBS already collected during generate() =====

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
    per_func_values_raw: List[torch.Tensor] = []
    # Build sanitized completions for selected rewards when tail is ambiguous
    terminated_flags_flat = [int(x) for x in completion_has_eos]
    truncated_flags_flat = [int(x) for x in completion_truncated]
    sanitized_completions: List[str] = []
    sanitizer_applied = 0
    for idx, txt in enumerate(completions_text):
        need_sanitize = (
            idx < len(terminated_flags_flat) and terminated_flags_flat[idx] == 0
        ) or (idx < len(truncated_flags_flat) and truncated_flags_flat[idx] == 1)
        if need_sanitize:
            sanitized = sanitize_tail_geometry_block(txt)
            if sanitized != txt:
                sanitizer_applied += 1
            sanitized_completions.append(sanitized)
        else:
            sanitized_completions.append(txt)

    # Rewards for which we apply tail sanitization by default (length-wise excluded)
    SANITIZE_REWARDS = {
        "wrappers",
        "coords",
        "separators",
        "caption_f1",
        "grounding_acc",
        "coverage",
        "bbox_giou",
        "quad_giou",
        "line_giou",
        "quad_l1",
        "line_l1",
    }
    clip_sigma = float(reward_clip_sigma) if reward_clip_sigma is not None else 5.0
    for fn, reward_name in zip(reward_fns, reward_names):
        try:
            values = fn(
                prompts=prompts_repeated,
                completions=(
                    sanitized_completions
                    if reward_name in SANITIZE_REWARDS
                    else completions_text
                ),
                meta=meta_repeated,
            )
        except TypeError:
            values = fn(
                prompts=prompts_repeated,
                completions=(
                    sanitized_completions
                    if reward_name in SANITIZE_REWARDS
                    else completions_text
                ),
            )
        values = [float(v) if v is not None else float("nan") for v in values]
        # Raw (pre-clip, pre-standardize)
        values_tensor_raw = torch.tensor(values, dtype=torch.float32, device=device)
        values_tensor_raw = torch.nan_to_num(
            values_tensor_raw, nan=0.0, posinf=0.0, neginf=0.0
        )
        per_func_values_raw.append(values_tensor_raw)
        # Processed (clip + optional standardize)
        values_tensor = values_tensor_raw.clone()
        values_tensor = torch.clamp(values_tensor, -clip_sigma, clip_sigma)
        if reward_standardizer is not None:
            values_tensor = reward_standardizer.update_and_standardize(
                reward_name, values_tensor
            )
            # Guard against NaN/Inf after standardization to avoid rank divergence
            values_tensor = torch.nan_to_num(
                values_tensor,
                nan=0.0,
                posinf=float(clip_sigma),
                neginf=-float(clip_sigma),
            )
        per_func_values.append(values_tensor)

    rewards_per_func = torch.stack(per_func_values, dim=1) if per_func_values else None
    rewards_per_func_raw = (
        torch.stack(per_func_values_raw, dim=1) if per_func_values_raw else None
    )
    if rewards_per_func is None:
        rewards = torch.zeros(len(prompts_repeated), device=device)
    else:
        weighted = rewards_per_func * reward_weights.view(1, -1)
        rewards = torch.nan_to_num(weighted, nan=0.0).sum(dim=1)
    # Aggregate raw (absolute) reward without clip/standardize
    if rewards_per_func_raw is None:
        raw_rewards = torch.zeros(len(prompts_repeated), device=device)
    else:
        raw_weighted = rewards_per_func_raw * reward_weights.view(1, -1)
        raw_rewards = torch.nan_to_num(raw_weighted, nan=0.0).sum(dim=1)

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

    # Optionally zero advantages for truncated/no‑EOS completions when masking is enabled
    try:
        if (
            bool(mask_truncated_completions)
            and len(completion_truncated) == advantages.numel()
        ):
            trunc_mask = torch.tensor(
                completion_truncated, dtype=torch.float32, device=device
            )
            # Keep gradients only for EOS-terminated completions
            advantages = advantages * (1.0 - trunc_mask)
    except Exception:
        pass

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

    # Prepare outputs
    # Build repeated prompts aligned with per-completion arrays
    prompts_repeated = [prompts[idx] for idx in prompt_index_repeat]

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
        "meta": meta_per_completion,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "images_per_sample": images_per_sample_tensor,
        "completion_lengths": torch.tensor(
            completion_lengths, dtype=torch.long, device=device
        ),
        "terminated_with_eos": torch.tensor(
            completion_has_eos, dtype=torch.long, device=device
        ).view(-1),
        "truncated_flags": torch.tensor(
            completion_truncated, dtype=torch.long, device=device
        ).view(-1),
        "generation_logps": generation_logps_tensor.to(
            device
        ),  # Log-probs from generation policy
    }
    # Log sanitizer diagnostics
    try:
        total_c = max(len(completions_text), 1)
        result["sanitizer/applied_ratio"] = float(sanitizer_applied) / float(total_c)
    except Exception:
        pass
    # Aggregate dynamic cap statistics for logging
    try:
        if dynamic_caps:
            caps_t = torch.tensor(dynamic_caps, dtype=torch.float32, device=device)
            result["dynamic_length/mean_cap"] = float(caps_t.mean().item())
            result["dynamic_length/max_cap"] = float(caps_t.max().item())
    except Exception:
        pass

    # Aggregate tokenizer-based length stats and cap-hit ratio
    try:
        if gen_len_tok_list:
            v = torch.tensor(gen_len_tok_list, dtype=torch.float32, device=device)
            result["completions/mean_len_tok"] = float(v.mean().item())
        if gt_len_tok_list:
            u = torch.tensor(gt_len_tok_list, dtype=torch.float32, device=device)
            # zeros may be placeholders; compute mean over nonzeros
            if (u > 0).any():
                result["gt/mean_len_tok"] = float(u[u > 0].mean().item())
        if cap_hit_flags:
            c = torch.tensor(cap_hit_flags, dtype=torch.float32, device=device)
            result["completions/cap_hit_ratio"] = float(c.mean().item())
        # Zero-length completion ratio (effective length after masking)
        if completion_lengths:
            zero_cnt = sum(1 for _l in completion_lengths if int(_l) <= 0)
            result["completions/zero_len_ratio"] = float(zero_cnt) / float(
                max(len(completion_lengths), 1)
            )
        # Duplication metrics across prompts (mean of per-sample ratios)
        if dup_exact_ratios:
            try:
                mean_dup = float(
                    sum(dup_exact_ratios) / float(max(len(dup_exact_ratios), 1))
                )
                result["completions/dup_exact_mean"] = mean_dup
                if mean_dup >= 0.5:
                    _LOGGER.warning(
                        "High duplication across prompts: dup_exact_mean=%.3f. Increase temperature/top_p, set repetition_penalty, or reduce steps_per_generation.",
                        mean_dup,
                    )
            except Exception:
                pass
    except Exception:
        pass

    # Log GT length computation success rate for diagnostic purposes
    if gt_length_computed_count > 0:
        _LOGGER.debug(
            "GT length computed for %d/%d samples (%.1f%%) - length_vs_gt reward will use accurate GT lengths",
            gt_length_computed_count,
            len(inputs),
            100.0 * gt_length_computed_count / max(len(inputs), 1),
        )
    elif len(inputs) > 0:
        _LOGGER.warning(
            "GT length not computed for any samples - length_vs_gt reward may be inaccurate. "
            "Ensure dataset samples contain 'objects' in meta dict."
        )

    if rewards_per_func is not None:
        result["rewards_per_func"] = rewards_per_func.to(device)
    if rewards_per_func_raw is not None:
        result["raw_rewards_per_func"] = rewards_per_func_raw.to(device)
        result["raw_rewards"] = raw_rewards.to(device)

    # Build a typed dataclass for downstream consumers (bridge back to dict)
    try:
        dc_extra: Dict[str, Any] = {}
        for k in (
            "sanitizer/applied_ratio",
            "dynamic_length/mean_cap",
            "dynamic_length/max_cap",
            "completions/mean_len_tok",
            "gt/mean_len_tok",
            "completions/cap_hit_ratio",
            "completions/zero_len_ratio",
            "completions/dup_exact_mean",
        ):
            if k in result:
                dc_extra[k] = result[k]

        gen_dc = GenerationResult(
            prompt_ids=result["prompt_ids"],
            prompt_mask=result["prompt_mask"],
            completion_ids=result["completion_ids"],
            completion_mask=result["completion_mask"],
            advantages=result["advantages"],
            rewards=result["rewards"],
            reward_names=list(result.get("reward_names", [])),
            raw_rewards=result.get("raw_rewards"),
            rewards_per_func=result.get("rewards_per_func"),
            raw_rewards_per_func=result.get("raw_rewards_per_func"),
            generation_logps=result.get("generation_logps"),
            pixel_values=result.get("pixel_values"),
            image_grid_thw=result.get("image_grid_thw"),
            images_per_sample=result.get("images_per_sample"),
            completion_lengths=result.get("completion_lengths"),
            terminated_with_eos=result.get("terminated_with_eos"),
            truncated_flags=result.get("truncated_flags"),
            prompts=list(result.get("prompts", [])),
            completions=list(result.get("completions", [])),
            meta=list(result.get("meta", [])),
            temperature=float(temperature),
            beta=0.0,
            extra=dc_extra,
        )
        return gen_dc.to_dict()
    except Exception:
        # Fallback to legacy dict if dataclass construction fails for any reason
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
        "raw_rewards_per_func",
        "raw_rewards",
        "completion_lengths",
        "terminated_with_eos",
        "truncated_flags",
        "generation_logps",
    ]
    list_keys = ["prompts", "completions", "meta"]
    scalar_keys = [
        "temperature",
        "advantages_global/std",
        "advantages_global/max_abs",
        "rewards_global/mean",
        "rewards_global/std",
    ]

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
