#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import Qwen2VLProcessor
from transformers.generation import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from src_new.models.wrapper import DetectionModel
from src_post.generation.generation import (
    decode_to_text,
    sft_style_preprocess_image,
    to_device_and_cast,
)
from src_post.prompting.conversation import GroupQCConversationBuilder
from src_post.rewards.compose import compose_reward
from src_post.tf.teacher_forcing import (
    decision_margin,
    kl_to_ref_with_cur_logits,
    sequence_entropy_from_logits,
    tf_decision_probs,
    tf_sum_logprob_and_logits_over_response,
)


class BaseCreditAssigner:
    def compute_loss_a(
        self,
        policy: DetectionModel,
        train_model: torch.nn.Module,
        processor: Qwen2VLProcessor,
        conv_builder: Any,
        images: List[Image.Image],
        context_lines: List[str],
        checklist: List[str],
        mission: Optional[str],
        gt_label: str,
        enc_b_baseline: Dict[str, torch.Tensor],
        tf_cfg: Dict[
            str, Any
        ],  # length_norm, logits_processors, stopping, device, K_A, max_images_tf, ddp_policy
        gen_cfg_a: GenerationConfig,
        reward_cfg: Dict[
            str, Any
        ],  # names, weights, group_reward_mode, use_ref_kl, ref_model, lambda_kl_stage_a, use_uncertainty_gate, entropy_threshold, use_mission_checklist, baseline_tf_p_pass, baseline_tf_p_fail
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError


class OffAssigner(BaseCreditAssigner):
    def compute_loss_a(
        self,
        policy: DetectionModel,
        train_model: torch.nn.Module,
        processor: Qwen2VLProcessor,
        conv_builder: Any,
        images: List[Image.Image],
        context_lines: List[str],
        checklist: List[str],
        mission: Optional[str],
        gt_label: str,
        enc_b_baseline: Dict[str, torch.Tensor],
        tf_cfg: Dict[str, Any],
        gen_cfg_a: GenerationConfig,
        reward_cfg: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dev = next(policy.parameters()).device
        return torch.tensor(0.0, dtype=torch.float32, device=dev), {}


class ConditionalAssigner(BaseCreditAssigner):
    def compute_loss_a(
        self,
        policy: DetectionModel,
        train_model: torch.nn.Module,
        processor: Qwen2VLProcessor,
        conv_builder: Any,
        images: List[Image.Image],
        context_lines: List[str],
        checklist: List[str],
        mission: Optional[str],
        gt_label: str,
        enc_b_baseline: Dict[str, torch.Tensor],
        tf_cfg: Dict[str, Any],
        gen_cfg_a: GenerationConfig,
        reward_cfg: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device: str = str(tf_cfg.get("device"))
        length_norm: bool = bool(tf_cfg.get("length_norm", True))
        logits_processors: LogitsProcessorList = tf_cfg.get("logits_processors")
        stopping = tf_cfg.get("stopping")
        K_A: int = int(tf_cfg.get("K_A", 3))
        max_images_tf: int = int(tf_cfg.get("max_images_tf", 1))
        ddp_policy = tf_cfg.get("ddp_policy")
        accum_scale: float = float(tf_cfg.get("accum_scale", 1.0))

        reward_names: List[str] = list(reward_cfg.get("names", []))
        reward_weights: List[float] = list(reward_cfg.get("weights", []))
        group_reward_mode: str = str(
            reward_cfg.get("group_reward_mode", "combined")
        ).lower()
        use_ref_kl: bool = bool(reward_cfg.get("use_ref_kl", False))
        ref_model: Optional[DetectionModel] = reward_cfg.get("ref_model")
        lambda_kl_stage_a: float = float(reward_cfg.get("lambda_kl_stage_a", 0.0))
        use_uncertainty_gate: bool = bool(reward_cfg.get("use_uncertainty_gate", False))
        entropy_threshold: float = float(reward_cfg.get("entropy_threshold", 3.5))
        use_mission_checklist: bool = bool(
            reward_cfg.get("use_mission_checklist", True)
        )
        baseline_tf_p_pass: float = float(reward_cfg.get("baseline_tf_p_pass", 0.0))
        baseline_tf_p_fail: float = float(reward_cfg.get("baseline_tf_p_fail", 0.0))

        # Prepare baseline margin for gating
        baseline_margin = decision_margin(
            enc_b_baseline, processor, train_model, length_norm=length_norm
        )

        dev = torch.device(device)
        loss_a = torch.tensor(0.0, dtype=torch.float32, device=dev)
        entropy_vals: List[float] = []
        best_single_delta_val: float = 0.0
        # Collect Stage-A K_A candidates (first pass) for diagnostics/logging
        diag_stage_a_samples: List[Dict[str, Any]] = []

        capped_images = list(images)[: max(0, max_images_tf)]
        deltas: List[Tuple[float, int]] = []
        # First pass: measure best margin delta per image
        for i, img in enumerate(capped_images):
            # Build Stage‑A encoding
            messages = conv_builder.build_stage_a_messages(mission)
            text_a = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            img_proc = sft_style_preprocess_image(img)
            # Enforce typed message carries exactly one image for Stage-A
            GroupQCConversationBuilder.validate_typed_image_count(messages, 1)
            enc_a = processor(
                text=[text_a], images=[img_proc], padding=True, return_tensors="pt"
            )
            enc_a = to_device_and_cast(enc_a, device)

            cand_token_ids: List[List[int]] = []
            rewards_i: List[float] = []
            margins_i: List[float] = []

            ka_cands: List[Dict[str, Any]] = []
            for _ in range(max(1, K_A)):
                with torch.no_grad():
                    out = policy.generate(
                        **enc_a,
                        generation_config=gen_cfg_a,
                        logits_processor=logits_processors,
                        stopping_criteria=stopping,
                    )
                new_ids = out[:, enc_a["input_ids"].size(1) :]
                toks = new_ids[0].tolist()
                raw = decode_to_text(processor, toks).strip()

                # Build Stage‑B variant (minimal vs checklist)
                context_variant = list(context_lines)
                context_variant[i] = raw
                if use_mission_checklist:
                    msgs_var = conv_builder.build_stage_b_messages(
                        summary_lines=context_variant, checklist_lines=checklist
                    )
                else:
                    msgs_var = conv_builder.build_stage_b_messages_minimal(
                        summary_lines=context_variant
                    )
                text_b_var = processor.apply_chat_template(
                    conversation=msgs_var, tokenize=False, add_generation_prompt=True
                )
                enc_b_var = processor(
                    text=[text_b_var], images=None, return_tensors="pt", padding=True
                )
                enc_b_var = to_device_and_cast(enc_b_var, device)

                # DDP sync
                if ddp_policy is not None and torch.distributed.is_initialized():
                    try:
                        ddp_policy._sync_params_and_buffers(authoritative_rank=0)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                try:
                    prob_pass, prob_fail = tf_decision_probs(
                        train_model, enc_b_var, processor, length_norm=length_norm
                    )
                except Exception:
                    prob_pass, prob_fail = 0.0, 0.0

                # Reward composition mode
                rn, rw = reward_names, reward_weights
                if group_reward_mode == "margin_only":
                    rn, rw = ["group_margin"], [1.0]
                elif group_reward_mode == "label_match":
                    # Deprecated; fallback to margin_only
                    rn, rw = ["group_margin"], [1.0]

                r_i = compose_reward(
                    gt_label=gt_label,
                    pred_label=None,
                    summary_lines=context_variant,
                    reason=None,
                    checklist_lines=checklist,
                    reward_names=rn,
                    reward_weights=rw,
                    tf_p_pass=prob_pass,
                    tf_p_fail=prob_fail,
                    mission=mission,
                )
                cand_token_ids.append(toks)
                rewards_i.append(float(r_i))
                # Margin for gating diagnostic
                try:
                    m = decision_margin(
                        enc_b_var, processor, train_model, length_norm=length_norm
                    )
                except Exception:
                    m = 0.0
                margins_i.append(float(m))
                # record candidate for diagnostics
                try:
                    ka_cands.append(
                        {
                            "text": raw,
                            "reward": float(r_i),
                            "margin": float(m),
                        }
                    )
                except Exception:
                    pass

            # Compute z‑score advantages and duplicate detection (Stage-A, first pass)
            with torch.no_grad():
                rewards_t = torch.tensor(rewards_i, dtype=torch.float32, device=dev)
                mean = rewards_t.mean()
                std = rewards_t.std(unbiased=False)
                try:
                    import statistics as _st

                    _std_val = float(
                        (_st.pvariance(rewards_i) ** 0.5) if len(rewards_i) > 1 else 0.0
                    )
                except Exception:
                    _std_val = float(std.item())
                try:
                    from logging import getLogger

                    _logger = getLogger("src_post.credit")
                    # identical raw captions across all K_A
                    _cap_counts: Dict[str, int] = {}
                    for _cap in [
                        decode_to_text(processor, t).strip() for t in cand_token_ids
                    ]:
                        _cap_counts[_cap] = _cap_counts.get(_cap, 0) + 1
                    if len(_cap_counts) == 1 and len(cand_token_ids) >= max(1, K_A):
                        _logger.warning(
                            "[stage-a/sample] All sampled captions identical across K_A (first pass); std may be ~0"
                        )
                except Exception:
                    pass
                if float(std.item()) < 1e-6:
                    adv_i = torch.zeros_like(rewards_t)
                else:
                    adv_i = (rewards_t - mean) / (std + 1e-6)
                adv_i = torch.clamp(
                    adv_i,
                    min=-float(tf_cfg.get("adv_clip", 1.5)),
                    max=float(tf_cfg.get("adv_clip", 1.5)),
                )
                # Track best single margin improvement vs baseline
                try:
                    best_margin = max(margins_i) if len(margins_i) > 0 else 0.0
                    delta = float(best_margin - baseline_margin)
                    deltas.append((delta, i))
                    if delta > best_single_delta_val:
                        best_single_delta_val = delta
                except Exception as e:
                    raise RuntimeError(f"Failed computing best_single_delta: {e}")

            # Mark per-image storage for later gating; defer backprop until top-M selection
            if i == len(capped_images) - 1:
                pass  # placeholder; next loop will perform backprop with selection
            # Append diagnostics for this image
            try:
                diag_stage_a_samples.append(
                    {
                        "image_index": int(i),
                        "candidates": ka_cands,
                    }
                )
            except Exception:
                pass

        # Select top-M images for backprop
        top_m = max(0, int(tf_cfg.get("top_m", 0)))
        selected_idx = (
            set(range(len(capped_images)))
            if top_m == 0
            else set(
                [j for _, j in sorted(deltas, key=lambda x: x[0], reverse=True)[:top_m]]
            )
        )

        # Second pass: regenerate and backprop only for selected images
        for i, img in enumerate(capped_images):
            if i not in selected_idx:
                continue
            messages = conv_builder.build_stage_a_messages(mission)
            text_a = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            img_proc = sft_style_preprocess_image(img)
            GroupQCConversationBuilder.validate_typed_image_count(messages, 1)
            enc_a = processor(
                text=[text_a], images=[img_proc], padding=True, return_tensors="pt"
            )
            enc_a = to_device_and_cast(enc_a, device)

            cand_token_ids = []
            margins_i = []
            rewards_i = []

            for _ in range(max(1, K_A)):
                with torch.no_grad():
                    out = policy.generate(
                        **enc_a,
                        generation_config=gen_cfg_a,
                        logits_processor=logits_processors,
                        stopping_criteria=stopping,
                    )
                new_ids = out[:, enc_a["input_ids"].size(1) :]
                toks = new_ids[0].tolist()
                raw = decode_to_text(processor, toks).strip()

                # Build Stage‑B variant
                context_variant = list(context_lines)
                context_variant[i] = raw
                if use_mission_checklist:
                    msgs_var = conv_builder.build_stage_b_messages(
                        summary_lines=context_variant, checklist_lines=checklist
                    )
                else:
                    msgs_var = conv_builder.build_stage_b_messages_minimal(
                        summary_lines=context_variant
                    )
                text_b_var = processor.apply_chat_template(
                    conversation=msgs_var, tokenize=False, add_generation_prompt=True
                )
                enc_b_var = processor(
                    text=[text_b_var], images=None, return_tensors="pt", padding=True
                )
                enc_b_var = to_device_and_cast(enc_b_var, device)

                try:
                    prob_pass, prob_fail = tf_decision_probs(
                        train_model, enc_b_var, processor, length_norm=length_norm
                    )
                except Exception:
                    prob_pass, prob_fail = 0.0, 0.0

                rn, rw = reward_names, reward_weights
                if group_reward_mode == "margin_only":
                    rn, rw = ["group_margin"], [1.0]
                elif group_reward_mode == "label_match":
                    # Deprecated; fallback to margin_only
                    rn, rw = ["group_margin"], [1.0]

                r_i = compose_reward(
                    gt_label=gt_label,
                    pred_label=None,
                    summary_lines=context_variant,
                    reason=None,
                    checklist_lines=checklist,
                    reward_names=rn,
                    reward_weights=rw,
                    tf_p_pass=prob_pass,
                    tf_p_fail=prob_fail,
                    mission=mission,
                )
                cand_token_ids.append(toks)
                rewards_i.append(float(r_i))
                try:
                    m = decision_margin(
                        enc_b_var, processor, train_model, length_norm=length_norm
                    )
                except Exception:
                    m = 0.0
                margins_i.append(float(m))

            with torch.no_grad():
                rewards_t = torch.tensor(rewards_i, dtype=torch.float32, device=dev)
                mean = rewards_t.mean()
                std = rewards_t.std(unbiased=False)
                # Duplicate detection for Stage-A (second pass)
                try:
                    from logging import getLogger

                    _logger = getLogger("src_post.credit")
                    _cap_counts2: Dict[str, int] = {}
                    for _cap in [
                        decode_to_text(processor, t).strip() for t in cand_token_ids
                    ]:
                        _cap_counts2[_cap] = _cap_counts2.get(_cap, 0) + 1
                    if len(_cap_counts2) == 1 and len(cand_token_ids) >= max(1, K_A):
                        _logger.warning(
                            "[stage-a/sample] All sampled captions identical across K_A (second pass); std may be ~0"
                        )
                except Exception:
                    pass
                if float(std.item()) < 1e-6:
                    adv_i = torch.zeros_like(rewards_t)
                else:
                    adv_i = (rewards_t - mean) / (std + 1e-6)
                adv_i = torch.clamp(
                    adv_i,
                    min=-float(tf_cfg.get("adv_clip", 1.5)),
                    max=float(tf_cfg.get("adv_clip", 1.5)),
                )

            for k in range(len(cand_token_ids)):
                logp_a_k, cur_logits_a_k, prompt_len_a_k = (
                    tf_sum_logprob_and_logits_over_response(
                        train_model, enc_a, cand_token_ids[k], length_norm
                    )
                )
                if use_uncertainty_gate:
                    try:
                        B, Lm1, _V = cur_logits_a_k.size()
                        mask = torch.zeros(
                            (B, Lm1), dtype=torch.bool, device=cur_logits_a_k.device
                        )
                        mask[:, int(prompt_len_a_k) - 1 :] = True
                        ent = sequence_entropy_from_logits(cur_logits_a_k, mask)
                        entropy_vals.append(float(ent))
                        if not (
                            ent > entropy_threshold and margins_i[k] > baseline_margin
                        ):
                            decay = float(
                                tf_cfg.get("uncertainty_decay_factor", 0.0) or 0.0
                            )
                            if decay <= 0.0:
                                adv_i[k] = torch.tensor(
                                    0.0, device=adv_i.device, dtype=adv_i.dtype
                                )
                            else:
                                adv_i[k] = decay * adv_i[k]
                    except Exception:
                        pass
                term_a = -(adv_i[k].detach()) * logp_a_k
                if use_ref_kl and (ref_model is not None) and (lambda_kl_stage_a > 0.0):
                    kl_a_k = kl_to_ref_with_cur_logits(
                        ref_model,
                        enc_a,
                        cand_token_ids[k],
                        cur_logits_a_k,
                        int(prompt_len_a_k),
                    )
                    term_a = term_a + float(lambda_kl_stage_a) * kl_a_k
                (term_a * accum_scale).backward()
                loss_a = loss_a + term_a.detach()

            for k in range(len(cand_token_ids)):
                logp_a_k, cur_logits_a_k, prompt_len_a_k = (
                    tf_sum_logprob_and_logits_over_response(
                        train_model, enc_a, cand_token_ids[k], length_norm
                    )
                )
                # Optional entropy gating
                if use_uncertainty_gate:
                    try:
                        # Mask response positions
                        B, Lm1, _V = cur_logits_a_k.size()
                        mask = torch.zeros(
                            (B, Lm1), dtype=torch.bool, device=cur_logits_a_k.device
                        )
                        mask[:, int(prompt_len_a_k) - 1 :] = True
                        ent = sequence_entropy_from_logits(cur_logits_a_k, mask)
                        entropy_vals.append(float(ent))
                        # If candidate not clearly cautious/informative, decay or zero advantage
                        if not (
                            ent > entropy_threshold and margins_i[k] > baseline_margin
                        ):
                            decay = float(
                                tf_cfg.get("uncertainty_decay_factor", 0.0) or 0.0
                            )
                            if decay <= 0.0:
                                adv_i[k] = torch.tensor(
                                    0.0, device=adv_i.device, dtype=adv_i.dtype
                                )
                            else:
                                adv_i[k] = decay * adv_i[k]
                    except Exception:
                        pass
                term_a = -(adv_i[k].detach()) * logp_a_k
                if use_ref_kl and (ref_model is not None) and (lambda_kl_stage_a > 0.0):
                    kl_a_k = kl_to_ref_with_cur_logits(
                        ref_model,
                        enc_a,
                        cand_token_ids[k],
                        cur_logits_a_k,
                        int(prompt_len_a_k),
                    )
                    term_a = term_a + float(lambda_kl_stage_a) * kl_a_k
                (term_a * accum_scale).backward()
                loss_a = loss_a + term_a.detach()

        diags: Dict[str, Any] = {}
        if entropy_vals:
            diags["phase_a_entropy_mean"] = float(
                sum(entropy_vals) / max(1, len(entropy_vals))
            )
        diags["best_single_delta"] = float(best_single_delta_val)
        # Attach Stage-A candidates (first pass) for logging to results JSONL
        try:
            diags["stage_a_candidates"] = diag_stage_a_samples
        except Exception:
            pass
        return loss_a, diags


class PairwiseFallbackAssigner(BaseCreditAssigner):
    def compute_loss_a(
        self,
        policy: DetectionModel,
        train_model: torch.nn.Module,
        processor: Qwen2VLProcessor,
        conv_builder: Any,
        images: List[Image.Image],
        context_lines: List[str],
        checklist: List[str],
        mission: Optional[str],
        gt_label: str,
        enc_b_baseline: Dict[str, torch.Tensor],
        tf_cfg: Dict[str, Any],
        gen_cfg_a: GenerationConfig,
        reward_cfg: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device: str = str(tf_cfg.get("device"))
        length_norm: bool = bool(tf_cfg.get("length_norm", True))
        logits_processors: LogitsProcessorList = tf_cfg.get("logits_processors")
        stopping = tf_cfg.get("stopping")
        ddp_policy = tf_cfg.get("ddp_policy")
        pairs_per_group: int = int(tf_cfg.get("pairs_per_group", 1))
        accum_scale: float = float(tf_cfg.get("accum_scale", 1.0))

        use_mission_checklist: bool = bool(
            reward_cfg.get("use_mission_checklist", True)
        )
        use_ref_kl: bool = bool(reward_cfg.get("use_ref_kl", False))
        ref_model: Optional[DetectionModel] = reward_cfg.get("ref_model")
        lambda_kl_stage_a: float = float(reward_cfg.get("lambda_kl_stage_a", 0.0))

        # Baseline margin
        baseline_margin = decision_margin(
            enc_b_baseline, processor, train_model, length_norm=length_norm
        )

        dev = torch.device(device)
        loss_a = torch.tensor(0.0, dtype=torch.float32, device=dev)

        # Deterministic pairs list i<j up to budget
        n = len(images)
        if n < 2 or pairs_per_group <= 0:
            return loss_a, {"pairwise_triggered": 0.0}
        pairs: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        pairs = pairs[: min(pairs_per_group, len(pairs))]
        if not pairs:
            return loss_a, {"pairwise_triggered": 0.0}

        total_pairs_used = 0
        import math as _math

        eps = 1e-9
        r_baseline = _math.log(
            max(eps, float(reward_cfg.get("baseline_tf_p_pass", 0.0))) + eps
        ) - _math.log(max(eps, float(reward_cfg.get("baseline_tf_p_fail", 0.0))) + eps)
        for i, j in pairs:
            # i
            msgs_i = conv_builder.build_stage_a_messages(mission)
            text_ai = processor.apply_chat_template(
                msgs_i, tokenize=False, add_generation_prompt=True
            )
            GroupQCConversationBuilder.validate_typed_image_count(msgs_i, 1)
            enc_ai = processor(
                text=[text_ai],
                images=[sft_style_preprocess_image(images[i])],
                padding=True,
                return_tensors="pt",
            )
            enc_ai = to_device_and_cast(enc_ai, device)
            with torch.no_grad():
                out_i = policy.generate(
                    **enc_ai,
                    generation_config=gen_cfg_a,
                    logits_processor=logits_processors,
                    stopping_criteria=stopping,
                )
            toks_i = out_i[:, enc_ai["input_ids"].size(1) :][0].tolist()
            cap_i = decode_to_text(processor, toks_i).strip()

            # j
            msgs_j = conv_builder.build_stage_a_messages(mission)
            text_aj = processor.apply_chat_template(
                msgs_j, tokenize=False, add_generation_prompt=True
            )
            GroupQCConversationBuilder.validate_typed_image_count(msgs_j, 1)
            enc_aj = processor(
                text=[text_aj],
                images=[sft_style_preprocess_image(images[j])],
                padding=True,
                return_tensors="pt",
            )
            enc_aj = to_device_and_cast(enc_aj, device)
            with torch.no_grad():
                out_j = policy.generate(
                    **enc_aj,
                    generation_config=gen_cfg_a,
                    logits_processor=logits_processors,
                    stopping_criteria=stopping,
                )
            toks_j = out_j[:, enc_aj["input_ids"].size(1) :][0].tolist()
            cap_j = decode_to_text(processor, toks_j).strip()

            # Build Stage‑B with both replaced
            context_pair = list(context_lines)
            context_pair[i] = cap_i
            context_pair[j] = cap_j
            if use_mission_checklist:
                msgs_b = conv_builder.build_stage_b_messages(
                    summary_lines=context_pair, checklist_lines=checklist
                )
            else:
                msgs_b = conv_builder.build_stage_b_messages_minimal(
                    summary_lines=context_pair
                )
            text_b = processor.apply_chat_template(
                conversation=msgs_b, tokenize=False, add_generation_prompt=True
            )
            enc_b = processor(
                text=[text_b], images=None, return_tensors="pt", padding=True
            )
            enc_b = to_device_and_cast(enc_b, device)

            try:
                p_pass, p_fail = tf_decision_probs(
                    train_model, enc_b, processor, length_norm=length_norm
                )
            except Exception:
                p_pass, p_fail = 0.0, 0.0

            r_pair = _math.log(p_pass + eps) - _math.log(p_fail + eps)
            rt = torch.tensor([r_baseline, r_pair], dtype=torch.float32, device=dev)
            mean = rt.mean()
            std = rt.std(unbiased=False)
            if float(std.item()) < 1e-6:
                adv_pair = torch.tensor(0.0, dtype=torch.float32, device=dev)
            else:
                advs = (rt - mean) / (std + 1e-6)
                adv_pair = advs[1]

            # Half advantage to each summary
            for enc_a_cand, toks_cand in ((enc_ai, toks_i), (enc_aj, toks_j)):
                logp_k, cur_logits_k, prompt_len_k = (
                    tf_sum_logprob_and_logits_over_response(
                        train_model, enc_a_cand, toks_cand, length_norm
                    )
                )
                term = -(0.5 * adv_pair.detach()) * logp_k
                if use_ref_kl and (ref_model is not None) and (lambda_kl_stage_a > 0.0):
                    kl_k = kl_to_ref_with_cur_logits(
                        ref_model,
                        enc_a_cand,
                        toks_cand,
                        cur_logits_k,
                        int(prompt_len_k),
                    )
                    term = term + float(lambda_kl_stage_a) * kl_k
                (term * accum_scale).backward()
                loss_a = loss_a + term.detach()
            total_pairs_used += 1

        return loss_a, {"pairwise_triggered": 1.0 if total_pairs_used > 0 else 0.0}


def get_credit_assigner(mode: str) -> BaseCreditAssigner:
    m = str(mode).strip().lower()
    if m == "off":
        return OffAssigner()
    if m == "conditional":
        return ConditionalAssigner()
    # Placeholder for 'joint' or others
    return OffAssigner()
