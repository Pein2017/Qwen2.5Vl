#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core GRPO group-processing loop extracted from runner."""

from __future__ import annotations

import logging
import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src_post.credit.credit_assignment import get_credit_assigner
from src_post.pipeline import (
    StageADiagnostics,
    StageASampler,
    StageBResult,
    StageBSampler,
)
from src_post.pipeline.dto import (
    StageAResult,
    StageBCandidate,
    StageBSamplingDiagnostics,
)
from src_post.rewards.utils import l1_normalize
from src_post.tf.teacher_forcing import (
    kl_to_ref_with_cur_logits,
    tf_decision_probs,
    tf_sum_logprob_and_logits_over_response,
)
from src_post.training.grpo_trainer import GRPOTrainer


@dataclass
class GroupMetrics:
    total_loss: float = 0.0
    total_items: int = 0
    reward_best_sum: float = 0.0
    reward_best_sq_sum: float = 0.0
    best_hit_count: float = 0.0
    any_hit_count: float = 0.0
    resp_len_sum: float = 0.0
    grad_norm_sum: float = 0.0
    kl_b_sum: float = 0.0
    kl_b_count: float = 0.0
    diag_fmt_sum: float = 0.0
    diag_cov_sum: float = 0.0
    diag_cln_sum: float = 0.0
    diag_tax_sum: float = 0.0
    diag_cons_sum: float = 0.0
    skip_std0_count: float = 0.0
    pairwise_trigger_count: float = 0.0
    entropy_sum: float = 0.0
    margin_best_sum: float = 0.0
    flip_hit_count: float = 0.0
    decision_ce_sum: float = 0.0
    decision_ce_count: float = 0.0
    fn_count: float = 0.0
    gt_fail_count: float = 0.0
    sb_clip_low_sum: float = 0.0
    sb_clip_high_sum: float = 0.0
    sb_clip_region_sum: float = 0.0
    sb_clip_den: float = 0.0
    sb_entropy_mask_true_sum: float = 0.0
    sb_reply_token_sum: float = 0.0

    def add_inplace(self, other: "GroupMetrics") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def to_list(self) -> List[float]:
        return [
            self.total_items,
            self.total_loss,
            self.reward_best_sum,
            self.reward_best_sq_sum,
            self.best_hit_count,
            self.any_hit_count,
            self.resp_len_sum,
            self.grad_norm_sum,
            self.kl_b_sum,
            self.kl_b_count,
            self.diag_fmt_sum,
            self.diag_cov_sum,
            self.diag_cln_sum,
            self.diag_tax_sum,
            self.diag_cons_sum,
            self.skip_std0_count,
            self.pairwise_trigger_count,
            self.entropy_sum,
            self.margin_best_sum,
            self.flip_hit_count,
            self.decision_ce_sum,
            self.decision_ce_count,
            self.fn_count,
            self.gt_fail_count,
        ]


@dataclass
class GroupTrainOutput:
    metrics: GroupMetrics
    stage_b_result: StageBResult
    record: Optional[Dict[str, Any]]
    context_lines: List[str]
    grad_norm: Optional[float]
    pairwise_triggered: float
    dt_group: float


logger = logging.getLogger("src_post.training.grpo_loop")


class GRPOLoop:
    """Encapsulates per-group GRPO training operations."""

    def __init__(
        self,
        *,
        cfg,
        device: str,
        trainer: GRPOTrainer,
        stage_a_sampler: StageASampler,
        stage_b_sampler: StageBSampler,
        conv_builder,
        processor,
        policy,
        train_model,
        ddp_policy: Optional[DDP],
        ref_model,
        gen_cfg_stage_a,
        gen_cfg_stage_b,
        logits_processors,
        stopping_stage_a,
        tb_writer=None,
        debug_verbose: bool = False,
        rank: int = 0,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.trainer = trainer
        self.stage_a_sampler = stage_a_sampler
        self.stage_b_sampler = stage_b_sampler
        self.conv_builder = conv_builder
        self.processor = processor
        self.policy = policy
        self.train_model = train_model
        self.ddp_policy = ddp_policy
        self.ref_model = ref_model
        self.gen_cfg_stage_a = gen_cfg_stage_a
        self.gen_cfg_stage_b = gen_cfg_stage_b
        self.logits_processors = logits_processors
        self.stopping_stage_a = stopping_stage_a
        self.tb_writer = tb_writer
        self.debug_verbose = debug_verbose
        self.rank = rank

    def _compute_stage_b_diagnostics(
        self, candidates: List[StageBCandidate]
    ) -> StageBSamplingDiagnostics:
        if not candidates:
            return StageBSamplingDiagnostics(
                reward_mean=0.0,
                reward_std=0.0,
                duplicate_count=0,
                raw_texts=[],
                reasons=[],
            )

        rewards = [float(c.reward) for c in candidates]
        texts = [c.text for c in candidates]
        reasons = [c.reason or "" for c in candidates]
        mean_val = statistics.mean(rewards)
        std_val = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        dup_count = len(texts) - len(set(texts))
        return StageBSamplingDiagnostics(
            reward_mean=float(mean_val),
            reward_std=float(std_val),
            duplicate_count=int(dup_count),
            raw_texts=texts,
            reasons=reasons,
        )

    def process_group(
        self,
        *,
        idx: int,
        sample: Dict[str, Any],
        accum_scale: float,
        debug_verbose: bool,
        sync_now: bool,
    ) -> GroupTrainOutput:
        cfg = self.cfg
        device = self.device
        trainer = self.trainer
        ddp_policy = self.ddp_policy

        metrics = GroupMetrics()
        grp_start_ts = time.time()

        images: List[Any] = sample["images"]
        gt_label: str = str(sample["label"]).strip().lower()

        stage_a_result: StageAResult = self.stage_a_sampler.generate(
            images=images,
            enable_diagnostics=bool(cfg.enable_phase_a_diagnostics),
        )
        stage_a_result.validate(len(images))
        context_lines = stage_a_result.summary_lines
        checklist = stage_a_result.checklist_lines
        if self.rank == 0:
            try:
                img_names = [Path(p).name for p in sample.get("image_paths", [])]
            except Exception:
                img_names = []

            def _short(s: str, m: int = 120) -> str:
                s = str(s).strip()
                return s if len(s) <= m else (s[: max(0, m - 1)].rstrip() + "…")

            logger.info(
                f"[stage-a] step={self.trainer.global_step} idx={idx} summaries (N={len(context_lines)}):"
            )
            for i, line in enumerate(context_lines):
                name = img_names[i] if i < len(img_names) else f"img{i}"
                logger.info(f"  ({i}) {name}: {_short(line)}")

        if bool(cfg.enable_phase_a_diagnostics):
            diagnostics = stage_a_result.diagnostics or StageADiagnostics()
            metrics.diag_fmt_sum += diagnostics.get("formatting")
            metrics.diag_cov_sum += diagnostics.get("coverage")
            metrics.diag_cln_sum += diagnostics.get("cleanliness")
            metrics.diag_tax_sum += diagnostics.get("taxonomy")
            metrics.diag_cons_sum += diagnostics.get("consistency")
            item_diag = diagnostics.item_level
        else:
            diagnostics = None
            item_diag = None

        if bool(cfg.use_mission_checklist):
            msgs_b = self.conv_builder.build_stage_b_messages(
                summary_lines=context_lines,
                checklist_lines=checklist,
            )
        else:
            msgs_b = self.conv_builder.build_stage_b_messages_minimal(
                summary_lines=context_lines
            )
        text_b = self.processor.apply_chat_template(
            conversation=msgs_b, tokenize=False, add_generation_prompt=True
        )
        self.conv_builder.validate_image_placeholder_count(text_b, 0)
        enc_b = self.processor(
            text=[text_b], images=None, return_tensors="pt", padding=True
        )
        for key, value in enc_b.items():
            if isinstance(value, torch.Tensor):
                enc_b[key] = value.to(device)

        tf_p_pass, tf_p_fail = tf_decision_probs(
            self.train_model, enc_b, self.processor, length_norm=bool(cfg.length_norm)
        )
        eps = 1e-9
        margin_best = float(
            math.log(float(tf_p_pass) + eps) - math.log(float(tf_p_fail) + eps)
        )
        metrics.margin_best_sum += margin_best
        decision_ce = (
            -math.log(float(tf_p_pass) + eps)
            if gt_label == "pass"
            else -math.log(float(tf_p_fail) + eps)
        )
        metrics.decision_ce_sum += float(decision_ce)
        metrics.decision_ce_count += 1.0

        prefix_fn = None
        try:
            from src_post.generation.generation import build_decision_prefix_constraint

            prompt_len_b = int(enc_b["input_ids"].size(1))
            prefix_fn = build_decision_prefix_constraint(self.processor, prompt_len_b)
        except Exception:
            prefix_fn = None

        raw_reward_weights = list(cfg.reward_weights)
        normalized_reward_weights = l1_normalize(raw_reward_weights)

        stage_b_output = self.stage_b_sampler.sample_and_score(
            enc_b=enc_b,
            gt_label=gt_label,
            context_lines=context_lines,
            checklist_lines=checklist,
            tf_p_pass=tf_p_pass,
            tf_p_fail=tf_p_fail,
            prefix_allowed_tokens_fn=prefix_fn,
            baseline_margin=margin_best,
            decision_ce=decision_ce,
        )
        stage_b_candidates = list(stage_b_output.candidates)
        total_stage_b_latency = float(stage_b_output.latency_sampling_sec)

        rewards_b = [float(c.reward) for c in stage_b_candidates]
        replies_text = [c.text for c in stage_b_candidates]
        replies_token_ids = [list(c.token_ids) for c in stage_b_candidates]
        pred_labels = [c.label for c in stage_b_candidates]
        reasons = [c.reason for c in stage_b_candidates]

        rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device)
        if rewards_t.numel() > 0 and float(rewards_t.std(unbiased=False).item()) < 1e-6:
            attempts = int(getattr(cfg, "max_resample_times", 0) or 0)
            tried = 0
            while tried < attempts:
                tried += 1
                extra = self.stage_b_sampler.sample_and_score(
                    enc_b=enc_b,
                    gt_label=gt_label,
                    context_lines=context_lines,
                    checklist_lines=checklist,
                    tf_p_pass=tf_p_pass,
                    tf_p_fail=tf_p_fail,
                    prefix_allowed_tokens_fn=prefix_fn,
                    baseline_margin=margin_best,
                    decision_ce=decision_ce,
                    num_samples=1,
                )
                if not extra.candidates:
                    break
                stage_b_candidates.extend(extra.candidates)
                total_stage_b_latency += float(extra.latency_sampling_sec)
                rewards_b = [float(c.reward) for c in stage_b_candidates]
                replies_text = [c.text for c in stage_b_candidates]
                replies_token_ids = [list(c.token_ids) for c in stage_b_candidates]
                pred_labels = [c.label for c in stage_b_candidates]
                reasons = [c.reason for c in stage_b_candidates]
                rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device)
                if float(rewards_t.std(unbiased=False).item()) >= 1e-6:
                    break

        generation_logps = [
            torch.tensor(c.generation_logps, dtype=torch.float32, device=device)
            if c.generation_logps is not None
            else None
            for c in stage_b_candidates
        ]

        diag_stage_b = self._compute_stage_b_diagnostics(stage_b_candidates)

        if rewards_b:
            best_idx = int(max(range(len(rewards_b)), key=lambda i: rewards_b[i]))
            best_reward = float(rewards_b[best_idx])
            best_label = pred_labels[best_idx] if len(pred_labels) > best_idx else None
        else:
            best_idx = 0
            best_reward = 0.0
            best_label = None

        metrics.reward_best_sum += best_reward
        metrics.reward_best_sq_sum += best_reward * best_reward
        metrics.best_hit_count += float(
            isinstance(best_label, str) and best_label == gt_label
        )
        any_hit = float(
            any((lbl == gt_label) for lbl in pred_labels if isinstance(lbl, str))
        )
        metrics.any_hit_count += any_hit
        metrics.flip_hit_count += any_hit
        metrics.gt_fail_count += 1.0 if gt_label == "fail" else 0.0
        if isinstance(best_label, str) and best_label == "pass" and gt_label == "fail":
            metrics.fn_count += 1.0
        if replies_token_ids:
            metrics.resp_len_sum += float(len(replies_token_ids[best_idx]))

        with torch.no_grad():
            if rewards_t.numel() == 0:
                rewards_t = torch.tensor([0.0], device=device, dtype=torch.float32)
            mean = rewards_t.mean()
            std = rewards_t.std(unbiased=False)
            if float(std.item()) < 1e-6:
                adv = torch.zeros_like(rewards_t)
                metrics.skip_std0_count += 1.0
            else:
                adv = (rewards_t - mean) / (std + 1e-6)
            adv = torch.clamp(adv, min=-cfg.adv_clip, max=cfg.adv_clip)

        loss_b_scalar = torch.tensor(0.0, dtype=torch.float32, device=device)
        for k in range(len(replies_token_ids)):
            logp_k, cur_logits_k, prompt_len_k = (
                tf_sum_logprob_and_logits_over_response(
                    self.train_model,
                    enc_b,
                    replies_token_ids[k],
                    cfg.length_norm,
                )
            )
            if bool(getattr(cfg, "enable_clipped_grpo", False)):
                from src_post.tf.grpo_loss import (
                    apply_entropy_mask_from_logits,
                    build_reply_mask,
                    compute_ratio_and_clip,
                    per_token_logps_from_logits,
                    reduce_loss,
                )

                prompt_ids = enc_b["input_ids"].to(cur_logits_k.device)
                reply_ids = torch.tensor(
                    [replies_token_ids[k]],
                    dtype=prompt_ids.dtype,
                    device=prompt_ids.device,
                )
                full_ids = torch.cat([prompt_ids, reply_ids], dim=1)
                targets = full_ids[:, 1:]
                reply_mask = build_reply_mask(int(prompt_len_k), cur_logits_k)
                cur_logps = per_token_logps_from_logits(cur_logits_k, targets)
                gen_logps_k = generation_logps[k]
                if gen_logps_k is not None:
                    old_raw = gen_logps_k.to(
                        device=cur_logps.device, dtype=cur_logps.dtype
                    )
                    if old_raw.ndim == 2 and old_raw.shape == cur_logps.shape:
                        old_logps = old_raw
                    elif old_raw.ndim == 1:
                        reply_tokens = int(reply_mask.sum().item())
                        if old_raw.numel() != reply_tokens:
                            raise ValueError(
                                "Generation log-prob count mismatch: "
                                f"expected {reply_tokens} reply tokens but got {old_raw.numel()}."
                            )
                        old_logps = cur_logps.detach().clone()
                        old_logps[reply_mask] = old_raw
                    else:
                        raise ValueError(
                            f"Unsupported generation_logps shape {tuple(old_raw.shape)}; "
                            "expected either [B,L] or [num_reply_tokens]."
                        )
                else:
                    old_logps = cur_logps.detach()
                coef_1, coef_2 = compute_ratio_and_clip(
                    cur_logps,
                    old_logps,
                    float(cfg.epsilon_low),
                    float(getattr(cfg, "epsilon_high", 0.0) or 0.0),
                )
                adv_scalar = float(adv[k].detach()) if k < adv.size(0) else 0.0
                adv_tok = torch.full_like(cur_logps, fill_value=adv_scalar)
                per_token_loss = -torch.minimum(coef_1 * adv_tok, coef_2 * adv_tok)
                if bool(getattr(cfg, "enable_entropy_mask_stage_b", False)):
                    top_q = getattr(cfg, "entropy_top_quantile_stage_b", None)
                    min_thr = getattr(cfg, "entropy_min_threshold_stage_b", None)
                    ent_mask = apply_entropy_mask_from_logits(
                        cur_logits_k,
                        reply_mask,
                        top_quantile=top_q,
                        min_threshold=min_thr,
                    )
                    per_token_loss = per_token_loss * ent_mask.to(per_token_loss.dtype)
                    metrics.sb_entropy_mask_true_sum += float(ent_mask.sum().item())
                term = reduce_loss(per_token_loss, reply_mask, keep_batch=True).sum()
                with torch.no_grad():
                    low = 1.0 - float(cfg.epsilon_low)
                    high = 1.0 + float(getattr(cfg, "epsilon_high", 0.0) or 0.0)
                    reply_mask_bool = reply_mask
                    is_low = (coef_1 < low) & (adv_tok < 0)
                    is_high = (coef_1 > high) & (adv_tok > 0)
                    is_region = is_low | is_high
                    is_low = is_low & reply_mask_bool
                    is_high = is_high & reply_mask_bool
                    is_region = is_region & reply_mask_bool
                    metrics.sb_clip_low_sum += float(is_low.sum().item())
                    metrics.sb_clip_high_sum += float(is_high.sum().item())
                    metrics.sb_clip_region_sum += float(is_region.sum().item())
                    sb_den_inc = float(reply_mask_bool.sum().item())
                    metrics.sb_clip_den += sb_den_inc
                    metrics.sb_reply_token_sum += sb_den_inc
            else:
                term = -(adv[k].detach()) * logp_k
            if (
                cfg.use_ref_kl
                and self.ref_model is not None
                and cfg.lambda_kl_stage_b > 0.0
            ):
                kl_k = kl_to_ref_with_cur_logits(
                    self.ref_model,
                    enc_b,
                    replies_token_ids[k],
                    cur_logits_k,
                    int(prompt_len_k),
                )
                metrics.kl_b_sum += float(kl_k.detach().item())
                metrics.kl_b_count += 1.0
                term = term + (cfg.lambda_kl_stage_b * kl_k)
            if bool(cfg.train_stage_b) and trainer.global_step >= int(
                cfg.freeze_stage_b_steps
            ):
                scale_b = float(cfg.stage_b_weight)
                with (
                    ddp_policy.no_sync()
                    if (ddp_policy is not None and not sync_now)
                    else nullcontext()
                ):
                    (term * (accum_scale * scale_b)).backward()
            loss_b_scalar = loss_b_scalar + term.detach()

        loss_a = torch.tensor(0.0, dtype=torch.float32, device=device)
        diags_a: Optional[Dict[str, Any]] = None
        pairwise_triggered_flag = 0
        pairwise_reason: Optional[str] = None
        assigner = get_credit_assigner(cfg.train_stage_a_mode)
        if cfg.train_stage_a_mode in {"conditional", "joint"}:
            tf_cfg = {
                "device": device,
                "length_norm": bool(cfg.length_norm),
                "logits_processors": self.logits_processors,
                "stopping": self.stopping_stage_a,
                "K_A": int(cfg.K_A),
                "max_images_tf": int(cfg.max_images_tf),
                "adv_clip": float(cfg.adv_clip),
                "ddp_policy": ddp_policy,
                "pairs_per_group": int(cfg.pairwise_pairs_per_group),
                "top_m": int(getattr(cfg, "stage_a_top_m", 0) or 0),
                "uncertainty_decay_factor": float(
                    getattr(cfg, "uncertainty_decay_factor", 0.0) or 0.0
                ),
                "accum_scale": float(accum_scale * float(cfg.stage_a_weight)),
            }
            reward_cfg = {
                "names": list(cfg.reward_fns),
                "weights": list(normalized_reward_weights),
                "raw_weights": raw_reward_weights,
                "group_reward_mode": str(cfg.group_reward_mode),
                "use_ref_kl": bool(cfg.use_ref_kl),
                "ref_model": self.ref_model,
                "lambda_kl_stage_a": float(cfg.lambda_kl_stage_a),
                "use_uncertainty_gate": bool(cfg.use_uncertainty_gate),
                "entropy_threshold": float(cfg.uncertainty_gate_min_entropy),
                "use_mission_checklist": bool(cfg.use_mission_checklist),
                "baseline_tf_p_pass": float(tf_p_pass),
                "baseline_tf_p_fail": float(tf_p_fail),
            }
            with (
                ddp_policy.no_sync()
                if (ddp_policy is not None and not sync_now)
                else nullcontext()
            ):
                loss_a, diags_a = assigner.compute_loss_a(
                    policy=self.policy,
                    train_model=self.train_model,
                    processor=self.processor,
                    conv_builder=self.conv_builder,
                    images=images,
                    context_lines=context_lines,
                    checklist=checklist,
                    mission=cfg.mission,
                    gt_label=gt_label,
                    enc_b_baseline=enc_b,
                    tf_cfg=tf_cfg,
                    gen_cfg_a=self.gen_cfg_stage_a,
                    reward_cfg=reward_cfg,
                )
            if isinstance(diags_a, dict) and "phase_a_entropy_mean" in diags_a:
                metrics.entropy_sum += float(diags_a.get("phase_a_entropy_mean", 0.0))
            if bool(cfg.pairwise_credit_enabled):
                try:
                    from src_post.credit.credit_assignment import (
                        PairwiseFallbackAssigner,
                    )

                    best_label_local = (
                        pred_labels[best_idx] if len(pred_labels) > best_idx else None
                    )
                    max_single_delta = (
                        float(diags_a.get("best_single_delta", 0.0))
                        if isinstance(diags_a, dict)
                        else 0.0
                    )
                    threshold = float(cfg.pairwise_delta_threshold)
                    if (
                        isinstance(best_label_local, str)
                        and best_label_local != gt_label
                        and int(cfg.pairwise_pairs_per_group) > 0
                        and (max_single_delta < threshold)
                    ):
                        pw = PairwiseFallbackAssigner()
                        with (
                            ddp_policy.no_sync()
                            if (ddp_policy is not None and not sync_now)
                            else nullcontext()
                        ):
                            loss_pw, diags_pw = pw.compute_loss_a(
                                policy=self.policy,
                                train_model=self.train_model,
                                processor=self.processor,
                                conv_builder=self.conv_builder,
                                images=images,
                                context_lines=context_lines,
                                checklist=checklist,
                                mission=cfg.mission,
                                gt_label=gt_label,
                                enc_b_baseline=enc_b,
                                tf_cfg=tf_cfg,
                                gen_cfg_a=self.gen_cfg_stage_a,
                                reward_cfg=reward_cfg,
                            )
                        loss_a = loss_a + loss_pw
                        if isinstance(diags_pw, dict):
                            pairwise_triggered_flag = int(
                                diags_pw.get("pairwise_triggered", 0.0)
                            )
                            pairwise_reason = str(
                                diags_pw.get("reason", "pairwise_triggered")
                            )
                except Exception as exc:
                    raise RuntimeError(f"Pairwise fallback failed: {exc}")
        else:
            diags_a = None

        num_images_in_group = int(sample.get("num_images", len(images)))
        stage_a_weight_eff = float(cfg.stage_a_weight) / max(1, num_images_in_group)
        loss = (cfg.stage_b_weight * loss_b_scalar) + (stage_a_weight_eff * loss_a)
        metrics.total_loss += float(loss.detach().item())
        metrics.total_items += 1
        metrics.pairwise_trigger_count += float(pairwise_triggered_flag)

        stage_b_result = StageBResult(
            candidates=stage_b_candidates,
            tf_p_pass=float(tf_p_pass),
            tf_p_fail=float(tf_p_fail),
            latency_sampling_sec=float(total_stage_b_latency),
            best_index=(
                int(
                    max(
                        range(len(stage_b_candidates)),
                        key=lambda i: stage_b_candidates[i].reward,
                    )
                )
                if stage_b_candidates
                else None
            ),
            baseline_margin=float(margin_best),
            decision_ce=float(decision_ce),
            diagnostics=diag_stage_b,
        )
        stage_b_result.validate()

        if self.rank == 0:
            try:
                raw_reasons = []
                for idx_txt, txt in enumerate(replies_text):
                    reason_val = reasons[idx_txt] if idx_txt < len(reasons) else None
                    if isinstance(reason_val, str) and reason_val.strip():
                        raw_reasons.append(reason_val.strip())
                        continue
                    lines = txt.splitlines()
                    reason_text = lines[1].strip() if len(lines) >= 2 else txt.strip()
                    if reason_text.startswith("原因:") or reason_text.startswith(
                        "原因："
                    ):
                        reason_text = reason_text[3:].strip()
                    raw_reasons.append(reason_text)
                n_cand = len(replies_text)
                dup = len(replies_text) - len(set(replies_text))
                dup_ratio = dup / n_cand if n_cand > 0 else 0.0
                labels_str = ", ".join(
                    [f"{p.label}:{p.reward:.3f}" for p in stage_b_candidates]
                )
                logger.info(
                    f"[stage-b/sample] step={self.trainer.global_step} idx={idx} K_B={n_cand} in {total_stage_b_latency:.2f}s "
                    f"| reward_mean={diag_stage_b.reward_mean:.3f} std={diag_stage_b.reward_std:.3f} "
                    f"| dup={dup}/{n_cand} ({dup_ratio:.2f}) | labels={labels_str}"
                )
                if bool(getattr(cfg, "log_all_candidates", False)):
                    for j, cand in enumerate(stage_b_candidates):
                        logger.info(
                            f"    cand[{j}]: label={cand.label} r={cand.reward:.3f} text={_short(cand.text, 200)}"
                        )
                # top/worst preview (truncated)
                if stage_b_result.best_candidate() is not None:
                    best = stage_b_result.best_candidate()
                    logger.info(
                        f"  top1: label={best.label} reward={best.reward:.3f} reason={best.trimmed_reason(120) or _short(best.text)}"
                    )
                if n_cand > 1:
                    order = sorted(
                        range(n_cand), key=lambda i: rewards_b[i], reverse=True
                    )
                    if len(order) > 1:
                        sec = stage_b_candidates[order[1]]
                        logger.info(
                            f"  top2: label={sec.label} reward={sec.reward:.3f} reason={sec.trimmed_reason(120) or _short(sec.text)}"
                        )
                    worst_i = min(range(n_cand), key=lambda i: rewards_b[i])
                    worst = stage_b_candidates[worst_i]
                    logger.info(
                        f"  worst: label={worst.label} reward={worst.reward:.3f} reason={worst.trimmed_reason(120) or _short(worst.text)}"
                    )
                if diag_stage_b.all_identical and n_cand >= max(1, int(cfg.K_B)):
                    logger.warning(
                        "[stage-b/sample] All sampled replies are identical across K_B; std ~0, learning signal weak"
                    )
                if debug_verbose and stage_b_result.best_candidate() is not None:
                    logger.info(
                        f"[debug] stage_b_raw_best={stage_b_result.best_candidate().text}"
                    )
            except Exception:
                pass

        record = {
            "group_index": idx,
            "group_id": sample.get("meta", {}).get("group_id", "-")
            if isinstance(sample, dict)
            else "-",
            "mission": cfg.mission,
            "gt_label": gt_label,
            "k_b": int(cfg.K_B),
            "k_a": int(cfg.K_A),
            "images": [
                {"image_id": Path(p).name} for p in sample.get("image_paths", [])
            ],
            "stage_b": {
                "best": {
                    "pred_label": stage_b_result.best_candidate().label
                    if stage_b_result.best_candidate()
                    else None,
                    "reward": stage_b_result.best_candidate().reward
                    if stage_b_result.best_candidate()
                    else 0.0,
                    "raw": stage_b_result.best_candidate().text
                    if stage_b_result.best_candidate()
                    else "",
                },
                "candidates": [
                    {
                        "raw": cand.text,
                        "reward": float(cand.reward),
                        "pred_label": cand.label,
                    }
                    for cand in stage_b_result.candidates
                ],
            },
            "stage_a": None,
            "used_minimal_prompt": not bool(cfg.use_mission_checklist),
            "pairwise_triggered": bool(pairwise_triggered_flag),
            "pairwise_reason": pairwise_reason,
        }

        if cfg.enable_phase_a_diagnostics and item_diag is not None:
            diag_slim = dict(item_diag)
            if "per_image" in diag_slim:
                for entry in diag_slim["per_image"]:
                    entry.pop("present_pass_tokens", None)
                    entry.pop("present_fail_tokens", None)
            record["diag_item_level"] = diag_slim
            try:
                from src_post.logging.diagnostics import classify_mismatch

                record["diag_mismatch"] = classify_mismatch(
                    gt_label=gt_label,
                    pred_label=record["stage_b"]["best"]["pred_label"],
                    diag=item_diag,
                )
            except Exception:
                pass

        if isinstance(diags_a, dict) and "stage_a_candidates" in diags_a:
            record["stage_a"] = diags_a["stage_a_candidates"]

        dt_group = time.time() - grp_start_ts

        return GroupTrainOutput(
            metrics=metrics,
            stage_b_result=stage_b_result,
            record=record,
            context_lines=context_lines,
            grad_norm=None,
            pairwise_triggered=float(pairwise_triggered_flag),
            dt_group=float(dt_group),
        )
