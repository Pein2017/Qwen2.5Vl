#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import GenerationConfig, Qwen2VLProcessor
from transformers.generation import LogitsProcessorList

from PIL import Image

from contextlib import nullcontext
from src_post.config import RLRunnerConfig, load_and_validate_config
from src_post.prompting.conversation import GroupQCConversationBuilder
from src_post.data.dataset_group_qc import RLGroupQCDataset
from src_post.generation.logits_processors import GeometryCoordMaskLogitsProcessor
from src_post.models import (
    apply_freeze_and_param_groups,
    load_detection_model,
    load_processor,
    wrap_ddp_if_needed,
)
from src_post.generation.generation import (
    build_stage_a_context_lines,
    build_stage_a_stopping,
    decode_to_text,
    to_device_and_cast,
)
from src_post.tf.teacher_forcing import (
    kl_to_ref_with_cur_logits,
    tf_decision_probs,
    tf_sum_logprob_and_logits_over_response,
)
from src_post.rewards.compose import compose_reward
from src_post.data.data_loader import build_epoch_indices, iter_batches
from src_post.logging.logging_utils import compute_eta, rank0_log, aggregate_training_metrics
from src_post.io.checkpoints import save_if_rank0
from src_post.logging.diagnostics import compute_phase_a_diagnostics
from src_post.credit.credit_assignment import get_credit_assigner

from src_new.models.wrapper import DetectionModel
from transformers.utils import logging as hf_logging


logger = logging.getLogger("src_post.runner")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def _get_dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def _maybe_init_ddp() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def _bind_device(device: str) -> str:
    rank, world_size, local_rank = _get_dist_info()
    if device.startswith("cuda") and torch.cuda.is_available():
        dev = f"cuda:{local_rank}" if world_size > 1 else (device if ":" in device else "cuda:0")
        torch.cuda.set_device(dev)
        return dev
    return device


class RLRunner:
    def __init__(self, cfg: RLRunnerConfig) -> None:
        self.cfg = cfg
        self._decision_ce_ema_state: Optional[float] = None
        # Sliding window buffers for stable metrics reporting (env-controlled)
        try:
            self._metrics_window: int = max(1, int(os.environ.get("METRICS_WINDOW_UPDATES", "50")))
        except Exception:
            self._metrics_window = 50
        self._win_buf: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self._metrics_window))

    def _update_window_metrics(self, scalars: Dict[str, float]) -> Dict[str, float]:
        keys_to_track: List[str] = [
            "loss",
            "reward_best_mean", "reward_best_std",
            "acc_best", "acc_any", "accuracy", "fn_rate",
            "resp_len_mean", "grad_norm_mean", "kl_b_mean",
            "phase_a_formatting", "phase_a_coverage", "phase_a_cleanliness", "phase_a_taxonomy", "phase_a_consistency",
            "skip_updates_std0", "pairwise_trigger_rate", "phase_a_entropy_mean",
            "reward_margin_best", "flip_rate", "decision_ce_mean", "decision_ce_ema",
        ]
        out: Dict[str, float] = {}
        for k in keys_to_track:
            if k in scalars and isinstance(scalars[k], (float, int)):
                self._win_buf[k].append(float(scalars[k]))
                # mean over window
                buf = self._win_buf[k]
                if len(buf) > 0:
                    out[f"{k}_win"] = float(sum(buf) / float(len(buf)))
        return out

    def run(self) -> None:
        cfg = self.cfg
        # Device + DDP
        device = _bind_device(str(cfg.device))
        _maybe_init_ddp()

        # Reduce verbosity on non-zero ranks
        try:
            rank, world_size, _ = _get_dist_info()
            if rank != 0:
                os.environ.setdefault("TQDM_DISABLE", "1")
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
                hf_logging.set_verbosity_error()
                for name, level in [
                    ("src_post", logging.WARNING),
                    ("src_new", logging.WARNING),
                    ("transformers", logging.ERROR),
                ]:
                    try:
                        logging.getLogger(name).setLevel(level)
                    except Exception:
                        pass
        except Exception:
            pass

        # Seeding
        try:
            import random, numpy as np
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
        except Exception:
            pass

        # IO setup
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        results_path = None
        writer = None
        rank, world_size, _ = _get_dist_info()
        if cfg.results_jsonl:
            Path(cfg.results_jsonl).parent.mkdir(parents=True, exist_ok=True)
            stem = Path(cfg.results_jsonl)
            results_path = str(stem.with_suffix("").as_posix()) + f".rank{rank}.jsonl"
            writer = open(results_path, "w", encoding="utf-8")
        metrics_path = None
        metrics_writer = None
        if cfg.metrics_jsonl:
            Path(cfg.metrics_jsonl).parent.mkdir(parents=True, exist_ok=True)
            mstem = Path(cfg.metrics_jsonl)
            metrics_path = str(mstem.with_suffix("").as_posix()) + f".rank{rank}.jsonl"
            metrics_writer = open(metrics_path, "w", encoding="utf-8")
        try:
            tb_writer = None
            if rank == 0 and cfg.tb_log_dir:
                base_dir = Path(cfg.tb_log_dir)
                rn = cfg.run_name
                if not rn:
                    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                    rn = f"{Path(cfg.output_dir).name}_{ts}"
                    logger.info(f"[tb] run_name not provided; derived run_name={rn}")
                tb_dir = base_dir / rn
                tb_dir.mkdir(parents=True, exist_ok=True)
                tb_writer = SummaryWriter(log_dir=str(tb_dir))
                logger.info(f"TensorBoard logging to: {tb_dir}")
        except Exception:
            tb_writer = None

        # Load processor & models
        logger.info("Loading processor & models ...")
        processor: Qwen2VLProcessor = load_processor(cfg.processor)
        policy: DetectionModel = load_detection_model(cfg.checkpoint, processor, device)
        # Disable KV cache during TF
        try:
            if hasattr(policy.base_model, "config"):
                policy.base_model.config.use_cache = False
                if hasattr(policy.base_model.config, "gradient_checkpointing"):
                    policy.base_model.config.gradient_checkpointing = False
            if hasattr(policy.base_model, "gradient_checkpointing_disable"):
                policy.base_model.gradient_checkpointing_disable()
        except Exception:
            pass
        # DDP model for training forwards
        ddp_policy: Optional[DDP] = wrap_ddp_if_needed(policy, device)
        train_model = ddp_policy if ddp_policy is not None else policy

        # Optional reference policy for KL
        ref_model: Optional[DetectionModel] = None
        if cfg.use_ref_kl:
            ref_model = load_detection_model(cfg.ref_checkpoint or cfg.checkpoint, processor, device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

        conv_builder = GroupQCConversationBuilder(processor=processor)

        # Logits processor for decode-time masking
        logits_processors = LogitsProcessorList()
        if cfg.mask_geometry_tokens or cfg.mask_coordinate_tokens:
            logits_processors.append(
                GeometryCoordMaskLogitsProcessor(
                    tokenizer=processor.tokenizer,
                    mask_geometry_tokens=cfg.mask_geometry_tokens,
                    mask_coordinate_tokens=cfg.mask_coordinate_tokens,
                )
            )

        # Stage-A stopping
        stopping_stage_a = build_stage_a_stopping(processor)

        # Generation configs
        stage_a_temp = float(getattr(cfg, 'temperature_stage_a', cfg.temperature))
        stage_a_top_p = float(getattr(cfg, 'top_p_stage_a', cfg.top_p))
        stage_b_temp = float(getattr(cfg, 'temperature_stage_b', cfg.temperature))
        stage_b_top_p = float(getattr(cfg, 'top_p_stage_b', cfg.top_p))
        gen_cfg_stage_a = GenerationConfig(
            do_sample=(stage_a_temp > 0.0),
            temperature=stage_a_temp,
            top_p=stage_a_top_p,
            max_new_tokens=max(1, min(cfg.max_new_tokens_stage_a, 64)),
            min_new_tokens=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=8,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        gen_cfg_stage_b = GenerationConfig(
            do_sample=(stage_b_temp > 0.0),
            temperature=stage_b_temp,
            top_p=stage_b_top_p,
            max_new_tokens=max(1, min(cfg.max_new_tokens_stage_b, 128)),
            min_new_tokens=1,
            repetition_penalty=1.3,
            no_repeat_ngram_size=16,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        # Dataset
        ds_path = cfg.train_data_dir or cfg.eval_data_dir
        if ds_path is None:
            logger.info("No dataset provided (train_data_dir or eval_data_dir); exiting.")
            return
        dataset = RLGroupQCDataset(jsonl_or_dir_path=ds_path, preload=False)
        logger.info(f"Loaded dataset: {ds_path} with {len(dataset)} groups")

        # Freeze and build optimizer param groups
        params_groups = apply_freeze_and_param_groups(policy, processor.tokenizer, cfg)
        try:
            optimizer = torch.optim.AdamW(params_groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=True)  # type: ignore[call-arg]
            logger.info("Using fused AdamW optimizer (fused=True)")
        except TypeError:
            optimizer = torch.optim.AdamW(params_groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            logger.info("Fused AdamW not available; falling back to standard AdamW")
        # LR scheduler (cosine/warmup) via HF get_scheduler
        try:
            from transformers import get_scheduler
            # Derive steps per rank based on current dataset sharding logic
            rank, world_size, _ = _get_dist_info()
            per_rank_indices = build_epoch_indices(
                dataset_len=len(dataset),
                world_size=world_size,
                rank=rank,
                limit_groups=int(cfg.limit_groups),
                seed=int(cfg.seed),
                epoch=0,
            )
            # Compute batches per epoch for this rank without consuming the dataset
            batches = list(iter_batches(per_rank_indices, int(cfg.batch_size), bool(cfg.drop_last)))
            updates_per_epoch_rank = int(len(batches))
            total_steps = max(1, int(cfg.epochs) * max(1, updates_per_epoch_rank))
            warmup_steps = max(0, int(getattr(cfg, 'warmup_ratio', 0.0) * total_steps))
            lr_scheduler = get_scheduler(
                name=str(getattr(cfg, 'lr_scheduler_type', 'cosine')),
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        except Exception:
            lr_scheduler = None

        # Writers and metrics buffers
        update = 0
        try:
            policy.train()
            if ddp_policy is not None:
                ddp_policy.train()
            if ref_model is not None:
                ref_model.eval()
        except Exception:
            pass

        # Grad accumulation
        grad_accum_steps = max(1, int(getattr(cfg, 'grad_accum_steps', 1)))
        accum_scale = 1.0 / float(grad_accum_steps)
        micro_step_counter = 0
        optimizer.zero_grad(set_to_none=True)

        # Env override for skip_save
        skip_save_env = os.environ.get("SKIP_SAVE", "0").strip().lower()
        skip_save = skip_save_env in ("1", "true", "yes") or bool(cfg.skip_save_checkpoints)

        # Epoch loop
        for epoch_round in range(int(max(1, cfg.epochs))):
            epoch_idx = epoch_round
            indices = build_epoch_indices(
                dataset_len=len(dataset),
                world_size=world_size,
                rank=rank,
                limit_groups=int(cfg.limit_groups),
                seed=int(cfg.seed),
                epoch=epoch_idx,
            )
            if rank == 0:
                logger.info(f"[data] Start epoch {epoch_round+1}/{cfg.epochs}: num_local={len(indices)} (drop_last={cfg.drop_last})")
            processed_so_far_local_epoch = 0
            epoch_total_local_len = len(indices)

            for batch_indices in iter_batches(indices, cfg.batch_size, cfg.drop_last):
                update_start_ts = time.time()
                # Accumulators per batch
                total_loss = 0.0
                total_items = 0
                reward_best_sum = 0.0
                reward_best_sq_sum = 0.0
                best_hit_count = 0
                any_hit_count = 0
                resp_len_sum = 0
                kl_b_sum = 0.0
                kl_b_count = 0
                grad_norm_sum = 0.0
                # Phase-A diagnostics sums (placeholders; computed via rewards if enabled)
                diag_fmt_sum = 0.0
                diag_cov_sum = 0.0
                diag_cln_sum = 0.0
                diag_tax_sum = 0.0
                diag_cons_sum = 0.0
                # New metrics accumulators
                skip_std0_count = 0.0
                pairwise_trigger_count = 0.0
                entropy_sum = 0.0
                margin_best_sum = 0.0
                flip_hit_count = 0.0
                decision_ce_sum = 0.0
                decision_ce_count = 0.0
                # Additional classification metrics
                fn_count = 0.0  # GT=fail but pred=pass
                gt_fail_count = 0.0

                for idx in batch_indices:
                    sample = dataset[idx]
                    images: List[Image.Image] = sample["images"]
                    gt_label: str = str(sample["label"]).strip().lower()

                    # 1) Stage-A greedy context lines
                    context_lines = build_stage_a_context_lines(
                        policy=policy,
                        processor=processor,
                        images=images,
                        gen_cfg=gen_cfg_stage_a,
                        logits_processors=logits_processors,
                        stopping=stopping_stage_a,
                        sanitize=bool(cfg.sanitize_stage_a),
                        mission=cfg.mission,
                    )

                    # 2) Stage-B GRPO step
                    checklist = conv_builder.rules_for_mission(cfg.mission)
                    # Phase-A diagnostics using reward functions (optional)
                    if cfg.enable_phase_a_diagnostics:
                        try:
                            diags = compute_phase_a_diagnostics(context_lines, checklist, mission=cfg.mission)
                            diag_fmt_sum += float(diags.get("formatting", 0.0))
                            diag_cov_sum += float(diags.get("coverage", 0.0))
                            diag_cln_sum += float(diags.get("cleanliness", 0.0))
                            diag_tax_sum += float(diags.get("taxonomy", 0.0))
                            diag_cons_sum += float(diags.get("consistency", 0.0))
                        except Exception as e:
                            # Fail-fast surface for diagnostics to not silently swallow systemic errors in rewards
                            raise RuntimeError(f"Phase-A diagnostics failed: {e}")

                    if bool(cfg.use_mission_checklist):
                        msgs_b = conv_builder.build_stage_b_messages(summary_lines=context_lines, checklist_lines=checklist)
                    else:
                        msgs_b = conv_builder.build_stage_b_messages_minimal(summary_lines=context_lines)
                    text_b = processor.apply_chat_template(conversation=msgs_b, tokenize=False, add_generation_prompt=True)
                    # Fail-fast: Stage-B must not contain any <image> placeholders
                    conv_builder.validate_image_placeholder_count(text_b, 0)
                    enc_b = processor(text=[text_b], images=None, return_tensors="pt", padding=True)
                    enc_b = to_device_and_cast(enc_b, device)

                    # Sync params/buffers before DDP forward
                    if ddp_policy is not None and dist.is_initialized():
                        try:
                            ddp_policy._sync_params_and_buffers(authoritative_rank=0)
                        except Exception:
                            pass

                    tf_p_pass, tf_p_fail = tf_decision_probs(train_model, enc_b, processor, length_norm=bool(cfg.length_norm))
                    # Baseline decision margin for logging
                    import math as _math
                    eps = 1e-9
                    margin_best = float(_math.log(float(tf_p_pass) + eps) - _math.log(float(tf_p_fail) + eps))
                    margin_best_sum += margin_best
                    # Decision CE for global guidance
                    if gt_label == "pass":
                        ce = -_math.log(float(tf_p_pass) + eps)
                    else:
                        ce = -_math.log(float(tf_p_fail) + eps)
                    decision_ce_sum += float(ce)
                    decision_ce_count += 1.0

                    replies_token_ids: List[List[int]] = []
                    replies_text: List[str] = []
                    rewards_b: List[float] = []
                    pred_labels: List[Optional[str]] = []

                    # Build a prefix constraint so Stage-B must start with a strict decision line
                    prefix_fn = None
                    try:
                        from src_post.generation.generation import build_decision_prefix_constraint
                        prompt_len_b = int(enc_b["input_ids"].size(1))
                        prefix_fn = build_decision_prefix_constraint(processor, prompt_len_b)
                    except Exception:
                        prefix_fn = None

                    for _ in range(max(1, int(cfg.K_B))):
                        with torch.no_grad():
                            gen_kwargs_b = {
                                "generation_config": gen_cfg_stage_b,
                                "logits_processor": logits_processors,
                            }
                            if prefix_fn is not None:
                                gen_kwargs_b["prefix_allowed_tokens_fn"] = prefix_fn
                            out = policy.generate(
                                **enc_b,
                                **gen_kwargs_b,
                            )
                        new_ids = out[:, enc_b["input_ids"].size(1):]
                        token_ids = new_ids[0].tolist()
                        if len(token_ids) == 0:
                            token_ids = [int(processor.tokenizer.eos_token_id)]
                        text_out = decode_to_text(processor, token_ids)
                        from src_post.prompting.span_parser import parse_stage_b_output
                        parsed = parse_stage_b_output(text_out)
                        pred_label = parsed.get("label")
                        reason = parsed.get("reason")
                        # Reward composition with group_reward_mode control
                        reward_names = list(cfg.reward_fns)
                        # Provide full Stage-B text for formatting reward
                        # and strict_decision_format
                        # (keys are optional; reward will fallback to 0.0 if missing)
                        # text_out included later via compose inputs extension
                        reward_weights = list(cfg.reward_weights)
                        if cfg.group_reward_mode == "margin_only":
                            # Ensure group_margin is included exclusively
                            reward_names = ["group_margin"]
                            reward_weights = [1.0]
                        elif cfg.group_reward_mode == "label_match":
                            reward_names = ["label_match"]
                            reward_weights = [1.0]
                        # else: combined → use config as-is
                        r = compose_reward(
                            gt_label=gt_label,
                            pred_label=pred_label,
                            summary_lines=context_lines,
                            reason=reason,
                            checklist_lines=checklist,
                            reward_names=reward_names,
                            reward_weights=reward_weights,
                            tf_p_pass=tf_p_pass,
                            tf_p_fail=tf_p_fail,
                            stage_b_text=text_out,
                            mission=cfg.mission,
                        )
                        if not (r == r and abs(r) != float("inf")):
                            r = 0.0
                        replies_token_ids.append(token_ids)
                        replies_text.append(text_out)
                        rewards_b.append(float(r))
                        pred_labels.append(pred_label)

                    with torch.no_grad():
                        rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device)
                        mean = rewards_t.mean()
                        std = rewards_t.std(unbiased=False)
                        if float(std.item()) < 1e-6:
                            adv = torch.zeros_like(rewards_t)
                            skip_std0_count += 1.0
                        else:
                            adv = (rewards_t - mean) / (std + 1e-6)
                        adv = torch.clamp(adv, min=-cfg.adv_clip, max=cfg.adv_clip)

                    # DDP no_sync for gradient accumulation: only sync on the last micro-step
                    sync_now = ((micro_step_counter + 1) % grad_accum_steps) == 0
                    loss_b_scalar = torch.tensor(0.0, dtype=torch.float32, device=device)
                    with (ddp_policy.no_sync() if (ddp_policy is not None and not sync_now) else nullcontext()):
                        for k in range(len(replies_token_ids)):
                            logp_k, cur_logits_k, prompt_len_k = tf_sum_logprob_and_logits_over_response(
                                train_model, enc_b, replies_token_ids[k], cfg.length_norm
                            )
                            term = - (adv[k].detach()) * logp_k
                            if cfg.use_ref_kl and ref_model is not None and cfg.lambda_kl_stage_b > 0.0:
                                kl_k = kl_to_ref_with_cur_logits(ref_model, enc_b, replies_token_ids[k], cur_logits_k, int(prompt_len_k))
                                try:
                                    kl_b_sum += float(kl_k.detach().item())
                                    kl_b_count += 1
                                except Exception:
                                    pass
                                term = term + (cfg.lambda_kl_stage_b * kl_k)
                            # Apply Stage‑B freeze/toggle at backward time
                            if bool(cfg.train_stage_b) and update >= int(cfg.freeze_stage_b_steps):
                                scale_b = float(cfg.stage_b_weight)
                                (term * (accum_scale * scale_b)).backward()
                            loss_b_scalar = loss_b_scalar + term.detach()

                    # Selection metrics
                    if len(rewards_b) > 0:
                        best_idx = int(max(range(len(rewards_b)), key=lambda i: rewards_b[i]))
                    else:
                        best_idx = 0
                    best_reward = float(rewards_b[best_idx]) if len(rewards_b) > 0 else 0.0
                    reward_best_sum += best_reward
                    reward_best_sq_sum += best_reward * best_reward
                    best_label = pred_labels[best_idx] if len(pred_labels) > 0 else None
                    best_hit_count += int(isinstance(best_label, str) and best_label == gt_label)
                    any_hit = int(any((lbl == gt_label) for lbl in pred_labels if isinstance(lbl, str)))
                    any_hit_count += any_hit
                    flip_hit_count += float(any_hit)
                    # FN counting: GT=fail but predicted pass (by best selection)
                    try:
                        gt_fail_count += 1.0 if gt_label == "fail" else 0.0
                        if isinstance(best_label, str) and (best_label == "pass") and (gt_label == "fail"):
                            fn_count += 1.0
                    except Exception:
                        pass
                    if len(replies_token_ids) > 0:
                        try:
                            resp_len_sum += int(len(replies_token_ids[best_idx]))
                        except Exception:
                            pass

                    # 3) Stage-A GRPO (conditional only supported here; joint can be added later)
                    loss_a = torch.tensor(0.0, dtype=torch.float32, device=device)
                    assigner = get_credit_assigner(cfg.train_stage_a_mode)
                    if cfg.train_stage_a_mode in {"conditional", "joint"}:
                        tf_cfg = {
                            "device": device,
                            "length_norm": bool(cfg.length_norm),
                            "logits_processors": logits_processors,
                            "stopping": stopping_stage_a,
                            "K_A": int(cfg.K_A),
                            "max_images_tf": int(cfg.max_images_tf),
                            "adv_clip": float(cfg.adv_clip),
                            "ddp_policy": ddp_policy,
                            "pairs_per_group": int(cfg.pairwise_pairs_per_group),
                            # Scale Stage-A gradients by stage_a_weight
                            "accum_scale": float(accum_scale * float(cfg.stage_a_weight)),
                        }
                        reward_cfg = {
                            "names": list(cfg.reward_fns),
                            "weights": list(cfg.reward_weights),
                            "group_reward_mode": str(cfg.group_reward_mode),
                            "use_ref_kl": bool(cfg.use_ref_kl),
                            "ref_model": ref_model,
                            "lambda_kl_stage_a": float(cfg.lambda_kl_stage_a),
                            "use_uncertainty_gate": bool(cfg.use_uncertainty_gate),
                            "entropy_threshold": float(cfg.uncertainty_gate_min_entropy),
                            "use_mission_checklist": bool(cfg.use_mission_checklist),
                            "baseline_tf_p_pass": float(tf_p_pass),
                            "baseline_tf_p_fail": float(tf_p_fail),
                        }
                        # Backward inside compute_loss_a should also honor no_sync
                        with (ddp_policy.no_sync() if (ddp_policy is not None and not sync_now) else nullcontext()):
                            loss_a, diags_a = assigner.compute_loss_a(
                                policy=policy,
                                train_model=train_model,
                                processor=processor,
                                conv_builder=conv_builder,
                                images=images,
                                context_lines=context_lines,
                                checklist=checklist,
                                mission=cfg.mission,
                                gt_label=gt_label,
                                enc_b_baseline=enc_b,
                                tf_cfg=tf_cfg,
                                gen_cfg_a=gen_cfg_stage_a,
                                reward_cfg=reward_cfg,
                            )
                        # Pairwise fallback when enabled and stage-B best != GT and deltas weak
                        pairwise_triggered_flag = 0
                        if bool(cfg.pairwise_credit_enabled):
                            try:
                                from src_post.credit.credit_assignment import PairwiseFallbackAssigner
                                # Heuristic trigger: reuse baseline vs pair margin condition here is approximate; runner lacks per-image deltas
                                # We gate on best_label mismatch as a proxy and rely on config to enable sparingly
                                best_idx = int(max(range(len(rewards_b)), key=lambda i: rewards_b[i])) if len(rewards_b) > 0 else 0
                                best_label = pred_labels[best_idx] if len(pred_labels) > 0 else None
                                # Require single-image deltas to be weak and best label mismatch
                                max_single_delta = float(diags_a.get("best_single_delta", 0.0)) if isinstance(diags_a, dict) else 0.0
                                threshold = float(cfg.pairwise_delta_threshold)
                                if (
                                    isinstance(best_label, str)
                                    and best_label != gt_label
                                    and int(cfg.pairwise_pairs_per_group) > 0
                                    and (max_single_delta < threshold)
                                ):
                                    pw = PairwiseFallbackAssigner()
                                    with (ddp_policy.no_sync() if (ddp_policy is not None and not sync_now) else nullcontext()):
                                        loss_pw, diags_pw = pw.compute_loss_a(
                                            policy=policy,
                                            train_model=train_model,
                                            processor=processor,
                                            conv_builder=conv_builder,
                                            images=images,
                                            context_lines=context_lines,
                                            checklist=checklist,
                                            mission=cfg.mission,
                                            gt_label=gt_label,
                                            enc_b_baseline=enc_b,
                                            tf_cfg=tf_cfg,
                                            gen_cfg_a=gen_cfg_stage_a,
                                            reward_cfg=reward_cfg,
                                        )
                                    loss_a = loss_a + loss_pw
                                    pairwise_triggered_flag = int(diags_pw.get("pairwise_triggered", 0.0))
                            except Exception as e:
                                raise RuntimeError(f"Pairwise fallback failed: {e}")
                        if isinstance(diags_a, dict) and "phase_a_entropy_mean" in diags_a:
                            try:
                                entropy_sum += float(diags_a["phase_a_entropy_mean"]) if bool(cfg.use_uncertainty_gate) else 0.0
                            except Exception:
                                raise RuntimeError("Invalid phase_a_entropy_mean diagnostic; expected float")
                        else:
                            pairwise_triggered_flag = 0

                    # 4) Combine and optimize
                    # Apply Stage‑B weight and scale Stage‑A by number of images to keep gradient magnitude stable under full coverage
                    try:
                        num_images_in_group = int(sample.get("num_images", len(images))) if isinstance(sample, dict) else len(images)
                    except Exception:
                        num_images_in_group = len(images)
                    stage_a_weight_eff = float(cfg.stage_a_weight) / max(1, int(num_images_in_group))
                    loss = (cfg.stage_b_weight * loss_b_scalar) + (stage_a_weight_eff * loss_a)
                    # Backward for combined scalar (only scaling for accum if there are any remaining grads not already applied)
                    # Note: Stage-B and Stage-A terms already backpropagated individually above. Here we do nothing to avoid double-backward.

                    # Step when reaching grad_accum_steps
                    did_step = False
                    if (micro_step_counter + 1) % grad_accum_steps == 0:
                        if ddp_policy is not None:
                            _gn = torch.nn.utils.clip_grad_norm_(ddp_policy.parameters(), cfg.max_grad_norm)
                        else:
                            _gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if lr_scheduler is not None:
                            try:
                                lr_scheduler.step()
                            except Exception:
                                pass
                        try:
                            gn_val = float(_gn.item()) if hasattr(_gn, "item") else float(_gn)
                            grad_norm_sum += gn_val
                        except Exception:
                            pass
                        did_step = True
                    micro_step_counter += 1

                    total_loss += float(loss.detach().item())
                    total_items += 1
                    # Accumulate per-item pairwise trigger count
                    pairwise_trigger_count += float(pairwise_triggered_flag)

                    # Write per-item results JSONL
                    if writer is not None:
                        per_image = []
                        for img_path, cap in zip(sample["image_paths"], context_lines):
                            per_image.append({"image_id": os.path.basename(img_path), "caption": cap})
                        best_idx = int(max(range(len(rewards_b)), key=lambda i: rewards_b[i])) if len(rewards_b) > 0 else 0
                        rec = {
                            "group_index": idx,
                            "mission": cfg.mission,
                            "gt_label": gt_label,
                            "pred_label": pred_labels[best_idx] if len(pred_labels) > 0 else None,
                            "reward": float(rewards_b[best_idx]) if len(rewards_b) > 0 else 0.0,
                            "k_b": int(cfg.K_B),
                            "k_a": int(cfg.K_A),
                            "images": per_image,
                            "stage_b_raw": replies_text[best_idx] if len(replies_text) > 0 else "",
                            "used_minimal_prompt": (not bool(cfg.use_mission_checklist)),
                            "pairwise_triggered": bool(pairwise_triggered_flag),
                        }
                        writer.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Aggregate per-batch metrics and log (rank0)
                if total_items > 0:
                    try:
                        processed_so_far_local_epoch += int(total_items)
                    except Exception:
                        pass
                    # Global reductions
                    dev = torch.device(device)
                    vec_local = torch.tensor([
                        float(total_items),
                        float(total_loss),
                        float(reward_best_sum),
                        float(reward_best_sq_sum),
                        float(best_hit_count),
                        float(any_hit_count),
                        float(resp_len_sum),
                        float(grad_norm_sum),
                        float(kl_b_sum),
                        float(kl_b_count),
                        float(diag_fmt_sum),
                        float(diag_cov_sum),
                        float(diag_cln_sum),
                        float(diag_tax_sum),
                        float(diag_cons_sum),
                        float(skip_std0_count),
                        float(pairwise_trigger_count),
                        float(entropy_sum),
                        float(margin_best_sum),
                        float(flip_hit_count),
                        float(decision_ce_sum),
                        float(decision_ce_count),
                        float(fn_count),
                        float(gt_fail_count),
                    ], device=dev, dtype=torch.float32)
                    if world_size > 1 and dist.is_initialized():
                        dist.all_reduce(vec_local, op=dist.ReduceOp.SUM)
                    # Aggregate into scalars
                    agg = aggregate_training_metrics(vec_local, world_size)
                    # Update sliding-window metrics (over last N updates)
                    agg_win = self._update_window_metrics(agg)
                    # ETA
                    dt_local = float(time.time() - update_start_ts)
                    dt_t = torch.tensor([dt_local], device=dev, dtype=torch.float32)
                    if world_size > 1 and dist.is_initialized():
                        dist.all_reduce(dt_t, op=dist.ReduceOp.MAX)
                    dt_global = float(dt_t.item())
                    ep_total_t = torch.tensor([float(epoch_total_local_len)], device=dev, dtype=torch.float32)
                    proc_so_far_t = torch.tensor([float(processed_so_far_local_epoch)], device=dev, dtype=torch.float32)
                    if world_size > 1 and dist.is_initialized():
                        dist.all_reduce(ep_total_t, op=dist.ReduceOp.SUM)
                        dist.all_reduce(proc_so_far_t, op=dist.ReduceOp.SUM)
                    epoch_total_g = int(ep_total_t.item())
                    processed_so_far_g = int(proc_so_far_t.item())
                    eta_hrs = compute_eta(
                        dt_max_sec=dt_global,
                        processed_so_far_g=processed_so_far_g,
                        epoch_total_g=epoch_total_g,
                        epoch_idx=epoch_idx,
                        epochs=int(cfg.epochs),
                        items_per_update_g=int(max(1.0, float(vec_local[0].item()))),
                    )
                    agg["eta_hours"] = float(eta_hrs)
                    # Update EMA (rank-0 only)
                    if rank == 0 and "decision_ce_mean" in agg:
                        beta = float(getattr(cfg, 'decision_ce_ema_beta', 0.9))
                        cur = float(agg.get("decision_ce_mean", 0.0))
                        if self._decision_ce_ema_state is None:
                            self._decision_ce_ema_state = cur
                        else:
                            self._decision_ce_ema_state = float(beta * self._decision_ce_ema_state + (1.0 - beta) * cur)
                        agg["decision_ce_ema"] = float(self._decision_ce_ema_state)
                    prefix = f"[{epoch_idx+1}/{int(cfg.epochs)}][{processed_so_far_g}/{epoch_total_g}] "

                    if rank == 0:
                        lrs = {str(g.get("name", f"group{i}")): float(g.get("lr", 0.0)) for i, g in enumerate(optimizer.param_groups)}
                        # Merge raw + windowed for logging/JSONL/TensorBoard
                        scalars_to_log = dict(agg)
                        scalars_to_log.update(agg_win)
                        rank0_log(logger, tb_writer, metrics_writer, update, prefix, scalars_to_log, lrs)
                        update += 1

            # Finalize after last epoch
            if (epoch_round + 1) == int(cfg.epochs):
                try:
                    save_if_rank0(policy, processor, cfg.output_dir, tag="grpo_final", skip_save=bool(skip_save))
                    if writer is not None:
                        writer.close()
                        logger.info(f"Results written: {results_path}")
                    if metrics_writer is not None:
                        metrics_writer.close()
                        logger.info(f"Metrics written: {metrics_path}")
                    try:
                        if tb_writer is not None:
                            tb_writer.flush(); tb_writer.close()
                    except Exception:
                        pass
                    if dist.is_initialized():
                        dist.barrier()
                except Exception as e:
                    logger.warning(f"Finalize error: {e}")

        # Destroy DDP
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Group QC GRPO Runner (config-first)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()
    cfg = load_and_validate_config(args.config)
    RLRunner(cfg).run()


if __name__ == "__main__":
    main()
