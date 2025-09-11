#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import GenerationConfig, Qwen2VLProcessor
from transformers.generation import LogitsProcessorList

from PIL import Image

from src_post.config import RLRunnerConfig, load_and_validate_config
from src_post.conversation import GroupQCConversationBuilder
from src_post.dataset_group_qc import RLGroupQCDataset
from src_post.logits_processors import GeometryCoordMaskLogitsProcessor
from src_post.models import (
    apply_freeze_and_param_groups,
    load_detection_model,
    load_processor,
    wrap_ddp_if_needed,
)
from src_post.generation import (
    build_stage_a_context_lines,
    build_stage_a_stopping,
    decode_to_text,
    to_device_and_cast,
)
from src_post.teacher_forcing import (
    kl_to_ref_with_cur_logits,
    tf_decision_probs,
    tf_sum_logprob_and_logits_over_response,
)
from src_post.rewards.compose import compose_reward
from src_post.data_loader import build_epoch_indices, iter_batches
from src_post.logging_utils import compute_eta, rank0_log, aggregate_training_metrics
from src_post.checkpoints import save_if_rank0
from src_post.diagnostics import compute_phase_a_diagnostics

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
        gen_cfg_stage_a = GenerationConfig(
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=max(1, min(cfg.max_new_tokens_stage_a, 64)),
            min_new_tokens=1,
            repetition_penalty=1.6,
            no_repeat_ngram_size=24,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        gen_cfg_stage_b = GenerationConfig(
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
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
        optimizer = torch.optim.AdamW(params_groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        # LR scheduler (cosine/warmup) via HF get_scheduler
        try:
            from transformers import get_scheduler
            total_steps = max(1, (int(cfg.num_updates) * max(1, int(cfg.batch_size))))
            warmup_steps = max(0, int(cfg.warmup_ratio * total_steps)) if hasattr(cfg, 'warmup_ratio') else 0
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
                    )

                    # 2) Stage-B GRPO step
                    checklist = conv_builder.rules_for_mission(cfg.mission)
                    # Phase-A diagnostics using reward functions (optional)
                    if cfg.enable_phase_a_diagnostics:
                        try:
                            diags = compute_phase_a_diagnostics(context_lines, checklist)
                            diag_fmt_sum += float(diags.get("formatting", 0.0))
                            diag_cov_sum += float(diags.get("coverage", 0.0))
                            diag_cln_sum += float(diags.get("cleanliness", 0.0))
                            diag_tax_sum += float(diags.get("taxonomy", 0.0))
                            diag_cons_sum += float(diags.get("consistency", 0.0))
                        except Exception:
                            pass

                    msgs_b = conv_builder.build_stage_b_messages(summary_lines=context_lines, checklist_lines=checklist)
                    text_b = processor.apply_chat_template(conversation=msgs_b, tokenize=False, add_generation_prompt=True)
                    enc_b = processor(text=[text_b], images=None, return_tensors="pt", padding=True)
                    enc_b = to_device_and_cast(enc_b, device)

                    # Sync params/buffers before DDP forward
                    if ddp_policy is not None and dist.is_initialized():
                        try:
                            ddp_policy._sync_params_and_buffers(authoritative_rank=0)
                        except Exception:
                            pass

                    tf_p_pass, tf_p_fail = tf_decision_probs(train_model, enc_b, processor, length_norm=bool(cfg.length_norm))

                    replies_token_ids: List[List[int]] = []
                    replies_text: List[str] = []
                    rewards_b: List[float] = []
                    pred_labels: List[Optional[str]] = []

                    for _ in range(max(1, int(cfg.K_B))):
                        with torch.no_grad():
                            out = policy.generate(
                                **enc_b,
                                generation_config=gen_cfg_stage_b,
                                logits_processor=logits_processors,
                            )
                        new_ids = out[:, enc_b["input_ids"].size(1):]
                        token_ids = new_ids[0].tolist()
                        if len(token_ids) == 0:
                            token_ids = [int(processor.tokenizer.eos_token_id)]
                        text_out = decode_to_text(processor, token_ids)
                        from src_post.span_parser import parse_stage_b_output
                        parsed = parse_stage_b_output(text_out)
                        pred_label = parsed.get("label")
                        reason = parsed.get("reason")
                        r = compose_reward(
                            gt_label=gt_label,
                            pred_label=pred_label,
                            summary_lines=context_lines,
                            reason=reason,
                            checklist_lines=checklist,
                            reward_names=cfg.reward_fns,
                            reward_weights=cfg.reward_weights,
                            tf_p_pass=tf_p_pass,
                            tf_p_fail=tf_p_fail,
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
                        else:
                            adv = (rewards_t - mean) / (std + 1e-6)
                        adv = torch.clamp(adv, min=-cfg.adv_clip, max=cfg.adv_clip)

                    loss_b_scalar = torch.tensor(0.0, dtype=torch.float32, device=device)
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
                        term.backward()
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
                    any_hit_count += int(any((lbl == gt_label) for lbl in pred_labels if isinstance(lbl, str)))
                    if len(replies_token_ids) > 0:
                        try:
                            resp_len_sum += int(len(replies_token_ids[best_idx]))
                        except Exception:
                            pass

                    # 3) Stage-A GRPO (conditional only supported here; joint can be added later)
                    loss_a = torch.tensor(0.0, dtype=torch.float32, device=device)
                    if cfg.train_stage_a_mode == "conditional":
                        capped_images = list(images)[:max(0, int(cfg.max_images_tf))]
                        for i, img in enumerate(capped_images):
                            img_proc = img  # preprocess is inside generation builder
                            messages = conv_builder.build_stage_a_messages(cfg.mission)
                            messages = [messages[0], {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "请只输出一行摘要"}]}]
                            text_a = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            from src_post.generation import sft_style_preprocess_image
                            img_proc = sft_style_preprocess_image(img)
                            enc_a = processor(text=[text_a], images=[img_proc], padding=True, return_tensors="pt")
                            enc_a = to_device_and_cast(enc_a, device)

                            cand_token_ids: List[List[int]] = []
                            rewards_i: List[float] = []
                            for _ in range(max(1, int(cfg.K_A))):
                                with torch.no_grad():
                                    out = policy.generate(
                                        **enc_a,
                                        generation_config=gen_cfg_stage_a,
                                        logits_processor=logits_processors,
                                        stopping_criteria=stopping_stage_a,
                                    )
                                new_ids = out[:, enc_a["input_ids"].size(1):]
                                toks = new_ids[0].tolist()
                                raw = decode_to_text(processor, toks).strip()
                                context_variant = list(context_lines)
                                context_variant[i] = raw
                                msgs_var = conv_builder.build_stage_b_messages(summary_lines=context_variant, checklist_lines=checklist)
                                text_b_var = processor.apply_chat_template(conversation=msgs_var, tokenize=False, add_generation_prompt=True)
                                enc_b_var = processor(text=[text_b_var], images=None, return_tensors="pt", padding=True)
                                enc_b_var = to_device_and_cast(enc_b_var, device)
                                # Sync before TF
                                if ddp_policy is not None and dist.is_initialized():
                                    try:
                                        ddp_policy._sync_params_and_buffers(authoritative_rank=0)
                                    except Exception:
                                        pass
                                try:
                                    prob_pass, prob_fail = tf_decision_probs(train_model, enc_b_var, processor, length_norm=bool(cfg.length_norm))
                                except Exception:
                                    prob_pass, prob_fail = 0.0, 0.0
                                r_i = compose_reward(
                                    gt_label=gt_label,
                                    pred_label=None,
                                    summary_lines=context_variant,
                                    reason=None,
                                    checklist_lines=checklist,
                                    reward_names=cfg.reward_fns,
                                    reward_weights=cfg.reward_weights,
                                    tf_p_pass=prob_pass,
                                    tf_p_fail=prob_fail,
                                )
                                cand_token_ids.append(toks)
                                rewards_i.append(float(r_i))

                            with torch.no_grad():
                                rewards_t = torch.tensor(rewards_i, dtype=torch.float32, device=device)
                                mean = rewards_t.mean(); std = rewards_t.std(unbiased=False)
                                if float(std.item()) < 1e-6:
                                    adv_i = torch.zeros_like(rewards_t)
                                else:
                                    adv_i = (rewards_t - mean) / (std + 1e-6)
                                adv_i = torch.clamp(adv_i, min=-cfg.adv_clip, max=cfg.adv_clip)

                            for k in range(len(cand_token_ids)):
                                logp_a_k, cur_logits_a_k, prompt_len_a_k = tf_sum_logprob_and_logits_over_response(
                                    train_model, enc_a, cand_token_ids[k], cfg.length_norm
                                )
                                term_a = (-(adv_i[k].detach()) * logp_a_k)
                                if cfg.use_ref_kl and ref_model is not None and cfg.lambda_kl_stage_a > 0.0:
                                    kl_a_k = kl_to_ref_with_cur_logits(ref_model, enc_a, cand_token_ids[k], cur_logits_a_k, int(prompt_len_a_k))
                                    term_a = term_a + cfg.lambda_kl_stage_a * kl_a_k
                                term_a.backward()
                                loss_a = loss_a + term_a.detach()

                    # 4) Combine and optimize
                    loss = loss_b_scalar + (cfg.stage_a_weight * loss_a)
                    # Clip and step
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

                    total_loss += float(loss.detach().item())
                    total_items += 1
                    try:
                        gn_val = float(_gn.item()) if hasattr(_gn, "item") else float(_gn)
                        grad_norm_sum += gn_val
                    except Exception:
                        pass

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
                            "images": per_image,
                            "stage_b_raw": replies_text[best_idx] if len(replies_text) > 0 else "",
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
                    ], device=dev, dtype=torch.float32)
                    if world_size > 1 and dist.is_initialized():
                        dist.all_reduce(vec_local, op=dist.ReduceOp.SUM)
                    # Aggregate into scalars
                    agg = aggregate_training_metrics(vec_local, world_size)
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
                    prefix = f"[{epoch_idx+1}/{int(cfg.epochs)}][{processed_so_far_g}/{epoch_total_g}] "

                    if rank == 0:
                        lrs = {str(g.get("name", f"group{i}")): float(g.get("lr", 0.0)) for i, g in enumerate(optimizer.param_groups)}
                        rank0_log(logger, tb_writer, metrics_writer, update, prefix, agg, lrs)
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
