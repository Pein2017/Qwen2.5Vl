#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import GenerationConfig, Qwen2VLProcessor
from transformers.generation import LogitsProcessorList

from src_new.models.wrapper import DetectionModel
from src_post.config import RLRunnerConfig, load_and_validate_config
from src_post.data.data_loader import (
    build_balanced_indices,
    build_epoch_indices,
    iter_batches,
)
from src_post.data.dataset_group_qc import RLGroupQCDataset
from src_post.generation.generation import build_stage_a_stopping
from src_post.io.checkpoints import save_if_rank0
from src_post.logging.logging_utils import (
    aggregate_training_metrics,
    compute_eta,
    rank0_log,
)
from src_post.logging.setup import configure_logging
from src_post.models import load_detection_model, load_processor, wrap_ddp_if_needed
from src_post.pipeline import (
    StageASampler,
    StageASamplerConfig,
    StageBCandidate,
    StageBSampler,
    StageBSamplerConfig,
    StageBSamplingDiagnostics,
)
from src_post.prompting.conversation import GroupQCConversationBuilder
from src_post.training.grpo_loop import GroupMetrics, GRPOLoop
from src_post.training.grpo_trainer import GRPOTrainer, TrainerComponents
from src_post.training.optim import build_optimizer, build_scheduler


logger = logging.getLogger("src_post.runner")


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
        dev = (
            f"cuda:{local_rank}"
            if world_size > 1
            else (device if ":" in device else "cuda:0")
        )
        torch.cuda.set_device(dev)
        return dev
    return device


def _compute_stage_b_diagnostics(
    candidates: List[StageBCandidate],
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
    text_values = [c.text for c in candidates]
    reason_values = [c.reason or "" for c in candidates]
    mean_val = statistics.mean(rewards)
    std_val = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    dup_count = len(text_values) - len(set(text_values))
    return StageBSamplingDiagnostics(
        reward_mean=float(mean_val),
        reward_std=float(std_val),
        duplicate_count=int(dup_count),
        raw_texts=text_values,
        reasons=reason_values,
    )


class RLRunner:
    def __init__(self, cfg: RLRunnerConfig) -> None:
        self.cfg = cfg
        self._decision_ce_ema_state: Optional[float] = None
        # placeholder for future trainer, metrics handled in trainer to avoid double tracking
        self._eta_groups_done: int = 0
        self._eta_time_sum_sec: float = 0.0

    def run(self) -> None:
        cfg = self.cfg
        # Device + DDP
        device = _bind_device(str(cfg.device))
        _maybe_init_ddp()

        # Debug toggles
        debug_verbose = (
            str(os.environ.get("DEBUG_GROUP_QC", "0")).strip().lower()
            in {"1", "true", "yes"}
        ) or bool(getattr(cfg, "debug_verbose", False))
        rank, world_size, _ = _get_dist_info()
        repeat_filter = configure_logging(rank, debug=debug_verbose)
        stop_after_updates_env = os.environ.get("STOP_AFTER_UPDATES", "0")
        try:
            stop_after_updates: int = (
                int(stop_after_updates_env)
                if str(stop_after_updates_env).strip()
                else 0
            )
        except Exception:
            stop_after_updates = 0
        if debug_verbose:
            logger.info(
                f"[debug] mission={cfg.mission} | K_B={cfg.K_B} K_A={cfg.K_A} adv_clip={cfg.adv_clip} "
                f"tempA/B={getattr(cfg, 'temperature_stage_a', cfg.temperature)}/{getattr(cfg, 'temperature_stage_b', cfg.temperature)} "
                f"top_pA/B={getattr(cfg, 'top_p_stage_a', cfg.top_p)}/{getattr(cfg, 'top_p_stage_b', cfg.top_p)} "
                f"len_norm={cfg.length_norm} | use_ref_kl={cfg.use_ref_kl} | group_reward_mode={cfg.group_reward_mode}"
            )
            logger.info(
                f"[debug] output_dir={cfg.output_dir} results={cfg.results_jsonl} metrics={cfg.metrics_jsonl} tb_log_dir={cfg.tb_log_dir} run_name={cfg.run_name}"
            )
            if stop_after_updates > 0:
                logger.info(
                    f"[debug] STOP_AFTER_UPDATES={stop_after_updates} (early-stop enabled)"
                )

        # Reduce verbosity on non-zero ranks
        if rank != 0:
            os.environ.setdefault("TQDM_DISABLE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        # Seeding
        try:
            import random

            import numpy as np

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
            ref_model = load_detection_model(
                cfg.ref_checkpoint or cfg.checkpoint, processor, device
            )
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

        conv_builder = GroupQCConversationBuilder(processor=processor)

        # Logits processor for decode-time masking
        logits_processors = LogitsProcessorList()

        # Stage-A stopping
        stopping_stage_a = build_stage_a_stopping(processor)

        # Generation configs
        stage_a_temp = float(getattr(cfg, "temperature_stage_a", cfg.temperature))
        stage_a_top_p = float(getattr(cfg, "top_p_stage_a", cfg.top_p))
        stage_b_temp = float(getattr(cfg, "temperature_stage_b", cfg.temperature))
        stage_b_top_p = float(getattr(cfg, "top_p_stage_b", cfg.top_p))
        # Optional overrides per user preference (remain None if not provided)
        rep_a = getattr(cfg, "repetition_penalty_stage_a", None)
        rep_b = getattr(cfg, "repetition_penalty_stage_b", None)
        min_a = getattr(cfg, "min_new_tokens_stage_a", None)
        min_b = getattr(cfg, "min_new_tokens_stage_b", None)
        ngram_a = getattr(cfg, "no_repeat_ngram_size_stage_a", None)
        ngram_b = getattr(cfg, "no_repeat_ngram_size_stage_b", None)

        gen_cfg_stage_a = GenerationConfig(
            do_sample=(stage_a_temp > 0.0),
            temperature=stage_a_temp,
            top_p=stage_a_top_p,
            max_new_tokens=max(1, min(cfg.max_new_tokens_stage_a, 64)),
            min_new_tokens=int(min_a) if isinstance(min_a, int) else 1,
            repetition_penalty=float(rep_a) if isinstance(rep_a, float) else 1.2,
            no_repeat_ngram_size=int(ngram_a) if isinstance(ngram_a, int) else 8,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        gen_cfg_stage_b = GenerationConfig(
            do_sample=(stage_b_temp > 0.0),
            temperature=stage_b_temp,
            top_p=stage_b_top_p,
            max_new_tokens=max(1, min(cfg.max_new_tokens_stage_b, 128)),
            min_new_tokens=int(min_b) if isinstance(min_b, int) else 1,
            repetition_penalty=float(rep_b) if isinstance(rep_b, float) else 1.3,
            no_repeat_ngram_size=int(ngram_b) if isinstance(ngram_b, int) else 16,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        stage_a_sampler = StageASampler(
            policy=policy,
            processor=processor,
            conv_builder=conv_builder,
            generation_config=gen_cfg_stage_a,
            base_logits_processors=list(logits_processors),
            stopping=stopping_stage_a,
            config=StageASamplerConfig(
                sanitize=bool(cfg.sanitize_stage_a),
                mission=cfg.mission,
                enable_diagnostics=bool(cfg.enable_phase_a_diagnostics),
            ),
        )
        stage_b_sampler = StageBSampler(
            policy=policy,
            processor=processor,
            generation_config=gen_cfg_stage_b,
            base_logits_processors=list(logits_processors),
            config=StageBSamplerConfig(
                K_B=int(cfg.K_B),
                reward_names=list(cfg.reward_fns),
                reward_weights=list(cfg.reward_weights),
                group_reward_mode=str(cfg.group_reward_mode),
                mission=cfg.mission,
                soft_overlong_penalty_enabled=bool(cfg.soft_overlong_penalty_enabled),
                soft_overlong_penalty_weight=cfg.soft_overlong_penalty_weight,
            ),
        )

        def _build_stage_b_diagnostics(
            candidates: List[StageBCandidate],
        ) -> StageBSamplingDiagnostics:
            if not candidates:
                return StageBSamplingDiagnostics(
                    reward_mean=0.0,
                    reward_std=0.0,
                    duplicate_count=0,
                    raw_texts=[],
                    reasons=[],
                )
            rewards_local = [float(c.reward) for c in candidates]
            mean_local = statistics.mean(rewards_local)
            std_local = (
                statistics.pstdev(rewards_local) if len(rewards_local) > 1 else 0.0
            )
            texts_local = [c.text for c in candidates]
            reasons_local = [c.reason or "" for c in candidates]
            dup_local = len(texts_local) - len(set(texts_local))
            return StageBSamplingDiagnostics(
                reward_mean=float(mean_local),
                reward_std=float(std_local),
                duplicate_count=int(dup_local),
                raw_texts=texts_local,
                reasons=reasons_local,
            )

        # Dataset
        ds_path = cfg.train_data_dir or cfg.eval_data_dir
        if ds_path is None:
            logger.info(
                "No dataset provided (train_data_dir or eval_data_dir); exiting."
            )
            return
        dataset = RLGroupQCDataset(jsonl_or_dir_path=ds_path, preload=False)
        logger.info(f"Loaded dataset: {ds_path} with {len(dataset)} groups")
        if debug_verbose:
            logger.info(f"[debug] Dataset path resolved to: {Path(ds_path).resolve()}")

        optimizer, param_group_infos = build_optimizer(policy, processor.tokenizer, cfg)
        rank, world_size, _ = _get_dist_info()
        lr_scheduler, updates_per_epoch_rank = build_scheduler(
            cfg=cfg,
            optimizer=optimizer,
            dataset_len=len(dataset),
            world_size=world_size,
            rank=rank,
            limit_groups=int(cfg.limit_groups),
        )
        if debug_verbose:
            logger.info(
                f"[debug] updates_per_epoch_rank={updates_per_epoch_rank} "
                f"scheduler={'enabled' if lr_scheduler is not None else 'disabled'}"
            )

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
        grad_accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1)))
        try:
            metrics_window = max(1, int(os.environ.get("METRICS_WINDOW_UPDATES", "50")))
        except Exception:
            metrics_window = 50
        trainer: Optional[GRPOTrainer] = None
        try:
            trainer = GRPOTrainer(
                components=TrainerComponents(
                    model=train_model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                ),
                grad_accum_steps=grad_accum_steps,
                metrics_window=metrics_window,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize GRPOTrainer: {exc}")
        if trainer is None:
            raise RuntimeError("GRPOTrainer initialization returned None")
        accum_scale = 1.0 / float(grad_accum_steps)
        trainer.zero_grad()
        grpo_loop = GRPOLoop(
            cfg=cfg,
            device=device,
            trainer=trainer,
            stage_a_sampler=stage_a_sampler,
            stage_b_sampler=stage_b_sampler,
            conv_builder=conv_builder,
            processor=processor,
            policy=policy,
            train_model=train_model,
            ddp_policy=ddp_policy,
            ref_model=ref_model,
            gen_cfg_stage_a=gen_cfg_stage_a,
            gen_cfg_stage_b=gen_cfg_stage_b,
            logits_processors=logits_processors,
            stopping_stage_a=stopping_stage_a,
            tb_writer=tb_writer,
            debug_verbose=debug_verbose,
            rank=rank,
        )

        # Env override for skip_save
        skip_save_env = os.environ.get("SKIP_SAVE", "0").strip().lower()
        skip_save = skip_save_env in ("1", "true", "yes") or bool(
            cfg.skip_save_checkpoints
        )

        # Epoch loop
        early_stop = False
        for epoch_round in range(int(max(1, cfg.epochs))):
            epoch_idx = epoch_round
            # reset ETA accumulators per epoch
            self._eta_groups_done = 0
            self._eta_time_sum_sec = 0.0
            if bool(getattr(cfg, "balance_pass_fail", False)):
                try:
                    label_map = dataset.get_label_indices()
                    p_idx = label_map.get("pass", [])
                    f_idx = label_map.get("fail", [])
                    indices = build_balanced_indices(
                        pass_indices=p_idx,
                        fail_indices=f_idx,
                        world_size=world_size,
                        rank=rank,
                        limit_groups=int(cfg.limit_groups),
                        seed=int(cfg.seed),
                        epoch=epoch_idx,
                    )
                except Exception:
                    indices = build_epoch_indices(
                        dataset_len=len(dataset),
                        world_size=world_size,
                        rank=rank,
                        limit_groups=int(cfg.limit_groups),
                        seed=int(cfg.seed),
                        epoch=epoch_idx,
                    )
            else:
                indices = build_epoch_indices(
                    dataset_len=len(dataset),
                    world_size=world_size,
                    rank=rank,
                    limit_groups=int(cfg.limit_groups),
                    seed=int(cfg.seed),
                    epoch=epoch_idx,
                )
            if rank == 0:
                logger.info(
                    f"[data] Start epoch {epoch_round + 1}/{cfg.epochs}: num_local={len(indices)} (drop_last={cfg.drop_last})"
                )
            processed_so_far_local_epoch = 0
            epoch_total_local_len = len(indices)

            for batch_indices in iter_batches(indices, cfg.batch_size, cfg.drop_last):
                if early_stop:
                    break
                update_start_ts = time.time()
                batch_metrics = GroupMetrics()
                last_stage_b_diag: Optional[StageBSamplingDiagnostics] = None
                last_stage_b_candidate_count = 0

                for idx in batch_indices:
                    sample = dataset[idx]
                    try:
                        gid = sample.get("meta", {}).get("group_id", "-")
                    except Exception:
                        gid = "-"
                    gt_label = str(sample["label"]).strip().lower()
                    try:
                        logger.info(
                            f"[group] idx={idx} gid={gid} gt={gt_label} images={len(sample['images'])} "
                            f"K_B={int(cfg.K_B)} K_A={int(cfg.K_A)} mode_A={cfg.train_stage_a_mode} "
                            f"pairwise={bool(cfg.pairwise_credit_enabled)}"
                        )
                        if debug_verbose:
                            img_names = [
                                Path(p).name for p in sample.get("image_paths", [])
                            ]
                            logger.info(f"[debug] image_paths={img_names}")
                    except Exception:
                        pass

                    sync_now = trainer.should_sync()
                    group_output = grpo_loop.process_group(
                        idx=idx,
                        sample=sample,
                        accum_scale=accum_scale,
                        debug_verbose=debug_verbose,
                        sync_now=sync_now,
                    )

                    batch_metrics.add_inplace(group_output.metrics)

                    stage_b_result = group_output.stage_b_result
                    last_stage_b_diag = stage_b_result.diagnostics
                    last_stage_b_candidate_count = len(stage_b_result.candidates)
                    if rank == 0:
                        best_candidate = stage_b_result.best_candidate()
                        try:
                            if best_candidate is not None:

                                def _short(s: str, m: int = 120) -> str:
                                    s = str(s).strip()
                                    return (
                                        s
                                        if len(s) <= m
                                        else (s[: max(0, m - 1)].rstrip() + "…")
                                    )

                                best_reason_lines = best_candidate.text.splitlines()
                                best_reason = (
                                    best_reason_lines[1].strip()
                                    if len(best_reason_lines) >= 2
                                    else best_candidate.text.strip()
                                )
                                if best_reason.startswith(
                                    "原因:"
                                ) or best_reason.startswith("原因："):
                                    best_reason = best_reason[3:].strip()
                                logger.info(
                                    f"[stage-b/best] idx={idx} label={best_candidate.label} reward={best_candidate.reward:.3f} reason={_short(best_reason)}"
                                )
                        except Exception:
                            pass

                    if writer is not None and group_output.record is not None:
                        # attach current ETA (hours) to record for visibility
                        try:
                            # average seconds per group so far on this rank
                            self._eta_groups_done += 1
                            self._eta_time_sum_sec += float(group_output.dt_group)
                            avg_sec_per_group = (
                                self._eta_time_sum_sec / max(1, self._eta_groups_done)
                            )
                            remaining_current = max(0, epoch_total_local_len - self._eta_groups_done)
                            remaining_future_epochs = max(0, int(cfg.epochs) - (epoch_idx + 1))
                            eta_sec = avg_sec_per_group * (
                                remaining_current + remaining_future_epochs * epoch_total_local_len
                            )
                            eta_hours = float(eta_sec / 3600.0)
                            group_output.record["eta_hours"] = eta_hours
                        except Exception:
                            pass
                        writer.write(json.dumps(group_output.record, ensure_ascii=False) + "\n")

                    logger.info(
                        f"[group] idx={idx} done in {group_output.dt_group:.2f}s"
                    )
                    # also print ETA in hours based on running average (rank-local, rough)
                    try:
                        avg_sec_per_group = (
                            self._eta_time_sum_sec / max(1, self._eta_groups_done)
                        )
                        remaining_current = max(0, epoch_total_local_len - self._eta_groups_done)
                        remaining_future_epochs = max(0, int(cfg.epochs) - (epoch_idx + 1))
                        eta_sec = avg_sec_per_group * (
                            remaining_current + remaining_future_epochs * epoch_total_local_len
                        )
                        eta_hours = float(eta_sec / 3600.0)
                        logger.info(
                            f"[group] idx={idx} eta={eta_hours:.2f}h (rank-local rough estimate)"
                        )
                    except Exception:
                        pass

                    if sync_now:
                        if ddp_policy is not None:
                            gn = torch.nn.utils.clip_grad_norm_(
                                ddp_policy.parameters(), cfg.max_grad_norm
                            )
                        else:
                            gn = torch.nn.utils.clip_grad_norm_(
                                policy.parameters(), cfg.max_grad_norm
                            )
                        trainer.step()
                        try:
                            batch_metrics.grad_norm_sum += float(
                                gn.item() if hasattr(gn, "item") else gn
                            )
                        except Exception:
                            pass
                    trainer.advance_accum()
                    continue

                if batch_metrics.total_items > 0:
                    try:
                        processed_so_far_local_epoch += int(batch_metrics.total_items)
                    except Exception:
                        pass
                    # Global reductions
                    dev = torch.device(device)
                    vec_local = torch.tensor(
                        batch_metrics.to_list(), device=dev, dtype=torch.float32
                    )
                    sb_vec_local = torch.tensor(
                        [
                            batch_metrics.sb_clip_low_sum,
                            batch_metrics.sb_clip_high_sum,
                            batch_metrics.sb_clip_region_sum,
                            batch_metrics.sb_clip_den,
                            batch_metrics.sb_entropy_mask_true_sum,
                            batch_metrics.sb_reply_token_sum,
                        ],
                        device=dev,
                        dtype=torch.float32,
                    )
                    if world_size > 1 and dist.is_initialized():
                        dist.all_reduce(vec_local, op=dist.ReduceOp.SUM)
                        dist.all_reduce(sb_vec_local, op=dist.ReduceOp.SUM)
                    (
                        sb_clip_low_sum,
                        sb_clip_high_sum,
                        sb_clip_region_sum,
                        sb_clip_den,
                        sb_entropy_mask_true_sum,
                        sb_reply_token_sum,
                    ) = [float(x) for x in sb_vec_local.tolist()]
                    # Aggregate into scalars
                    agg = aggregate_training_metrics(vec_local, world_size)
                    # Update sliding-window metrics (over last N updates)
                    agg_win = trainer.update_metrics(agg)
                    # ETA
                    dt_local = float(time.time() - update_start_ts)
                    dt_t = torch.tensor([dt_local], device=dev, dtype=torch.float32)
                    if world_size > 1 and dist.is_initialized():
                        dist.all_reduce(dt_t, op=dist.ReduceOp.MAX)
                    dt_global = float(dt_t.item())
                    ep_total_t = torch.tensor(
                        [float(epoch_total_local_len)], device=dev, dtype=torch.float32
                    )
                    proc_so_far_t = torch.tensor(
                        [float(processed_so_far_local_epoch)],
                        device=dev,
                        dtype=torch.float32,
                    )
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
                        beta = float(getattr(cfg, "decision_ce_ema_beta", 0.9))
                        cur = float(agg.get("decision_ce_mean", 0.0))
                        if self._decision_ce_ema_state is None:
                            self._decision_ce_ema_state = cur
                        else:
                            self._decision_ce_ema_state = float(
                                beta * self._decision_ce_ema_state + (1.0 - beta) * cur
                            )
                        agg["decision_ce_ema"] = float(self._decision_ce_ema_state)
                    prefix = f"[{epoch_idx + 1}/{int(cfg.epochs)}][{processed_so_far_g}/{epoch_total_g}] "

                    if rank == 0:
                        # Advance step counter first for intuitive 1-based logging
                        update += 1
                        lrs = {
                            str(g.get("name", f"group{i}")): float(g.get("lr", 0.0))
                            for i, g in enumerate(optimizer.param_groups)
                        }
                        # Merge raw + windowed for logging/JSONL/TensorBoard
                        scalars_to_log = dict(agg)
                        scalars_to_log.update(agg_win)
                        # New Stage-B metrics
                        if sb_clip_den > 0.0:
                            scalars_to_log["sb_clip_low_mean"] = float(
                                sb_clip_low_sum / sb_clip_den
                            )
                            scalars_to_log["sb_clip_high_mean"] = float(
                                sb_clip_high_sum / sb_clip_den
                            )
                            scalars_to_log["sb_clip_region_mean"] = float(
                                sb_clip_region_sum / sb_clip_den
                            )
                        if sb_reply_token_sum > 0.0:
                            scalars_to_log["sb_entropy_mask_ratio"] = float(
                                sb_entropy_mask_true_sum / sb_reply_token_sum
                            )
                        suppressed = repeat_filter.suppressed_summary(reset=True)
                        if suppressed:
                            scalars_to_log["warnings/suppressed"] = float(
                                sum(suppressed.values())
                            )
                        if last_stage_b_diag is not None:
                            scalars_to_log["stage_b_reward_mean"] = float(
                                last_stage_b_diag.reward_mean
                            )
                            scalars_to_log["stage_b_reward_std"] = float(
                                last_stage_b_diag.reward_std
                            )
                            scalars_to_log["stage_b_duplicate_count"] = float(
                                last_stage_b_diag.duplicate_count
                            )
                            total_candidates = max(1, last_stage_b_candidate_count)
                            scalars_to_log["stage_b_duplicate_ratio"] = float(
                                last_stage_b_diag.duplicate_count / total_candidates
                            )
                        # Tunable logging cadence via cfg.log_step (default=10) + env + edge conditions
                        try:
                            step_mod = int(getattr(cfg, "log_step", 10) or 10)
                        except Exception:
                            step_mod = 10
                        if step_mod < 1:
                            step_mod = 1
                        force_every_step_env = str(
                            os.environ.get("FORCE_LOG_EVERY_STEP", "0")
                        ).strip().lower() in {"1", "true", "yes"}
                        end_of_epoch = processed_so_far_g >= epoch_total_g
                        # Always log the first few steps for visibility
                        do_log = (
                            force_every_step_env
                            or (update <= 3)
                            or ((update % step_mod) == 0)
                            or end_of_epoch
                        )
                        if do_log:
                            rank0_log(
                                logger,
                                tb_writer,
                                metrics_writer,
                                update,
                                prefix,
                                scalars_to_log,
                                lrs,
                            )
                        # Early stop when reaching STOP_AFTER_UPDATES
                        if stop_after_updates > 0 and update >= stop_after_updates:
                            logger.info(
                                f"[debug] STOP_AFTER_UPDATES reached at update={update}; early stopping..."
                            )
                            early_stop = True
                    # Respect early-stop across ranks
                    if world_size > 1 and dist.is_initialized():
                        flag = torch.tensor([1.0 if early_stop else 0.0], device=dev)
                        dist.all_reduce(flag, op=dist.ReduceOp.SUM)
                        early_stop = flag.item() > 0.0
                # end batch agg
            # end batch loop
            if early_stop:
                break
        # end epoch loop

        # Finalize after last epoch
        if True:
            try:
                save_if_rank0(
                    policy,
                    processor,
                    cfg.output_dir,
                    tag="grpo_final",
                    skip_save=bool(skip_save),
                )
                if writer is not None:
                    writer.close()
                    logger.info(f"Results written: {results_path}")
                if metrics_writer is not None:
                    metrics_writer.close()
                    logger.info(f"Metrics written: {metrics_path}")
                try:
                    if tb_writer is not None:
                        tb_writer.flush()
                        tb_writer.close()
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
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML/JSON config file"
    )
    args = parser.parse_args()
    cfg = load_and_validate_config(args.config)
    RLRunner(cfg).run()


if __name__ == "__main__":
    main()
