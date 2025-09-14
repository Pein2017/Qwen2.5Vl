#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
from typing import Dict, Optional, TypedDict
import os

import torch
import torch.distributed as dist


class AggregatedMetrics(TypedDict, total=False):
    loss: float
    reward_best_mean: float
    reward_best_std: float
    acc_best: float
    acc_any: float
    resp_len_mean: float
    grad_norm_mean: float
    eta_hours: float
    kl_b_mean: float
    phase_a_formatting: float
    phase_a_coverage: float
    phase_a_cleanliness: float
    phase_a_taxonomy: float
    phase_a_consistency: float
    skip_updates_std0: float
    pairwise_trigger_rate: float
    phase_a_entropy_mean: float
    reward_margin_best: float
    flip_rate: float
    decision_ce_mean: float
    decision_ce_ema: float
    # New classification metrics
    accuracy: float
    fn_rate: float


def reduce_sum(t: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def reduce_max(t: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


def compute_eta(
    dt_max_sec: float,
    processed_so_far_g: int,
    epoch_total_g: int,
    epoch_idx: int,
    epochs: int,
    items_per_update_g: int,
) -> float:
    remaining_epoch_groups = max(0, int(epoch_total_g - processed_so_far_g))
    remaining_future_epochs = max(0, int(max(1, epochs)) - (epoch_idx + 1))
    remaining_groups_all = int(remaining_epoch_groups + remaining_future_epochs * epoch_total_g)
    if items_per_update_g <= 0:
        return 0.0
    eta_sec = dt_max_sec * (float(remaining_groups_all) / float(items_per_update_g))
    return eta_sec / 3600.0


def aggregate_training_metrics(vec_local: torch.Tensor, world_size: int) -> AggregatedMetrics:
    """Aggregate the runner's metrics vector across ranks and compute means/std.

    Expects vec_local layout:
        [ total_items,
          total_loss,
          reward_best_sum,
          reward_best_sq_sum,
          best_hit_count,
          any_hit_count,
          resp_len_sum,
          grad_norm_sum,
          kl_b_sum,
          kl_b_count,
          diag_fmt_sum,
          diag_cov_sum,
          diag_cln_sum,
          diag_tax_sum,
          diag_cons_sum,
          skip_std0_count,
          pairwise_trigger_count,
          entropy_sum,
          margin_best_sum,
          flip_hit_count,
          decision_ce_sum,
          decision_ce_count,
          fn_count,
          gt_fail_count ]
    """
    dev = vec_local.device
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(vec_local, op=dist.ReduceOp.SUM)
    vals = [float(x) for x in vec_local.tolist()]
    (
        tot_items_g,
        tot_loss_g,
        rew_best_sum_g,
        rew_best_sq_sum_g,
        best_hit_g,
        any_hit_g,
        resp_len_sum_g,
        grad_norm_sum_g,
        kl_b_sum_g,
        kl_b_count_g,
        diag_fmt_sum_g,
        diag_cov_sum_g,
        diag_cln_sum_g,
        diag_tax_sum_g,
        diag_cons_sum_g,
        skip_std0_count_g,
        pairwise_trigger_count_g,
        entropy_sum_g,
        margin_best_sum_g,
        flip_hit_count_g,
        decision_ce_sum_g,
        decision_ce_count_g,
        fn_count_g,
        gt_fail_count_g,
    ) = vals

    tot_items_g = max(1.0, tot_items_g)
    avg_loss = tot_loss_g / tot_items_g
    avg_reward_best = rew_best_sum_g / tot_items_g
    var_best = max(0.0, (rew_best_sq_sum_g / tot_items_g) - (avg_reward_best * avg_reward_best))
    std_best = var_best ** 0.5
    acc_best = best_hit_g / tot_items_g
    acc_any = any_hit_g / tot_items_g
    # New: overall accuracy and false negative rate
    accuracy = acc_best
    fn_rate = (fn_count_g / max(1.0, gt_fail_count_g)) if gt_fail_count_g > 0 else 0.0
    avg_resp_len = resp_len_sum_g / tot_items_g
    avg_grad_norm = grad_norm_sum_g / tot_items_g
    avg_kl_b = (kl_b_sum_g / max(1.0, kl_b_count_g)) if kl_b_count_g > 0 else None
    # Phase‑A diagnostics means
    fmt_mean = diag_fmt_sum_g / tot_items_g
    cov_mean = diag_cov_sum_g / tot_items_g
    cln_mean = diag_cln_sum_g / tot_items_g
    tax_mean = diag_tax_sum_g / tot_items_g
    cons_mean = diag_cons_sum_g / tot_items_g
    # New metrics
    skip_updates_std0 = skip_std0_count_g / tot_items_g
    pairwise_trigger_rate = pairwise_trigger_count_g / tot_items_g
    phase_a_entropy_mean = entropy_sum_g / tot_items_g
    reward_margin_best = margin_best_sum_g / tot_items_g
    flip_rate = flip_hit_count_g / tot_items_g
    decision_ce_mean = (decision_ce_sum_g / max(1.0, decision_ce_count_g)) if decision_ce_count_g > 0 else 0.0

    out: AggregatedMetrics = {
        "loss": float(avg_loss),
        "reward_best_mean": float(avg_reward_best),
        "reward_best_std": float(std_best),
        "acc_best": float(acc_best),
        "acc_any": float(acc_any),
        "accuracy": float(accuracy),
        "fn_rate": float(fn_rate),
        "resp_len_mean": float(avg_resp_len),
        "grad_norm_mean": float(avg_grad_norm),
        "phase_a_formatting": float(fmt_mean),
        "phase_a_coverage": float(cov_mean),
        "phase_a_cleanliness": float(cln_mean),
        "phase_a_taxonomy": float(tax_mean),
        "phase_a_consistency": float(cons_mean),
        "skip_updates_std0": float(skip_updates_std0),
        "pairwise_trigger_rate": float(pairwise_trigger_rate),
        "phase_a_entropy_mean": float(phase_a_entropy_mean),
        "reward_margin_best": float(reward_margin_best),
        "flip_rate": float(flip_rate),
        "decision_ce_mean": float(decision_ce_mean),
    }
    if avg_kl_b is not None:
        out["kl_b_mean"] = float(avg_kl_b)
    return out


def rank0_log(
    logger: logging.Logger,
    tb_writer: Optional[object],
    metrics_writer: Optional[object],
    step: int,
    prefix: str,
    scalars: Dict[str, float],
    lrs: Dict[str, float],
) -> None:
    use_window_only = str(os.environ.get("METRICS_WINDOW_ONLY", "1")).strip().lower() in {"1", "true", "yes"}
    def pick(key: str, default: float = 0.0) -> float:
        if use_window_only and (f"{key}_win" in scalars):
            return float(scalars.get(f"{key}_win", default))
        return float(scalars.get(key, default))

    msg = prefix + (
        f"[train][grpo][rank0] update={step} "
        f"loss={pick('loss'):.4f} "
        f"reward_best={pick('reward_best_mean'):.3f}±{pick('reward_best_std'):.3f} "
        f"acc_best={pick('acc_best'):.3f} any_hit={pick('acc_any'):.3f} "
        f"acc={pick('accuracy'):.3f} fn_rate={pick('fn_rate'):.3f} "
        f"resp_len={pick('resp_len_mean'):.1f} grad_norm={pick('grad_norm_mean'):.2f} "
        f"eta_hours={scalars.get('eta_hours', 0):.2f}"
    )
    if 'kl_b_mean' in scalars:
        val = pick('kl_b_mean')
        msg += f" kl_b={val:.4f}"
    if 'skip_updates_std0' in scalars:
        msg += f" skip_std0={pick('skip_updates_std0'):.3f} pairwise={pick('pairwise_trigger_rate'):.3f}"
    if 'phase_a_formatting' in scalars:
        msg += (
            f", phase_A: fmt={pick('phase_a_formatting'):.3f} "
            f"cov={pick('phase_a_coverage'):.3f} cln={pick('phase_a_cleanliness'):.3f} "
            f"tax={pick('phase_a_taxonomy'):.3f} cons={pick('phase_a_consistency'):.3f}"
        )
    if 'phase_a_entropy_mean' in scalars:
        msg += f" ent={pick('phase_a_entropy_mean'):.3f}"
    if 'reward_margin_best' in scalars:
        msg += f" margin_best={pick('reward_margin_best'):.3f} flip_rate={pick('flip_rate'):.3f}"
    if 'decision_ce_mean' in scalars:
        msg += f" dec_ce={pick('decision_ce_mean'):.3f}"
    if 'decision_ce_ema' in scalars:
        msg += f" dec_ce_ema={pick('decision_ce_ema'):.3f}"
    logger.info(msg)
    if metrics_writer is not None:
        try:
            rec = dict(scalars)
            rec['update'] = int(step)
            rec['lrs'] = dict(lrs)
            metrics_writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
    if tb_writer is not None:
        try:
            # Prefer windowed values for TB when enabled
            tb_values = {
                "train/loss": pick('loss'),
                "train/reward_best_mean": pick('reward_best_mean'),
                "train/reward_best_std": pick('reward_best_std'),
                "train/acc_best": pick('acc_best'),
                "train/acc_any": pick('acc_any'),
                "train/accuracy": pick('accuracy'),
                "train/fn_rate": pick('fn_rate'),
                "train/resp_len_mean": pick('resp_len_mean'),
                "train/grad_norm_mean": pick('grad_norm_mean'),
                "train/eta_hours": scalars.get('eta_hours', 0),
                "train/decision_ce_mean": pick('decision_ce_mean') if 'decision_ce_mean' in scalars or f"decision_ce_mean_win" in scalars else None,
                "train/decision_ce_ema": pick('decision_ce_ema') if 'decision_ce_ema' in scalars or f"decision_ce_ema_win" in scalars else None,
                "train/skip_std0": pick('skip_updates_std0') if 'skip_updates_std0' in scalars or f"skip_updates_std0_win" in scalars else None,
                "train/pairwise_trigger_rate": pick('pairwise_trigger_rate') if 'pairwise_trigger_rate' in scalars or f"pairwise_trigger_rate_win" in scalars else None,
                "train/reward_margin_best": pick('reward_margin_best') if 'reward_margin_best' in scalars or f"reward_margin_best_win" in scalars else None,
                "train/flip_rate": pick('flip_rate') if 'flip_rate' in scalars or f"flip_rate_win" in scalars else None,
            }
            for k, v in tb_values.items():
                if v is not None:
                    tb_writer.add_scalar(k, float(v), step)
            # Optional
            if 'kl_b_mean' in scalars:
                tb_writer.add_scalar('train/kl_b_mean', float(pick('kl_b_mean')), step)
            for k, v in {
                "phase_a/formatting": pick('phase_a_formatting') if 'phase_a_formatting' in scalars or f"phase_a_formatting_win" in scalars else None,
                "phase_a/coverage": pick('phase_a_coverage') if 'phase_a_coverage' in scalars or f"phase_a_coverage_win" in scalars else None,
                "phase_a/cleanliness": pick('phase_a_cleanliness') if 'phase_a_cleanliness' in scalars or f"phase_a_cleanliness_win" in scalars else None,
                "phase_a/taxonomy": pick('phase_a_taxonomy') if 'phase_a_taxonomy' in scalars or f"phase_a_taxonomy_win" in scalars else None,
                "phase_a/consistency": pick('phase_a_consistency') if 'phase_a_consistency' in scalars or f"phase_a_consistency_win" in scalars else None,
            }.items():
                if v is not None:
                    tb_writer.add_scalar(k, float(v), step)
            for name, lr in lrs.items():
                tb_writer.add_scalar(f"lr/{name}", float(lr), step)
        except Exception:
            pass
