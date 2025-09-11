#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
from typing import Dict, Optional, TypedDict

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
          diag_cons_sum ]
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
    ) = vals

    tot_items_g = max(1.0, tot_items_g)
    avg_loss = tot_loss_g / tot_items_g
    avg_reward_best = rew_best_sum_g / tot_items_g
    var_best = max(0.0, (rew_best_sq_sum_g / tot_items_g) - (avg_reward_best * avg_reward_best))
    std_best = var_best ** 0.5
    acc_best = best_hit_g / tot_items_g
    acc_any = any_hit_g / tot_items_g
    avg_resp_len = resp_len_sum_g / tot_items_g
    avg_grad_norm = grad_norm_sum_g / tot_items_g
    avg_kl_b = (kl_b_sum_g / max(1.0, kl_b_count_g)) if kl_b_count_g > 0 else None
    # Phase‑A diagnostics means
    fmt_mean = diag_fmt_sum_g / tot_items_g
    cov_mean = diag_cov_sum_g / tot_items_g
    cln_mean = diag_cln_sum_g / tot_items_g
    tax_mean = diag_tax_sum_g / tot_items_g
    cons_mean = diag_cons_sum_g / tot_items_g

    out: AggregatedMetrics = {
        "loss": float(avg_loss),
        "reward_best_mean": float(avg_reward_best),
        "reward_best_std": float(std_best),
        "acc_best": float(acc_best),
        "acc_any": float(acc_any),
        "resp_len_mean": float(avg_resp_len),
        "grad_norm_mean": float(avg_grad_norm),
        "phase_a_formatting": float(fmt_mean),
        "phase_a_coverage": float(cov_mean),
        "phase_a_cleanliness": float(cln_mean),
        "phase_a_taxonomy": float(tax_mean),
        "phase_a_consistency": float(cons_mean),
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
    msg = prefix + (
        f"[train][grpo][rank0] update={step} "
        f"loss={scalars.get('loss', 0):.4f} "
        f"reward_best={scalars.get('reward_best_mean', 0):.3f}±{scalars.get('reward_best_std', 0):.3f} "
        f"acc_best={scalars.get('acc_best', 0):.3f} any_hit={scalars.get('acc_any', 0):.3f} "
        f"resp_len={scalars.get('resp_len_mean', 0):.1f} grad_norm={scalars.get('grad_norm_mean', 0):.2f} "
        f"eta_hours={scalars.get('eta_hours', 0):.2f}"
    )
    if 'kl_b_mean' in scalars:
        msg += f" kl_b={scalars['kl_b_mean']:.4f}"
    if 'phase_a_formatting' in scalars:
        msg += (
            f", phase_A: fmt={scalars['phase_a_formatting']:.3f} "
            f"cov={scalars['phase_a_coverage']:.3f} cln={scalars['phase_a_cleanliness']:.3f} "
            f"tax={scalars['phase_a_taxonomy']:.3f} cons={scalars['phase_a_consistency']:.3f}"
        )
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
            for k, v in {
                "train/loss": scalars.get('loss', 0),
                "train/reward_best_mean": scalars.get('reward_best_mean', 0),
                "train/reward_best_std": scalars.get('reward_best_std', 0),
                "train/acc_best": scalars.get('acc_best', 0),
                "train/acc_any": scalars.get('acc_any', 0),
                "train/resp_len_mean": scalars.get('resp_len_mean', 0),
                "train/grad_norm_mean": scalars.get('grad_norm_mean', 0),
                "train/eta_hours": scalars.get('eta_hours', 0),
            }.items():
                tb_writer.add_scalar(k, float(v), step)
            # Optional
            if 'kl_b_mean' in scalars:
                tb_writer.add_scalar('train/kl_b_mean', float(scalars['kl_b_mean']), step)
            for k, v in {
                "phase_a/formatting": scalars.get('phase_a_formatting', None),
                "phase_a/coverage": scalars.get('phase_a_coverage', None),
                "phase_a/cleanliness": scalars.get('phase_a_cleanliness', None),
                "phase_a/taxonomy": scalars.get('phase_a_taxonomy', None),
                "phase_a/consistency": scalars.get('phase_a_consistency', None),
            }.items():
                if v is not None:
                    tb_writer.add_scalar(k, float(v), step)
            for name, lr in lrs.items():
                tb_writer.add_scalar(f"lr/{name}", float(lr), step)
        except Exception:
            pass
