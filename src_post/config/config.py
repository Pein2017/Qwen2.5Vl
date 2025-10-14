#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Typed configuration payload definitions for src_post.

This module mirrors the legacy `RLRunnerConfig` structure. Future refactors
will introduce additional sectioned dataclasses, but the runner still consumes
this single dataclass for compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RLRunnerConfig:
    """Typed configuration payload consumed by the RL runner."""

    # Required paths (explicit)
    checkpoint: str
    processor: str
    output_dir: str
    # Dataset (one of the two must be provided)
    train_data_dir: Optional[str]
    eval_data_dir: Optional[str]
    # Sampling balance
    balance_pass_fail: bool
    # Optional outputs
    results_jsonl: Optional[str]
    metrics_jsonl: Optional[str]
    # Mission
    mission: Optional[str]
    # Runtime
    device: str
    seed: int
    limit_groups: int
    skip_save_checkpoints: bool
    # Logging
    tb_log_dir: Optional[str]
    run_name: Optional[str]
    log_step: int
    # Generation
    temperature: float
    top_p: float  # recommended Stage-A baseline: temperature=0.01, top_p=0.5
    max_new_tokens_stage_a: int
    max_new_tokens_stage_b: int
    # Optional stage-specific generation overrides
    temperature_stage_a: Optional[float]
    temperature_stage_b: Optional[float]
    top_p_stage_a: Optional[float]
    top_p_stage_b: Optional[float]
    repetition_penalty_stage_a: Optional[float]
    repetition_penalty_stage_b: Optional[float]
    min_new_tokens_stage_a: Optional[int]
    min_new_tokens_stage_b: Optional[int]
    no_repeat_ngram_size_stage_a: Optional[int]
    no_repeat_ngram_size_stage_b: Optional[int]
    mask_geometry_tokens: bool
    mask_coordinate_tokens: bool
    sanitize_stage_a: bool
    # GRPO sampling
    K_B: int
    K_A: int
    adv_clip: float
    length_norm: bool
    # Training
    train: bool
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    epochs: int
    drop_last: bool
    # Selective unfreeze + per-group LRs
    llm_top_k_block: int
    vision_top_k_block: int
    freeze_patch_embed: bool
    aligner_lr: Optional[float]
    llm_lr: Optional[float]
    vision_lr: Optional[float]
    # Legacy/compat knobs (kept to avoid breakage; not primary path)
    train_lm_head: bool
    train_last_n_layers: int
    lr_lm_head: Optional[float]
    lr_last_layers: Optional[float]
    # KL to reference policy
    use_ref_kl: bool
    ref_checkpoint: Optional[str]
    lambda_kl_stage_b: float
    lambda_kl_stage_a: float
    # Stage‑A training mode
    train_stage_a_mode: str  # 'off' | 'conditional' | 'joint'
    stage_a_weight: float
    K_set: int
    max_images_tf: int
    # Rewards
    reward_fns: List[str]
    reward_weights: List[float]
    # Diagnostics
    enable_phase_a_diagnostics: bool
    # New: Stage‑B control and prompt bias
    train_stage_b: bool
    stage_b_weight: float
    freeze_stage_b_steps: int
    use_mission_checklist: bool
    # New: group reward strategy
    group_reward_mode: str  # 'margin_only' | 'combined'
    # New: pairwise fallback
    pairwise_credit_enabled: bool
    pairwise_pairs_per_group: int
    pairwise_delta_threshold: float
    # New: uncertainty gate
    use_uncertainty_gate: bool
    uncertainty_gate_min_entropy: float
    # New: optimizer/metrics extras
    grad_accum_steps: int
    decision_ce_ema_beta: float
    # New: toggles
    train_aligner: bool

    # ---- Enhancements (optional toggles; safe defaults) ----
    # Stage-B: clipped GRPO
    enable_clipped_grpo: bool
    epsilon_low: Optional[float]
    epsilon_high: Optional[float]
    loss_type_stage_b: Optional[str]  # {'grpo','bnpo','dr_grpo'} when enabled
    # Stage-B: entropy mask
    enable_entropy_mask_stage_b: bool
    entropy_top_quantile_stage_b: Optional[float]
    entropy_min_threshold_stage_b: Optional[float]
    # Sampling stability
    max_resample_times: int
    # Stage-A focusing & gating
    stage_a_top_m: int
    uncertainty_decay_factor: float
    pairwise_select: str  # {'heuristic','entropy','delta'}
    # Logging & shaping
    log_all_candidates: bool
    soft_overlong_penalty_enabled: bool
    soft_overlong_penalty_weight: Optional[float]
    # Stage-A token bias (optional)
    stage_a_bias_fail_enabled: bool
    stage_a_bias_fail_value: float
    stage_a_bias_pass_enabled: bool
    stage_a_bias_pass_value: float
    # Checkpoint saving (interval + retention)
    save_step: int
    save_limit: int

    # ---- Prompt enhancements (optional toggles; safe defaults) ----
    # Stage-B diversity & uncertainty guidance
    enable_stage_b_diversity_prompt: bool = False
    enable_stage_b_uncertainty_prompt: bool = False
    # Stage-A negative cues emphasis
    enable_stage_a_negative_cues_prompt: bool = False
    stage_a_negative_examples: Optional[List[str]] = None
