#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

# Stable entry-points re-export (explicit)
from .config import RLRunnerConfig, load_and_validate_config
from .models import load_detection_model, load_processor, apply_freeze_and_param_groups, wrap_ddp_if_needed
from .rewards import REGISTRY as REWARD_REGISTRY

# Subpackages (compat layer)
from .data import RLGroupQCDataset, build_epoch_indices, iter_batches
from .prompting import GroupQCConversationBuilder
from .generation import (
    to_device_and_cast,
    decode_to_text,
    sft_style_preprocess_image,
    build_stage_a_stopping,
    build_stage_a_context_lines,
)
from .tf import (
    tf_sum_logprob_over_response,
    tf_sum_logprob_and_logits_over_response,
    tf_decision_probs,
    kl_to_ref_over_response,
    kl_to_ref_with_cur_logits,
    decision_margin,
    sequence_entropy_from_logits,
)
from .credit import (
    BaseCreditAssigner,
    OffAssigner,
    ConditionalAssigner,
    PairwiseFallbackAssigner,
    get_credit_assigner,
)
from .logging import aggregate_training_metrics, compute_eta, rank0_log
from .io import save_if_rank0

__all__ = [
    # Config & models
    "RLRunnerConfig",
    "load_and_validate_config",
    "load_detection_model",
    "load_processor",
    "apply_freeze_and_param_groups",
    "wrap_ddp_if_needed",
    "REWARD_REGISTRY",
    # Data
    "RLGroupQCDataset",
    "build_epoch_indices",
    "iter_batches",
    # Prompting
    "GroupQCConversationBuilder",
    # Generation
    "to_device_and_cast",
    "decode_to_text",
    "sft_style_preprocess_image",
    "build_stage_a_stopping",
    "build_stage_a_context_lines",
    # TF
    "tf_sum_logprob_over_response",
    "tf_sum_logprob_and_logits_over_response",
    "tf_decision_probs",
    "kl_to_ref_over_response",
    "kl_to_ref_with_cur_logits",
    "decision_margin",
    "sequence_entropy_from_logits",
    # Credit
    "BaseCreditAssigner",
    "OffAssigner",
    "ConditionalAssigner",
    "PairwiseFallbackAssigner",
    "get_credit_assigner",
    # Logging
    "aggregate_training_metrics",
    "compute_eta",
    "rank0_log",
    # IO
    "save_if_rank0",
]
