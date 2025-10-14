#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from src_post.tf.teacher_forcing import (
    decision_margin,
    kl_to_ref_over_response,
    kl_to_ref_with_cur_logits,
    sequence_entropy_from_logits,
    tf_decision_probs,
    tf_sum_logprob_and_logits_over_response,
    tf_sum_logprob_over_response,
)


__all__ = [
    "tf_sum_logprob_over_response",
    "tf_sum_logprob_and_logits_over_response",
    "tf_decision_probs",
    "kl_to_ref_over_response",
    "kl_to_ref_with_cur_logits",
    "decision_margin",
    "sequence_entropy_from_logits",
]
