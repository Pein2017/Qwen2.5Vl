#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-B sampling and reward composition utilities."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.generation import LogitsProcessorList

from src_post.generation.generation import decode_to_text
from src_post.pipeline.dto import (
    StageBCandidate,
    StageBResult,
    StageBSamplingDiagnostics,
)
from src_post.prompting.span_parser import parse_stage_b_output
from src_post.rewards.compose import compose_reward
from src_post.rewards.utils import l1_normalize


@dataclass
class StageBSamplerConfig:
    """Static configuration for Stage-B sampling."""

    K_B: int
    reward_names: List[str]
    reward_weights: List[float]
    group_reward_mode: str
    mission: Optional[str]
    soft_overlong_penalty_enabled: bool = False
    soft_overlong_penalty_weight: Optional[float] = None


class StageBSampler:
    """Encapsulates Stage-B generation, reward composition, and diagnostics."""

    def __init__(
        self,
        policy,
        processor,
        generation_config,
        base_logits_processors: Optional[Sequence] = None,
        config: Optional[StageBSamplerConfig] = None,
        reward_composer: Callable[..., float] = compose_reward,
        parse_fn: Callable[[str], Dict[str, Optional[str]]] = parse_stage_b_output,
    ) -> None:
        self._policy = policy
        self._processor = processor
        self._generation_config = generation_config
        self._base_logits_processors = list(base_logits_processors or [])
        self._config = config or StageBSamplerConfig(
            K_B=1,
            reward_names=["group_margin"],
            reward_weights=[1.0],
            group_reward_mode="margin_only",
            mission=None,
        )
        self._reward_composer = reward_composer
        self._parse_fn = parse_fn

    def _clone_logits_processors(self) -> LogitsProcessorList:
        return LogitsProcessorList(list(self._base_logits_processors))

    def _normalize_rewards(
        self, reward_names: Iterable[str], reward_weights: Iterable[float]
    ) -> Tuple[List[str], List[float]]:
        names = list(reward_names)
        weights = list(reward_weights)
        mode = (self._config.group_reward_mode or "").strip().lower()
        if mode == "margin_only":
            return ["group_margin"], [1.0]
        normalized = l1_normalize(weights)
        return names, normalized

    def sample_and_score(
        self,
        enc_b: Dict[str, torch.Tensor],
        gt_label: str,
        context_lines: List[str],
        checklist_lines: List[str],
        tf_p_pass: float,
        tf_p_fail: float,
        prefix_allowed_tokens_fn: Optional[Callable] = None,
        baseline_margin: Optional[float] = None,
        decision_ce: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> StageBResult:
        """Sample K_B Stage-B candidates and compose rewards."""

        sample_count = max(
            1,
            int(num_samples) if num_samples is not None else int(self._config.K_B),
        )
        logits_processors = self._clone_logits_processors()
        gen_kwargs = {
            "generation_config": self._generation_config,
            "logits_processor": logits_processors,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if prefix_allowed_tokens_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        candidates: List[StageBCandidate] = []
        raw_texts: List[str] = []
        reasons: List[str] = []
        rewards: List[float] = []

        start_ts = time.time()
        for _ in range(sample_count):
            with torch.no_grad():
                out = self._policy.generate(
                    **enc_b,
                    **gen_kwargs,
                )
            sequences = out.sequences if hasattr(out, "sequences") else out[0]
            new_ids = sequences[:, enc_b["input_ids"].size(1) :]
            token_ids = new_ids[0].tolist()
            if not token_ids:
                token_ids = [int(self._processor.tokenizer.eos_token_id)]
            text_out = decode_to_text(self._processor, token_ids)
            parsed = self._parse_fn(text_out)
            pred_label = parsed.get("label")
            reason = parsed.get("reason")

            log_probs: Optional[List[float]] = None
            try:
                scores = out.scores if hasattr(out, "scores") else None
                if scores is not None and len(scores) == len(token_ids):
                    log_probs = []
                    for step_idx, score in enumerate(scores):
                        log_soft = F.log_softmax(score[0], dim=-1)
                        tid = token_ids[step_idx]
                        log_probs.append(float(log_soft[tid].detach().cpu().item()))
            except Exception:
                log_probs = None

            reward_names, reward_weights = self._normalize_rewards(
                self._config.reward_names, self._config.reward_weights
            )
            hit_max = False
            try:
                max_len_b = int(self._generation_config.max_new_tokens)
                hit_max = len(token_ids) >= max_len_b and (
                    self._processor.tokenizer.eos_token_id not in token_ids
                )
            except Exception:
                hit_max = False
            meta = {
                "stage_b_hit_max": bool(hit_max),
                "soft_overlong_penalty_weight": float(
                    self._config.soft_overlong_penalty_weight or 0.0
                )
                if bool(self._config.soft_overlong_penalty_enabled)
                else 0.0,
            }
            reward = self._reward_composer(
                gt_label=str(gt_label),
                pred_label=pred_label,
                summary_lines=context_lines,
                reason=reason,
                checklist_lines=checklist_lines,
                reward_names=reward_names,
                reward_weights=reward_weights,
                tf_p_pass=tf_p_pass,
                tf_p_fail=tf_p_fail,
                stage_b_text=text_out,
                mission=self._config.mission,
                meta=meta,
            )
            if not (reward == reward and abs(reward) != float("inf")):
                reward = 0.0

            candidates.append(
                StageBCandidate(
                    text=text_out,
                    token_ids=list(token_ids),
                    reward=float(reward),
                    label=pred_label,
                    reason=reason,
                    generation_logps=log_probs,
                )
            )
            raw_texts.append(text_out)
            reasons.append(reason or "")
            rewards.append(float(reward))

        latency = time.time() - start_ts
        diag: Optional[StageBSamplingDiagnostics] = None
        if rewards:
            mean = statistics.mean(rewards)
            std = (
                statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            )
            duplicate_count = len(raw_texts) - len(set(raw_texts))
            diag = StageBSamplingDiagnostics(
                reward_mean=float(mean),
                reward_std=float(std),
                duplicate_count=int(duplicate_count),
                raw_texts=list(raw_texts),
                reasons=list(reasons),
            )
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
        else:
            best_idx = None
            diag = StageBSamplingDiagnostics(
                reward_mean=0.0,
                reward_std=0.0,
                duplicate_count=0,
                raw_texts=[],
                reasons=[],
            )

        result = StageBResult(
            candidates=candidates,
            tf_p_pass=float(tf_p_pass),
            tf_p_fail=float(tf_p_fail),
            latency_sampling_sec=float(latency),
            best_index=best_idx,
            baseline_margin=baseline_margin,
            decision_ce=decision_ce,
            diagnostics=diag,
        )
        result.validate()
        return result
