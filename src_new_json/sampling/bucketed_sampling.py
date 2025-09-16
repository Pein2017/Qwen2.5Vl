# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from random import Random
from collections import Counter


@dataclass(frozen=True)
class EpisodeSpec:
    target_idx: int
    context_idx: Optional[int]


@dataclass(frozen=True)
class SamplingMetrics:
    pair_episodes: int
    single_episodes: int
    target_is_current_pct: float
    avg_contrast_score: float  # kept as placeholder for compatibility
    coverage_context_unique_pct: float
    context_usage_histogram: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingConfig:
    candidate_pool_size: int
    max_teacher_uses_per_epoch: int
    temperature: float  # hardness temperature for weighting
    target_assignment: str  # {"random","current","opposite"}
    cross_bucket_explore_prob: float = 0.0
    # Large similarity pool controls (upper-limit only)
    pool_fraction: float = 0.3
    pool_max: int = 4096


class BucketedSamplingEngine:
    """Object-type overlap sampling with optional hardness weighting.

    - Each sample i acts as student once per epoch.
    - With Bernoulli(teacher_ratio), draw 1 teacher from a candidate pool C_i
      formed by the union of all samples that share at least one object type with i.
    - Candidate pool size is bounded ABOVE by min(pool_max, pool_fraction * N). No minimum enforced.
    - If hardness weights are provided, select teacher with softmax over normalized
      hardness using temperature; else uniform.
    - Optional small cross-bucket explore fallback when pool is empty.
    """

    @staticmethod
    def _map_prefix_to_bucket(prefix: str) -> Optional[str]:
        if prefix in ("BBU设备", "BBU"):
            return "bbu"
        if prefix in ("挡风板", "Shield"):
            return "bbu_shield"
        if prefix in ("ConnectPoint", "连接点", "螺丝", "光纤插头"):
            return "connect_point"
        if prefix in ("Fiber", "光纤"):
            return "fiber"
        if prefix in ("Wire", "电线"):
            return "wire"
        if prefix in ("Label", "标签"):
            return "label"
        return None

    @staticmethod
    def _types_of(sample: Dict[str, Any]) -> List[str]:
        if ("objects" not in sample) or (not isinstance(sample["objects"], list)):
            return []
        objs = sample["objects"]
        present: Dict[str, bool] = {}
        for obj in objs:
            d = obj["desc"] if (isinstance(obj, dict) and ("desc" in obj)) else ""
            if not isinstance(d, str) or not d:
                continue
            prefix = d.split("/")[0].strip()
            bucket = BucketedSamplingEngine._map_prefix_to_bucket(prefix)
            if bucket is not None:
                present[bucket] = True
        return list(present.keys())

    @staticmethod
    def _build_type_inverted_index(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, List[int]], List[List[str]]]:
        type_to_ids: Dict[str, List[int]] = {}
        sample_types: List[List[str]] = []
        for idx, s in enumerate(samples):
            types_i = BucketedSamplingEngine._types_of(s)
            sample_types.append(types_i)
            for t in types_i:
                type_to_ids.setdefault(t, []).append(idx)
        return type_to_ids, sample_types

    def build_epoch_map(
        self,
        samples: List[Dict[str, Any]],
        cfg: SamplingConfig,
        base_seed: int,
        epoch_idx: int,
        teacher_ratio: float,
        is_eval: bool,
        sample_weights: Optional[List[float]] = None,
    ) -> Tuple[Dict[int, EpisodeSpec], SamplingMetrics]:
        n = len(samples)
        if is_eval or teacher_ratio <= 0.0 or n == 0:
            empty_hist: Dict[int, int] = {}
            return {}, SamplingMetrics(
                pair_episodes=0,
                single_episodes=n,
                target_is_current_pct=0.0,
                avg_contrast_score=0.0,
                coverage_context_unique_pct=0.0,
                context_usage_histogram=empty_hist,
            )

        rng = Random(int(base_seed) + int(epoch_idx))

        # Build inverted index once
        type_to_ids, sample_types = self._build_type_inverted_index(samples)

        # Determine upper-limit pool cap
        frac = max(0.0, min(1.0, float(cfg.pool_fraction)))
        frac_cap = int(round(frac * n))
        upper_cap = max(0, min(int(cfg.pool_max), frac_cap if frac_cap > 0 else int(cfg.pool_max)))

        context_use_count: Dict[int, int] = {}
        episode_map: Dict[int, EpisodeSpec] = {}

        def _weighted_choice(pool_indices: List[int]) -> Optional[int]:
            if not pool_indices:
                return None
            if sample_weights is None or len(sample_weights) != n:
                return rng.choice(pool_indices)
            # Temperature-scaled softmax over normalized hardness weights
            tau = max(1e-6, float(cfg.temperature))
            vals = []
            maxv = None
            for j in pool_indices:
                w = max(0.0, float(sample_weights[j]))
                vals.append(w)
                maxv = w if maxv is None else max(maxv, w)
            # Numerical stability
            exps = []
            if maxv is None:
                return rng.choice(pool_indices)
            for v in vals:
                exps.append(pow(2.718281828, (v - maxv) / tau))
            s = sum(exps)
            if s <= 0:
                return rng.choice(pool_indices)
            # Draw based on cumulative probability
            r = rng.random() * s
            c = 0.0
            for idx_local, j in enumerate(pool_indices):
                c += exps[idx_local]
                if r <= c:
                    return j
            return pool_indices[-1]

        for i in range(n):
            if rng.random() >= float(teacher_ratio):
                continue

            # Construct object-type overlap pool
            t_i = sample_types[i]
            pool: List[int] = []
            for t in t_i:
                if t in type_to_ids:
                    pool.extend(type_to_ids[t])
            # Deduplicate and exclude self
            pool = [j for j in sorted(set(pool)) if j != i]

            # If empty, optional cross-explore or global fallback
            if not pool:
                if rng.random() < float(cfg.cross_bucket_explore_prob):
                    pool = [j for j in range(n) if j != i]
                else:
                    pool = [j for j in range(n) if j != i]

            # Apply only upper limit if cap > 0
            if upper_cap > 0 and len(pool) > upper_cap:
                rng.shuffle(pool)
                pool = pool[:upper_cap]
            elif len(pool) == 0:
                continue

            j = _weighted_choice(pool)
            if j is None:
                continue
            context_use_count[j] = (context_use_count[j] + 1) if (j in context_use_count) else 1
            episode_map[i] = EpisodeSpec(target_idx=i, context_idx=j)

        pair_eps = len(episode_map)
        single_eps = n - pair_eps
        used_contexts = sum(1 for v in context_use_count.values() if v > 0)
        coverage_pct = (used_contexts / max(1, n)) * 100.0
        target_is_current_pct = 100.0 if pair_eps > 0 else 0.0
        hist = dict(sorted(Counter(context_use_count.values()).items()))

        metrics = SamplingMetrics(
            pair_episodes=pair_eps,
            single_episodes=single_eps,
            target_is_current_pct=float(target_is_current_pct),
            avg_contrast_score=0.0,
            coverage_context_unique_pct=float(coverage_pct),
            context_usage_histogram=hist,
        )
        return episode_map, metrics
