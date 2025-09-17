# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from random import Random
from collections import Counter


@dataclass(frozen=True)
class EpisodeSpec:
    target_idx: int
    context_idx: Optional[int]


@dataclass(frozen=True)
class SamplerMetrics:
    pair_episodes: int
    single_episodes: int
    target_is_current_pct: float
    coverage_context_unique_pct: float
    context_usage_histogram: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RandomBucketConfig:
    cross_bucket_explore_prob: float = 0.0


class RandomBucketSampler:
    """Random teacher sampler with simple type buckets.

    - Each sample is the student once per epoch.
    - With probability `teacher_ratio`, sample exactly one teacher from the OVERLAP POOL defined as
      the union of all samples that share ANY object type with the target (based on desc prefixes → buckets),
      excluding self; fallback to global pool when overlap is empty.
    - Optional `cross_bucket_explore_prob` allows sampling from global pool regardless of overlap pool.
    - Order: teacher → student. No loss awareness, no usage caps, no scoring.
    """

    def _get_logger(self):
        try:
            from src_new.utils.rank_aware_logging import get_rank_aware_logger as _get
            return _get("sampler.random_bucket")
        except Exception:
            import logging as _logging
            return _logging.getLogger("sampler.random_bucket")

    @staticmethod
    def _map_prefix_to_bucket(prefix: str) -> Optional[str]:
        if prefix in ("BBU设备", "BBU", "基带处理单元"):
            return "bbu"
        if prefix in ("挡风板", "Shield", "防护板", "BBU挡风板"):
            return "bbu_shield"
        if prefix in ("ConnectPoint", "连接点", "螺丝", "光纤插头", "螺丝、光纤插头"):
            return "connect_point"
        if prefix in ("Fiber", "光纤", "光缆", "光纤线"):
            return "fiber"
        if prefix in ("Wire", "电线", "线缆", "电缆"):
            return "wire"
        if prefix in ("Label", "标签", "标签贴纸", "文字标签"):
            return "label"
        return None

    @staticmethod
    def _major_type_of(sample: Dict[str, Any]) -> str:
        objs = sample.get("objects", []) or []
        counts: Dict[str, int] = {}
        order: List[str] = []
        for obj in objs:
            d = obj.get("desc", "")
            if not isinstance(d, str) or not d:
                continue
            prefix = d.split("/")[0].strip()
            bucket = RandomBucketSampler._map_prefix_to_bucket(prefix)
            if bucket is None:
                continue
            if bucket not in counts:
                counts[bucket] = 0
                order.append(bucket)
            counts[bucket] += 1
        if not counts:
            return "misc"
        max_cnt = max(counts.values())
        tied = {t for t, c in counts.items() if c == max_cnt}
        for t in order:
            if t in tied:
                return t
        return "misc"

    @staticmethod
    def _extract_types_from_sample(sample: Dict[str, Any]) -> Set[str]:
        """Return set of object-type buckets present in a sample based on desc prefixes."""
        types: Set[str] = set()
        objs = sample.get("objects", []) or []
        for obj in objs:
            d = obj.get("desc", "")
            if not isinstance(d, str) or not d:
                continue
            prefix = str(d).split("/")[0].strip()
            b = RandomBucketSampler._map_prefix_to_bucket(prefix)
            if b is not None:
                types.add(b)
        if not types:
            types.add("misc")
        return types

    @staticmethod
    def _build_overlap_type_index(samples: List[Dict[str, Any]]) -> Tuple[List[Set[str]], Dict[str, List[int]]]:
        """Build per-sample type sets and an index mapping type→list of sample indices."""
        per_sample_types: List[Set[str]] = []
        type_to_indices: Dict[str, List[int]] = {}
        for idx, s in enumerate(samples):
            tset = RandomBucketSampler._extract_types_from_sample(s)
            per_sample_types.append(tset)
            for t in tset:
                if t not in type_to_indices:
                    type_to_indices[t] = []
                type_to_indices[t].append(idx)
        return per_sample_types, type_to_indices

    def build_epoch_map(
        self,
        samples: List[Dict[str, Any]],
        cfg: RandomBucketConfig,
        base_seed: int,
        epoch_idx: int,
        teacher_ratio: float,
        is_eval: bool,
    ) -> Tuple[Dict[int, EpisodeSpec], SamplerMetrics]:
        logger = self._get_logger()
        n = len(samples)
        if is_eval or teacher_ratio <= 0.0 or n == 0:
            empty_hist: Dict[int, int] = {}
            return {}, SamplerMetrics(
                pair_episodes=0,
                single_episodes=n,
                target_is_current_pct=0.0,
                coverage_context_unique_pct=0.0,
                context_usage_histogram=empty_hist,
            )

        rng = Random(int(base_seed) + int(epoch_idx))
        # Overlap-based type index (per-sample type sets + reverse index)
        per_sample_types, type_to_indices = self._build_overlap_type_index(samples)
        try:
            logger.debug(
                f"[sampler] epoch={epoch_idx} teacher_ratio={teacher_ratio:.2f} cross_prob={getattr(cfg, 'cross_bucket_explore_prob', 0.0):.3f} "
                f"unique_types={len(type_to_indices)}"
            )
        except Exception:
            pass
        context_use_count: Dict[int, int] = {}
        episode_map: Dict[int, EpisodeSpec] = {}

        for i in range(n):
            if rng.random() >= float(teacher_ratio):
                continue

            use_cross = rng.random() < float(getattr(cfg, "cross_bucket_explore_prob", 0.0))
            if use_cross:
                pool = [j for j in range(n) if j != i]
            else:
                # Union of all samples that share ANY type with target i
                pool_set: Set[int] = set()
                for t in per_sample_types[i]:
                    for j in type_to_indices.get(t, []):
                        if j != i:
                            pool_set.add(j)
                pool = list(pool_set)
                if not pool:
                    pool = [j for j in range(n) if j != i]

            if not pool:
                continue

            j = rng.choice(pool)
            context_use_count[j] = context_use_count.get(j, 0) + 1
            episode_map[i] = EpisodeSpec(target_idx=i, context_idx=j)
            try:
                logger.debug(
                    f"[sampler] pick: target={i} types={sorted(list(per_sample_types[i]))} "
                    f"pool_size={len(pool)} cross={use_cross} -> context={j}"
                )
            except Exception:
                pass

        pair_eps = len(episode_map)
        single_eps = n - pair_eps
        used_contexts = sum(1 for v in context_use_count.values() if v > 0)
        coverage_pct = (used_contexts / max(1, n)) * 100.0
        target_is_current_pct = 100.0 if pair_eps > 0 else 0.0
        hist = dict(sorted(Counter(context_use_count.values()).items()))

        metrics = SamplerMetrics(
            pair_episodes=pair_eps,
            single_episodes=single_eps,
            target_is_current_pct=float(target_is_current_pct),
            coverage_context_unique_pct=float(coverage_pct),
            context_usage_histogram=hist,
        )
        try:
            logger.debug(
                f"[sampler] summary: pairs={pair_eps} singles={single_eps} unique_contexts={used_contexts}/{n} coverage={coverage_pct:.1f}%"
            )
        except Exception:
            pass
        return episode_map, metrics
