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
class PairingMetrics:
    pair_episodes: int
    single_episodes: int
    target_is_current_pct: float
    avg_contrast_score: float  # loss-based模式下记为0.0（占位）
    coverage_context_unique_pct: float
    context_usage_histogram: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PairingConfig:
    candidate_pool_size: int
    max_teacher_uses_per_epoch: int
    temperature: float  # 占位，loss-based不使用
    target_assignment: str  # {"random","current","opposite"}
    cross_bucket_explore_prob: float = 0.0  # 小概率跨桶探索（默认关闭，安全）


class ContrastivePairingEngine:
    """Minimal random pairing engine (bucketed, with replacement).

    - 遍历全部样本，保证每个样本都作为 student（target）被访问一次。
    - 以 teacher_ratio 的伯努利概率，为当前样本从“同主类型桶”均匀随机抽取 1 个 teacher（可重复、允许多次被选中），排除 self；若该桶空，则退化到全量随机（排除 self）。
    - 支持小概率跨桶探索：以 cross_bucket_explore_prob 覆盖同桶限制，直接在全量中随机抽 teacher（排除 self）。
    - 对话顺序固定为：teacher → student（target 必在后）。
    - 不依赖任何 loss/业务特征；不设使用上限；不做打分。
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
    def _major_type_of(sample: Dict[str, Any]) -> str:
        """Majority vote over all objects' desc prefixes with stable tie-break.

        - 统计每个对象映射到的桶的频次；取最高频作为主类型。
        - 若出现并列，按样本中对象出现顺序优先（第一个出现的并列类型获胜）。
        - 若无任何可映射类型，返回 "misc"。
        """
        objs = sample.get("objects", []) or []
        counts: Dict[str, int] = {}
        order: List[str] = []
        for obj in objs:
            d = obj.get("desc", "")
            if not isinstance(d, str) or not d:
                continue
            prefix = d.split("/")[0].strip()
            bucket = ContrastivePairingEngine._map_prefix_to_bucket(prefix)
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
        # stable tie-break: first occurrence in the sample's object order
        for t in order:
            if t in tied:
                return t
        return "misc"

    @staticmethod
    def _build_type_buckets(samples: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        buckets: Dict[str, List[int]] = {}
        for idx, s in enumerate(samples):
            mt = ContrastivePairingEngine._major_type_of(s)
            buckets.setdefault(mt, []).append(idx)
        return buckets

    def build_epoch_map(
        self,
        samples: List[Dict[str, Any]],
        cfg: PairingConfig,
        base_seed: int,
        epoch_idx: int,
        teacher_ratio: float,
        is_eval: bool,
        sample_weights: Optional[List[float]] = None,
    ) -> Tuple[Dict[int, EpisodeSpec], PairingMetrics]:
        n = len(samples)
        if is_eval or teacher_ratio <= 0.0 or n == 0:
            empty_hist: Dict[int, int] = {}
            return {}, PairingMetrics(
                pair_episodes=0,
                single_episodes=n,
                target_is_current_pct=0.0,
                avg_contrast_score=0.0,
                coverage_context_unique_pct=0.0,
                context_usage_histogram=empty_hist,
            )

        rng = Random(int(base_seed) + int(epoch_idx))

        # 桶索引
        type_buckets = self._build_type_buckets(samples)
        context_use_count: Dict[int, int] = {}
        episode_map: Dict[int, EpisodeSpec] = {}

        # 遍历每个样本，作为 student（target）各过一次
        for i in range(n):
            if rng.random() >= float(teacher_ratio):
                # 单轮
                continue

            use_cross = rng.random() < float(getattr(cfg, "cross_bucket_explore_prob", 0.0))
            if use_cross:
                pool = [j for j in range(n) if j != i]
            else:
                mt = self._major_type_of(samples[i])
                pool = [j for j in type_buckets.get(mt, []) if j != i]
                if not pool:
                    pool = [j for j in range(n) if j != i]

            if not pool:
                # 数据集中仅 1 条样本之类的极端情况
                continue

            j = rng.choice(pool)
            # 记录一次使用次数（仅用于直方图/覆盖率指标）
            context_use_count[j] = context_use_count.get(j, 0) + 1

            # 固定当前样本为 target；对话顺序 teacher→student
            episode_map[i] = EpisodeSpec(target_idx=i, context_idx=j)

        # 指标
        pair_eps = len(episode_map)
        single_eps = n - pair_eps
        used_contexts = sum(1 for v in context_use_count.values() if v > 0)
        coverage_pct = (used_contexts / max(1, n)) * 100.0
        target_is_current_pct = 100.0 if pair_eps > 0 else 0.0
        hist = dict(sorted(Counter(context_use_count.values()).items()))

        metrics = PairingMetrics(
            pair_episodes=pair_eps,
            single_episodes=single_eps,
            target_is_current_pct=float(target_is_current_pct),
            avg_contrast_score=0.0,
            coverage_context_unique_pct=float(coverage_pct),
            context_usage_histogram=hist,
        )
        return episode_map, metrics
