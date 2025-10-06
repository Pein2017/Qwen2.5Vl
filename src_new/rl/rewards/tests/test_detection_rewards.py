#!/usr/bin/env python3
from __future__ import annotations

from src_new.rl.rewards.detection_rewards import (
    parse_dense_caption,
    reward_bbox_giou,
    reward_coverage,
    reward_geometry_sanity,
    reward_quad_l1,
    reward_line_l1,
    reward_ordering,
)


def test_parse_dense_caption_extracts_objects():
    text = (
        "<|object_ref_start|>螺丝<|object_ref_end|><|box_start|>[10, 20, 30, 40]<|box_end|>"
        "<|object_ref_start|>光纤<|object_ref_end|><|line_start|>[0, 0, 10, 10]<|line_end|>"
    )
    objs = parse_dense_caption(text)
    assert len(objs) == 2
    assert objs[0]["bbox_2d"] == [10, 20, 30, 40]
    assert "line" in objs[1]


def test_reward_coverage_matches_counts():
    text = "<|object_ref_start|>A<|object_ref_end|><|box_start|>[0,0,1,1]<|box_end|>"
    meta = {"objects": [{"bbox_2d": [0, 0, 1, 1]}]}
    assert reward_coverage(text, meta=meta) == 1.0

    meta_miss = {"objects": [{"bbox_2d": [0, 0, 1, 1]}, {"bbox_2d": [2, 2, 3, 3]}]}
    val = reward_coverage(text, meta=meta_miss)
    assert 0.0 < val < 1.0


def test_reward_geometry_sanity_bounds():
    text = "<|object_ref_start|>A<|object_ref_end|><|box_start|>[0,0,5,5]<|box_end|>"
    meta = {"width": 10, "height": 10}
    assert reward_geometry_sanity(text, meta=meta) == 1.0

    bad = "<|object_ref_start|>A<|object_ref_end|><|box_start|>[-1,0,5,5]<|box_end|>"
    assert reward_geometry_sanity(bad, meta=meta) == 0.0


def test_reward_bbox_giou_mapping_range():
    # Perfect match -> mapped to 1.0
    text = "<|object_ref_start|>A<|object_ref_end|><|box_start|>[0,0,10,10]<|box_end|>"
    meta = {"objects": [{"bbox_2d": [0, 0, 10, 10]}]}
    assert reward_bbox_giou(text, meta=meta) == 1.0

    # Disjoint boxes -> GIoU negative -> mapped into [0,1)
    text2 = "<|object_ref_start|>A<|object_ref_end|><|box_start|>[100,100,110,110]<|box_end|>"
    score = reward_bbox_giou(text2, meta=meta)
    assert 0.0 <= score < 0.5  # mapped from negative GIoU


def test_reward_quad_l1_and_ordering():
    # Pred quad equals GT -> zero L1 → score 1.0, ordering ok
    text = (
        "<|object_ref_start|>Q<|object_ref_end|>"
        "<|quad_start|>[0,0, 10,0, 10,10, 0,10]<|quad_end|>"
    )
    meta = {"objects": [{"quad": [0,0, 10,0, 10,10, 0,10]}], "width": 100, "height": 100}
    assert reward_quad_l1(text, meta=meta) == 1.0
    assert reward_ordering(text) == 1.0

    # Shuffled vertices (non-canonical) → ordering < 1.0
    bad_order = (
        "<|object_ref_start|>Q<|object_ref_end|>"
        "<|quad_start|>[10,10, 10,0, 0,0, 0,10]<|quad_end|>"
    )
    ord_score = reward_ordering(bad_order)
    assert 0.0 <= ord_score < 1.0


def test_reward_line_l1_and_ordering():
    text = (
        "<|object_ref_start|>L<|object_ref_end|>"
        "<|line_start|>[0,0, 10,10]<|line_end|>"
    )
    meta = {"objects": [{"line": [0,0, 10,10]}], "width": 100, "height": 100}
    assert reward_line_l1(text, meta=meta) == 1.0
    assert reward_ordering(text) == 1.0

    # Reverse direction should fail ordering
    reversed_line = (
        "<|object_ref_start|>L<|object_ref_end|>"
        "<|line_start|>[10,10, 0,0]<|line_end|>"
    )
    ord_score = reward_ordering(reversed_line)
    assert 0.0 <= ord_score < 1.0
