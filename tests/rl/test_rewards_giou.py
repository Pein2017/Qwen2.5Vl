import importlib

import pytest


def _has_shapely():
    try:
        import shapely  # noqa: F401

        return True
    except Exception:
        return False


def _mk_meta(objects):
    return {"objects": objects, "width": 1000, "height": 800}


@pytest.mark.parametrize("shift", [0, 5, 20])
def test_quad_giou_matches_bbox_when_shapely_missing(monkeypatch, shift):
    # Force import path
    mod = importlib.import_module("src_new.rl.rewards.detection_rewards")

    # If shapely exists, temporarily disable it to test fallback
    if _has_shapely():
        monkeypatch.setattr(mod, "Polygon", None, raising=False)
        monkeypatch.setattr(mod, "unary_union", None, raising=False)

    # Perfect overlap when shift=0, otherwise reduced
    text = (
        "<|object_ref_start|>A<|object_ref_end|>"
        "<|quad_start|>[100, 100, 200, 100, 200, 200, 100, 200]<|quad_end|>"
    )
    meta = _mk_meta(
        [
            {
                "quad": [
                    100 + shift,
                    100 + shift,
                    200 + shift,
                    100 + shift,
                    200 + shift,
                    200 + shift,
                    100 + shift,
                    200 + shift,
                ]
            }
        ]
    )

    val = mod.reward_quad_giou(text, meta=meta)
    assert 0.0 <= val <= 1.0
    if shift == 0:
        assert pytest.approx(val, rel=0, abs=1e-6) == 1.0
    else:
        assert val < 1.0


@pytest.mark.parametrize("buffer_frac", [0.005, 0.02])
def test_line_giou_identical_vs_shift(buffer_frac):
    mod = importlib.import_module("src_new.rl.rewards.detection_rewards")

    # Two identical simple horizontal lines
    text = (
        "<|object_ref_start|>L<|object_ref_end|>"
        "<|line_start|>[100, 100, 300, 100]<|line_end|>"
    )
    meta_same = _mk_meta([{"line": [100, 100, 300, 100]}])

    val_same = mod.reward_line_giou(text, meta=meta_same, buffer_frac=buffer_frac)
    assert 0.0 <= val_same <= 1.0
    # Should be close to 1 for identical; allow tolerance for fallback path
    assert val_same > 0.8

    # Shift GT down; overlap should decrease with small buffer
    meta_shift = _mk_meta([{"line": [100, 120, 300, 120]}])
    val_shift = mod.reward_line_giou(text, meta=meta_shift, buffer_frac=buffer_frac)
    assert 0.0 <= val_shift <= 1.0
    assert val_shift <= val_same


def test_line_giou_fallback_endpoints(monkeypatch):
    mod = importlib.import_module("src_new.rl.rewards.detection_rewards")
    # Force fallback by disabling shapely usage
    monkeypatch.setattr(mod, "LineString", None, raising=False)
    monkeypatch.setattr(mod, "unary_union", None, raising=False)

    text = (
        "<|object_ref_start|>L<|object_ref_end|>"
        "<|line_start|>[10, 10, 110, 10]<|line_end|>"
    )
    meta = _mk_meta([{"line": [10, 10, 110, 10]}])
    v = mod.reward_line_giou(text, meta=meta, buffer_frac=0.01)
    assert 0.0 <= v <= 1.0
    assert v > 0.8
