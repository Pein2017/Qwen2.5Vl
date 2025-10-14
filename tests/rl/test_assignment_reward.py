import importlib


def _mk_meta(objects):
    return {"objects": objects, "width": 1000, "height": 800}


def test_assignment_basic_bbox_identical():
    mod = importlib.import_module("src_new.rl.rewards.assignment")
    text = (
        "<|object_ref_start|>A<|object_ref_end|>"
        "<|box_start|>[100, 100, 200, 200]<|box_end|>"
        "<|object_ref_start|>B<|object_ref_end|>"
        "<|box_start|>[300, 300, 360, 360]<|box_end|>"
    )
    meta = _mk_meta(
        [
            {"bbox_2d": [100, 100, 200, 200], "desc": "A"},
            {"bbox_2d": [300, 300, 360, 360], "desc": "B"},
        ]
    )
    v = mod.assignment_f1(text, meta=meta, lambda_caption=0.3)
    assert 0.0 <= v <= 1.0
    assert v > 0.8


def test_assignment_mismatch_penalizes_fp_fn():
    mod = importlib.import_module("src_new.rl.rewards.assignment")
    # One pred, zero GT → FP
    text = (
        "<|object_ref_start|>A<|object_ref_end|>"
        "<|box_start|>[10, 10, 30, 30]<|box_end|>"
    )
    meta = _mk_meta([])
    v = mod.assignment_f1(text, meta=meta)
    assert 0.0 <= v <= 1.0
    assert v < 0.5


def test_assignment_caption_impact():
    mod = importlib.import_module("src_new.rl.rewards.assignment")
    # Same boxes, different captions should lower reward when lambda_caption>0
    text_pred = (
        "<|object_ref_start|>X<|object_ref_end|>"
        "<|box_start|>[50, 50, 150, 150]<|box_end|>"
    )
    meta_match = _mk_meta([{"bbox_2d": [50, 50, 150, 150], "desc": "foo bar"}])
    meta_mismatch = _mk_meta([{"bbox_2d": [50, 50, 150, 150], "desc": "baz qux"}])
    v_match = mod.assignment_f1(text_pred, meta=meta_match, lambda_caption=0.5)
    v_mismatch = mod.assignment_f1(text_pred, meta=meta_mismatch, lambda_caption=0.5)
    assert 0.0 <= v_match <= 1.0 and 0.0 <= v_mismatch <= 1.0
    assert v_match >= v_mismatch
