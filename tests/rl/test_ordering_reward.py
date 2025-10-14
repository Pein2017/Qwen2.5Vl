import importlib


def test_ordering_quad_clockwise_top_left_start():
    mod = importlib.import_module("src_new.rl.rewards.detection_rewards")
    # Properly ordered quad (top-left start then clockwise)
    text_ok = (
        "<|object_ref_start|>Q<|object_ref_end|>"
        "<|quad_start|>[100, 100, 200, 100, 200, 200, 100, 200]<|quad_end|>"
    )
    v_ok = mod.reward_ordering(text_ok, meta=None)
    assert 0.0 <= v_ok <= 1.0
    assert v_ok >= 0.99

    # Mis-ordered (reverse direction)
    text_bad = (
        "<|object_ref_start|>Q<|object_ref_end|>"
        "<|quad_start|>[100, 100, 100, 200, 200, 200, 200, 100]<|quad_end|>"
    )
    v_bad = mod.reward_ordering(text_bad, meta=None)
    assert 0.0 <= v_bad <= 1.0
    assert v_bad < v_ok


def test_ordering_line_direction_and_backtracking():
    mod = importlib.import_module("src_new.rl.rewards.detection_rewards")
    # Canonical: left-to-right horizontal line
    text_ok = (
        "<|object_ref_start|>L<|object_ref_end|>"
        "<|line_start|>[10, 10, 50, 10, 90, 10]<|line_end|>"
    )
    v_ok = mod.reward_ordering(text_ok, meta=None)
    assert 0.0 <= v_ok <= 1.0
    assert v_ok >= 0.99

    # Reversed direction
    text_rev = (
        "<|object_ref_start|>L<|object_ref_end|>"
        "<|line_start|>[90, 10, 50, 10, 10, 10]<|line_end|>"
    )
    v_rev = mod.reward_ordering(text_rev, meta=None)
    assert 0.0 <= v_rev <= 1.0
    assert v_rev < v_ok

    # Backtracking zig-zag should be penalized
    text_zig = (
        "<|object_ref_start|>L<|object_ref_end|>"
        "<|line_start|>[10, 10, 50, 10, 40, 10, 80, 10]<|line_end|>"
    )
    v_zig = mod.reward_ordering(text_zig, meta=None)
    assert 0.0 <= v_zig <= 1.0
    assert v_zig < v_ok
