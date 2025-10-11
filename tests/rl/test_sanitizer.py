from src_new.rl.rewards.sanitizer import sanitize_tail_geometry_block


def test_sanitize_removes_only_rightmost_block_box():
    text = (
        "<|object_ref_start|>A<|object_ref_end|>"
        "<|box_start|>[1, 2, 3, 4]<|box_end|>"
        "<|object_ref_start|>B<|object_ref_end|>"
        "<|box_start|>[5, 6, 7, 8]<|box_end|>"
    )
    out = sanitize_tail_geometry_block(text)
    assert "[5, 6, 7, 8]" not in out
    assert "[1, 2, 3, 4]" in out
    # Ensure preceding content kept
    assert "<|object_ref_start|>A<|object_ref_end|>" in out
    assert "<|object_ref_start|>B<|object_ref_end|>" in out


def test_sanitize_mixed_types_removes_rightmost():
    text = (
        "<|object_ref_start|>X<|object_ref_end|>"
        "<|quad_start|>[10, 10, 20, 10, 20, 20, 10, 20]<|quad_end|>"
        "<|object_ref_start|>Y<|object_ref_end|>"
        "<|line_start|>[100, 100, 200, 200]<|line_end|>"
    )
    out = sanitize_tail_geometry_block(text)
    # Rightmost is the line block; it should be removed, quad remains
    assert "[100, 100, 200, 200]" not in out
    assert "[10, 10, 20, 10, 20, 20, 10, 20]" in out


def test_sanitize_no_block_returns_original():
    text = "<|object_ref_start|>desc<|object_ref_end|>"
    out = sanitize_tail_geometry_block(text)
    assert out == text
