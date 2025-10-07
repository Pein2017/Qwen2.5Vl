import pytest

from src_new.rl.rewards.format_rewards import length_vs_gt


@pytest.mark.parametrize(
    "gt_len,gen_len,lower,upper,gamma,expected_range",
    [
        (100, 50, 0.7, 1.2, 3.0, (0.0, 0.8)),  # below lower → linear ramp < 1.0
        (100, 90, 0.7, 1.2, 3.0, (0.9, 1.0)),  # inside window → near 1.0
        (100, 140, 0.7, 1.2, 3.0, (0.1, 0.7)),  # above upper → exp decay < 1.0
    ],
)
def test_length_vs_gt_basic(gt_len, gen_len, lower, upper, gamma, expected_range):
    meta = {"gt_len_tokenizer": gt_len, "gen_len_tokenizer": gen_len}
    score = length_vs_gt("dummy", meta=meta, lower=lower, upper=upper, gamma=gamma)
    lo, hi = expected_range
    assert lo <= score <= hi


def test_length_vs_gt_tail_penalty():
    # Construct a text with many digits to trigger tail penalty
    gt_len = 100
    # gen longer than alpha*gt_len; use heavy numeric tail
    gen_len = 150
    numeric_tail_text = "0123456789 " * 50
    meta = {"gt_len_tokenizer": gt_len, "gen_len_tokenizer": gen_len}
    base = length_vs_gt("short text", meta=meta, alpha=1.1, tail_numeric_weight=0.0)
    penalized = length_vs_gt(
        numeric_tail_text, meta=meta, alpha=1.1, tail_numeric_weight=0.4
    )
    assert penalized <= base
