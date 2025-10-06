#!/usr/bin/env python3
import unittest
from src_new.rl.rewards.format_rewards import (
    check_wrappers,
    check_ascii_separators,
    check_coords_counts,
    check_banned_vocab,
    compute_reward,
)


class TestFormatRewards(unittest.TestCase):
    def test_wrappers_ok(self):
        text = "<|object_ref_start|>x<|object_ref_end|><|box_start|>[1, 2, 3, 4]<|box_end|>"
        self.assertEqual(check_wrappers(text), 1.0)

    def test_ascii_separators_fail_on_chinese(self):
        text = "<|object_ref_start|>x<|object_ref_end|><|box_start|>[1，2，3，4]<|box_end|>"
        self.assertEqual(check_ascii_separators(text), 0.0)

    def test_coords_counts_quad(self):
        ok = "<|object_ref_start|>x<|object_ref_end|><|quad_start|>[1, 2, 3, 4, 5, 6, 7, 8]<|quad_end|>"
        bad = "<|object_ref_start|>x<|object_ref_end|><|quad_start|>[1, 2, 3, 4]<|quad_end|>"
        self.assertEqual(check_coords_counts(ok), 1.0)
        self.assertEqual(check_coords_counts(bad), 0.0)

    def test_banned_vocab(self):
        self.assertEqual(check_banned_vocab("PPDU"), 0.0)
        self.assertEqual(check_banned_vocab("ok"), 1.0)

    def test_compute_reward_mix(self):
        text = "<|object_ref_start|>x<|object_ref_end|><|line_start|>[1, 2, 3, 4]<|line_end|>"
        w = {"parse": 0.5, "wrappers": 0.5}
        r = compute_reward(text, w)
        self.assertTrue(0.9 <= r <= 1.0)


if __name__ == "__main__":
    unittest.main()
