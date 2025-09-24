#!/usr/bin/env python3
import unittest
import torch
from src_rl.tools.parity_check import parity_check


class TestParityCheck(unittest.TestCase):
    def test_shapes_equal(self):
        a = {"input_ids": torch.ones(1, 5, dtype=torch.long), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
        b = {"input_ids": torch.ones(1, 5, dtype=torch.long), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
        ok, msg = parity_check(a, b)
        self.assertTrue(ok)

    def test_prefix_diff(self):
        a = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}
        b = {"input_ids": torch.tensor([[1, 9, 3, 4]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}
        ok, msg = parity_check(a, b)
        self.assertFalse(ok)
        self.assertIn("input_ids differ", msg)


if __name__ == "__main__":
    unittest.main()
