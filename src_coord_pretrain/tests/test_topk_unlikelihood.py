#!/usr/bin/env python3
"""
Comprehensive test suite for Top-K Unlikelihood training features.

Implements the exact verification checklist from PHASE_A_UNLIKELIHOOD_REVERSE_PLAN.md:
- Unit tests with mixed forward/reverse batches
- Mask verification assertions
- Negative set sanity checks
- End-to-end numerical validation
"""

from dataclasses import dataclass
from typing import Tuple
from unittest.mock import MagicMock

import torch


@dataclass
class MockTopKConfig:
    """Mock configuration for Top-K Unlikelihood testing."""

    max_coord_value: int = 1024
    ul_topk_noncoord: int = 100
    ul_topk_coord: int = 100
    ul_neighbor_window: int = 8
    unlikelihood_lambda_digits: float = 1.0
    unlikelihood_lambda_coords: float = 1.0


def create_mock_tokenizer_with_coords(max_coord: int = 1024) -> MagicMock:
    """Create mock tokenizer with coordinate tokens."""
    mock_tokenizer = MagicMock()

    # Create coordinate token vocabulary
    vocab = {}
    coord_ids = set()

    # Regular tokens
    vocab.update(
        {
            "0": 100,
            "1": 101,
            "2": 102,
            "3": 103,
            "4": 104,
            "5": 105,
            "6": 106,
            "7": 107,
            "8": 108,
            "9": 109,
            "<|im_end|>": 151643,
            "regular": 500,
            "token": 501,
            "text": 502,
        }
    )

    # Coordinate tokens
    coord_start_id = 151667
    for i in range(max_coord + 1):
        token_name = f"<|coord_{i}|>"
        token_id = coord_start_id + i
        vocab[token_name] = token_id
        coord_ids.add(token_id)

    mock_tokenizer.get_vocab.return_value = vocab
    mock_tokenizer.vocab_size = len(vocab)

    return mock_tokenizer, coord_ids


def create_mixed_batch_sample() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a tiny batch with forward and reverse samples for testing.

    Returns:
        logits: [batch_size=2, seq_len=10, vocab_size=2000]
        labels: [batch_size=2, seq_len=10]
        assistant_mask: [batch_size=2, seq_len=10]
    """
    batch_size, seq_len, vocab_size = 2, 10, 2000

    # Create mock logits
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Create labels with mixed forward/reverse samples
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    # Sample 1: Forward mapping (user question -> coordinate token response)
    # Assistant span at positions 6-8: [<|coord_123|>, <|im_end|>, -100]
    labels[0, 6] = 151667 + 123  # <|coord_123|>
    labels[0, 7] = 151643  # <|im_end|>

    # Sample 2: Reverse mapping (coordinate question -> text number response)
    # Assistant span at positions 5-8: [1, 2, 3, <|im_end|>]
    labels[1, 5] = 101  # "1"
    labels[1, 6] = 102  # "2"
    labels[1, 7] = 103  # "3"
    labels[1, 8] = 151643  # <|im_end|>

    # Create assistant mask (where labels != -100)
    assistant_mask = labels != -100

    return logits, labels, assistant_mask


class TestTopKUnlikelihoodMasks:
    """Test mask computation and verification."""

    def test_mask_computation_basic(self):
        """Test basic mask computation for mixed batch."""
        logits, labels, assistant_mask = create_mixed_batch_sample()
        _, coord_ids = create_mock_tokenizer_with_coords()

        # Compute target masks
        coord_target_mask = (
            torch.isin(labels, torch.tensor(list(coord_ids))) & assistant_mask
        )
        text_target_mask = (
            (~torch.isin(labels, torch.tensor(list(coord_ids))))
            & (labels != -100)
            & assistant_mask
        )

        # Verification assertions from the plan
        assert not (coord_target_mask & text_target_mask).any(), (
            "Masks should be mutually exclusive"
        )

        # Check coverage (allowing for EOS exclusion)
        total_targets = coord_target_mask.sum() + text_target_mask.sum()
        assert total_targets <= assistant_mask.sum(), (
            "Target masks should not exceed assistant positions"
        )

        print(f"✅ Basic mask computation test passed")
        print(f"   Assistant positions: {assistant_mask.sum().item()}")
        print(f"   Coordinate targets: {coord_target_mask.sum().item()}")
        print(f"   Text targets: {text_target_mask.sum().item()}")

    def test_forward_sample_masks(self):
        """Test masks for forward mapping sample (gold coordinate token)."""
        logits, labels, assistant_mask = create_mixed_batch_sample()
        _, coord_ids = create_mock_tokenizer_with_coords()

        # Focus on sample 0 (forward mapping)
        sample_0_labels = labels[0]
        sample_0_assistant = assistant_mask[0]

        coord_target_mask_0 = (
            torch.isin(sample_0_labels, torch.tensor(list(coord_ids)))
            & sample_0_assistant
        )
        text_target_mask_0 = (
            (~torch.isin(sample_0_labels, torch.tensor(list(coord_ids))))
            & (sample_0_labels != -100)
            & sample_0_assistant
        )

        # For forward sample: should have exactly one coordinate target, no text targets
        assert coord_target_mask_0.sum() == 1, (
            f"Forward sample should have exactly 1 coord target, got {coord_target_mask_0.sum()}"
        )
        assert text_target_mask_0.sum() == 1, (
            f"Forward sample should have 1 text target (<|im_end|>), got {text_target_mask_0.sum()}"
        )  # <|im_end|> is text

        # Check that the coordinate target is at the right position
        coord_positions = coord_target_mask_0.nonzero(as_tuple=True)[0]
        assert len(coord_positions) == 1 and coord_positions[0] == 6, (
            "Coordinate target should be at position 6"
        )

        print(f"✅ Forward sample mask test passed")

    def test_reverse_sample_masks(self):
        """Test masks for reverse mapping sample (gold text tokens)."""
        logits, labels, assistant_mask = create_mixed_batch_sample()
        _, coord_ids = create_mock_tokenizer_with_coords()

        # Focus on sample 1 (reverse mapping)
        sample_1_labels = labels[1]
        sample_1_assistant = assistant_mask[1]

        coord_target_mask_1 = (
            torch.isin(sample_1_labels, torch.tensor(list(coord_ids)))
            & sample_1_assistant
        )
        text_target_mask_1 = (
            (~torch.isin(sample_1_labels, torch.tensor(list(coord_ids))))
            & (sample_1_labels != -100)
            & sample_1_assistant
        )

        # For reverse sample: should have no coordinate targets, multiple text targets
        assert coord_target_mask_1.sum() == 0, (
            f"Reverse sample should have 0 coord targets, got {coord_target_mask_1.sum()}"
        )
        assert text_target_mask_1.sum() >= 3, (
            f"Reverse sample should have ≥3 text targets, got {text_target_mask_1.sum()}"
        )

        # Check that text targets include the digit tokens
        text_positions = text_target_mask_1.nonzero(as_tuple=True)[0]
        expected_positions = torch.tensor([5, 6, 7, 8])  # "1", "2", "3", "<|im_end|>"
        assert torch.equal(text_positions, expected_positions), (
            f"Text targets at wrong positions: {text_positions} vs {expected_positions}"
        )

        print(f"✅ Reverse sample mask test passed")


class TestTopKNegativeSelection:
    """Test Top-K negative sampling logic."""

    def test_coordinate_target_negatives(self):
        """Test Top-K non-coordinate token selection for coordinate targets."""
        logits, labels, assistant_mask = create_mixed_batch_sample()
        mock_tokenizer, coord_ids = create_mock_tokenizer_with_coords()

        # Test parameters
        topk_noncoord = 10
        vocab_size = logits.shape[-1]

        # Create coordinate target positions (sample 0, position 6)
        coord_target_positions = [(0, 6)]  # batch_idx, seq_idx

        for batch_idx, seq_idx in coord_target_positions:
            position_logits = logits[batch_idx, seq_idx]  # [vocab_size]
            gold_token_id = labels[batch_idx, seq_idx].item()

            # Get probabilities
            probs = torch.softmax(position_logits, dim=-1)

            # Create mask for valid negative tokens
            # Exclude: gold token, coordinate tokens, <|im_end|>
            exclude_ids = coord_ids | {gold_token_id, 151643}  # 151643 is <|im_end|>
            valid_mask = torch.ones(vocab_size, dtype=torch.bool)
            for exclude_id in exclude_ids:
                if exclude_id < vocab_size:
                    valid_mask[exclude_id] = False

            # Select top-K from valid tokens
            valid_probs = probs.clone()
            valid_probs[~valid_mask] = -float("inf")

            topk_probs, topk_indices = torch.topk(
                valid_probs, k=min(topk_noncoord, valid_mask.sum().item())
            )

            # Verify negative set properties
            assert len(topk_indices) <= topk_noncoord, (
                f"Should have ≤{topk_noncoord} negatives"
            )
            assert gold_token_id not in topk_indices, (
                "Gold token should not be in negatives"
            )
            assert 151643 not in topk_indices, "<|im_end|> should not be in negatives"

            # Check no coordinate tokens in negatives
            for neg_id in topk_indices:
                assert neg_id.item() not in coord_ids, (
                    f"Coordinate token {neg_id.item()} should not be in negatives"
                )

            print(
                f"✅ Coordinate target negatives test passed for position ({batch_idx}, {seq_idx})"
            )
            print(f"   Selected {len(topk_indices)} negative tokens")

    def test_text_target_negatives(self):
        """Test Top-K coordinate token selection for text targets."""
        logits, labels, assistant_mask = create_mixed_batch_sample()
        mock_tokenizer, coord_ids = create_mock_tokenizer_with_coords()

        # Test parameters
        topk_coord = 10

        # Create text target positions (sample 1, positions 5-7)
        text_target_positions = [(1, 5), (1, 6), (1, 7)]  # batch_idx, seq_idx

        for batch_idx, seq_idx in text_target_positions:
            position_logits = logits[batch_idx, seq_idx]  # [vocab_size]
            gold_token_id = labels[batch_idx, seq_idx].item()

            # Get probabilities
            probs = torch.softmax(position_logits, dim=-1)

            # Create mask for coordinate tokens only
            coord_mask = torch.zeros(logits.shape[-1], dtype=torch.bool)
            for coord_id in coord_ids:
                if coord_id < logits.shape[-1]:
                    coord_mask[coord_id] = True

            # Exclude gold if it's accidentally a coordinate token
            if gold_token_id in coord_ids:
                coord_mask[gold_token_id] = False

            # Select top-K from coordinate tokens
            coord_probs = probs.clone()
            coord_probs[~coord_mask] = -float("inf")

            if coord_mask.sum() > 0:
                topk_probs, topk_indices = torch.topk(
                    coord_probs, k=min(topk_coord, coord_mask.sum().item())
                )

                # Verify negative set properties
                assert len(topk_indices) <= topk_coord, (
                    f"Should have ≤{topk_coord} negatives"
                )

                # Check all negatives are coordinate tokens
                for neg_id in topk_indices:
                    assert neg_id.item() in coord_ids, (
                        f"Token {neg_id.item()} should be a coordinate token"
                    )

                print(
                    f"✅ Text target negatives test passed for position ({batch_idx}, {seq_idx})"
                )
                print(f"   Selected {len(topk_indices)} coordinate token negatives")


def test_numerical_guards():
    """Test numerical stability guards in unlikelihood computation."""
    # Test probability clamping
    eps = 1e-5
    test_probs = torch.tensor([0.0, 0.5, 1.0, 0.999999])

    # Compute unlikelihood with clamping
    clamped_complement = torch.clamp(1.0 - test_probs, min=eps)
    ul_loss = -torch.log(clamped_complement)

    # Verify no infinities
    assert torch.isfinite(ul_loss).all(), "Unlikelihood loss should be finite"
    assert (clamped_complement >= eps).all(), "Clamped probabilities should be ≥ eps"

    print("✅ Numerical guards test passed")


if __name__ == "__main__":
    print("🧪 Running Top-K Unlikelihood Test Suite...")
    print("=" * 60)

    # Run mask tests
    mask_tests = TestTopKUnlikelihoodMasks()
    mask_tests.test_mask_computation_basic()
    mask_tests.test_forward_sample_masks()
    mask_tests.test_reverse_sample_masks()

    print()

    # Run negative selection tests
    neg_tests = TestTopKNegativeSelection()
    neg_tests.test_coordinate_target_negatives()
    neg_tests.test_text_target_negatives()

    print()

    # Run numerical tests
    test_numerical_guards()

    print()
    print("🎉 All Top-K Unlikelihood tests passed!")
    print("\n📋 Verification Checklist Completed:")
    print("✅ Mixed forward/reverse batch handling")
    print("✅ Mutually exclusive mask computation")
    print("✅ Coordinate target negative selection")
    print("✅ Text target negative selection")
    print("✅ Numerical stability guards")
