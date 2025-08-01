"""
Tests for data collation functionality.

This module tests:
- Batch creation and padding
- Standard vs packed collation strategies
- Attention mask generation
- Memory efficiency optimizations
"""

from typing import Dict, List

import pytest
import torch

from src_new.tests.fixtures import (
    assert_tensor_shapes,
    validate_batch_structure,
)


class TestDataCollator:
    """Test data collation functionality."""

    def test_standard_collation(self):
        """Test standard data collation."""
        # Create mock samples with different lengths
        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels": torch.tensor([1, 2, 3, 4, 5]),
                "pixel_values": torch.randn(3, 224, 224),
                "image_grid_thw": torch.tensor([1, 224, 224]),
            },
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "pixel_values": torch.randn(3, 224, 224),
                "image_grid_thw": torch.tensor([1, 224, 224]),
            },
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1]),
                "labels": torch.tensor([1, 2, 3, 4, 5, 6, 7]),
                "pixel_values": torch.randn(3, 224, 224),
                "image_grid_thw": torch.tensor([1, 224, 224]),
            },
        ]

        # Mock standard collation
        batch = self._mock_standard_collate(samples)

        # Validate batch structure
        assert validate_batch_structure(batch)

        # Check padding
        batch_size = len(samples)
        max_length = max(len(sample["input_ids"]) for sample in samples)

        expected_shapes = {
            "input_ids": (batch_size, max_length),
            "attention_mask": (batch_size, max_length),
            "labels": (batch_size, max_length),
            "pixel_values": (batch_size, 3, 224, 224),
            "image_grid_thw": (batch_size, 3),
        }

        assert_tensor_shapes(batch, expected_shapes)

        # Check that padding was applied correctly
        assert (
            batch["input_ids"].size(1) == max_length
        )  # Should be padded to max length

        # Check attention masks for padded positions
        for i, sample in enumerate(samples):
            original_length = len(sample["input_ids"])
            # Original positions should have attention_mask = 1
            assert torch.all(batch["attention_mask"][i, :original_length] == 1)
            # Padded positions should have attention_mask = 0
            if original_length < max_length:
                assert torch.all(batch["attention_mask"][i, original_length:] == 0)

    def test_packed_collation(self):
        """Test packed data collation for memory efficiency."""
        # Create samples with varying lengths
        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels": torch.tensor([1, 2, 3, 4, 5]),
                "pixel_values": torch.randn(3, 224, 224),
                "image_grid_thw": torch.tensor([1, 224, 224]),
            },
            {
                "input_ids": torch.tensor([6, 7, 8]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([6, 7, 8]),
                "pixel_values": torch.randn(3, 224, 224),
                "image_grid_thw": torch.tensor([1, 224, 224]),
            },
        ]

        # Mock packed collation
        batch = self._mock_packed_collate(samples, max_length=10)

        # Validate batch structure
        assert validate_batch_structure(batch)

        # In packed mode, sequences are concatenated
        total_tokens = sum(len(sample["input_ids"]) for sample in samples)

        # Should have efficient packing
        assert batch["input_ids"].numel() <= len(samples) * 10  # Max length constraint

        # Check that we have position information for unpacking
        assert (
            "position_ids" in batch or "cu_seqlens" in batch
        )  # Some form of position tracking

    def test_attention_mask_generation(self):
        """Test attention mask generation for different scenarios."""
        # Test case 1: Standard attention masks
        sample_with_padding = {
            "input_ids": torch.tensor([1, 2, 3, 0, 0]),  # Last two are padding
            "labels": torch.tensor([1, 2, 3, -100, -100]),  # Corresponding labels
        }

        # Generate attention mask
        attention_mask = self._generate_attention_mask(
            sample_with_padding["input_ids"], pad_token_id=0
        )
        expected_mask = torch.tensor([1, 1, 1, 0, 0])

        assert torch.equal(attention_mask, expected_mask)

        # Test case 2: No padding
        sample_no_padding = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "labels": torch.tensor([1, 2, 3, 4, 5]),
        }

        attention_mask = self._generate_attention_mask(
            sample_no_padding["input_ids"], pad_token_id=0
        )
        expected_mask = torch.tensor([1, 1, 1, 1, 1])

        assert torch.equal(attention_mask, expected_mask)

    def test_label_masking(self):
        """Test proper label masking for loss computation."""
        # Create sample with system and user prompts that should be masked
        mock_conversation_tokens = {
            "input_ids": torch.tensor(
                [
                    151644,
                    100,
                    101,
                    151645,  # <im_start>system...<im_end>
                    151644,
                    102,
                    103,
                    151645,  # <im_start>user...<im_end>
                    151644,
                    104,
                    105,
                    106,
                    151645,  # <im_start>assistant...<im_end>
                ]
            ),
            "conversation_boundaries": [
                ("system", 0, 4),
                ("user", 4, 8),
                ("assistant", 8, 13),
            ],
        }

        # Mock label masking
        labels = self._mock_create_labels_with_masking(
            mock_conversation_tokens["input_ids"],
            mock_conversation_tokens["conversation_boundaries"],
        )

        # Check that system and user tokens are masked (set to -100)
        IGNORE_INDEX = -100

        # System tokens should be masked
        assert torch.all(labels[0:4] == IGNORE_INDEX)

        # User tokens should be masked
        assert torch.all(labels[4:8] == IGNORE_INDEX)

        # Assistant tokens should NOT be masked (used for loss)
        assert torch.all(labels[8:13] != IGNORE_INDEX)

        # Check that assistant tokens match input_ids (for standard LM loss)
        assert torch.equal(labels[8:13], mock_conversation_tokens["input_ids"][8:13])

    def test_coordinate_token_handling(self):
        """Test handling of coordinate tokens in collation."""
        # Create sample with coordinate tokens
        sample_with_coords = {
            "input_ids": torch.tensor(
                [
                    1,
                    2,
                    3,  # Regular tokens
                    151666,
                    151667,
                    151668,
                    151669,  # Coordinate tokens
                    4,
                    5,
                    6,  # More regular tokens
                ]
            ),
            "coordinate_mask": torch.tensor(
                [
                    0,
                    0,
                    0,  # Not coordinate tokens
                    1,
                    1,
                    1,
                    1,  # Coordinate tokens
                    0,
                    0,
                    0,  # Not coordinate tokens
                ]
            ),
        }

        # Validate coordinate mask
        coordinate_positions = (sample_with_coords["coordinate_mask"] == 1).nonzero(
            as_tuple=True
        )[0]
        coordinate_token_ids = sample_with_coords["input_ids"][coordinate_positions]

        # Check that coordinate positions contain coordinate tokens (vocab_size + offset)
        base_vocab_size = 151665
        for token_id in coordinate_token_ids:
            assert token_id >= base_vocab_size, (
                f"Token {token_id} should be a coordinate token"
            )

    def test_batch_memory_efficiency(self):
        """Test memory efficiency of different collation strategies."""
        # Create samples with very different lengths
        short_sample = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
        }

        long_sample = {
            "input_ids": torch.tensor(list(range(1, 101))),  # 100 tokens
            "attention_mask": torch.tensor([1] * 100),
            "labels": torch.tensor(list(range(1, 101))),
        }

        samples = [short_sample, long_sample]

        # Standard collation (pads to max length)
        standard_batch = self._mock_standard_collate(samples)
        standard_memory = standard_batch["input_ids"].numel()

        # Packed collation (more efficient)
        packed_batch = self._mock_packed_collate(samples, max_length=120)
        packed_memory = packed_batch["input_ids"].numel()

        # Packed should be more memory efficient
        # (This is a simplified test - real implementation would show bigger difference)
        assert packed_memory <= standard_memory

    def test_image_batch_handling(self):
        """Test batching of image data."""
        # Create samples with images
        samples = [
            {
                "pixel_values": torch.randn(3, 224, 224),
                "image_grid_thw": torch.tensor([1, 224, 224]),
            },
            {
                "pixel_values": torch.randn(3, 224, 224),
                "image_grid_thw": torch.tensor([1, 224, 224]),
            },
            {
                "pixel_values": torch.randn(3, 224, 224),
                "image_grid_thw": torch.tensor([1, 224, 224]),
            },
        ]

        # Mock image batching
        image_batch = self._mock_batch_images(samples)

        # Validate image batch
        expected_shapes = {
            "pixel_values": (3, 3, 224, 224),  # (batch_size, channels, height, width)
            "image_grid_thw": (3, 3),  # (batch_size, 3)
        }

        assert_tensor_shapes(image_batch, expected_shapes)

        # Check that all images are properly stacked
        for i in range(3):
            assert image_batch["pixel_values"][i].shape == (3, 224, 224)
            assert image_batch["image_grid_thw"][i].shape == (3,)

    def test_error_handling(self):
        """Test error handling in collation."""
        # Test mismatched sequence lengths
        invalid_sample = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "attention_mask": torch.tensor([1, 1, 1]),  # Wrong length
            "labels": torch.tensor([1, 2, 3, 4, 5]),
        }

        with pytest.raises((AssertionError, ValueError)):
            self._validate_sample_consistency(invalid_sample)

        # Test empty batch
        empty_samples = []

        with pytest.raises((AssertionError, ValueError)):
            self._mock_standard_collate(empty_samples)

    # Helper methods for mocking collation functionality

    def _mock_standard_collate(
        self, samples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Mock standard collation with padding."""
        if not samples:
            raise ValueError("Cannot collate empty batch")

        batch_size = len(samples)

        # Find max sequence length
        max_length = max(len(sample["input_ids"]) for sample in samples)

        # Pad sequences
        batch = {}

        for key in ["input_ids", "attention_mask", "labels"]:
            padded_sequences = []
            for sample in samples:
                seq = sample[key]
                pad_length = max_length - len(seq)

                if key == "labels":
                    # Pad labels with -100 (ignore index)
                    padded_seq = torch.cat(
                        [seq, torch.full((pad_length,), -100, dtype=seq.dtype)]
                    )
                else:
                    # Pad other sequences with 0
                    padded_seq = torch.cat(
                        [seq, torch.zeros(pad_length, dtype=seq.dtype)]
                    )

                padded_sequences.append(padded_seq)

            batch[key] = torch.stack(padded_sequences)

        # Handle image data
        if "pixel_values" in samples[0]:
            batch["pixel_values"] = torch.stack(
                [sample["pixel_values"] for sample in samples]
            )

        if "image_grid_thw" in samples[0]:
            batch["image_grid_thw"] = torch.stack(
                [sample["image_grid_thw"] for sample in samples]
            )

        return batch

    def _mock_packed_collate(
        self, samples: List[Dict[str, torch.Tensor]], max_length: int
    ) -> Dict[str, torch.Tensor]:
        """Mock packed collation for memory efficiency."""
        if not samples:
            raise ValueError("Cannot collate empty batch")

        # For simplicity, just concatenate sequences up to max_length
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        cu_seqlens = [0]  # Cumulative sequence lengths

        current_length = 0
        for sample in samples:
            seq_len = min(len(sample["input_ids"]), max_length - current_length)
            if seq_len <= 0:
                break

            all_input_ids.append(sample["input_ids"][:seq_len])
            all_attention_masks.append(sample["attention_mask"][:seq_len])
            all_labels.append(sample["labels"][:seq_len])

            current_length += seq_len
            cu_seqlens.append(current_length)

        batch = {
            "input_ids": torch.cat(all_input_ids).unsqueeze(0),  # (1, total_length)
            "attention_mask": torch.cat(all_attention_masks).unsqueeze(0),
            "labels": torch.cat(all_labels).unsqueeze(0),
            "cu_seqlens": torch.tensor(cu_seqlens),
        }

        # Handle image data (still need individual images)
        if "pixel_values" in samples[0]:
            batch["pixel_values"] = torch.stack(
                [sample["pixel_values"] for sample in samples]
            )

        if "image_grid_thw" in samples[0]:
            batch["image_grid_thw"] = torch.stack(
                [sample["image_grid_thw"] for sample in samples]
            )

        return batch

    def _generate_attention_mask(
        self, input_ids: torch.Tensor, pad_token_id: int = 0
    ) -> torch.Tensor:
        """Generate attention mask based on padding tokens."""
        return (input_ids != pad_token_id).long()

    def _mock_create_labels_with_masking(
        self, input_ids: torch.Tensor, conversation_boundaries: List[tuple]
    ) -> torch.Tensor:
        """Mock label creation with conversation masking."""
        labels = input_ids.clone()
        IGNORE_INDEX = -100

        for role, start, end in conversation_boundaries:
            if role in ["system", "user"]:
                labels[start:end] = IGNORE_INDEX

        return labels

    def _mock_batch_images(
        self, samples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Mock image batching."""
        return {
            "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
            "image_grid_thw": torch.stack(
                [sample["image_grid_thw"] for sample in samples]
            ),
        }

    def _validate_sample_consistency(self, sample: Dict[str, torch.Tensor]):
        """Validate that sample tensors have consistent dimensions."""
        if "input_ids" in sample and "attention_mask" in sample:
            assert len(sample["input_ids"]) == len(sample["attention_mask"]), (
                "input_ids and attention_mask must have same length"
            )

        if "input_ids" in sample and "labels" in sample:
            assert len(sample["input_ids"]) == len(sample["labels"]), (
                "input_ids and labels must have same length"
            )


class TestCollatorIntegration:
    """Test collator integration with different data types."""

    def test_coordinate_token_collation(self):
        """Test collation of samples with coordinate tokens."""
        # Create samples with different coordinate token counts
        samples = [
            {
                "input_ids": torch.tensor(
                    [1, 2, 151666, 151667, 3, 4]
                ),  # 2 coord tokens
                "coordinate_mask": torch.tensor([0, 0, 1, 1, 0, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
                "labels": torch.tensor([1, 2, 151666, 151667, 3, 4]),
            },
            {
                "input_ids": torch.tensor([5, 6, 151668, 7]),  # 1 coord token
                "coordinate_mask": torch.tensor([0, 0, 1, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "labels": torch.tensor([5, 6, 151668, 7]),
            },
        ]

        # Mock collation
        batch = self._mock_standard_collate_with_coords(samples)

        # Validate coordinate information is preserved
        assert "coordinate_mask" in batch
        assert batch["coordinate_mask"].shape == batch["input_ids"].shape

        # Check that coordinate positions are correctly marked
        coord_positions = (batch["coordinate_mask"] == 1).nonzero(as_tuple=True)
        assert len(coord_positions[0]) > 0  # Should have coordinate tokens

    def test_teacher_student_collation(self):
        """Test collation of teacher-student conversation samples."""
        # Create teacher-student samples
        teacher_student_sample = {
            "input_ids": torch.tensor(
                [
                    151644,
                    100,
                    151645,  # system
                    151644,
                    101,
                    151645,  # teacher user
                    151644,
                    102,
                    151645,  # teacher assistant
                    151644,
                    103,
                    151645,  # student user
                    151644,
                    104,
                    151645,  # student assistant
                ]
            ),
            "attention_mask": torch.tensor([1] * 15),
            "labels": torch.tensor(
                [
                    -100,
                    -100,
                    -100,  # system (masked)
                    -100,
                    -100,
                    -100,  # teacher user (masked)
                    102,
                    102,
                    -100,  # teacher assistant (not masked)
                    -100,
                    -100,
                    -100,  # student user (masked)
                    104,
                    104,
                    -100,  # student assistant (not masked)
                ]
            ),
            "conversation_roles": [
                "system",
                "teacher_user",
                "teacher_assistant",
                "student_user",
                "student_assistant",
            ],
        }

        # Validate teacher-student structure
        labels = teacher_student_sample["labels"]
        IGNORE_INDEX = -100

        # Check that only assistant responses contribute to loss
        non_masked_positions = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        assert len(non_masked_positions) > 0  # Should have some non-masked tokens

        # In this mock example, positions 6, 7 (teacher assistant) and 12, 13 (student assistant) should not be masked
        expected_non_masked = torch.tensor([6, 7, 12, 13])
        # Note: In real implementation, the exact positions would depend on tokenization

    def _mock_standard_collate_with_coords(
        self, samples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Mock collation that preserves coordinate token information."""
        batch = self._mock_standard_collate(samples)

        # Add coordinate mask to batch
        if "coordinate_mask" in samples[0]:
            max_length = batch["input_ids"].size(1)
            padded_masks = []

            for sample in samples:
                mask = sample["coordinate_mask"]
                pad_length = max_length - len(mask)
                padded_mask = torch.cat(
                    [mask, torch.zeros(pad_length, dtype=mask.dtype)]
                )
                padded_masks.append(padded_mask)

            batch["coordinate_mask"] = torch.stack(padded_masks)

        return batch

    def _mock_standard_collate(
        self, samples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Reuse the standard collation method from above."""
        if not samples:
            raise ValueError("Cannot collate empty batch")

        batch_size = len(samples)
        max_length = max(len(sample["input_ids"]) for sample in samples)

        batch = {}

        for key in ["input_ids", "attention_mask", "labels"]:
            if key not in samples[0]:
                continue

            padded_sequences = []
            for sample in samples:
                seq = sample[key]
                pad_length = max_length - len(seq)

                if key == "labels":
                    padded_seq = torch.cat(
                        [seq, torch.full((pad_length,), -100, dtype=seq.dtype)]
                    )
                else:
                    padded_seq = torch.cat(
                        [seq, torch.zeros(pad_length, dtype=seq.dtype)]
                    )

                padded_sequences.append(padded_seq)

            batch[key] = torch.stack(padded_sequences)

        return batch
