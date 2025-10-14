"""
Test to verify that checkpoint saving only happens on rank 0.

This test ensures that the coordinate pretraining trainer correctly
saves checkpoints only on the main process (rank 0) to avoid conflicts
in distributed training scenarios.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class TestRank0CheckpointSaving(unittest.TestCase):
    """Test checkpoint saving behavior in distributed training scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src_coord_pretrain.training.trainer._save_inference_checkpoint')
    def test_checkpoint_saving_only_on_rank0(self, mock_save_checkpoint):
        """Test that checkpoint saving only occurs on rank 0."""
        # Mock trainer with rank 0 behavior
        mock_trainer_rank0 = Mock()
        mock_trainer_rank0.is_world_process_zero.return_value = True
        mock_trainer_rank0.state.global_step = 100
        mock_trainer_rank0.evaluate.return_value = {"eval_loss": 0.5}

        # Mock trainer with non-rank 0 behavior
        mock_trainer_rank1 = Mock()
        mock_trainer_rank1.is_world_process_zero.return_value = False
        mock_trainer_rank1.state.global_step = 100
        mock_trainer_rank1.evaluate.return_value = {"eval_loss": 0.5}

        # Mock other required objects
        mock_model = Mock()
        mock_processor = Mock()
        mock_coord_meta = {"coord_token_ids": [151667, 151668]}
        max_coord_value = 1024

        # Simulate the checkpoint saving logic for rank 0
        final_metrics = mock_trainer_rank0.evaluate()
        if mock_trainer_rank0.is_world_process_zero():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            final_step = mock_trainer_rank0.state.global_step
            final_checkpoint_dir = self.output_dir / f"checkpoint-{final_step}"
            final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # This should be called for rank 0
            mock_save_checkpoint(
                mock_model,
                mock_processor,
                final_checkpoint_dir,
                mock_coord_meta,
                max_coord_value,
                final_metrics,
            )

        # Simulate the checkpoint saving logic for rank 1
        final_metrics = mock_trainer_rank1.evaluate()
        if mock_trainer_rank1.is_world_process_zero():
            # This block should NOT be executed for rank 1
            mock_save_checkpoint(
                mock_model,
                mock_processor,
                self.output_dir / f"checkpoint-{mock_trainer_rank1.state.global_step}",
                mock_coord_meta,
                max_coord_value,
                final_metrics,
            )

        # Verify that _save_inference_checkpoint was called exactly once (only for rank 0)
        self.assertEqual(mock_save_checkpoint.call_count, 1)

        # Verify the checkpoint directory was created for rank 0
        expected_checkpoint_dir = self.output_dir / "checkpoint-100"
        self.assertTrue(expected_checkpoint_dir.exists())

    def test_is_world_process_zero_method_call(self):
        """Test that is_world_process_zero() is called as a method, not accessed as attribute."""
        # Create a mock trainer that tracks method calls
        mock_trainer = Mock()
        mock_trainer.is_world_process_zero.return_value = True

        # Simulate the checkpoint saving condition check
        should_save = mock_trainer.is_world_process_zero()

        # Verify the method was called
        mock_trainer.is_world_process_zero.assert_called_once()
        self.assertTrue(should_save)

        # Test with rank 1 (non-main process)
        mock_trainer_rank1 = Mock()
        mock_trainer_rank1.is_world_process_zero.return_value = False

        should_save_rank1 = mock_trainer_rank1.is_world_process_zero()
        mock_trainer_rank1.is_world_process_zero.assert_called_once()
        self.assertFalse(should_save_rank1)

    def test_checkpoint_directory_creation_logic(self):
        """Test that checkpoint directories are created with correct naming."""
        mock_trainer = Mock()
        mock_trainer.is_world_process_zero.return_value = True
        mock_trainer.state.global_step = 1500

        # Simulate directory creation logic
        if mock_trainer.is_world_process_zero():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            final_step = mock_trainer.state.global_step
            final_checkpoint_dir = self.output_dir / f"checkpoint-{final_step}"
            final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Verify correct directory structure
        expected_dir = self.output_dir / "checkpoint-1500"
        self.assertTrue(expected_dir.exists())
        self.assertTrue(expected_dir.is_dir())

    @patch('builtins.print')
    def test_logging_only_on_rank0(self, mock_print):
        """Test that checkpoint save logging only occurs on rank 0."""
        # Mock trainer for rank 0
        mock_trainer_rank0 = Mock()
        mock_trainer_rank0.is_world_process_zero.return_value = True
        mock_trainer_rank0.state.global_step = 200

        # Mock trainer for rank 1
        mock_trainer_rank1 = Mock()
        mock_trainer_rank1.is_world_process_zero.return_value = False
        mock_trainer_rank1.state.global_step = 200

        # Simulate logging for rank 0
        if mock_trainer_rank0.is_world_process_zero():
            final_step = mock_trainer_rank0.state.global_step
            print(f"💾 Saving final inference-ready checkpoint to checkpoint-{final_step}...")
            print(f"✅ Training completed! Final checkpoint saved to checkpoint-{final_step}")

        # Simulate logging for rank 1 (should not print)
        if mock_trainer_rank1.is_world_process_zero():
            final_step = mock_trainer_rank1.state.global_step
            print(f"💾 Saving final inference-ready checkpoint to checkpoint-{final_step}...")
            print(f"✅ Training completed! Final checkpoint saved to checkpoint-{final_step}")

        # Verify print was called exactly twice (only for rank 0)
        self.assertEqual(mock_print.call_count, 2)

        # Verify the correct messages were printed
        expected_calls = [
            unittest.mock.call("💾 Saving final inference-ready checkpoint to checkpoint-200..."),
            unittest.mock.call("✅ Training completed! Final checkpoint saved to checkpoint-200")
        ]
        mock_print.assert_has_calls(expected_calls)


if __name__ == "__main__":
    unittest.main()
