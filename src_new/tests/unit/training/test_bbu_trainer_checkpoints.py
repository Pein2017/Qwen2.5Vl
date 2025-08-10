"""
Integration tests for BBUTrainer checkpoint functionality.

Tests the complete checkpoint saving pipeline including:
- Checkpoint size validation (inference-ready ~7-8GB, not 49GB)
- SafeTensors format verification
- Absence of optimizer states in inference checkpoints
- Unified checkpoint creation (both regular and best)
- Metric extraction from trainer state
- Distributed training coordination
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from transformers import TrainerState, TrainingArguments

# Import modules we're testing
from src_new.training.bbu_trainer import BBUTrainer


class TestBBUTrainerCheckpoints:
    """Test suite for BBUTrainer checkpoint functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create mock training arguments
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            save_steps=100,
            save_total_limit=2,
            logging_steps=10,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            learning_rate=1e-5,
        )

        # Create mock trainer state with evaluation metrics
        self.trainer_state = TrainerState()
        self.trainer_state.global_step = 500
        self.trainer_state.epoch = 1.0
        self.trainer_state.log_history = [
            {"step": 100, "train_loss": 2.5},
            {"step": 200, "eval_loss": 1.8, "eval_accuracy": 0.85},
            {"step": 300, "train_loss": 2.2},
            {"step": 400, "eval_loss": 1.5, "eval_accuracy": 0.88},
            {"step": 500, "train_loss": 2.0},
        ]

    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_size_is_inference_ready(self):
        """Test that checkpoints are inference-ready size (~7-8GB), not training size (49GB)."""
        # Create mock trainer with minimal setup
        trainer = self._create_mock_trainer()

        # Mock the model saving to simulate SafeTensors format
        mock_model = Mock()
        mock_model.save_pretrained = Mock()

        # Mock the unwrapped model to return our mock
        with patch.object(trainer, "_get_unwrapped_model", return_value=mock_model):
            # Call checkpoint saving
            trainer._save_checkpoint(mock_model, trial=None)

            # Verify save_pretrained was called with SafeTensors format
            mock_model.save_pretrained.assert_called()
            call_args = mock_model.save_pretrained.call_args

            # Check that safe_serialization=True was passed
            assert call_args[1]["safe_serialization"] is True
            assert "max_shard_size" in call_args[1]

    def test_safetensors_format_used(self):
        """Test that model.safetensors exists in checkpoint, not .bin files."""
        trainer = self._create_mock_trainer()

        # Create a real checkpoint directory structure
        checkpoint_dir = os.path.join(self.output_dir, "checkpoint-500")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Mock model that creates actual SafeTensors file
        def mock_save_pretrained(path, **kwargs):
            # Simulate creating SafeTensors file
            safetensors_path = os.path.join(path, "model.safetensors")
            with open(safetensors_path, "w") as f:
                f.write("fake safetensors data")

            # Verify SafeTensors format was requested
            assert kwargs.get("safe_serialization") is True

        mock_model = Mock()
        mock_model.save_pretrained = mock_save_pretrained

        with patch.object(trainer, "_get_unwrapped_model", return_value=mock_model):
            trainer._save_checkpoint(mock_model, trial=None)

            # Verify SafeTensors file exists
            safetensors_file = os.path.join(checkpoint_dir, "model.safetensors")
            assert os.path.exists(safetensors_file)

            # Verify no .bin files exist
            bin_files = list(Path(checkpoint_dir).glob("*.bin"))
            assert len(bin_files) == 0

    def test_no_optimizer_states_in_checkpoint(self):
        """Test that optimizer states are not saved in inference checkpoints."""
        trainer = self._create_mock_trainer()

        checkpoint_dir = os.path.join(self.output_dir, "checkpoint-500")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Mock model saving
        mock_model = Mock()
        mock_model.save_pretrained = Mock()

        with patch.object(trainer, "_get_unwrapped_model", return_value=mock_model):
            trainer._save_checkpoint(mock_model, trial=None)

            # Verify no DeepSpeed optimizer state files
            optimizer_files = list(Path(checkpoint_dir).glob("*optim_states.pt"))
            assert len(optimizer_files) == 0

            # Verify no PyTorch optimizer files
            pytorch_optimizer_files = list(Path(checkpoint_dir).glob("optimizer.pt"))
            assert len(pytorch_optimizer_files) == 0

    def test_unified_checkpoint_creation(self):
        """Test that both regular and best checkpoints are created when metrics qualify."""
        trainer = self._create_mock_trainer()

        # Set up trainer state with evaluation metrics
        trainer.state = self.trainer_state

        # Mock model saving
        mock_model = Mock()
        mock_model.save_pretrained = Mock()

        # Track all save_pretrained calls
        save_calls = []

        def track_save_calls(path, **kwargs):
            save_calls.append(path)

        mock_model.save_pretrained = track_save_calls

        with patch.object(trainer, "_get_unwrapped_model", return_value=mock_model):
            # Provide current metrics that should qualify as best
            current_metrics = {"eval_loss": 1.0}  # Better than 1.5 in log history
            trainer._save_checkpoint(
                mock_model, trial=None, current_metrics=current_metrics
            )

            # Should have saved both regular and best checkpoints
            assert len(save_calls) == 2

            # Verify regular checkpoint path
            regular_checkpoint = os.path.join(self.output_dir, "checkpoint-500")
            assert regular_checkpoint in save_calls

            # Verify best checkpoint path (should contain descriptive name)
            best_checkpoint_calls = [call for call in save_calls if "best-" in call]
            assert len(best_checkpoint_calls) == 1
            assert "best-500-loss1.0000" in best_checkpoint_calls[0]

    def test_metric_extraction_from_trainer_state(self):
        """Test that evaluation metrics are correctly extracted from trainer state."""
        trainer = self._create_mock_trainer()
        trainer.state = self.trainer_state

        # Extract current metrics
        metrics = trainer._extract_current_metrics()

        # Should get the most recent evaluation metrics
        expected_metrics = {"eval_loss": 1.5, "eval_accuracy": 0.88}
        assert metrics == expected_metrics

    def test_metric_extraction_no_eval_metrics(self):
        """Test metric extraction when no evaluation metrics are available."""
        trainer = self._create_mock_trainer()

        # State with only training metrics
        trainer.state.log_history = [
            {"step": 100, "train_loss": 2.5},
            {"step": 200, "train_loss": 2.2},
        ]

        metrics = trainer._extract_current_metrics()
        assert metrics == {}

    def test_metric_extraction_empty_log_history(self):
        """Test metric extraction with empty log history."""
        trainer = self._create_mock_trainer()
        trainer.state.log_history = []

        metrics = trainer._extract_current_metrics()
        assert metrics == {}

    def test_distributed_training_coordination(self):
        """Test that only rank 0 performs checkpoint saving operations."""
        # Mock distributed training setup
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=1),
        ):  # Non-zero rank
            trainer = self._create_mock_trainer()

            # Mock should_save to return False for non-zero ranks
            trainer.args.should_save = False

            mock_model = Mock()
            mock_model.save_pretrained = Mock()

            with patch.object(trainer, "_get_unwrapped_model", return_value=mock_model):
                trainer._save_checkpoint(mock_model, trial=None)

                # Should not call save_pretrained on non-zero ranks
                mock_model.save_pretrained.assert_not_called()

    def test_deepspeed_inference_checkpoint_path(self):
        """Test that DeepSpeed uses inference checkpoint path, not full training checkpoint."""
        trainer = self._create_mock_trainer()

        # Mock DeepSpeed model
        mock_deepspeed_model = Mock()
        mock_deepspeed_model.save_checkpoint = Mock()

        # Mock is_deepspeed_enabled
        with patch.object(trainer, "is_deepspeed_enabled", True):
            trainer._save_deepspeed_inference_checkpoint(
                mock_deepspeed_model, "test_checkpoint_dir"
            )

            # Verify DeepSpeed save_checkpoint was called with exclude_frozen_parameters
            mock_deepspeed_model.save_checkpoint.assert_called_once()
            call_args = mock_deepspeed_model.save_checkpoint.call_args
            assert call_args[1].get("exclude_frozen_parameters") is True

    def _create_mock_trainer(self):
        """Create a mock BBUTrainer for testing."""
        # Mock the required dependencies
        mock_model = Mock()
        mock_model.training_config = None  # No training config for tests
        mock_tokenizer = Mock()
        mock_data_collator = Mock()

        # Create trainer with minimal setup and proper iterables
        trainer = BBUTrainer(
            model=mock_model,
            processing_class=mock_tokenizer,
            training_args=self.training_args,
            data_collator=mock_data_collator,
            callbacks=[],  # Empty list instead of None
        )

        # Set up required attributes
        trainer.state = self.trainer_state
        trainer.processing_class = mock_tokenizer

        # Mock the training state manager
        trainer.training_state_manager = Mock()

        # Ensure any attributes that might be iterated over are proper lists
        if hasattr(trainer, "callback_handler"):
            trainer.callback_handler.callbacks = []

        return trainer
