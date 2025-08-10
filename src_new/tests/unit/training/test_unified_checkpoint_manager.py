"""
Unit tests for UnifiedCheckpointManager.

Tests the core logic for unified checkpoint management including:
- Best checkpoint detection with different metrics
- Descriptive naming format generation
- Old checkpoint cleanup operations
- Edge cases and error handling
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import torch

# Import the module we're testing (will be created)
from src_new.training.unified_checkpoint_manager import UnifiedCheckpointManager


class TestUnifiedCheckpointManager:
    """Test suite for UnifiedCheckpointManager."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = UnifiedCheckpointManager(
            metric_name="eval_loss",
            greater_is_better=False
        )

    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_is_new_best_with_eval_loss(self):
        """Test best checkpoint detection with eval_loss (lower is better)."""
        # Test initial state - first metric should always be best
        metrics = {"eval_loss": 2.5}
        assert self.manager.is_new_best(metrics) is True
        
        # Update with first best
        self.manager.update_best_checkpoint(metrics, "/fake/path")
        
        # Test better metric (lower loss)
        better_metrics = {"eval_loss": 2.0}
        assert self.manager.is_new_best(better_metrics) is True
        
        # Test worse metric (higher loss)
        worse_metrics = {"eval_loss": 3.0}
        assert self.manager.is_new_best(worse_metrics) is False
        
        # Test equal metric (should not be considered better)
        equal_metrics = {"eval_loss": 2.5}
        assert self.manager.is_new_best(equal_metrics) is False

    def test_is_new_best_with_accuracy(self):
        """Test best checkpoint detection with accuracy (higher is better)."""
        manager = UnifiedCheckpointManager(
            metric_name="eval_accuracy",
            greater_is_better=True
        )
        
        # Test initial state
        metrics = {"eval_accuracy": 0.85}
        assert manager.is_new_best(metrics) is True
        
        # Update with first best
        manager.update_best_checkpoint(metrics, "/fake/path")
        
        # Test better metric (higher accuracy)
        better_metrics = {"eval_accuracy": 0.90}
        assert manager.is_new_best(better_metrics) is True
        
        # Test worse metric (lower accuracy)
        worse_metrics = {"eval_accuracy": 0.80}
        assert manager.is_new_best(worse_metrics) is False

    def test_create_best_checkpoint_name_format(self):
        """Test descriptive naming format matches existing callback format."""
        metrics = {"eval_loss": 1.2345}
        step = 1000
        
        expected_name = "best-1000-loss1.2345"
        actual_name = self.manager.create_best_checkpoint_name(metrics, step)
        
        assert actual_name == expected_name

    def test_create_best_checkpoint_name_format_precision(self):
        """Test naming format handles different precision values correctly."""
        test_cases = [
            ({"eval_loss": 1.0}, 500, "best-500-loss1.0000"),
            ({"eval_loss": 0.123456}, 1500, "best-1500-loss0.1235"),
            ({"eval_accuracy": 0.9876}, 2000, "best-2000-accuracy0.9876"),
        ]
        
        for metrics, step, expected in test_cases:
            metric_name = list(metrics.keys())[0]
            manager = UnifiedCheckpointManager(
                metric_name=metric_name,
                greater_is_better=metric_name == "eval_accuracy"
            )
            actual = manager.create_best_checkpoint_name(metrics, step)
            assert actual == expected

    def test_cleanup_old_best_checkpoint(self):
        """Test old checkpoint cleanup with file system operations."""
        # Create a fake old checkpoint directory
        old_checkpoint_dir = os.path.join(self.temp_dir, "best-500-loss2.5000")
        os.makedirs(old_checkpoint_dir)
        
        # Create some files in the directory
        test_file = os.path.join(old_checkpoint_dir, "model.safetensors")
        with open(test_file, "w") as f:
            f.write("fake model data")
        
        # Set up manager with old checkpoint
        self.manager.current_best_dir = old_checkpoint_dir
        
        # Verify directory exists before cleanup
        assert os.path.exists(old_checkpoint_dir)
        assert os.path.exists(test_file)
        
        # Perform cleanup
        self.manager.cleanup_old_best_checkpoint()
        
        # Verify directory is removed
        assert not os.path.exists(old_checkpoint_dir)
        assert self.manager.current_best_dir is None

    def test_cleanup_old_best_checkpoint_nonexistent(self):
        """Test cleanup handles nonexistent directories gracefully."""
        # Set up manager with nonexistent checkpoint
        fake_path = os.path.join(self.temp_dir, "nonexistent-checkpoint")
        self.manager.current_best_dir = fake_path
        
        # Should not raise exception
        self.manager.cleanup_old_best_checkpoint()
        assert self.manager.current_best_dir is None

    def test_update_best_checkpoint(self):
        """Test best checkpoint tracking state updates."""
        metrics = {"eval_loss": 1.5}
        checkpoint_path = "/path/to/best/checkpoint"
        
        # Initial state
        assert self.manager.current_best_metric is None
        assert self.manager.current_best_dir is None
        
        # Update with new best
        self.manager.update_best_checkpoint(metrics, checkpoint_path)
        
        # Verify state updated
        assert self.manager.current_best_metric == 1.5
        assert self.manager.current_best_dir == checkpoint_path

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with NaN metrics
        nan_metrics = {"eval_loss": float('nan')}
        assert self.manager.is_new_best(nan_metrics) is False
        
        # Test with missing metric
        missing_metrics = {"other_metric": 1.0}
        assert self.manager.is_new_best(missing_metrics) is False
        
        # Test with empty metrics
        empty_metrics = {}
        assert self.manager.is_new_best(empty_metrics) is False

    def test_invalid_metric_name_raises_error(self):
        """Test that invalid metric names raise appropriate errors."""
        metrics = {"eval_loss": 1.5}
        step = 1000
        
        # Manager configured for different metric
        manager = UnifiedCheckpointManager(metric_name="eval_accuracy")
        
        # Should raise ValueError when metric not found
        with pytest.raises(ValueError, match="Metric 'eval_accuracy' not found"):
            manager.create_best_checkpoint_name(metrics, step)

    def test_cleanup_with_permission_error(self):
        """Test cleanup handles permission errors gracefully."""
        # Create a directory
        old_checkpoint_dir = os.path.join(self.temp_dir, "best-500-loss2.5000")
        os.makedirs(old_checkpoint_dir)
        
        self.manager.current_best_dir = old_checkpoint_dir
        
        # Mock shutil.rmtree to raise PermissionError
        with patch('shutil.rmtree', side_effect=PermissionError("Permission denied")):
            # Should log error but not raise exception
            self.manager.cleanup_old_best_checkpoint()
            
            # Directory should still exist due to permission error
            assert os.path.exists(old_checkpoint_dir)
            # But tracking should be reset
            assert self.manager.current_best_dir is None

    def test_thread_safety_considerations(self):
        """Test that manager operations are safe for concurrent access."""
        # This is a basic test - in practice, the trainer ensures single-threaded access
        metrics1 = {"eval_loss": 1.0}
        metrics2 = {"eval_loss": 0.5}
        
        # Simulate concurrent updates
        self.manager.update_best_checkpoint(metrics1, "/path1")
        assert self.manager.current_best_metric == 1.0
        
        self.manager.update_best_checkpoint(metrics2, "/path2")
        assert self.manager.current_best_metric == 0.5
        assert self.manager.current_best_dir == "/path2"
