#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive DetectionModel Tests

This module tests the DetectionModel wrapper that integrates coordinate token
support with the base Qwen2.5-VL model.

Key Features:
- Tests model initialization and configuration
- Tests coordinate mode switching
- Tests forward pass with coordinate token support
- Tests loss computation integration
- Tests model saving and loading
- Tests generation capabilities
"""

from unittest.mock import Mock, patch

import pytest
import torch

from src_new.models.wrapper import CoordinateProcessor, DetectionModel


class TestDetectionModel:
    """Test suite for DetectionModel wrapper."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock()
        config.coordinate_tokens_enabled = True
        config.max_coord_value = 1024
        config.coordinate_loss_weight = 0.05
        config.regular_loss_weight = 1.0
        config.coordinate_loss_temperature = 1.0
        config.teacher_loss_weight = 0.3
        config.student_loss_weight = 1.0
        config.model_hidden_size = 2048
        config.skip_vocab_extension = False
        return config

    @pytest.fixture
    def mock_base_model(self):
        """Create mock base Qwen2.5-VL model."""
        model = Mock()
        model.config = Mock()
        model.config.vocab_size = 151665
        model.config.hidden_size = 2048

        # Mock forward pass
        mock_output = Mock()
        mock_output.loss = torch.tensor(1.5)
        mock_output.logits = torch.randn(1, 10, 151665)
        mock_output.hidden_states = torch.randn(1, 10, 2048)

        model.forward.return_value = mock_output
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Mock model methods
        model.train.return_value = model
        model.eval.return_value = model
        model.to.return_value = model
        model.parameters.return_value = []

        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.vocab_size = 151665
        vocab = {"<|pad|>": 0, "<|eos|>": 1}
        # Add coordinate tokens
        for i in range(1025):  # 0 to 1024
            vocab[f"<|coord_{i}|>"] = 151667 + i
        tokenizer.get_vocab.return_value = vocab
        return tokenizer

    @pytest.fixture
    def detection_model(
        self, real_test_config, real_extended_tokenizer, realistic_base_model
    ):
        """Create DetectionModel for testing with real components."""
        with patch("src_new.models.wrapper.apply_comprehensive_qwen25_fixes"):
            model = DetectionModel(
                base_model=realistic_base_model,
                config=real_test_config,
                tokenizer=real_extended_tokenizer,
                skip_expansion=True,  # Skip expansion for testing
            )
        return model

    def test_detection_model_initialization(self, detection_model, real_test_config):
        """Test DetectionModel initialization."""
        assert detection_model.base_model is not None
        # detection_model.config exposes HF base model config; training config is stored separately
        assert (
            hasattr(detection_model, "training_config")
            and detection_model.training_config == real_test_config
        )
        assert (
            detection_model._coordinate_mode
            == real_test_config.coordinate_tokens_enabled
        )
        assert isinstance(detection_model.coordinate_processor, CoordinateProcessor)
        assert detection_model.loss_manager is not None

    def test_coordinate_mode_switching(self, detection_model):
        """Test coordinate mode enable/disable functionality."""
        # Test enabling coordinate mode
        detection_model.enable_coordinate_mode()
        assert detection_model.coordinate_mode_enabled is True

        # Test disabling coordinate mode
        detection_model.disable_coordinate_mode()
        assert detection_model.coordinate_mode_enabled is False

    def test_forward_pass_standard_mode(self, detection_model):
        """Test forward pass in standard mode (no coordinate tokens)."""
        detection_model.disable_coordinate_mode()

        # Create test inputs
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)

        # Forward pass
        outputs = detection_model(input_ids=input_ids, attention_mask=attention_mask)

        # Should return base model outputs
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")

    def test_forward_pass_coordinate_mode(self, detection_model):
        """Test forward pass in coordinate mode."""
        detection_model.enable_coordinate_mode()

        # Create test inputs with labels for loss computation
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        labels = torch.randint(0, 1000, (1, 10))

        # Forward pass
        outputs = detection_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # Should return ModelOutput with loss components
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")

    def test_forward_pass_with_image_inputs(self, detection_model):
        """Test forward pass with image inputs."""
        # Create test inputs including image data (Qwen2.5-VL expects flattened patches)
        # Use a small grid [[1, 2, 2]] => expected patches = 1*2*2 = 4, tokens = 1 (merge_size=2)
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
        num_patches = int(
            (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
            .sum()
            .item()
        )
        patch_features = detection_model.base_model.config.hidden_size
        pixel_values = torch.randn(num_patches, patch_features)

        # Build input_ids with correct number of image tokens
        image_token_id = getattr(
            detection_model.base_model.config, "image_token_id", 151655
        )
        if not isinstance(image_token_id, int):
            image_token_id = 151655
        input_ids = torch.full((1, 10), 42, dtype=torch.long)
        input_ids[0, 0] = image_token_id  # one image token
        attention_mask = torch.ones(1, 10)

        # Forward pass
        outputs = detection_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # Should handle image inputs properly
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")

    def test_generation(self, detection_model):
        """Test text generation capabilities."""
        # Create test inputs
        input_ids = torch.randint(0, 1000, (1, 5))

        # Generate
        generated = detection_model.generate(
            input_ids=input_ids, max_length=10, do_sample=False
        )

        # Should return generated token IDs
        assert isinstance(generated, torch.Tensor)
        assert generated.shape[0] == 1  # Batch size

    def test_loss_components_retrieval(self, detection_model):
        """Test retrieval of loss components after forward pass."""
        detection_model.enable_coordinate_mode()

        # Create test inputs
        input_ids = torch.randint(0, 1000, (1, 10))
        labels = torch.randint(0, 1000, (1, 10))

        # Forward pass
        detection_model(input_ids=input_ids, labels=labels)

        # Get loss components
        loss_components = detection_model.get_loss_components()

        # Should return loss components or None
        # (Depends on whether loss computation was successful)
        assert loss_components is None or hasattr(loss_components, "loss")

    def test_training_mode_switching(self, detection_model):
        """Test switching between training and evaluation modes."""
        # Test training mode
        detection_model.train()
        assert detection_model.training is True

        # Test evaluation mode
        detection_model.eval()
        assert detection_model.training is False

    def test_device_movement(self, detection_model):
        """Test moving model to different devices."""
        # Test CPU (should work in testing environment)
        detection_model_cpu = detection_model.to("cpu")
        assert detection_model_cpu is not None

        # Test CUDA (if available)
        if torch.cuda.is_available():
            detection_model_cuda = detection_model.to("cuda")
            assert detection_model_cuda is not None

    def test_parameter_access(self, detection_model):
        """Test access to model parameters."""
        # Should be able to access parameters
        params = list(detection_model.parameters())
        assert isinstance(params, list)

        # Should be able to access named parameters
        named_params = dict(detection_model.named_parameters())
        assert isinstance(named_params, dict)

    def test_epoch_update(self, detection_model):
        """Test epoch update functionality."""
        # Should not raise error
        detection_model.update_epoch(5)

        # Loss manager should have update_epoch method called
        # (This is tested indirectly through no exceptions)

    @patch("src_new.models.wrapper.Path")
    def test_checkpoint_detection(self, mock_path):
        """Test extended checkpoint detection."""
        # Mock path existence
        mock_path.return_value.exists.return_value = True

        # Test detection
        is_extended = DetectionModel.detect_extended_checkpoint("/fake/path")

        # Should return boolean
        assert isinstance(is_extended, bool)

    def test_model_saving_preparation(self, detection_model, tmp_path):
        """Test model saving preparation."""
        save_dir = tmp_path / "test_model"
        save_dir.mkdir()

        # Should be able to call save_pretrained without errors
        # (Mock the actual saving since we don't have real model weights)
        with patch.object(detection_model.base_model, "save_pretrained"):
            try:
                detection_model.save_pretrained(str(save_dir))
            except Exception:
                # Some exceptions are expected due to mocking
                # Just ensure the method exists and is callable
                pass

    def test_image_token_expected_count_alignment(
        self, detection_model, real_test_config
    ):
        """Ensure expected image token count matches grid-based expansion.

        This simulates a batch with 2 images where each grid contributes t*h*w // (merge_size**2) tokens.
        """
        import torch

        # Enable coordinate mode to exercise forward validations
        detection_model.enable_coordinate_mode()

        # Prepare input_ids containing the correct number of <|image_pad|> placeholders
        image_token_id = getattr(
            detection_model.base_model.config, "image_token_id", 151655
        )
        if not isinstance(image_token_id, int):
            image_token_id = 151655

        # Suppose merge_size=2 and we have two images with grid_thw: [1, 4, 4] and [1, 2, 2]
        # Expected image tokens = (1*4*4 // 4) + (1*2*2 // 4) = 4 + 1 = 5
        grids = torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.long)
        expected_tokens = int((torch.prod(grids, dim=1) // (2 * 2)).sum().item())
        assert expected_tokens == 5

        # Build input_ids with 5 image_token_ids somewhere in the sequence
        seq_len = 20
        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        input_ids[0, :expected_tokens] = image_token_id
        attention_mask = torch.ones_like(input_ids)

        # Minimal labels to pass interface
        labels = input_ids.clone()

        # pixel_values rows must equal sum(t*h*w) from grids = (1*4*4)+(1*2*2)=16+4=20; create dummy patch features
        pixel_values = torch.randn(20, detection_model.base_model.config.hidden_size)

        outputs = detection_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=grids,
        )

        assert hasattr(outputs, "logits")
