#!/usr/bin/env python3
"""
Test script to validate the refactored coordinate token loss system.

This script validates:
1. Loss Manager coordinate token vs standard LLM mode separation
2. Coordinate Loss Computer enhanced validation and debugging
3. Model Wrapper proper loss component attachment
4. Training Coordinator integration and logging consistency

Run with: python temporal/test_coordinate_loss_refactor.py
"""

import sys
import os
sys.path.insert(0, '/data3/Qwen2.5-VL-main')

import torch
import json
from typing import Dict, Any
from unittest.mock import MagicMock, patch

# Import the refactored components
from src.config import init_config
from src.training.loss_manager import LossManager
from src.utils.coordinate_loss_computer import CoordinateLossComputer, create_coordinate_loss_computer
from src.utils.coordinate_token_manager import CoordinateTokenManager, create_coordinate_token_manager
from src.models.wrapper import Qwen25VLWithDetection, CoordinateConfig
from src.training.training_coordinator import TrainingCoordinator

def create_mock_tokenizer():
    """Create a mock tokenizer for testing."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {f"token_{i}": i for i in range(152000)}
    mock_tokenizer.convert_tokens_to_ids.return_value = 151648  # Mock box_start_id
    mock_tokenizer.additional_special_tokens = []
    mock_tokenizer.add_special_tokens.return_value = 2048
    return mock_tokenizer

def create_mock_model_outputs(coordinate_enabled: bool = True):
    """Create mock model outputs with or without coordinate losses."""
    outputs = MagicMock()
    outputs.loss = torch.tensor(0.5, requires_grad=True)
    outputs.logits = torch.randn(2, 100, 154000, requires_grad=True)  # Extended vocab
    outputs.hidden_states = [torch.randn(2, 100, 3584) for _ in range(28)]
    outputs.past_key_values = None
    outputs.attentions = None
    
    if coordinate_enabled:
        # Mock coordinate losses from model wrapper
        outputs._coordinate_loss = 0.023
        outputs._focal_loss = 0.015
        outputs._regular_loss = 0.089
        outputs._l1_loss = 0.012
        outputs._giou_loss = 0.008
        outputs._detection_loss = 0.035
        outputs._total_tokens = 85
        outputs._coordinate_tokens = 12
        outputs._regular_tokens = 73
    else:
        # Standard LLM mode - no coordinate losses
        outputs._coordinate_loss = 0.0
        outputs._focal_loss = 0.0
        outputs._regular_loss = 0.0
        outputs._l1_loss = 0.0
        outputs._giou_loss = 0.0
        outputs._detection_loss = 0.0
        outputs._total_tokens = 0
        outputs._coordinate_tokens = 0
        outputs._regular_tokens = 0
    
    return outputs

def create_test_inputs():
    """Create test inputs for loss computation."""
    return {
        "input_ids": torch.randint(0, 152000, (2, 100)),
        "labels": torch.randint(0, 152000, (2, 100)),
        "attention_mask": torch.ones(2, 100),
        "teacher_assistant_spans": [[(10, 20), (30, 40)], [(15, 25)]],
        "student_assistant_spans": [[(50, 70)], [(60, 80), (85, 95)]],
        "ground_truth_objects": [],
    }

def test_loss_manager_coordinate_mode():
    """Test Loss Manager in coordinate token mode."""
    print("🧪 Testing Loss Manager - Coordinate Token Mode")
    
    # Initialize config for coordinate tokens
    config_data = {
        "coordinate_tokens_enabled": True,
        "teacher_loss_weight": 0.3,
        "student_loss_weight": 1.0,
    }
    init_config(config_data)
    
    # Create loss manager
    tokenizer = create_mock_tokenizer()
    loss_manager = LossManager(tokenizer)
    
    # Test with coordinate token outputs
    model_outputs = create_mock_model_outputs(coordinate_enabled=True)
    inputs = create_test_inputs()
    
    total_loss, loss_components = loss_manager.compute_total_loss(
        model_outputs, inputs, is_training=True
    )
    
    # Validate results
    assert isinstance(total_loss, torch.Tensor), "Total loss should be tensor"
    assert not torch.isnan(total_loss), "Total loss should not be NaN"
    assert total_loss.item() > 0, "Total loss should be positive"
    
    # Check coordinate loss components
    assert "coordinate_loss" in loss_components, "Should have coordinate_loss"
    assert "focal_loss" in loss_components, "Should have focal_loss"
    assert "regular_loss" in loss_components, "Should have regular_loss"
    assert "l1_loss" in loss_components, "Should have l1_loss"
    assert "giou_loss" in loss_components, "Should have giou_loss"
    
    # Validate non-zero coordinate losses
    coord_loss_sum = (
        loss_components["coordinate_loss"] + 
        loss_components["focal_loss"] + 
        loss_components["l1_loss"] + 
        loss_components["giou_loss"]
    )
    assert coord_loss_sum > 0, "Coordinate losses should be non-zero"
    
    print(f"   ✅ Coordinate mode: total_loss={total_loss.item():.6f}")
    print(f"   ✅ Coordinate losses: coord={loss_components['coordinate_loss']:.6f}, "
          f"focal={loss_components['focal_loss']:.6f}")
    
    return True

def test_loss_manager_standard_mode():
    """Test Loss Manager in standard LLM mode."""
    print("🧪 Testing Loss Manager - Standard LLM Mode")
    
    # Initialize config for standard mode
    config_data = {
        "coordinate_tokens_enabled": False,
        "teacher_loss_weight": 0.3,
        "student_loss_weight": 1.0,
    }
    init_config(config_data)
    
    # Create loss manager
    tokenizer = create_mock_tokenizer()
    loss_manager = LossManager(tokenizer)
    
    # Test with standard outputs
    model_outputs = create_mock_model_outputs(coordinate_enabled=False)
    inputs = create_test_inputs()
    
    total_loss, loss_components = loss_manager.compute_total_loss(
        model_outputs, inputs, is_training=True
    )
    
    # Validate results
    assert isinstance(total_loss, torch.Tensor), "Total loss should be tensor"
    assert not torch.isnan(total_loss), "Total loss should not be NaN"
    
    # Check that coordinate losses are zero in standard mode
    assert loss_components["coordinate_loss"] == 0.0, "Coordinate loss should be zero"
    assert loss_components["focal_loss"] == 0.0, "Focal loss should be zero"
    assert loss_components["regular_loss"] == 0.0, "Regular loss should be zero"
    assert loss_components["l1_loss"] == 0.0, "L1 loss should be zero"
    assert loss_components["giou_loss"] == 0.0, "GIoU loss should be zero"
    
    print(f"   ✅ Standard mode: total_loss={total_loss.item():.6f}")
    print(f"   ✅ Coordinate losses properly zeroed")
    
    return True

def test_coordinate_loss_computer():
    """Test enhanced Coordinate Loss Computer."""
    print("🧪 Testing Enhanced Coordinate Loss Computer")
    
    # Create coordinate token manager
    tokenizer = create_mock_tokenizer()
    coordinate_config = {
        "enable_coordinate_tokens": True,
        "max_coord_value": 2048,
        "coordinate_loss_weight": 1.0,
        "regular_loss_weight": 1.0,
        "soft_expectation_temperature": 1.0,
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2.0,
    }
    
    coordinate_manager = create_coordinate_token_manager(
        tokenizer, 152000, coordinate_config
    )
    
    # Create loss computer
    loss_computer = create_coordinate_loss_computer(coordinate_manager)
    
    # Test coordinate-aware loss computation
    logits = torch.randn(2, 50, 154000, requires_grad=True)
    labels = torch.randint(0, 154000, (2, 50))
    
    total_loss, loss_components = loss_computer.compute_coordinate_aware_loss(
        logits, labels
    )
    
    # Validate results
    assert isinstance(total_loss, torch.Tensor), "Total loss should be tensor"
    assert total_loss.requires_grad, "Total loss should require gradients"
    assert not torch.isnan(total_loss), "Total loss should not be NaN"
    
    # Check loss components structure
    required_components = ["coordinate_loss", "focal_loss", "regular_loss", "l1_loss", "giou_loss"]
    for component in required_components:
        assert component in loss_components, f"Should have {component}"
    
    print(f"   ✅ Enhanced loss computation: total_loss={total_loss.item():.6f}")
    print(f"   ✅ Loss components validated")
    
    return True

def test_model_wrapper_loss_attachment():
    """Test Model Wrapper loss component attachment."""
    print("🧪 Testing Model Wrapper Loss Attachment")
    
    # Create mock tokenizer and coordinate config
    tokenizer = create_mock_tokenizer()
    coordinate_config = CoordinateConfig(
        enable_coordinate_tokens=True,
        max_coord_value=2048,
    )
    
    # Create wrapper (mock base model creation)
    with patch('src.models.wrapper.Qwen2_5_VLForConditionalGeneration.from_pretrained') as mock_from_pretrained:
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value.weight.shape = [152000, 3584]
        mock_model.get_output_embeddings.return_value.weight.shape = [152000, 3584]
        mock_model.parameters.return_value = [torch.randn(100, 100)]
        mock_from_pretrained.return_value = mock_model
        
        wrapper = Qwen25VLWithDetection(
            base_model_path="/fake/path",
            num_queries=100,
            max_caption_length=32,
            tokenizer=tokenizer,
            coordinate_config=coordinate_config,
        )
        
        # Test loss component initialization
        wrapper._initialize_loss_tracking_components()
        wrapper._ensure_loss_tracking_initialized()
        
        # Validate all tracking components exist
        required_attrs = [
            '_last_coordinate_loss', '_last_focal_loss', '_last_regular_loss',
            '_last_l1_loss', '_last_giou_loss', '_last_detection_loss',
            '_last_total_tokens', '_last_coordinate_tokens', '_last_regular_tokens'
        ]
        
        for attr in required_attrs:
            assert hasattr(wrapper, attr), f"Should have {attr}"
            assert getattr(wrapper, attr) == 0.0 or getattr(wrapper, attr) == 0, f"{attr} should be initialized"
        
        # Test loss attachment
        mock_outputs = MagicMock()
        wrapper._attach_coordinate_losses_to_outputs(mock_outputs)
        wrapper._validate_loss_attachment(mock_outputs)
        
        print(f"   ✅ Loss tracking components initialized")
        print(f"   ✅ Loss attachment validated")
    
    return True

def test_training_coordinator_integration():
    """Test Training Coordinator integration."""
    print("🧪 Testing Training Coordinator Integration")
    
    # Initialize config
    config_data = {
        "coordinate_tokens_enabled": True,
        "teacher_loss_weight": 0.3,
        "student_loss_weight": 1.0,
        "weight_decay": 0.01,
    }
    init_config(config_data)
    
    # Create mock model and coordinator
    mock_model = MagicMock()
    mock_model.named_parameters.return_value = [
        ("test_param", torch.randn(10, 10, requires_grad=True))
    ]
    
    tokenizer = create_mock_tokenizer()
    coordinator = TrainingCoordinator(mock_model, tokenizer)
    
    # Test coordinate token validation
    coord_enabled = coordinator._validate_coordinate_token_config()
    assert coord_enabled == True, "Should detect coordinate tokens enabled"
    
    # Test loss computation
    model_outputs = create_mock_model_outputs(coordinate_enabled=True)
    inputs = create_test_inputs()
    
    total_loss, loss_components = coordinator.compute_loss(
        model_outputs, inputs, is_training=True
    )
    
    # Validate results
    assert isinstance(total_loss, torch.Tensor), "Total loss should be tensor"
    assert not torch.isnan(total_loss), "Total loss should not be NaN"
    
    # Test averaged losses
    averaged_losses = coordinator.get_averaged_losses_and_reset()
    assert "coordinate_loss" in averaged_losses, "Should have coordinate_loss in averaged"
    
    # Test status summary
    status = coordinator.get_status_summary()
    assert "coordinate_metrics" in status, "Should have coordinate_metrics in status"
    assert status["training_state"]["coordinate_tokens_enabled"] == True
    
    print(f"   ✅ Coordinator integration: total_loss={total_loss.item():.6f}")
    print(f"   ✅ Status summary includes coordinate metrics")
    
    return True

def run_all_tests():
    """Run all validation tests."""
    print("🚀 Starting Coordinate Token Loss System Validation")
    print("=" * 60)
    
    tests = [
        test_loss_manager_coordinate_mode,
        test_loss_manager_standard_mode,
        test_coordinate_loss_computer,
        test_model_wrapper_loss_attachment,
        test_training_coordinator_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"   ✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"   ❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"   ❌ {test.__name__} FAILED: {e}")
        print()
    
    print("=" * 60)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Coordinate token loss system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)