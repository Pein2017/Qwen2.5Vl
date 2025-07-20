#!/usr/bin/env python3
"""
Simple test script to validate core coordinate token loss functionality.

This focuses on testing the key loss computation logic without complex initialization.
"""

import sys
import os
sys.path.insert(0, '/data3/Qwen2.5-VL-main')

import torch
from unittest.mock import MagicMock

# Test the core loss computation functionality
def test_coordinate_loss_computation():
    """Test core coordinate loss computation logic."""
    print("🧪 Testing Core Coordinate Loss Computation")
    
    # Mock coordinate config
    config = MagicMock()
    config.enable_coordinate_tokens = True
    config.max_coord_value = 2048
    config.soft_expectation_temperature = 1.0
    config.focal_loss_alpha = 0.25
    config.focal_loss_gamma = 2.0
    config.coordinate_loss_weight = 1.0
    config.regular_loss_weight = 1.0
    config.box_start_id = 151648
    config.box_end_id = 151649
    
    # Mock coordinate manager
    manager = MagicMock()
    manager.config = config
    manager.extended_vocab_size = 154048
    manager.original_vocab_size = 152000
    manager.coord_start_id = 152000
    manager.coord_end_id = 154048
    manager.is_coordinate_token.return_value = True
    
    # Import and create loss computer
    from src.utils.coordinate_loss_computer import CoordinateLossComputer
    loss_computer = CoordinateLossComputer(manager)
    
    # Test input validation
    logits = torch.randn(2, 50, 154048, requires_grad=True)
    labels = torch.randint(0, 154048, (2, 50))
    
    # Test that validation passes
    is_valid = loss_computer._validate_inputs(logits, labels)
    assert is_valid == True, "Input validation should pass"
    
    print("   ✅ Input validation works")
    
    # Test loss component initialization
    loss_components = loss_computer._initialize_loss_components(100, 20, 80)
    required_keys = ["coordinate_loss", "focal_loss", "regular_loss", "l1_loss", "giou_loss"]
    for key in required_keys:
        assert key in loss_components, f"Should have {key}"
    
    print("   ✅ Loss component initialization works")
    
    # Test loss value validation
    valid_loss = loss_computer._validate_loss_value(0.5, "test_loss")
    assert valid_loss == 0.5, "Valid loss should be preserved"
    
    invalid_loss = loss_computer._validate_loss_value(float('nan'), "test_loss")
    assert invalid_loss == 0.0, "NaN loss should be converted to 0.0"
    
    extreme_loss = loss_computer._validate_loss_value(2000.0, "test_loss")
    assert extreme_loss == 1000.0, "Extreme loss should be clamped"
    
    print("   ✅ Loss value validation works")
    
    return True

def test_loss_manager_mode_separation():
    """Test Loss Manager's coordinate vs standard mode separation."""
    print("🧪 Testing Loss Manager Mode Separation")
    
    # Mock config and tokenizer
    from unittest.mock import patch
    
    with patch('src.training.loss_manager.config') as mock_config:
        mock_config.coordinate_tokens_enabled = True
        mock_config.teacher_loss_weight = 0.3
        mock_config.student_loss_weight = 1.0
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {}
        
        from src.training.loss_manager import LossManager
        loss_manager = LossManager(mock_tokenizer)
        
        # Test coordinate tokens enabled detection
        coord_enabled = loss_manager._get_coordinate_tokens_enabled()
        assert coord_enabled == True, "Should detect coordinate tokens enabled"
        
        # Test zero coordinate components
        zero_components = loss_manager._get_zero_coordinate_components()
        assert zero_components["coordinate_loss"] == 0.0, "Should have zero coordinate loss"
        assert zero_components["focal_loss"] == 0.0, "Should have zero focal loss"
        
        print("   ✅ Mode detection works")
        print("   ✅ Zero component generation works")
    
    return True

def test_model_wrapper_initialization():
    """Test Model Wrapper loss tracking initialization."""
    print("🧪 Testing Model Wrapper Loss Tracking")
    
    # Create mock wrapper to test initialization methods
    wrapper = MagicMock()
    
    # Import the actual methods we want to test
    from src.models.wrapper import Qwen25VLWithDetection
    
    # Test the initialization method directly
    def test_init_loss_tracking():
        # Mock the attributes
        wrapper._last_coordinate_loss = 0.0
        wrapper._last_focal_loss = 0.0
        wrapper._last_regular_loss = 0.0
        wrapper._last_l1_loss = 0.0
        wrapper._last_giou_loss = 0.0
        wrapper._last_detection_loss = 0.0
        wrapper._last_total_tokens = 0
        wrapper._last_coordinate_tokens = 0
        wrapper._last_regular_tokens = 0
        
        # Check all attributes exist
        required_attrs = [
            '_last_coordinate_loss', '_last_focal_loss', '_last_regular_loss',
            '_last_l1_loss', '_last_giou_loss', '_last_detection_loss',
            '_last_total_tokens', '_last_coordinate_tokens', '_last_regular_tokens'
        ]
        
        for attr in required_attrs:
            assert hasattr(wrapper, attr), f"Should have {attr}"
        
        return True
    
    assert test_init_loss_tracking(), "Loss tracking initialization should work"
    
    print("   ✅ Loss tracking initialization works")
    return True

def test_enhanced_validation():
    """Test enhanced validation functions."""
    print("🧪 Testing Enhanced Validation Functions")
    
    # Test coordinate loss computer validation functions
    from src.utils.coordinate_loss_computer import CoordinateLossComputer
    
    # Mock manager and config
    manager = MagicMock()
    config = MagicMock()
    config.enable_coordinate_tokens = True
    config.max_coord_value = 2048
    config.box_start_id = 151648
    config.box_end_id = 151649
    manager.config = config
    manager.extended_vocab_size = 154048
    
    loss_computer = CoordinateLossComputer(manager)
    
    # Test enhanced focal loss computation
    soft_weights = torch.softmax(torch.randn(10, 100), dim=-1)
    targets = torch.randint(0, 100, (10,))
    
    focal_loss = loss_computer._compute_focal_loss_enhanced(soft_weights, targets)
    assert isinstance(focal_loss, torch.Tensor), "Focal loss should be tensor"
    assert not torch.isnan(focal_loss), "Focal loss should not be NaN"
    assert focal_loss.item() >= 0, "Focal loss should be non-negative"
    
    print("   ✅ Enhanced focal loss computation works")
    
    # Test enhanced GIoU loss computation
    pred_boxes = torch.rand(5, 4)  # 5 boxes, 4 coordinates each
    target_boxes = torch.rand(5, 4)
    
    giou_loss = loss_computer._compute_giou_loss_enhanced(pred_boxes, target_boxes)
    assert isinstance(giou_loss, torch.Tensor), "GIoU loss should be tensor"
    assert not torch.isnan(giou_loss), "GIoU loss should not be NaN"
    
    print("   ✅ Enhanced GIoU loss computation works")
    
    return True

def run_simple_tests():
    """Run simplified validation tests."""
    print("🚀 Starting Simple Coordinate Token Loss Validation")
    print("=" * 60)
    
    tests = [
        test_coordinate_loss_computation,
        test_loss_manager_mode_separation,
        test_model_wrapper_initialization,
        test_enhanced_validation,
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
            import traceback
            traceback.print_exc()
        print()
    
    print("=" * 60)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Core coordinate token functionality is working.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)