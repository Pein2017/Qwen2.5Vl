#!/usr/bin/env python3
"""
Test script to verify the unified loss structure works correctly
"""
import sys
sys.path.append('src')

import torch
from transformers import Qwen2VLProcessor
from src.training.loss_manager import LossManager
from src.utils.coordinate_loss_computer import CoordinateLossComputer
from src.utils.coordinate_token_manager import create_coordinate_token_manager

def test_loss_structure():
    """Test that the new loss structure works without coordinate_loss summation."""
    print("🧪 Testing unified loss structure...")
    
    # Mock tokenizer for testing
    class MockTokenizer:
        def get_vocab(self):
            return {'<pad>': 0, '<unk>': 1}
        
        def convert_tokens_to_ids(self, token):
            if token == '<coord_0>':
                return 151665
            elif token == '<|box_start|>':
                return 151648
            elif token == '<|box_end|>':
                return 151649
            return 1
    
    tokenizer = MockTokenizer()
    
    # Mock config to avoid dependency issues
    class MockConfig:
        coordinate_tokens_enabled = True
        teacher_loss_weight = 0.1
        student_loss_weight = 0.9
    
    # Test Loss Manager
    print("  Testing LossManager...")
    loss_manager = LossManager(tokenizer=tokenizer)
    
    # Monkey patch the config method to avoid config loading
    loss_manager._get_coordinate_tokens_enabled = lambda: True
    loss_manager._get_validated_loss_weights = lambda: (0.1, 0.9)
    
    # Create mock model outputs without coordinate_loss
    class MockOutputs:
        def __init__(self):
            self.loss = torch.tensor(1.0)
            self.logits = torch.randn(2, 10, 1000)
            # Individual loss components (no coordinate_loss summation)
            self._focal_loss = 0.1
            self._regular_loss = 0.2
            self._l1_loss = 0.3
            self._giou_loss = 0.4
            self._total_tokens = 100
            self._coordinate_tokens = 20
            self._regular_tokens = 80
    
    mock_outputs = MockOutputs()
    mock_inputs = {
        'labels': torch.randint(0, 1000, (2, 10)),
        'teacher_assistant_spans': [[(1, 3)], [(2, 4)]],
        'student_assistant_spans': [[(4, 6)], [(5, 7)]],
    }
    
    # Test loss computation
    total_loss, loss_components = loss_manager.compute_total_loss(
        mock_outputs, mock_inputs, is_training=True
    )
    
    print(f"    ✅ Total loss computed: {total_loss.item():.3f}")
    print(f"    ✅ Loss components: {list(loss_components.keys())}")
    
    # Verify coordinate_loss is not in components
    assert 'coordinate_loss' not in loss_components, "coordinate_loss should be removed!"
    assert 'focal_loss' in loss_components, "focal_loss should be present"
    assert 'l1_loss' in loss_components, "l1_loss should be present"  
    assert 'giou_loss' in loss_components, "giou_loss should be present"
    
    print("    ✅ coordinate_loss summation successfully removed")
    print("    ✅ Individual loss components preserved")
    
    # Test averaging
    averaged_losses = loss_manager.get_averaged_losses()
    print(f"    ✅ Averaged losses: {list(averaged_losses.keys())}")
    assert 'coordinate_loss' not in averaged_losses, "coordinate_loss should not be in averaged losses!"
    
    print("🎉 All tests passed! Unified loss structure is working correctly.")
    print()
    print("📊 Expected training log structure:")
    expected_structure = {
        'loss': 'total_combined_loss',
        'lm_loss': 'base_language_modeling_loss', 
        'teacher_lm_loss': 'teacher_spans_loss',
        'student_lm_loss': 'student_spans_loss',
        'focal_loss': 'coordinate_focal_loss',
        'regular_loss': 'regular_token_ce_loss',
        'l1_loss': 'coordinate_l1_loss',
        'giou_loss': 'coordinate_giou_loss',
        # 'coord_l1_loss': 'same_as_l1_loss (legacy_alias)',
        # 'coord_giou_loss': 'same_as_giou_loss (legacy_alias)',
    }
    
    for key, desc in expected_structure.items():
        print(f"  '{key}': {desc}")

if __name__ == "__main__":
    test_loss_structure()