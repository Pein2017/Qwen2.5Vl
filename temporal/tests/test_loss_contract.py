#!/usr/bin/env python3
"""
Test loss computation contract between model wrapper and loss manager
Verify that teacher-student loss differentiation works correctly
"""

import sys
import os
sys.path.insert(0, '/data3/Qwen2.5-VL-main')

import torch
from src.config.global_config import init_config, reset_config
from src.training.loss_manager import LossManager

def create_mock_model_outputs():
    """Create mock model outputs with loss attributes for testing"""
    class MockOutputs:
        def __init__(self):
            self.loss = torch.tensor(5.0, requires_grad=True)  # Original combined loss
            self.logits = torch.randn(2, 10, 1000)  # Mock logits
            
            # Loss components that model wrapper would attach
            self._llm_loss = torch.tensor(3.0, requires_grad=True)
            self._geometry_focal_loss = torch.tensor(0.5, requires_grad=True)
            self._coordinate_l1_loss = torch.tensor(1.0, requires_grad=True)
            self._geometry_bbox_giou_loss = torch.tensor(0.3, requires_grad=True)
            self._geometry_square_polygon_loss = torch.tensor(0.1, requires_grad=True)
            self._geometry_square_corner_loss = torch.tensor(0.05, requires_grad=True)
            self._geometry_line_smoothness_loss = torch.tensor(0.02, requires_grad=True)
            self._geometry_line_ordering_loss = torch.tensor(0.03, requires_grad=True)
    
    return MockOutputs()

def create_mock_inputs():
    """Create mock inputs with teacher-student spans"""
    return {
        "input_ids": torch.randint(0, 1000, (2, 100)),
        "labels": torch.randint(0, 1000, (2, 100)),
        # Mock teacher-student spans (simplified)
        "teacher_assistant_spans": [[(10, 30)], [(40, 60)]],  # Teacher spans per sample
        "student_assistant_spans": [[(70, 90)], [(20, 40)]]   # Student spans per sample
    }

def test_loss_contract():
    """Test that loss computation contract works correctly"""
    
    print("🧪 Testing loss computation contract...")
    
    try:
        # Reset any existing config
        reset_config()
        
        # Initialize config
        config = init_config("configs/bbu_v2.yaml")
        
        # Create mock model for loss manager
        class MockModel:
            def get_last_coordinate_losses(self):
                return {
                    '_geometry_focal_loss': 0.5,
                    '_coordinate_l1_loss': 1.0,
                    '_geometry_bbox_giou_loss': 0.3,
                    '_geometry_square_polygon_loss': 0.1,
                    '_geometry_square_corner_loss': 0.05,
                    '_geometry_line_smoothness_loss': 0.02,
                    '_geometry_line_ordering_loss': 0.03
                }
        
        # Create mock tokenizer
        class MockTokenizer:
            def get_vocab(self):
                return {"test": 0}
        
        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()
        
        # Initialize loss manager
        loss_manager = LossManager(
            tokenizer=mock_tokenizer,
            model=mock_model,
            teacher_loss_weight=config.teacher_loss_weight,
            student_loss_weight=config.student_loss_weight
        )
        
        # Create mock data
        mock_outputs = create_mock_model_outputs()
        mock_inputs = create_mock_inputs()
        
        print("📊 Input data:")
        print(f"   Original total loss: {mock_outputs.loss.item():.3f}")
        print(f"   LLM loss: {mock_outputs._llm_loss.item():.3f}")
        print(f"   Coordinate losses: focal={mock_outputs._geometry_focal_loss.item():.3f}, l1={mock_outputs._coordinate_l1_loss.item():.3f}")
        print(f"   Teacher weight: {config.teacher_loss_weight}, Student weight: {config.student_loss_weight}")
        
        # Compute losses using loss manager (model_outputs, inputs)
        final_loss, loss_components = loss_manager.compute_total_loss(mock_outputs, mock_inputs)
        
        print("🎯 Loss computation results:")
        print(f"   Final total loss: {final_loss.item():.3f}")
        print(f"   Teacher LLM loss: {loss_components['teacher_lm_loss']:.3f}")
        print(f"   Student LLM loss: {loss_components['student_lm_loss']:.3f}")
        print(f"   Weighted teacher loss: {loss_components['weighted_teacher_loss']:.3f}")
        print(f"   Weighted student loss: {loss_components['weighted_student_loss']:.3f}")
        
        # Verify that final loss is different from original (due to teacher-student weighting)
        if abs(final_loss.item() - mock_outputs.loss.item()) > 0.01:
            print("✅ Loss recomputation working - final loss differs from original")
        else:
            print("❌ FAIL: Final loss same as original - teacher-student weighting not applied")
            return False
        
        # Verify that final loss requires gradients
        if final_loss.requires_grad:
            print("✅ Final loss requires gradients for backpropagation")
        else:
            print("❌ FAIL: Final loss doesn't require gradients")
            return False
        
        # Verify loss components are present
        expected_components = [
            "llm_loss", "teacher_lm_loss", "student_lm_loss", 
            "weighted_teacher_loss", "weighted_student_loss", "final_total_loss"
        ]
        
        for component in expected_components:
            if component in loss_components:
                print(f"✅ {component}: {loss_components[component]:.3f}")
            else:
                print(f"❌ FAIL: Missing loss component: {component}")
                return False
        
        print("✅ Loss computation contract validated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Loss contract test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        reset_config()

if __name__ == "__main__":
    success = test_loss_contract()
    sys.exit(0 if success else 1)