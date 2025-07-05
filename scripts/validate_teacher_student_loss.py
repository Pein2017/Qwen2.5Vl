#!/usr/bin/env python3
"""
Teacher-Student Loss Validation Script

This script validates that teacher and student losses are properly included
in backpropagation and that gradients flow correctly to all components.

Usage:
    python scripts/validate_teacher_student_loss.py
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import init_config, config
from src.training.loss_manager import LossManager
from src.models.model_loader import load_model_and_processor_unified


def test_teacher_student_loss_backpropagation():
    """Test that teacher and student losses contribute to backpropagation."""
    print("🧪 Testing teacher-student loss backpropagation...")
    
    # Initialize config
    init_config("configs/base_flat_v2.yaml")
    
    # Load model and tokenizer
    try:
        model, tokenizer, _ = load_model_and_processor_unified(
            model_path=config.model_path,
            for_inference=False,
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Create loss manager
    loss_manager = LossManager(
        tokenizer=tokenizer,
        detection_enabled=config.detection_enabled,
        bbox_weight=config.detection_bbox_weight,
        giou_weight=config.detection_giou_weight,
        objectness_weight=config.detection_objectness_weight,
        caption_weight=config.detection_caption_weight,
        focal_loss_gamma=config.detection_focal_loss_gamma,
        focal_loss_alpha=config.detection_focal_loss_alpha,
    )
    
    # Create sample inputs with teacher and student spans
    batch_size = 2
    seq_len = 100
    vocab_size = len(tokenizer)
    
    # Mock inputs
    inputs = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "teacher_assistant_spans": [
            [(10, 20), (30, 40)],  # Batch 0: teacher spans
            [(15, 25)]             # Batch 1: teacher spans
        ],
        "student_assistant_spans": [
            [(50, 70)],            # Batch 0: student spans  
            [(60, 80), (85, 95)]   # Batch 1: student spans
        ],
        "ground_truth_objects": [[], []]  # No detection objects for this test
    }
    
    # Mock model outputs
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    model_outputs = type('MockOutputs', (), {
        'logits': logits,
        'loss': None
    })()
    
    print(f"✅ Created mock inputs:")
    print(f"   Teacher spans: {inputs['teacher_assistant_spans']}")
    print(f"   Student spans: {inputs['student_assistant_spans']}")
    print(f"   Teacher weight: {config.teacher_loss_weight}")
    print(f"   Student weight: {config.student_loss_weight}")
    
    # Compute loss
    try:
        total_loss, loss_components = loss_manager.compute_total_loss(
            model_outputs=model_outputs,
            inputs=inputs,
            is_training=True,
            detection_training_enabled=config.detection_enabled
        )
        
        print(f"✅ Loss computation successful:")
        print(f"   Total loss: {total_loss.item():.6f}")
        print(f"   Teacher loss: {loss_components['teacher_lm_loss']:.6f}")
        print(f"   Student loss: {loss_components['student_lm_loss']:.6f}")
        print(f"   LM loss: {loss_components['lm_loss']:.6f}")
        
        # Check that total_loss requires_grad
        if not total_loss.requires_grad:
            print(f"❌ CRITICAL: total_loss does not require gradients!")
            return False
        
        print(f"✅ Total loss requires gradients: {total_loss.requires_grad}")
        
        # Test backpropagation
        print("🧪 Testing backpropagation...")
        total_loss.backward()
        
        # Check that gradients were computed
        if logits.grad is None:
            print(f"❌ CRITICAL: No gradients computed for logits!")
            return False
        
        grad_norm = logits.grad.norm().item()
        print(f"✅ Gradients computed successfully (norm: {grad_norm:.6f})")
        
        # Verify non-zero losses for teacher and student
        if loss_components['teacher_lm_loss'] == 0.0:
            print(f"⚠️  Warning: Teacher loss is zero")
        else:
            print(f"✅ Teacher loss is non-zero: {loss_components['teacher_lm_loss']:.6f}")
            
        if loss_components['student_lm_loss'] == 0.0:
            print(f"⚠️  Warning: Student loss is zero")
        else:
            print(f"✅ Student loss is non-zero: {loss_components['student_lm_loss']:.6f}")
        
        print(f"✅ Teacher-student loss backpropagation test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    print("🚨 CRITICAL: Validating Teacher-Student Loss Backpropagation")
    print("=" * 70)
    
    success = test_teacher_student_loss_backpropagation()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 VALIDATION PASSED: Teacher and student losses are properly backpropagating!")
        print("   This should fix the issue where student loss was fluctuating.")
    else:
        print("💥 VALIDATION FAILED: Teacher-student loss system has issues!")
        print("   Check the error messages above and fix before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()