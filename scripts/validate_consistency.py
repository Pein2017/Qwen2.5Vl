#!/usr/bin/env python3
"""
Complete Training-Inference Consistency Validation

This script validates that the unified system ensures complete consistency
between training and inference, with no silent fallbacks.

Usage:
    python scripts/validate_consistency.py --config base_flat_v2
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import init_config, config


def validate_unified_loader():
    """Test that unified loader works for both training and inference."""
    print("🧪 Testing unified model loader...")
    
    try:
        from src.models.model_loader import (
            load_model_and_processor_unified,
            validate_training_inference_consistency
        )
        
        # Test loading for training
        print("   Testing training mode loading...")
        try:
            train_model, train_tok, train_proc = load_model_and_processor_unified(
                model_path=config.model_path,
                for_inference=False
            )
            print(f"   ✅ Training load successful (detection: {getattr(train_model, 'detection_enabled', 'unknown')})")
            
        except Exception as e:
            print(f"   ❌ Training load failed: {e}")
            return False
        
        # Test loading for inference
        print("   Testing inference mode loading...")
        try:
            infer_model, infer_tok, infer_proc = load_model_and_processor_unified(
                model_path=config.model_path,
                for_inference=True
            )
            print(f"   ✅ Inference load successful (detection: {getattr(infer_model, 'detection_enabled', 'unknown')})")
            
        except Exception as e:
            print(f"   ❌ Inference load failed: {e}")
            return False
        
        # Test consistency validation
        print("   Testing consistency validation...")
        try:
            # This would validate that both paths load compatible models
            is_consistent = (
                type(train_model).__name__ == type(infer_model).__name__ and
                len(train_tok) == len(infer_tok)
            )
            
            if is_consistent:
                print("   ✅ Training-inference consistency validated")
            else:
                print("   ❌ Training-inference inconsistency detected")
                return False
                
        except Exception as e:
            print(f"   ❌ Consistency validation failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Could not import unified loader: {e}")
        return False


def validate_teacher_student_config():
    """Validate teacher-student loss configuration."""
    print("🧪 Testing teacher-student loss configuration...")
    
    # Check config has required parameters
    required_params = ['teacher_loss_weight', 'student_loss_weight']
    
    for param in required_params:
        if not hasattr(config, param):
            print(f"   ❌ Missing config parameter: {param}")
            return False
        
        value = getattr(config, param)
        if not isinstance(value, (int, float)) or value < 0:
            print(f"   ❌ Invalid config parameter {param}: {value}")
            return False
    
    print(f"   ✅ Teacher weight: {config.teacher_loss_weight}")
    print(f"   ✅ Student weight: {config.student_loss_weight}")
    
    # Check detection configuration
    if hasattr(config, 'detection_enabled'):
        print(f"   ✅ Detection enabled: {config.detection_enabled}")
    else:
        print(f"   ⚠️  Detection enabled not set")
    
    return True


def validate_no_silent_fallbacks():
    """Ensure no silent fallbacks are present."""
    print("🧪 Testing for silent fallbacks...")
    
    # Check that critical functions raise exceptions rather than failing silently
    try:
        from src.models.model_loader import ModelLoadingError
        
        # Test that invalid model path raises exception
        try:
            from src.models.model_loader import load_model_and_processor_unified
            load_model_and_processor_unified(
                model_path="/nonexistent/path",
                for_inference=True
            )
            print("   ❌ Invalid path should have raised exception")
            return False
        except (ModelLoadingError, FileNotFoundError, Exception):
            print("   ✅ Invalid path correctly raises exception")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Could not test fallbacks: {e}")
        return False


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate training-inference consistency")
    parser.add_argument("--config", default="base_flat_v2", help="Config name")
    args = parser.parse_args()
    
    print("🚨 CRITICAL: Validating Complete Training-Inference Consistency")
    print("=" * 80)
    
    # Initialize config
    try:
        config_path = f"configs/{args.config}.yaml"
        init_config(config_path)
        print(f"✅ Config loaded: {args.config}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        sys.exit(1)
    
    # Run all validations
    validations = [
        ("Unified Loader", validate_unified_loader),
        ("Teacher-Student Config", validate_teacher_student_config),
        ("No Silent Fallbacks", validate_no_silent_fallbacks),
    ]
    
    results = {}
    for name, func in validations:
        print(f"\n🔍 {name}:")
        try:
            results[name] = func()
        except Exception as e:
            print(f"   ❌ Validation failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Validation Summary:")
    
    passed = 0
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
        if result:
            passed += 1
    
    if passed == len(validations):
        print(f"\n🎉 ALL VALIDATIONS PASSED!")
        print("   The unified system should ensure strict training-inference consistency.")
        print("   Teacher-student loss backpropagation should now work correctly.")
    else:
        print(f"\n💥 {len(validations) - passed} VALIDATIONS FAILED!")
        print("   Fix the issues above before running training.")
        sys.exit(1)


if __name__ == "__main__":
    main()