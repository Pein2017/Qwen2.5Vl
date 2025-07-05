#!/usr/bin/env python3
"""
Teacher Ratio Validation Script

This script validates that the teacher_ratio configuration is working correctly
and provides detailed analysis of teacher assignment patterns.

Usage:
    python scripts/validate_teacher_ratio.py --config base_flat_v2 --samples 1000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import init_config, config
from src.chat_processor import ChatProcessor
from src.data import BBUDataset
from src.teacher_pool import create_teacher_pool_manager


def test_teacher_ratio_distribution(num_samples: int = 1000):
    """Test that teacher assignment follows configured ratio."""
    print(f"🧪 Testing teacher ratio distribution over {num_samples} samples...")
    
    # Initialize config
    init_config("configs/base_flat_v2.yaml")
    
    # Create chat processor
    from transformers import AutoTokenizer, AutoProcessor
    
    try:
        processor = AutoProcessor.from_pretrained(config.model_path)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = 'left'  # Flash Attention compatibility
        
        chat_processor = ChatProcessor(
            tokenizer=tokenizer,
            image_processor=processor.image_processor,
            merge_size=getattr(config, 'merge_size', 324),
            max_length=getattr(config, 'max_total_length', 12000),
            use_training_prompts=True,
            language="chinese",
        )
        
    except Exception as e:
        print(f"❌ Failed to create chat processor: {e}")
        return False
    
    # Create teacher pool manager
    try:
        teacher_pool_manager = create_teacher_pool_manager()
        print(f"✅ Teacher pool loaded with {len(teacher_pool_manager)} teachers")
    except Exception as e:
        print(f"❌ Failed to create teacher pool: {e}")
        return False
    
    # Create training dataset
    try:
        train_dataset = BBUDataset(
            data_path=getattr(config, 'train_data_path', 'data/train.jsonl'),
            chat_processor=chat_processor,
            teacher_pool_manager=teacher_pool_manager,
            teacher_ratio=getattr(config, 'teacher_ratio', 0.7),
            is_training=True,
        )
        
        print(f"✅ Dataset created with {len(train_dataset)} samples")
        print(f"   Configured teacher ratio: {train_dataset.teacher_ratio}")
        
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False
    
    # Process samples to trigger teacher assignment
    print(f"🔄 Processing {min(num_samples, len(train_dataset))} samples...")
    
    processed_samples = 0
    for i in range(min(num_samples, len(train_dataset))):
        try:
            # This triggers teacher assignment in _sample_teachers_for_student
            sample = train_dataset[i]
            processed_samples += 1
            
            # Log progress
            if (i + 1) % 200 == 0:
                print(f"   Processed {i + 1} samples...")
                
        except Exception as e:
            print(f"⚠️  Error processing sample {i}: {e}")
            continue
    
    # Get final statistics
    summary = train_dataset.get_teacher_assignment_summary()
    
    print("\n" + "="*70)
    print("📊 Teacher Assignment Analysis Results:")
    print("="*70)
    print(f"✅ Total samples processed: {summary['total_samples_processed']}")
    print(f"🎯 Samples with teacher: {summary['samples_with_teacher']}")
    print(f"🎯 Samples without teacher: {summary['samples_without_teacher']}")
    print(f"📈 Actual teacher ratio: {summary['actual_teacher_ratio']:.3f}")
    print(f"⚙️  Configured teacher ratio: {summary['configured_teacher_ratio']:.3f}")
    print(f"📏 Ratio accuracy (deviation): {summary['ratio_accuracy']:.3f}")
    
    # Validate results
    expected_ratio = summary['configured_teacher_ratio']
    actual_ratio = summary['actual_teacher_ratio']
    deviation = abs(actual_ratio - expected_ratio)
    
    # Allow 5% deviation for statistical randomness
    tolerance = 0.05
    
    if deviation <= tolerance:
        print(f"✅ PASS: Teacher ratio within acceptable tolerance ({deviation:.3f} <= {tolerance})")
        success = True
    else:
        print(f"❌ FAIL: Teacher ratio deviation too high ({deviation:.3f} > {tolerance})")
        success = False
    
    # Additional validation
    if summary['total_samples_processed'] != processed_samples:
        print(f"⚠️  Warning: Sample count mismatch ({summary['total_samples_processed']} vs {processed_samples})")
    
    return success


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate teacher ratio implementation")
    parser.add_argument("--config", default="base_flat_v2", help="Config name")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to test")
    args = parser.parse_args()
    
    print("🚨 VALIDATION: Teacher Ratio Implementation")
    print("="*70)
    
    success = test_teacher_ratio_distribution(args.samples)
    
    print("\n" + "="*70)
    if success:
        print("🎉 VALIDATION PASSED: Teacher ratio is working correctly!")
        print("   The configured ratio is being respected in practice.")
        print("   Training logs will show teacher effectiveness analysis.")
    else:
        print("💥 VALIDATION FAILED: Teacher ratio implementation has issues!")
        print("   Check the configuration and teacher assignment logic.")
        sys.exit(1)


if __name__ == "__main__":
    main()