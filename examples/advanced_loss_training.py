#!/usr/bin/env python3
"""
Example script demonstrating advanced loss functions for Qwen2.5-VL training.

This script shows how to:
1. Use WeightedCrossEntropyLoss for few-shot learning
2. Use ObjectDetectionAugmentedLoss for integrating detection losses
3. Configure different focus areas for training
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core import Config, create_trainer


def train_with_weighted_loss():
    """Example: Training with weighted loss for few-shot learning."""
    print("🎯 Training with Weighted Loss for Few-Shot Learning")
    print("=" * 60)

    config = Config()

    # Configure for weighted loss
    config.loss_type = "weighted"
    config.example_weight = 0.3  # Lower weight for examples (guidance only)
    config.main_weight = 1.0  # Full weight for main prediction
    config.vision_weight = 1.2  # Higher weight for vision tokens
    config.text_weight = 1.0  # Standard weight for text
    config.label_smoothing = 0.1  # Add robustness

    # Focus areas for few-shot learning
    config.focus_areas = {
        "bbox_precision": True,
        "spatial_relationships": True,
        "object_classification": True,
        "few_shot_transfer": True,  # Key for few-shot learning
    }

    # Training settings optimized for few-shot learning
    config.learning_rate = 5e-8  # Lower LR for stability
    config.num_train_epochs = 25  # More epochs for few-shot
    config.warmup_ratio = 0.15  # Longer warmup

    print("✅ Configuration:")
    print(f"   - Loss type: {config.loss_type}")
    print(f"   - Example weight: {config.example_weight}")
    print(f"   - Vision weight: {config.vision_weight}")
    print(f"   - Label smoothing: {config.label_smoothing}")

    # Create and train
    trainer = create_trainer(config)
    trainer.train()


def train_with_detection_augmented_loss():
    """Example: Training with detection-augmented loss."""
    print("🎯 Training with Detection-Augmented Loss")
    print("=" * 60)

    config = Config()

    # Configure for detection-augmented loss
    config.loss_type = "detection_augmented"
    config.lm_weight = 1.0  # Language modeling
    config.bbox_weight = 0.5  # Bounding box regression
    config.giou_weight = 0.3  # GIoU for better localization
    config.class_weight = 0.2  # Classification accuracy
    config.hungarian_matching = True  # Optimal assignment

    # Focus areas for object detection
    config.focus_areas = {
        "bbox_precision": True,  # Critical for detection
        "spatial_relationships": True,
        "object_classification": True,
        "few_shot_transfer": False,  # Less important for detection
    }

    # Training settings optimized for detection
    config.learning_rate = 8e-8  # Balanced LR
    config.num_train_epochs = 30  # More epochs for detection
    config.gradient_accumulation_steps = 4  # Larger effective batch

    print("✅ Configuration:")
    print(f"   - Loss type: {config.loss_type}")
    print(f"   - LM weight: {config.lm_weight}")
    print(f"   - BBox weight: {config.bbox_weight}")
    print(f"   - GIoU weight: {config.giou_weight}")
    print(f"   - Hungarian matching: {config.hungarian_matching}")

    # Create and train
    trainer = create_trainer(config)
    trainer.train()


def train_with_progressive_loss():
    """Example: Progressive training with changing loss weights."""
    print("🎯 Progressive Training with Adaptive Loss Weights")
    print("=" * 60)

    # Stage 1: Focus on few-shot learning
    print("\n📚 Stage 1: Few-Shot Learning Focus")
    config_stage1 = Config()
    config_stage1.loss_type = "weighted"
    config_stage1.example_weight = 0.5  # Higher weight for examples initially
    config_stage1.vision_weight = 1.5  # Strong vision focus
    config_stage1.num_train_epochs = 10
    config_stage1.output_dir = "output/stage1_few_shot"

    trainer_stage1 = create_trainer(config_stage1)
    trainer_stage1.train()

    # Stage 2: Transition to detection focus
    print("\n🎯 Stage 2: Detection Integration")
    config_stage2 = Config()
    config_stage2.loss_type = "detection_augmented"
    config_stage2.model_path = "output/stage1_few_shot"  # Load from stage 1
    config_stage2.bbox_weight = 0.7  # Higher detection weight
    config_stage2.giou_weight = 0.5
    config_stage2.num_train_epochs = 15
    config_stage2.output_dir = "output/stage2_detection"

    trainer_stage2 = create_trainer(config_stage2)
    trainer_stage2.train()

    # Stage 3: Fine-tuning with balanced weights
    print("\n⚖️  Stage 3: Balanced Fine-tuning")
    config_stage3 = Config()
    config_stage3.loss_type = "detection_augmented"
    config_stage3.model_path = "output/stage2_detection"  # Load from stage 2
    config_stage3.lm_weight = 1.0
    config_stage3.bbox_weight = 0.4  # Balanced weights
    config_stage3.giou_weight = 0.3
    config_stage3.class_weight = 0.3
    config_stage3.learning_rate = 2e-8  # Lower LR for fine-tuning
    config_stage3.num_train_epochs = 10
    config_stage3.output_dir = "output/stage3_balanced"

    trainer_stage3 = create_trainer(config_stage3)
    trainer_stage3.train()


def analyze_loss_components():
    """Example: Analyzing which parts should be focused more."""
    print("🔍 Loss Component Analysis")
    print("=" * 60)

    # Different configurations to test
    configs = {
        "bbox_focused": {
            "bbox_weight": 0.8,
            "giou_weight": 0.6,
            "class_weight": 0.2,
            "description": "Focus on precise bounding box localization",
        },
        "classification_focused": {
            "bbox_weight": 0.3,
            "giou_weight": 0.2,
            "class_weight": 0.8,
            "description": "Focus on object type recognition",
        },
        "spatial_focused": {
            "bbox_weight": 0.5,
            "giou_weight": 0.8,
            "class_weight": 0.4,
            "description": "Focus on spatial relationships and IoU",
        },
        "balanced": {
            "bbox_weight": 0.5,
            "giou_weight": 0.3,
            "class_weight": 0.3,
            "description": "Balanced approach",
        },
    }

    print("📊 Recommended configurations based on your data characteristics:")
    print()

    for name, cfg in configs.items():
        print(f"🎯 {name.upper()}:")
        print(f"   Description: {cfg['description']}")
        print(f"   BBox weight: {cfg['bbox_weight']}")
        print(f"   GIoU weight: {cfg['giou_weight']}")
        print(f"   Class weight: {cfg['class_weight']}")
        print("   Use when: ", end="")

        if name == "bbox_focused":
            print("Your data has precise annotations but varied object types")
        elif name == "classification_focused":
            print("Your bounding boxes are approximate but object types are critical")
        elif name == "spatial_focused":
            print("Spatial relationships and overlap detection are most important")
        else:
            print("You want overall good performance across all aspects")
        print()


if __name__ == "__main__":
    print("🚀 Qwen2.5-VL Advanced Loss Functions Examples")
    print("=" * 60)

    # Show analysis first
    analyze_loss_components()

    # Ask user which example to run
    print("Choose an example to run:")
    print("1. Weighted Loss for Few-Shot Learning")
    print("2. Detection-Augmented Loss")
    print("3. Progressive Training (3 stages)")
    print("4. Just show analysis (default)")

    choice = input("\nEnter choice (1-4, default=4): ").strip()

    if choice == "1":
        train_with_weighted_loss()
    elif choice == "2":
        train_with_detection_augmented_loss()
    elif choice == "3":
        train_with_progressive_loss()
    else:
        print(
            "✅ Analysis complete. Modify the configurations above for your specific needs."
        )
        print("\n💡 Quick Start Tips:")
        print("   - For few-shot learning: Use weighted loss with example_weight=0.3")
        print(
            "   - For precise detection: Use detection_augmented with high bbox_weight"
        )
        print(
            "   - For limited data: Start with weighted loss, then transition to detection"
        )
        print("   - For production: Use progressive training for best results")
