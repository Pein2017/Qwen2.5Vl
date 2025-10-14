#!/usr/bin/env python3
"""
Ratio-Based Learning Rate Training for Qwen2.5-VL

This script demonstrates how to train with ratio-based learning rates
where vision:language:aligner = 1:2:5 and base_lr equals aligner LR.

Usage:
    python examples/ratio_based_training.py
"""

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Config, create_trainer


def main():
    """Train with ratio-based learning rates (1:2:5)."""
    print("🚀 Starting Ratio-Based Learning Rate Training")
    print("   Vision:Language:Aligner = 1:2:5")

    # Create configuration
    config = Config()

    # Enable all modules for fine-tuning
    config.tune_vision = True  # Fine-tune vision encoder
    config.tune_mlp = True  # Fine-tune vision-language connector (aligner)
    config.tune_llm = True  # Fine-tune language model

    # Configure ratio-based learning rates
    config.use_lr_ratios = True
    config.base_lr = 5e-7  # Base learning rate (equals aligner LR)
    config.vision_ratio = 1  # Vision encoder ratio
    config.llm_ratio = 2  # Language model ratio
    config.mlp_ratio = 5  # Aligner/connector ratio (base)

    # Training settings
    config.num_train_epochs = 30
    config.per_device_train_batch_size = 1
    config.gradient_accumulation_steps = 8  # Effective batch size = 8
    config.output_dir = "output/ratio_based_1_2_5"
    config.run_name = "ratio_based_1_2_5"

    # Calculate and display actual learning rates
    actual_vision_lr = config.base_lr * (config.vision_ratio / config.mlp_ratio)
    actual_llm_lr = config.base_lr * (config.llm_ratio / config.mlp_ratio)
    actual_mlp_lr = config.base_lr

    print("\n📊 Configuration:")
    print(f"   - Base LR (aligner): {config.base_lr}")
    print(
        f"   - Ratios: Vision:{config.vision_ratio}, Language:{config.llm_ratio}, Aligner:{config.mlp_ratio}"
    )
    print("   - Actual Learning Rates:")
    print(f"     • Vision encoder: {actual_vision_lr:.2e}")
    print(f"     • Language model: {actual_llm_lr:.2e}")
    print(f"     • Aligner/connector: {actual_mlp_lr:.2e}")
    print(f"   - Epochs: {config.num_train_epochs}")
    print(f"   - Effective batch size: {config.gradient_accumulation_steps}")
    print(f"   - Output directory: {config.output_dir}")

    # Create and run trainer
    print("\n🏃 Creating trainer...")
    trainer = create_trainer(config)

    print("🚀 Starting training...")
    trainer.train()

    print("✅ Training completed!")
    print(f"📁 Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
