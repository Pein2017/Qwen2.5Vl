#!/usr/bin/env python3
"""
Example: Differential Learning Rates for Qwen2.5-VL Fine-tuning

This example demonstrates how to configure different learning rates for different modules:
- Vision encoder (lower LR for pre-trained features)
- Vision-language connector (higher LR for adaptation)
- Language model (standard LR for text understanding)

Usage:
    python examples/differential_learning_rates.py
"""

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Config


def example_full_finetuning():
    """Example: Full fine-tuning with single learning rate (current default)."""
    print("🔧 Example 1: Full Fine-tuning (Single LR)")

    config = Config()

    # Configure modules to fine-tune
    config.tune_vision = True  # Fine-tune vision encoder
    config.tune_mlp = True  # Fine-tune vision-language connector
    config.tune_llm = True  # Fine-tune language model

    # Single learning rate for all modules
    config.use_differential_lr = False
    config.learning_rate = 2e-7

    # Training settings
    config.num_train_epochs = 5
    config.output_dir = "output/full_finetuning"

    print(f"   - Vision encoder: {'✅' if config.tune_vision else '❌'}")
    print(f"   - Vision-language connector: {'✅' if config.tune_mlp else '❌'}")
    print(f"   - Language model: {'✅' if config.tune_llm else '❌'}")
    print(f"   - Learning rate: {config.learning_rate}")

    return config


def example_differential_learning_rates():
    """Example: Differential learning rates for different modules."""
    print("\n🎯 Example 2: Differential Learning Rates")

    config = Config()

    # Configure modules to fine-tune
    config.tune_vision = True
    config.tune_mlp = True
    config.tune_llm = True

    # Enable differential learning rates
    config.use_differential_lr = True
    config.vision_lr = 1e-7  # Lower LR for pre-trained vision encoder
    config.mlp_lr = 5e-7  # Higher LR for connector (needs more adaptation)
    config.llm_lr = 2e-7  # Standard LR for language model

    # Training settings
    config.num_train_epochs = 5
    config.output_dir = "output/differential_lr"

    print(
        f"   - Vision encoder: {'✅' if config.tune_vision else '❌'} (LR: {config.vision_lr})"
    )
    print(
        f"   - Vision-language connector: {'✅' if config.tune_mlp else '❌'} (LR: {config.mlp_lr})"
    )
    print(
        f"   - Language model: {'✅' if config.tune_llm else '❌'} (LR: {config.llm_lr})"
    )

    return config


def example_connector_only():
    """Example: Fine-tune only the vision-language connector (most efficient)."""
    print("\n⚡ Example 3: Connector-Only Fine-tuning")

    config = Config()

    # Only fine-tune the connector
    config.tune_vision = False  # Freeze vision encoder
    config.tune_mlp = True  # Fine-tune connector only
    config.tune_llm = False  # Freeze language model

    # Higher learning rate since only connector is being trained
    config.use_differential_lr = False
    config.learning_rate = 1e-6  # Higher LR for focused training

    # Training settings
    config.num_train_epochs = 10  # More epochs since fewer parameters
    config.output_dir = "output/connector_only"

    print(f"   - Vision encoder: {'✅' if config.tune_vision else '❌'}")
    print(f"   - Vision-language connector: {'✅' if config.tune_mlp else '❌'}")
    print(f"   - Language model: {'✅' if config.tune_llm else '❌'}")
    print(f"   - Learning rate: {config.learning_rate}")
    print("   - Strategy: Efficient training with minimal parameters")

    return config


def example_vision_focused():
    """Example: Vision-focused fine-tuning (vision + connector)."""
    print("\n👁️ Example 4: Vision-Focused Fine-tuning")

    config = Config()

    # Fine-tune vision components only
    config.tune_vision = True  # Fine-tune vision encoder
    config.tune_mlp = True  # Fine-tune connector
    config.tune_llm = False  # Freeze language model

    # Differential learning rates for vision components
    config.use_differential_lr = True
    config.vision_lr = 5e-8  # Very low LR for vision encoder
    config.mlp_lr = 1e-6  # Higher LR for connector
    config.llm_lr = 0  # Not used (LLM frozen)

    # Training settings
    config.num_train_epochs = 8
    config.output_dir = "output/vision_focused"

    print(
        f"   - Vision encoder: {'✅' if config.tune_vision else '❌'} (LR: {config.vision_lr})"
    )
    print(
        f"   - Vision-language connector: {'✅' if config.tune_mlp else '❌'} (LR: {config.mlp_lr})"
    )
    print(f"   - Language model: {'✅' if config.tune_llm else '❌'}")
    print("   - Strategy: Focus on improving visual understanding")

    return config


def example_language_focused():
    """Example: Language-focused fine-tuning (language + connector)."""
    print("\n📝 Example 5: Language-Focused Fine-tuning")

    config = Config()

    # Fine-tune language components
    config.tune_vision = False  # Freeze vision encoder
    config.tune_mlp = True  # Fine-tune connector
    config.tune_llm = True  # Fine-tune language model

    # Differential learning rates for language components
    config.use_differential_lr = True
    config.vision_lr = 0  # Not used (vision frozen)
    config.mlp_lr = 5e-7  # Standard LR for connector
    config.llm_lr = 2e-7  # Standard LR for language model

    # Training settings
    config.num_train_epochs = 6
    config.output_dir = "output/language_focused"

    print(f"   - Vision encoder: {'✅' if config.tune_vision else '❌'}")
    print(
        f"   - Vision-language connector: {'✅' if config.tune_mlp else '❌'} (LR: {config.mlp_lr})"
    )
    print(
        f"   - Language model: {'✅' if config.tune_llm else '❌'} (LR: {config.llm_lr})"
    )
    print("   - Strategy: Focus on improving text generation and reasoning")

    return config


def example_ratio_based_learning_rates():
    """Example: Ratio-based learning rates (vision:language:aligner = 1:2:5)."""
    print("\n🎯 Example 6: Ratio-Based Learning Rates (1:2:5)")

    config = Config()

    # Configure modules to fine-tune (tune all modules)
    config.tune_vision = True  # Fine-tune vision encoder
    config.tune_mlp = True  # Fine-tune vision-language connector (aligner)
    config.tune_llm = True  # Fine-tune language model

    # Enable ratio-based learning rates
    config.use_lr_ratios = True
    config.base_lr = 5e-7  # Base learning rate (equals aligner LR)
    config.vision_ratio = 1  # Vision encoder ratio
    config.llm_ratio = 2  # Language model ratio
    config.mlp_ratio = 5  # Aligner/connector ratio (base)

    # Training settings
    config.num_train_epochs = 5
    config.output_dir = "output/ratio_based_lr"

    # Calculate actual learning rates for display
    actual_vision_lr = config.base_lr * (config.vision_ratio / config.mlp_ratio)
    actual_llm_lr = config.base_lr * (config.llm_ratio / config.mlp_ratio)
    actual_mlp_lr = config.base_lr

    print(f"   - Base LR (aligner): {config.base_lr}")
    print(
        f"   - Ratios: Vision:{config.vision_ratio}, Language:{config.llm_ratio}, Aligner:{config.mlp_ratio}"
    )
    print(f"   - Vision encoder: ✅ (LR: {actual_vision_lr:.2e})")
    print(f"   - Vision-language connector: ✅ (LR: {actual_mlp_lr:.2e})")
    print(f"   - Language model: ✅ (LR: {actual_llm_lr:.2e})")
    print("   - Strategy: Balanced training with proportional learning rates")

    return config


def main():
    """Run all examples and show configurations."""
    print("🚀 Qwen2.5-VL Differential Learning Rate Examples\n")

    examples = [
        example_full_finetuning,
        example_differential_learning_rates,
        example_connector_only,
        example_vision_focused,
        example_language_focused,
        example_ratio_based_learning_rates,
    ]

    configs = []
    for example_func in examples:
        config = example_func()
        configs.append(config)

    print("\n" + "=" * 60)
    print("📊 RECOMMENDATIONS")
    print("=" * 60)

    print("\n🎯 For Telecommunications Quality Inspection:")
    print("   1. Start with Connector-Only (Example 3) for quick iteration")
    print("   2. Use Vision-Focused (Example 4) if visual understanding is poor")
    print("   3. Use Language-Focused (Example 5) if text generation needs improvement")
    print("   4. Use Differential LR (Example 2) for best overall performance")
    print("   5. Use Full Fine-tuning (Example 1) only with large datasets")
    print(
        "   6. ⭐ Use Ratio-Based LR (Example 6) for balanced training with 1:2:5 ratio"
    )

    print("\n⚡ Learning Rate Guidelines:")
    print("   - Vision Encoder: 1e-8 to 1e-7 (pre-trained, needs gentle updates)")
    print("   - Connector: 1e-7 to 1e-6 (needs adaptation, higher LR)")
    print("   - Language Model: 1e-7 to 5e-7 (standard fine-tuning range)")
    print(
        "   - Ratio-Based: Set base_lr = aligner LR, ratios = vision:language:aligner = 1:2:5"
    )

    print("\n💾 Memory Usage (approximate):")
    print("   - Connector-Only: ~4-6GB (3B model), ~8-12GB (7B model)")
    print("   - Vision-Focused: ~6-10GB (3B model), ~12-18GB (7B model)")
    print("   - Language-Focused: ~8-12GB (3B model), ~16-24GB (7B model)")
    print("   - Full Fine-tuning: ~10-16GB (3B model), ~20-32GB (7B model)")
    print("   - Ratio-Based (1:2:5): ~10-16GB (3B model), ~20-32GB (7B model)")

    print("\n🔧 To use these configurations:")
    print("   1. Copy the desired example configuration")
    print("   2. Modify paths and hyperparameters as needed")
    print("   3. Run training with: trainer = create_trainer(config); trainer.train()")

    # Example of how to actually run training
    print("\n" + "=" * 60)
    print("🏃 EXAMPLE TRAINING CODE")
    print("=" * 60)

    print("""
# Example 1: Run connector-only fine-tuning
from src.core import Config, create_trainer

config = Config()
config.tune_vision = False
config.tune_mlp = True
config.tune_llm = False
config.learning_rate = 1e-6
config.num_train_epochs = 10
config.output_dir = "output/connector_only"

# Create and run trainer
trainer = create_trainer(config)
trainer.train()

# Example 2: Run ratio-based learning rates (1:2:5)
config = Config()
config.tune_vision = True
config.tune_mlp = True
config.tune_llm = True
config.use_lr_ratios = True
config.base_lr = 5e-7  # Base LR equals aligner LR
config.vision_ratio = 1
config.llm_ratio = 2
config.mlp_ratio = 5
config.output_dir = "output/ratio_based_lr"

# This will result in:
# - Vision LR: 1e-7 (5e-7 * 1/5)
# - Language LR: 2e-7 (5e-7 * 2/5)
# - Aligner LR: 5e-7 (5e-7 * 5/5)

trainer = create_trainer(config)
trainer.train()
""")


if __name__ == "__main__":
    main()
