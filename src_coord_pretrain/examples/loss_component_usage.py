#!/usr/bin/env python3
"""
Example usage of the enhanced loss component logging system.

This demonstrates how to:
1. Use the built-in loss components (llm_loss, unlikelihood_loss)
2. Register additional custom loss components
3. Monitor loss components during training
"""


import torch


def example_custom_loss_function(outputs, inputs):
    """Example custom loss function for demonstration.

    This could be any additional loss you want to add, such as:
    - Regularization losses
    - Auxiliary task losses
    - Custom coordinate losses
    - Contrastive losses
    """
    # Example: L2 regularization on logits
    logits = outputs.logits
    l2_loss = torch.mean(logits ** 2) * 0.001
    return l2_loss


def example_coordinate_consistency_loss(outputs, inputs):
    """Example coordinate consistency loss."""
    # This is just a placeholder - implement your actual coordinate consistency logic
    logits = outputs.logits
    # Example: penalize inconsistent coordinate predictions
    consistency_loss = torch.tensor(0.01, device=logits.device)
    return consistency_loss


def demonstrate_loss_logging():
    """Demonstrate how the enhanced loss logging works."""

    print("🔧 Enhanced Loss Component Logging System")
    print("=" * 50)

    print("\n📊 Built-in Loss Components:")
    print("- llm_loss: Standard cross-entropy loss from HuggingFace trainer")
    print("- unlikelihood_loss: Digit and coordinate suppression loss")
    print("- loss: Total combined loss")

    print("\n📈 Automatic Logging Metrics:")
    print("- Individual loss values: llm_loss, unlikelihood_loss, custom_loss")
    print("- Loss ratios: llm_loss_ratio, unlikelihood_loss_ratio")
    print("- Absolute values: llm_loss_abs, unlikelihood_loss_abs")
    print("- Summary stats: auxiliary_loss_total, auxiliary_loss_ratio")

    print("\n🔌 Extensible Framework:")
    print("- Register custom loss functions with trainer.register_loss_component()")
    print("- Automatic integration with logging system")
    print("- Error handling for custom loss functions")

    print("\n💡 Usage Example:")
    print("""
# In your training script:
trainer = PhaseATrainer(...)

# Register custom loss components
trainer.register_loss_component(
    name="l2_regularization",
    loss_function=example_custom_loss_function,
    weight=0.1
)

trainer.register_loss_component(
    name="coordinate_consistency",
    loss_function=example_coordinate_consistency_loss,
    weight=0.05
)

# Training will automatically log all components:
# - llm_loss, llm_loss_ratio, llm_loss_abs
# - unlikelihood_loss, unlikelihood_loss_ratio, unlikelihood_loss_abs
# - l2_regularization, l2_regularization_ratio, l2_regularization_abs
# - coordinate_consistency, coordinate_consistency_ratio, coordinate_consistency_abs
# - auxiliary_loss_total, auxiliary_loss_ratio
# - loss (total)
""")


def example_training_logs():
    """Show what the enhanced training logs look like."""

    print("\n📋 Example Training Log Output:")
    print("-" * 40)

    # Simulate training log
    example_logs = {
        "epoch": 1.0,
        "step": 100,
        "loss": 0.845,
        "llm_loss": 0.723,
        "llm_loss_ratio": 0.856,
        "llm_loss_abs": 0.723,
        "unlikelihood_loss": 0.089,
        "unlikelihood_loss_ratio": 0.105,
        "unlikelihood_loss_abs": 0.089,
        "l2_regularization": 0.033,
        "l2_regularization_ratio": 0.039,
        "l2_regularization_abs": 0.033,
        "auxiliary_loss_total": 0.122,
        "auxiliary_loss_ratio": 0.144,
        "learning_rate": 1e-5,
        "current_phase": "A",
        "phase_a_progress": 0.25,
    }

    for key, value in example_logs.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n🎯 Key Benefits:")
    print("- Monitor individual loss components for debugging")
    print("- Track loss balance and ratios")
    print("- Identify which components are contributing most to total loss")
    print("- Easy to add new loss components without modifying core code")
    print("- Automatic error handling for custom loss functions")


def configuration_examples():
    """Show configuration examples for different loss setups."""

    print("\n⚙️  Configuration Examples:")
    print("-" * 30)

    print("\n1. Basic Setup (Built-in losses only):")
    print("""
# coord_bootstrap.yaml
unlikelihood_enabled: true
unlikelihood_lambda_digits: 1.0
unlikelihood_lambda_coords: 1.0

# Logs: llm_loss, unlikelihood_loss, loss
""")

    print("\n2. With Custom Loss Components:")
    print("""
# In training script:
trainer.register_loss_component("regularization", reg_loss_fn, 0.1)
trainer.register_loss_component("consistency", consistency_loss_fn, 0.05)

# Logs: llm_loss, unlikelihood_loss, regularization, consistency, loss
""")

    print("\n3. Monitoring Loss Balance:")
    print("""
# Watch these metrics to ensure balanced training:
- llm_loss_ratio: Should be 0.7-0.9 (main task dominates)
- auxiliary_loss_ratio: Should be 0.1-0.3 (auxiliary tasks help but don't overwhelm)
- Individual component ratios: Check if any component is too dominant
""")


if __name__ == "__main__":
    demonstrate_loss_logging()
    example_training_logs()
    configuration_examples()

    print("\n🎉 Enhanced loss logging system is ready!")
    print("See src_coord_pretrain/training/trainer.py for implementation details.")
