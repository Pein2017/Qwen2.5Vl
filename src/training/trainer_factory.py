"""
Trainer Factory for BBU Training System

This module provides factory functions for creating trainers with the new
refactored architecture. It integrates the domain-specific configuration
system with the training coordinator and loss manager.

Key Features:
- Support for both legacy and new configuration systems
- Automatic trainer setup with proper component integration
- Clean model and dataset creation interfaces
- Comprehensive error handling and validation
"""

from transformers.training_args import TrainingArguments

from src.core import CheckpointManager, DataProcessor
from src.logger_utils import get_training_logger

from .trainer import BBUTrainer
from .training_coordinator import TrainingCoordinator


def create_trainer_with_coordinator(
    training_args: TrainingArguments, config=None
) -> BBUTrainer:
    """
    Create BBU trainer with new training coordinator system.

    Args:
        training_args: HuggingFace training arguments
        config: Explicit configuration object (new system)

    Returns:
        Configured BBUTrainer instance
    """
    logger = get_training_logger()
    logger.info("🏭 Creating trainer with new coordinator system...")

    # Use explicit configuration if provided, otherwise fall back to global config
    if config is None:
        from src.config import get_config

        config = get_config()
        logger.info("📄 Using global configuration system (fallback)")
    else:
        logger.info("📄 Using explicit configuration system")

    cfg = config

    # Model, tokenizer, and image processor loaded via unified loader

    # Create model using unified model loader
    logger.info("🤖 Loading model...")
    from src.models.model_loader import load_model_and_processor_unified

    model, tokenizer, image_processor = load_model_and_processor_unified(
        model_path=config.model_path,
        for_inference=False,
        attn_implementation=config.attn_implementation,
        config=config,
    )

    # Create datasets and collator using DataProcessor
    logger.info("📊 Creating datasets...")
    data_processor = DataProcessor(tokenizer, image_processor, model, config=config)
    train_dataset, eval_dataset = data_processor.create_datasets()

    # Create data collator using DataProcessor
    logger.info("📦 Creating data collator...")
    data_collator = data_processor.create_data_collator()

    # Create training coordinator
    logger.info("🎯 Creating training coordinator...")
    coordinator = TrainingCoordinator(
        model=model, tokenizer=tokenizer, config_obj=config
    )

    # Setup training
    coordinator.setup_training()

    # Create trainer with coordinator
    trainer = BBUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        cfg=cfg,
        image_processor=image_processor,
        # Pass coordinator for integration
        training_coordinator=coordinator,
    )

    logger.info("✅ Trainer created with coordinator system")

    # Log training setup summary
    status = coordinator.get_status_summary()
    logger.info("📊 Training Setup Summary:")
    logger.info(
        f"   Trainable parameters: {status['parameter_statistics']['trainable_parameters']:,}"
    )
    logger.info(f"   Enabled components: {', '.join(status['enabled_components'])}")
    logger.info(
        f"   Detection training: {'enabled' if status['training_state']['detection_training_enabled'] else 'disabled'}"
    )

    # Log any configuration warnings
    for warning in status["configuration_warnings"]:
        logger.warning(f"⚠️  {warning}")

    return trainer


def safe_save_model_for_hf_trainer(trainer: BBUTrainer, output_dir: str):
    """
    Safely save model with proper HuggingFace compatibility.

    Args:
        trainer: The trainer instance
        output_dir: Directory to save the model
    """
    # Use CheckpointManager for centralized saving logic
    checkpoint_manager = CheckpointManager()

    # Get model path from explicit config to pass to checkpoint manager
    from src.config.explicit_config import get_explicit_config

    config = get_explicit_config()
    model_path = config.model_path

    success = checkpoint_manager.save_model_safely(trainer, output_dir, model_path)
    if not success:
        raise RuntimeError(f"Failed to save model to {output_dir}")


# Convenience function for backward compatibility
def create_trainer(
    training_args: TrainingArguments, use_new_system: bool = False
) -> BBUTrainer:
    """
    Create trainer with optional new system support.

    Args:
        training_args: HuggingFace training arguments
        use_new_system: Whether to use new coordinator-based system

    Returns:
        Configured BBUTrainer instance
    """
    if use_new_system:
        return create_trainer_with_coordinator(training_args)
    else:
        raise ValueError("Legacy system is no longer supported")
