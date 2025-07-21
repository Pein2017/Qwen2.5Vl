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

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.training_args import TrainingArguments

from src.config import get_config
from src.core import CheckpointManager, DataProcessor
from src.logger_utils import get_training_logger
from src.models.wrapper import Qwen25VLWithDetection

from .trainer import BBUTrainer
from .training_coordinator import TrainingCoordinator


def create_trainer_with_coordinator(training_args: TrainingArguments) -> BBUTrainer:
    """
    Create BBU trainer with new training coordinator system.

    Args:
        training_args: HuggingFace training arguments

    Returns:
        Configured BBUTrainer instance
    """
    logger = get_training_logger()
    logger.info("🏭 Creating trainer with new coordinator system...")

    # Use unified configuration
    config = get_config()
    cfg = config
    logger.info("📄 Using unified configuration system")

    # Create tokenizer and processor first
    logger.info("🔤 Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    image_processor = Qwen2VLImageProcessor.from_pretrained(config.model_path)

    # Create coordinate config from global config
    from src.models.wrapper import CoordinateConfig

    coordinate_config = CoordinateConfig(
        enable_coordinate_tokens=config.coordinate_tokens_enabled,
        max_coord_value=getattr(config, "coordinate_config_max_coord_value", 2048),
        coord_token_init_std=getattr(
            config, "coordinate_config_coord_token_init_std", 0.01
        ),
        coordinate_loss_weight=getattr(
            config, "coordinate_config_coordinate_loss_weight", 1.0
        ),
        regular_loss_weight=getattr(
            config, "coordinate_config_regular_loss_weight", 1.0
        ),
        soft_expectation_temperature=getattr(
            config, "coordinate_config_soft_expectation_temperature", 1.0
        ),
        focal_loss_alpha=getattr(config, "coordinate_config_focal_loss_alpha", 0.25),
        focal_loss_gamma=getattr(config, "coordinate_config_focal_loss_gamma", 2.0),
    )

    # Create model with tokenizer and coordinate config
    logger.info("🤖 Loading model...")
    model = Qwen25VLWithDetection.from_pretrained(
        config.model_path,
        tokenizer=tokenizer,
        coordinate_config=coordinate_config,
        attn_implementation=config.attn_implementation,
        torch_dtype=getattr(torch, config.torch_dtype),
    )

    # Create datasets and collator using DataProcessor
    logger.info("📊 Creating datasets...")
    data_processor = DataProcessor(tokenizer, image_processor)
    train_dataset, eval_dataset = data_processor.create_datasets()

    # Create data collator using DataProcessor
    logger.info("📦 Creating data collator...")
    data_collator = data_processor.create_data_collator()

    # Create training coordinator
    logger.info("🎯 Creating training coordinator...")
    coordinator = TrainingCoordinator(model=model, tokenizer=tokenizer)

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
    
    # Get model path from config to pass to checkpoint manager
    config = get_config()
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
