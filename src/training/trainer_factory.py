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

from typing import Optional, Any, Dict
from pathlib import Path

import torch
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLProcessor,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.config import config, get_config_manager
from src.logger_utils import get_training_logger
from src.core import ModelFactory, DataProcessor, CheckpointManager

from .trainer import BBUTrainer, create_trainer as legacy_create_trainer
from .training_coordinator import TrainingCoordinator


def create_trainer_with_coordinator(
    training_args: TrainingArguments,
    use_new_config: bool = False
) -> BBUTrainer:
    """
    Create BBU trainer with new training coordinator system.
    
    Args:
        training_args: HuggingFace training arguments
        use_new_config: Whether to use new domain-specific config system
        
    Returns:
        Configured BBUTrainer instance
    """
    logger = get_training_logger()
    logger.info("🏭 Creating trainer with new coordinator system...")
    
    # Get configuration
    if use_new_config:
        try:
            config_manager = get_config_manager()
            cfg = config_manager
            logger.info("✅ Using new domain-specific configuration system")
        except RuntimeError:
            logger.warning("⚠️  New config system not available, falling back to legacy")
            cfg = config
            use_new_config = False
    else:
        cfg = config
        logger.info("📄 Using legacy configuration system")
    
    # Create model using ModelFactory
    logger.info("🤖 Loading model...")
    model_factory = ModelFactory(use_new_config=use_new_config)
    model = model_factory.create_model()
    
    # Create tokenizer and processor using ModelFactory
    logger.info("🔤 Loading tokenizer and processor...")
    tokenizer, image_processor = model_factory.create_tokenizer_and_processor()
    
    # Create datasets and collator using DataProcessor
    logger.info("📊 Creating datasets...")
    data_processor = DataProcessor(tokenizer, image_processor, use_new_config=use_new_config)
    train_dataset, eval_dataset = data_processor.create_datasets()
    
    # Create data collator using DataProcessor
    logger.info("📦 Creating data collator...")
    data_collator = data_processor.create_data_collator()
    
    # Create training coordinator
    logger.info("🎯 Creating training coordinator...")
    coordinator = TrainingCoordinator(
        model=model,
        tokenizer=tokenizer,
        use_domain_config=use_new_config
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
        cfg=cfg if not use_new_config else None,
        image_processor=image_processor,
        # Pass coordinator for integration
        training_coordinator=coordinator
    )
    
    logger.info("✅ Trainer created with coordinator system")
    
    # Log training setup summary
    status = coordinator.get_status_summary()
    logger.info("📊 Training Setup Summary:")
    logger.info(f"   Trainable parameters: {status['parameter_statistics']['trainable_parameters']:,}")
    logger.info(f"   Enabled components: {', '.join(status['enabled_components'])}")
    logger.info(f"   Detection training: {'enabled' if status['training_state']['detection_training_enabled'] else 'disabled'}")
    
    # Log any configuration warnings
    for warning in status['configuration_warnings']:
        logger.warning(f"⚠️  {warning}")
    
    return trainer




def create_legacy_trainer(training_args: TrainingArguments) -> BBUTrainer:
    """
    Create trainer using the legacy system (for backward compatibility).
    
    Args:
        training_args: HuggingFace training arguments
        
    Returns:
        Configured BBUTrainer instance using legacy system
    """
    logger = get_training_logger()
    logger.info("🏭 Creating trainer with legacy system...")
    
    # Use the existing trainer creation function
    return legacy_create_trainer(training_args)


def safe_save_model_for_hf_trainer(trainer: BBUTrainer, output_dir: str):
    """
    Safely save model with proper HuggingFace compatibility.
    
    Args:
        trainer: The trainer instance
        output_dir: Directory to save the model
    """
    # Use CheckpointManager for centralized saving logic
    checkpoint_manager = CheckpointManager(use_new_config=False)  # Auto-detect from trainer
    
    success = checkpoint_manager.save_model_safely(trainer, output_dir)
    if not success:
        raise RuntimeError(f"Failed to save model to {output_dir}")


# Convenience function for backward compatibility
def create_trainer(training_args: TrainingArguments, use_new_system: bool = False) -> BBUTrainer:
    """
    Create trainer with optional new system support.
    
    Args:
        training_args: HuggingFace training arguments
        use_new_system: Whether to use new coordinator-based system
        
    Returns:
        Configured BBUTrainer instance
    """
    if use_new_system:
        return create_trainer_with_coordinator(training_args, use_new_config=True)
    else:
        return create_legacy_trainer(training_args)