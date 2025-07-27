#!/usr/bin/env python3
"""
Test script for differential learning rates implementation.

This script validates that coordinate tokens are properly detected and assigned
to the correct parameter group with the appropriate learning rate.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import init_config
from src.models.model_loader import load_model_and_processor_unified
from src.training.parameter_manager import ParameterGroupManager
from src.training.training_coordinator import TrainingCoordinator
from src.logger_utils import get_training_logger


def test_differential_learning_rates():
    """Test differential learning rates implementation."""
    logger = get_training_logger()
    logger.info("🧪 Testing differential learning rates implementation...")
    
    # Initialize configuration
    config_path = "configs/bbu_v2.yaml"
    config = init_config(config_path)
    
    logger.info(f"📄 Loaded config from {config_path}")
    logger.info(f"   coordinate_lr: {config.coordinate_lr}")
    logger.info(f"   llm_lr: {config.llm_lr}")
    logger.info(f"   coordinate_tokens_enabled: {config.coordinate_config_enable_coordinate_tokens}")
    
    # Load model and tokenizer
    logger.info("🤖 Loading model and tokenizer...")
    model, tokenizer, image_processor = load_model_and_processor_unified(
        model_path=config.model_path,
        for_inference=False,
        attn_implementation=config.attn_implementation
    )
    
    # Check model coordinate token setup
    logger.info("🔍 Checking model coordinate token setup...")
    if hasattr(model, 'coordinate_tokens_enabled'):
        logger.info(f"   coordinate_tokens_enabled: {model.coordinate_tokens_enabled}")
    if hasattr(model, 'original_vocab_size'):
        logger.info(f"   original_vocab_size: {model.original_vocab_size}")
    if hasattr(model, 'extended_vocab_size'):
        logger.info(f"   extended_vocab_size: {model.extended_vocab_size}")
    
    # Test parameter manager
    logger.info("🔧 Testing parameter manager...")
    param_manager = ParameterGroupManager(
        model=model, 
        base_weight_decay=config.weight_decay
    )
    
    # Create optimizer groups
    optimizer_groups = param_manager.create_optimizer_groups()
    
    logger.info(f"📊 Created {len(optimizer_groups)} parameter groups:")
    coordinate_group_found = False
    coordinate_params_count = 0
    
    for group in optimizer_groups:
        param_count = len(group['params'])
        logger.info(f"   - {group['name']}: {param_count} params, lr={group['lr']:.2e}")
        
        if group['name'] == 'coordinate':
            coordinate_group_found = True
            coordinate_params_count = param_count
    
    # Validate coordinate token detection
    if coordinate_group_found:
        logger.info(f"✅ Coordinate parameter group found with {coordinate_params_count} parameters")
        if coordinate_params_count > 0:
            logger.info("✅ Coordinate parameters successfully detected!")
        else:
            logger.warning("⚠️ Coordinate group found but no parameters assigned")
    else:
        logger.error("❌ Coordinate parameter group not found!")
    
    # Test training coordinator
    logger.info("🎯 Testing training coordinator...")
    coordinator = TrainingCoordinator(
        model=model,
        tokenizer=tokenizer,
        config_obj=config
    )
    
    # Validate configuration
    warnings = coordinator.validate_configuration()
    if warnings:
        logger.warning("⚠️ Configuration warnings found:")
        for warning in warnings:
            logger.warning(f"   - {warning}")
    else:
        logger.info("✅ Configuration validation passed!")
    
    # Setup training
    setup_info = coordinator.setup_training()
    
    logger.info("📊 Training setup summary:")
    logger.info(f"   Parameter groups: {len(setup_info['optimizer_groups'])}")
    logger.info(f"   Trainable parameters: {setup_info['parameter_statistics']['trainable_parameters']:,}")
    
    # Check specific coordinate token parameters
    logger.info("🔍 Checking specific coordinate token parameters...")
    coordinate_params_found = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if this would be categorized as coordinate
        category = param_manager._categorize_parameter(name)
        if category == "coordinate":
            coordinate_params_found.append((name, param.shape))
    
    if coordinate_params_found:
        logger.info(f"✅ Found {len(coordinate_params_found)} coordinate parameters:")
        for name, shape in coordinate_params_found[:5]:  # Show first 5
            logger.info(f"   - {name}: {shape}")
        if len(coordinate_params_found) > 5:
            logger.info(f"   ... and {len(coordinate_params_found) - 5} more")
    else:
        logger.error("❌ No coordinate parameters found!")
    
    # Test parameter detection methods
    logger.info("🧪 Testing parameter detection methods...")
    test_params = [
        "base_model.model.embed_tokens.weight",
        "base_model.lm_head.weight", 
        "extended_embeddings.weight",
        "extended_lm_head.weight",
        "base_model.visual.patch_embed.proj.weight",
        "base_model.model.layers.0.self_attn.q_proj.weight"
    ]
    
    for param_name in test_params:
        category = param_manager._categorize_parameter(param_name)
        logger.info(f"   {param_name} -> {category}")
    
    # Summary
    logger.info("📋 Test Summary:")
    logger.info(f"   ✅ Configuration loaded: coordinate_lr = {config.coordinate_lr}")
    logger.info(f"   ✅ Model loaded with coordinate tokens: {getattr(model, 'coordinate_tokens_enabled', False)}")
    logger.info(f"   ✅ Parameter groups created: {len(optimizer_groups)}")
    logger.info(f"   ✅ Coordinate group found: {coordinate_group_found}")
    logger.info(f"   ✅ Coordinate parameters detected: {len(coordinate_params_found)}")
    
    if coordinate_group_found and len(coordinate_params_found) > 0 and config.coordinate_lr > 0:
        logger.info("🎉 Differential learning rates implementation is working correctly!")
        return True
    else:
        logger.error("❌ Differential learning rates implementation has issues!")
        return False


def test_learning_rate_ratios():
    """Test learning rate ratios and recommendations."""
    logger = get_training_logger()
    logger.info("📊 Testing learning rate ratios...")
    
    config = init_config("configs/bbu_v2.yaml")
    
    # Calculate ratios
    base_lr = config.llm_lr
    coord_ratio = config.coordinate_lr / base_lr if base_lr > 0 else 0
    vision_ratio = config.vision_lr / base_lr if base_lr > 0 else 0
    merger_ratio = config.merger_lr / base_lr if base_lr > 0 else 0
    
    logger.info(f"Learning rate ratios (relative to llm_lr = {base_lr:.2e}):")
    logger.info(f"   coordinate_lr: {config.coordinate_lr:.2e} ({coord_ratio:.1f}x)")
    logger.info(f"   vision_lr: {config.vision_lr:.2e} ({vision_ratio:.1f}x)")
    logger.info(f"   merger_lr: {config.merger_lr:.2e} ({merger_ratio:.1f}x)")
    
    # Validate ratios
    recommendations = []
    if coord_ratio < 2:
        recommendations.append("Consider increasing coordinate_lr to 2-5x llm_lr for better spatial learning")
    if vision_ratio > 1:
        recommendations.append("Consider reducing vision_lr to preserve pretrained features")
    if merger_ratio < 5:
        recommendations.append("Consider increasing merger_lr for better vision-language fusion")
    
    if recommendations:
        logger.info("💡 Recommendations:")
        for rec in recommendations:
            logger.info(f"   - {rec}")
    else:
        logger.info("✅ Learning rate ratios look good!")


if __name__ == "__main__":
    print("🧪 Testing Differential Learning Rates Implementation")
    print("=" * 60)
    
    try:
        # Test main implementation
        success = test_differential_learning_rates()
        
        print("\n" + "=" * 60)
        
        # Test learning rate ratios
        test_learning_rate_ratios()
        
        print("\n" + "=" * 60)
        
        if success:
            print("🎉 All tests passed! Differential learning rates are working correctly.")
            sys.exit(0)
        else:
            print("❌ Some tests failed. Please check the implementation.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
