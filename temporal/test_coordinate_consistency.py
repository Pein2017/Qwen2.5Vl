#!/usr/bin/env python3
"""
Test Coordinate System Consistency

This script validates that coordinate token configurations are consistent
across all components of the Qwen2.5-VL training system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import yaml
from src.config.coordinate_validator import CoordinateConfigValidator
from src.config.config_manager import ConfigManager
from src.utils.coordinate_token_manager import CoordinateTokenManager, CoordinateTokenConfig
from src.logger_utils import get_logger


def test_coordinate_config_consistency():
    """Test coordinate configuration consistency across all components."""
    logger = get_logger("coordinate_consistency_test")
    
    # 1. Load configuration from YAML
    config_path = Path(__file__).parent.parent / "configs" / "base_flat_det.yaml"
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # 2. Create runtime config using the traditional config loading
    from src.config import init_config
    init_config(str(config_path))
    from src.config import config
    runtime_config = config
    
    # 3. Extract coordinate configurations
    coordinate_configs = {}
    
    # From YAML file
    yaml_coord_config = {}
    for key, value in yaml_config.items():
        if key.startswith('coordinate_config_'):
            yaml_coord_config[key] = value
    
    # From runtime config - access underlying DirectConfig
    runtime_coord_config = {}
    print(f"Runtime config type: {type(runtime_config)}")
    
    # Get the actual underlying config
    from src.config.global_config import config as direct_config
    if direct_config is not None:
        for key in yaml_coord_config.keys():
            if hasattr(direct_config, key):
                runtime_coord_config[key] = getattr(direct_config, key)
    
    print(f"Runtime coordinate config keys: {list(runtime_coord_config.keys())}")
    
    coordinate_configs['yaml'] = yaml_coord_config
    coordinate_configs['runtime'] = runtime_coord_config
    
    # 4. Validate using coordinator validator
    validator = CoordinateConfigValidator()
    result = validator.validate_config_consistency(coordinate_configs)
    
    # 5. Generate report
    report = validator.generate_validation_report(result)
    print(report)
    
    # 6. Test coordinate token manager creation
    if yaml_coord_config.get('coordinate_config_enable_coordinate_tokens', False):
        print("\n🔧 Testing coordinate token manager creation...")
        
        try:
            from transformers import AutoTokenizer
            
            # Load tokenizer
            model_path = runtime_config.model_path
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Create coordinate token manager
            coord_config = CoordinateTokenConfig(
                enable_coordinate_tokens=yaml_coord_config['coordinate_config_enable_coordinate_tokens'],
                max_coord_value=yaml_coord_config['coordinate_config_max_coord_value'],
                box_start_id=yaml_coord_config.get('coordinate_config_box_start_id', 151648),
                box_end_id=yaml_coord_config.get('coordinate_config_box_end_id', 151649),
                coordinate_loss_weight=yaml_coord_config['coordinate_config_coordinate_loss_weight'],
                regular_loss_weight=yaml_coord_config['coordinate_config_regular_loss_weight'],
                soft_expectation_temperature=yaml_coord_config['coordinate_config_soft_expectation_temperature'],
                focal_loss_alpha=yaml_coord_config['coordinate_config_focal_loss_alpha'],
                focal_loss_gamma=yaml_coord_config['coordinate_config_focal_loss_gamma'],
                enable_validation=yaml_coord_config.get('coordinate_config_enable_validation', True),
                enable_caching=yaml_coord_config.get('coordinate_config_enable_caching', True),
                batch_processing=yaml_coord_config.get('coordinate_config_batch_processing', True),
            )
            
            coord_manager = CoordinateTokenManager(
                tokenizer=tokenizer,
                config=coord_config,
                original_vocab_size=len(tokenizer)
            )
            
            print("✅ Coordinate token manager created successfully")
            print(f"   Extended vocab size: {coord_manager.extended_vocab_size}")
            print(f"   Coordinate token range: [{coord_manager.coord_start_id}, {coord_manager.coord_end_id})")
            
            # Test format conversion
            test_json = '[{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "test_object"}]'
            coord_format = coord_manager.convert_json_to_coordinate_format(test_json)
            print(f"   JSON to coordinate format: {coord_format}")
            
            back_to_json = coord_manager.convert_coordinate_to_json_format(coord_format)
            print(f"   Back to JSON: {back_to_json}")
            
        except Exception as e:
            print(f"❌ Error creating coordinate token manager: {e}")
            result.is_valid = False
    
    # 7. Test model loader consistency
    print("\n🔧 Testing model loader consistency...")
    try:
        from src.models.model_loader import load_model_and_processor_unified
        
        # This will validate that all required parameters are present
        model, tokenizer, processor = load_model_and_processor_unified(
            model_path=runtime_config.model_path,
            for_inference=False
        )
        
        print("✅ Model loader consistency validated")
        print(f"   Model type: {'Detection' if hasattr(model, 'detection_enabled') and model.detection_enabled else 'Base'}")
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        
        # Clean up
        del model
        
    except Exception as e:
        print(f"❌ Error in model loader consistency: {e}")
        result.is_valid = False
    
    # 8. Final result
    print("\n" + "="*60)
    if result.is_valid:
        print("✅ COORDINATE SYSTEM CONSISTENCY TEST PASSED")
        return True
    else:
        print("❌ COORDINATE SYSTEM CONSISTENCY TEST FAILED")
        return False


if __name__ == "__main__":
    success = test_coordinate_config_consistency()
    sys.exit(0 if success else 1)