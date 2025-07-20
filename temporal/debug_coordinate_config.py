#!/usr/bin/env python3
"""
Debug script to check coordinate token configuration values
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

from src.config import config, init_config

def debug_coordinate_config():
    """Debug coordinate token configuration."""
    
    print("🔍 Debugging Coordinate Token Configuration")
    print("=" * 50)
    
    # Initialize config
    try:
        init_config("configs/base_flat_det.yaml")
        print("✅ Config initialized successfully")
    except Exception as e:
        print(f"❌ Config initialization failed: {e}")
        return
    
    # Check coordinate token settings
    print(f"\n📊 Coordinate Token Settings:")
    
    # Check if coordinate_tokens_enabled exists
    if hasattr(config, 'coordinate_tokens_enabled'):
        print(f"   coordinate_tokens_enabled: {config.coordinate_tokens_enabled}")
    else:
        print("   ❌ coordinate_tokens_enabled not found in config")
    
    # Check if coordinate_config_max_coord_value exists
    if hasattr(config, 'coordinate_config_max_coord_value'):
        print(f"   coordinate_config_max_coord_value: {config.coordinate_config_max_coord_value}")
    else:
        print("   ❌ coordinate_config_max_coord_value not found in config")
    
    # Check nested coordinate_config
    if hasattr(config, 'coordinate_config'):
        print(f"   coordinate_config exists: {type(config.coordinate_config)}")
        if hasattr(config.coordinate_config, 'max_coord_value'):
            print(f"   coordinate_config.max_coord_value: {config.coordinate_config.max_coord_value}")
    else:
        print("   ❌ coordinate_config not found in config")
    
    # Check all coordinate-related attributes
    print(f"\n🔍 All coordinate-related attributes:")
    for attr in dir(config):
        if 'coord' in attr.lower():
            value = getattr(config, attr)
            print(f"   {attr}: {value}")
    
    print(f"\n📝 Full config dump (coordinate-related fields):")
    # Dump all config fields to see what's available
    for field_name in config.__dataclass_fields__:
        if 'coord' in field_name.lower():
            value = getattr(config, field_name)
            print(f"   {field_name}: {value}")

if __name__ == "__main__":
    debug_coordinate_config()