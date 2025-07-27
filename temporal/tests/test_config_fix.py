#!/usr/bin/env python3
"""
Test configuration system fix
Verify that coordinate tokens are properly enabled via unified configuration
"""

import sys
import os
sys.path.insert(0, '/data3/Qwen2.5-VL-main')

from src.config.global_config import init_config, reset_config

def test_config_fix():
    """Test that configuration system works with unified coordinate token field"""
    
    print("🧪 Testing configuration system fix...")
    
    try:
        # Reset any existing config
        reset_config()
        
        # Initialize config
        config = init_config("configs/bbu_v2.yaml")
        
        # Test coordinate token configuration
        print(f"✅ Config loaded successfully")
        print(f"   coordinate_config_enable_coordinate_tokens: {config.coordinate_config_enable_coordinate_tokens}")
        print(f"   coordinate_config_max_coord_value: {config.coordinate_config_max_coord_value}")
        
        # Verify deprecated field is gone
        if hasattr(config, 'coordinate_tokens_enabled'):
            print("❌ FAIL: Deprecated coordinate_tokens_enabled field still exists")
            return False
        
        # Test coordinate token validation works
        if config.coordinate_config_enable_coordinate_tokens:
            print("✅ Coordinate tokens properly enabled")
            print(f"   Max coord value: {config.coordinate_config_max_coord_value}")
            print(f"   Loss weights: coord={config.coordinate_config_coordinate_loss_weight}, regular={config.coordinate_config_regular_loss_weight}")
        
        print("✅ Configuration system fix validated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    finally:
        reset_config()

if __name__ == "__main__":
    success = test_config_fix()
    sys.exit(0 if success else 1)