#!/root/miniconda3/envs/ms/bin/python
"""
Test config loading exactly as the training coordinator does it
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

from src.config.global_config import init_config, reset_config

def test_coordinator_config_access():
    print("🔍 Testing config access exactly like training coordinator...")
    
    # Initialize config
    reset_config()
    config = init_config('configs/bbu_v2.yaml')
    
    # Test the global config singleton import (like training coordinator does)
    print("\n1. Testing global config singleton access:")
    from src.config import config as global_config
    
    print(f"   global_config.coordinate_tokens_enabled = {getattr(global_config, 'coordinate_tokens_enabled', 'MISSING')}")
    print(f"   config.coordinate_tokens_enabled = {getattr(config, 'coordinate_tokens_enabled', 'MISSING')}")
    print(f"   global_config id: {id(global_config)}")
    print(f"   config id: {id(config)}")
    print(f"   Same object? {global_config is config}")
    
    # Test both ways to access config (like training coordinator does in different lines)
    print("\n2. Testing different access patterns:")
    
    # Method 1: getattr with default False (like training coordinator line 330)
    coordinate_enabled_getattr = getattr(global_config, "coordinate_tokens_enabled", False)
    print(f"   getattr(config, 'coordinate_tokens_enabled', False) = {coordinate_enabled_getattr}")
    print(f"   Type: {type(coordinate_enabled_getattr)}")
    
    # Method 2: direct access 
    try:
        coordinate_enabled_direct = global_config.coordinate_tokens_enabled
        print(f"   config.coordinate_tokens_enabled = {coordinate_enabled_direct}")
        print(f"   Type: {type(coordinate_enabled_direct)}")
    except AttributeError as e:
        print(f"   Direct access failed: {e}")
    
    # Method 3: hasattr check
    has_attr = hasattr(global_config, 'coordinate_tokens_enabled')
    print(f"   hasattr(config, 'coordinate_tokens_enabled') = {has_attr}")
    
    # Check if there's any None value issue
    print(f"\n3. Raw attribute value inspection:")
    print(f"   Raw value: {repr(getattr(global_config, 'coordinate_tokens_enabled', 'ATTR_MISSING'))}")
    print(f"   Is None? {getattr(global_config, 'coordinate_tokens_enabled', 'MISSING') is None}")
    print(f"   Is False? {getattr(global_config, 'coordinate_tokens_enabled', 'MISSING') is False}")
    print(f"   Is True? {getattr(global_config, 'coordinate_tokens_enabled', 'MISSING') is True}")
    
    # Test the exact condition used in the validation
    print(f"\n4. Testing validation condition:")
    coordinate_enabled = getattr(global_config, "coordinate_tokens_enabled", False)
    if coordinate_enabled:
        print(f"   ✅ Coordinate tokens are enabled: {coordinate_enabled}")
    else:
        print(f"   ❌ Coordinate tokens are disabled: {coordinate_enabled}")
        print(f"   This would trigger the CONFIG_ISSUE error!")

if __name__ == "__main__":
    test_coordinator_config_access()