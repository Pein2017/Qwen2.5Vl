#!/root/miniconda3/envs/ms/bin/python
"""
Test config loading to diagnose why coordinate_tokens_enabled is False at runtime
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

from src.config.global_config import init_config, reset_config
import yaml

def test_config_loading():
    print("🔍 Testing config loading...")
    
    # 1. Load raw YAML to see what's actually in the file
    print("\n1. Raw YAML content:")
    with open('configs/bbu_v2.yaml', 'r') as f:
        raw_yaml = yaml.safe_load(f)
    
    coordinate_related = {k: v for k, v in raw_yaml.items() if 'coordinate' in k.lower()}
    print(f"   Coordinate-related fields in YAML: {coordinate_related}")
    
    # 2. Initialize config and check what gets loaded
    print("\n2. Initializing config...")
    reset_config()  # Reset if already initialized
    config = init_config('configs/bbu_v2.yaml')
    
    # 3. Check specific coordinate fields
    print("\n3. Config object coordinate fields:")
    print(f"   coordinate_tokens_enabled = {getattr(config, 'coordinate_tokens_enabled', 'MISSING')}")
    print(f"   coordinate_config_enable_coordinate_tokens = {getattr(config, 'coordinate_config_enable_coordinate_tokens', 'MISSING')}")
    print(f"   coordinate_lr = {getattr(config, 'coordinate_lr', 'MISSING')}")
    
    # 4. List all coordinate-related attributes in config object
    coord_attrs = [attr for attr in dir(config) if 'coord' in attr.lower()]
    print(f"\n4. All coordinate-related config attributes:")
    for attr in coord_attrs:
        value = getattr(config, attr)
        print(f"   {attr} = {value}")
    
    # 5. Check field types
    print(f"\n5. Field type information:")
    print(f"   coordinate_tokens_enabled type: {type(getattr(config, 'coordinate_tokens_enabled', None))}")
    print(f"   coordinate_config_enable_coordinate_tokens type: {type(getattr(config, 'coordinate_config_enable_coordinate_tokens', None))}")

if __name__ == "__main__":
    test_config_loading()