#!/root/miniconda3/envs/ms/bin/python
"""
Test that the fix works - coordinator should now get the correct config
"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

from src.config.global_config import init_config, reset_config

def test_fixed_coordinator_config():
    print("🔍 Testing fixed coordinator config access...")
    
    # Initialize config like train.py does
    reset_config()
    config = init_config('configs/bbu_v2.yaml')
    
    # Test the fixed approach (using get_config)
    print("\n1. Testing fixed get_config() approach:")
    from src.config import get_config
    
    retrieved_config = get_config()
    print(f"   get_config().coordinate_tokens_enabled = {getattr(retrieved_config, 'coordinate_tokens_enabled', 'MISSING')}")
    print(f"   retrieved_config id: {id(retrieved_config)}")
    print(f"   original config id: {id(config)}")
    print(f"   Same object? {retrieved_config is config}")
    
    # Test validation condition with fixed approach
    print(f"\n2. Testing validation condition with get_config():")
    coordinate_enabled = getattr(retrieved_config, "coordinate_tokens_enabled", False)
    if coordinate_enabled:
        print(f"   ✅ Coordinate tokens are enabled: {coordinate_enabled}")
        print(f"   This should now work correctly!")
    else:
        print(f"   ❌ Still disabled: {coordinate_enabled}")
        print(f"   Fix didn't work...")

if __name__ == "__main__":
    test_fixed_coordinator_config()