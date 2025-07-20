#!/usr/bin/env python3
"""Test the DirectConfig fix"""

import sys
sys.path.append('/data3/Qwen2.5-VL-main')

try:
    from src.config.global_config import init_config, reset_config
    
    print("🧪 Testing DirectConfig fix...")
    reset_config()
    config = init_config("/data3/Qwen2.5-VL-main/configs/base_flat_det.yaml")
    
    print("✅ DirectConfig loaded successfully!")
    print(f"   - Collator type: {config.collator_type}")
    print(f"   - Remove unused columns: {config.remove_unused_columns}")
    print(f"   - Pin memory: {config.pin_memory}")
    
    # Verify batching_strategy is no longer referenced
    if hasattr(config, 'batching_strategy'):
        print("❌ batching_strategy still exists")
    else:
        print("✅ batching_strategy successfully removed")
        
    print("\n🎉 Configuration fix successful!")
    
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)