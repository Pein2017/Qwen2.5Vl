#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.config import init_config

# Load YAML config
config_path = Path(__file__).parent.parent / "configs" / "base_flat_det.yaml"
with open(config_path, 'r') as f:
    yaml_config = yaml.safe_load(f)

# Initialize config
init_config(str(config_path))
from src.config import config

print("YAML Configuration:")
for key, value in yaml_config.items():
    if 'coordinate_config' in key:
        print(f"  {key}: {value}")

print("\nRuntime Configuration:")
for attr in dir(config):
    if 'coordinate_config' in attr:
        print(f"  {attr}: {getattr(config, attr)}")

# Test model loader
print("\nTesting model loader...")
try:
    from src.models.model_loader import load_model_and_processor_unified
    
    model, tokenizer, processor = load_model_and_processor_unified(
        model_path=config.model_path,
        for_inference=False
    )
    
    print("✅ Model loaded successfully!")
    print(f"   Model type: {'Detection' if hasattr(model, 'detection_enabled') and model.detection_enabled else 'Base'}")
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    # Clean up
    del model
    
except Exception as e:
    print(f"❌ Error in model loader: {e}")