#!/usr/bin/env python3
"""
Configuration Validation Script

This script validates training/inference configuration consistency to prevent
common issues that can cause training convergence with poor inference results.

Usage:
    python scripts/validate_config.py --config base_flat_v2
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import init_config, config


def check_detection_consistency():
    """Check detection configuration consistency (legacy - detection now uses coordinate tokens)."""
    issues = []
    
    # Check if we have coordinate token data
    data_files = ["data/train.jsonl", "data/val.jsonl"]
    for data_file in data_files:
        if Path(data_file).exists():
            # Sample first line to check data format
            with open(data_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        sample = json.loads(first_line)
                        if 'objects' in sample and sample['objects']:
                            # Check if coordinate tokens are enabled
                            coordinate_tokens_enabled = getattr(config, 'coordinate_tokens_enabled', False)
                            if not coordinate_tokens_enabled:
                                issues.append(f"⚠️  INFO: {data_file} contains detection data but coordinate tokens not enabled")
                    except json.JSONDecodeError:
                        issues.append(f"⚠️  {data_file} has invalid JSON format")
    
    return issues


def check_model_path_consistency():
    """Check model path consistency."""
    issues = []
    
    if hasattr(config, 'model_path'):
        model_path = Path(config.model_path)
        if not model_path.exists():
            issues.append(f"❌ Model path does not exist: {config.model_path}")
        
        # Check if it's a base model vs checkpoint
        if "checkpoint" in str(model_path):
            issues.append(f"ℹ️  Using checkpoint path: {config.model_path}")
        elif "Qwen" in str(model_path):
            issues.append(f"ℹ️  Using base model path: {config.model_path}")
    
    return issues


def check_data_consistency():
    """Check data configuration consistency."""
    issues = []
    
    # Check if data files exist
    data_files = ["data/train.jsonl", "data/val.jsonl"]
    for data_file in data_files:
        if not Path(data_file).exists():
            issues.append(f"⚠️  Data file missing: {data_file}")
    
    return issues


def check_training_parameters():
    """Check training parameter sanity."""
    issues = []
    
    # Check learning rate consistency
    if hasattr(config, 'learning_rate') and hasattr(config, 'adapter_lr'):
        if config.learning_rate == 0 and config.adapter_lr > 0:
            issues.append(f"ℹ️  Using adapter-only training (LR=0, adapter_lr={config.adapter_lr})")
        elif config.learning_rate > 0:
            issues.append(f"ℹ️  Using full model training (LR={config.learning_rate})")
    
    # Check coordinate token parameters if enabled
    if getattr(config, 'coordinate_tokens_enabled', False):
        required_params = [
            'coordinate_tokens_max_coord_value', 'coordinate_tokens_temperature'
        ]
        for param in required_params:
            if not hasattr(config, param):
                issues.append(f"❌ Missing coordinate token parameter: {param} (using defaults)")
            elif getattr(config, param) <= 0:
                issues.append(f"⚠️  Detection parameter {param} = {getattr(config, param)} (should be > 0)")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate training configuration")
    parser.add_argument("--config", required=True, help="Config name (e.g., base_flat_v2)")
    args = parser.parse_args()
    
    try:
        # Initialize config
        config_path = f"configs/{args.config}.yaml"
        init_config(config_path)
        
        print(f"🔍 Validating configuration: {args.config}")
        print("=" * 60)
        
        # Run all checks
        all_issues = []
        all_issues.extend(check_detection_consistency())
        all_issues.extend(check_model_path_consistency())
        all_issues.extend(check_data_consistency())
        all_issues.extend(check_training_parameters())
        
        # Report results
        if not all_issues:
            print("✅ Configuration validation passed!")
            print("   No issues detected.")
        else:
            print(f"Found {len(all_issues)} issues:")
            for issue in all_issues:
                print(f"   {issue}")
            
            # Check for critical issues
            critical_issues = [issue for issue in all_issues if "CRITICAL" in issue or "❌" in issue]
            if critical_issues:
                print(f"\n🚨 {len(critical_issues)} CRITICAL issues found!")
                print("   These issues may cause training-inference mismatches.")
                sys.exit(1)
            else:
                print(f"\nℹ️  All issues are warnings or info messages.")
        
        print("\n📊 Configuration Summary:")
        print(f"   Coordinate tokens enabled: {getattr(config, 'coordinate_tokens_enabled', False)}")
        print(f"   Model path: {getattr(config, 'model_path', 'Unknown')}")
        print(f"   Learning rate: {getattr(config, 'learning_rate', 'Unknown')}")
        print(f"   Adapter LR: {getattr(config, 'adapter_lr', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()