#!/usr/bin/env python3
"""
Integration validation script for src_coord_pretrain → src_new pipeline.

Validates that checkpoints from src_coord_pretrain are compatible with src_new.
"""

import json
import sys
from pathlib import Path
from typing import Dict


def validate_checkpoint_structure(checkpoint_dir: Path) -> Dict[str, bool]:
    """Validate that checkpoint contains all required files for src_new integration."""

    required_files = {
        # Model files
        "config.json": "Model configuration",
        "tokenizer.json": "Tokenizer vocabulary",
        "tokenizer_config.json": "Tokenizer configuration",
        "preprocessor_config.json": "Image processor configuration",
        # Coordinate token files
        "coordinate_config.json": "Coordinate system configuration",
        "coord_token_ids.json": "Coordinate token ID mappings",
        # Training artifacts
        "metrics-final.json": "Final training metrics",
        # Model weights (SafeTensors format)
        "model.safetensors.index.json": "Model weight index (sharded)",
    }

    results = {}
    for filename, description in required_files.items():
        file_path = checkpoint_dir / filename
        results[filename] = file_path.exists()

        if not results[filename]:
            print(f"❌ Missing: {filename} ({description})")
        else:
            print(f"✅ Found: {filename}")

    # Check for model weight files
    safetensors_files = list(checkpoint_dir.glob("model-*.safetensors"))
    if safetensors_files:
        print(f"✅ Found {len(safetensors_files)} SafeTensors model weight files")
        results["model_weights"] = True
    else:
        print("❌ No SafeTensors model weight files found")
        results["model_weights"] = False

    return results


def validate_coordinate_config(checkpoint_dir: Path) -> bool:
    """Validate coordinate configuration for src_new compatibility."""

    coord_config_path = checkpoint_dir / "coordinate_config.json"
    if not coord_config_path.exists():
        print("❌ coordinate_config.json not found")
        return False

    try:
        with open(coord_config_path, "r") as f:
            coord_config = json.load(f)

        required_keys = [
            "coordinate_tokens_enabled",
            "max_coord_value",
            "coordinate_token_count",
            "vocab_size_after",
        ]

        for key in required_keys:
            if key not in coord_config:
                print(f"❌ Missing key in coordinate_config.json: {key}")
                return False

        # Validate values
        if not coord_config["coordinate_tokens_enabled"]:
            print("❌ coordinate_tokens_enabled is False")
            return False

        if coord_config["max_coord_value"] <= 0:
            print(f"❌ Invalid max_coord_value: {coord_config['max_coord_value']}")
            return False

        expected_token_count = coord_config["max_coord_value"] + 1
        if coord_config["coordinate_token_count"] != expected_token_count:
            print(
                f"❌ Token count mismatch: expected {expected_token_count}, got {coord_config['coordinate_token_count']}"
            )
            return False

        print("✅ Coordinate configuration is valid")
        return True

    except Exception as e:
        print(f"❌ Error reading coordinate_config.json: {e}")
        return False


def validate_coordinate_tokens(checkpoint_dir: Path) -> bool:
    """Validate coordinate token mappings."""

    coord_ids_path = checkpoint_dir / "coord_token_ids.json"
    if not coord_ids_path.exists():
        print("❌ coord_token_ids.json not found")
        return False

    try:
        with open(coord_ids_path, "r") as f:
            coord_ids = json.load(f)

        if not coord_ids:
            print("❌ Empty coordinate token mappings")
            return False

        # Check for expected coordinate tokens
        expected_tokens = [f"<|coord_{i}|>" for i in range(min(10, len(coord_ids)))]
        for token in expected_tokens:
            if token not in coord_ids:
                print(f"❌ Missing coordinate token: {token}")
                return False

        # Check token ID ranges
        token_ids = list(coord_ids.values())
        if not token_ids:
            print("❌ No token IDs found")
            return False

        min_id, max_id = min(token_ids), max(token_ids)
        print(
            f"✅ Coordinate tokens: {len(coord_ids)} tokens, ID range: {min_id}-{max_id}"
        )
        return True

    except Exception as e:
        print(f"❌ Error reading coord_token_ids.json: {e}")
        return False


def validate_training_metrics(checkpoint_dir: Path) -> bool:
    """Validate training metrics for quality assessment."""

    metrics_path = checkpoint_dir / "metrics-final.json"
    if not metrics_path.exists():
        print("❌ metrics-final.json not found")
        return False

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Check for key metrics
        if "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            print(f"📊 Final evaluation loss: {eval_loss:.4f}")

            # Reasonable loss threshold (adjust based on experience)
            if eval_loss > 2.0:
                print(
                    f"⚠️  High evaluation loss: {eval_loss:.4f} (may indicate training issues)"
                )
            else:
                print("✅ Evaluation loss looks reasonable")

        # Check for enhanced metrics (if available)
        enhanced_metrics = [
            "eval_identity_accuracy",
            "eval_reverse_accuracy",
            "eval_strictness_accuracy",
        ]

        for metric in enhanced_metrics:
            if metric in metrics:
                value = metrics[metric]
                print(f"📊 {metric}: {value:.3f}")

        return True

    except Exception as e:
        print(f"❌ Error reading metrics-final.json: {e}")
        return False


def main():
    """Main validation function."""

    if len(sys.argv) != 2:
        print("Usage: python validate_integration.py <checkpoint_directory>")
        sys.exit(1)

    checkpoint_dir = Path(sys.argv[1])
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        sys.exit(1)

    print(f"🔍 Validating checkpoint integration: {checkpoint_dir}")
    print("=" * 60)

    # Run validation checks
    checks = [
        ("Checkpoint Structure", lambda: validate_checkpoint_structure(checkpoint_dir)),
        (
            "Coordinate Configuration",
            lambda: validate_coordinate_config(checkpoint_dir),
        ),
        (
            "Coordinate Token Mappings",
            lambda: validate_coordinate_tokens(checkpoint_dir),
        ),
        ("Training Metrics", lambda: validate_training_metrics(checkpoint_dir)),
    ]

    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        try:
            result = check_func()
            if isinstance(result, dict):
                # For structure check, count successes
                passed = all(result.values())
            else:
                passed = result

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All integration checks passed!")
        print("✅ Checkpoint is ready for src_new pipeline")
        sys.exit(0)
    else:
        print("❌ Some integration checks failed")
        print("⚠️  Checkpoint may not be compatible with src_new pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
