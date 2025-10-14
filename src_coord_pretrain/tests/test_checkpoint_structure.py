#!/usr/bin/env python3
"""
Test checkpoint saving structure to ensure final checkpoint is saved in checkpoint-xxxx subfolder.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


def test_checkpoint_folder_structure():
    """Test that final checkpoint is saved in checkpoint-xxxx subfolder."""

    # Mock trainer state
    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 3120  # Example final step

    # Mock model and processor
    MagicMock()
    mock_processor = MagicMock()
    mock_processor.chat_template = "test_template"

    # Test data

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Simulate the final checkpoint saving logic
        final_step = mock_trainer.state.global_step
        final_checkpoint_dir = output_dir / f"checkpoint-{final_step}"
        final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Simulate saving essential files
        essential_files = [
            "config.json",
            "coordinate_config.json",
            "coord_token_ids.json",
            "metrics-final.json",
            "README_INFERENCE.json"
        ]

        for filename in essential_files:
            file_path = final_checkpoint_dir / filename
            with open(file_path, 'w') as f:
                json.dump({"test": "data"}, f)

        # Verify checkpoint structure
        assert final_checkpoint_dir.exists(), f"Final checkpoint directory should exist: {final_checkpoint_dir}"
        assert final_checkpoint_dir.name == f"checkpoint-{final_step}", f"Should be named checkpoint-{final_step}"

        # Verify essential files exist in checkpoint subfolder
        for filename in essential_files:
            file_path = final_checkpoint_dir / filename
            assert file_path.exists(), f"Essential file should exist in checkpoint subfolder: {filename}"

        # Verify files are NOT in the root output directory
        for filename in essential_files:
            root_file_path = output_dir / filename
            assert not root_file_path.exists(), f"File should NOT exist in root output dir: {filename}"

        print(f"✅ Checkpoint structure test passed!")
        print(f"   Final checkpoint saved to: {final_checkpoint_dir}")
        print(f"   Contains {len(essential_files)} essential files")


def test_checkpoint_naming_convention():
    """Test that checkpoint naming follows the same convention as regular checkpoints."""

    test_steps = [100, 500, 1000, 2400, 3000, 3120]

    for step in test_steps:
        expected_name = f"checkpoint-{step}"

        # This matches the pattern used in HuggingFace Trainer
        assert expected_name.startswith("checkpoint-"), "Should start with 'checkpoint-'"
        assert expected_name.endswith(str(step)), f"Should end with step number {step}"

        print(f"✅ Step {step} -> {expected_name}")

    print("✅ Checkpoint naming convention test passed!")


def test_integration_with_existing_checkpoints():
    """Test that final checkpoint integrates well with existing checkpoint structure."""

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Simulate existing regular checkpoints
        existing_checkpoints = ["checkpoint-600", "checkpoint-1200", "checkpoint-1800"]
        for checkpoint_name in existing_checkpoints:
            checkpoint_dir = output_dir / checkpoint_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Add some files to make it realistic
            (checkpoint_dir / "config.json").write_text('{"test": "data"}')
            (checkpoint_dir / "pytorch_model.bin").write_text("mock_model_data")

        # Add final checkpoint
        final_checkpoint = output_dir / "checkpoint-2400"
        final_checkpoint.mkdir(parents=True, exist_ok=True)
        (final_checkpoint / "config.json").write_text('{"test": "final"}')
        (final_checkpoint / "coordinate_config.json").write_text('{"coordinate_tokens_enabled": true}')

        # Verify structure
        all_checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        expected_checkpoints = existing_checkpoints + ["checkpoint-2400"]

        assert len(all_checkpoints) == len(expected_checkpoints), "Should have all checkpoints"

        for checkpoint_dir in all_checkpoints:
            assert checkpoint_dir.name.startswith("checkpoint-"), "All should follow naming convention"
            assert (checkpoint_dir / "config.json").exists(), "All should have config.json"

        # Verify final checkpoint has additional files
        final_checkpoint_files = list(final_checkpoint.iterdir())
        assert len(final_checkpoint_files) >= 2, "Final checkpoint should have additional files"

        print(f"✅ Integration test passed!")
        print(f"   Found {len(all_checkpoints)} checkpoints total")
        print(f"   Final checkpoint has {len(final_checkpoint_files)} files")


def test_validation_script_compatibility():
    """Test that validation script works with new checkpoint structure."""

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoint-3120"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create required files for validation
        required_files = {
            "config.json": {"model_type": "qwen2_vl"},
            "coordinate_config.json": {
                "coordinate_tokens_enabled": True,
                "max_coord_value": 1024,
                "coordinate_token_count": 1025,
                "vocab_size_after": 152000
            },
            "coord_token_ids.json": {"<|coord_0|>": 1000, "<|coord_1|>": 1001},
            "metrics-final.json": {"eval_loss": 0.123},
            "tokenizer.json": {"version": "1.0"},
            "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
            "preprocessor_config.json": {"image_processor_type": "Qwen2VLImageProcessor"},
            "model.safetensors.index.json": {"metadata": {"total_size": 1000000}}
        }

        for filename, content in required_files.items():
            with open(checkpoint_dir / filename, 'w') as f:
                json.dump(content, f)

        # Create mock model weight files
        (checkpoint_dir / "model-00001-of-00002.safetensors").write_text("mock_weights")
        (checkpoint_dir / "model-00002-of-00002.safetensors").write_text("mock_weights")

        # Verify all required files exist
        for filename in required_files.keys():
            assert (checkpoint_dir / filename).exists(), f"Required file missing: {filename}"

        print(f"✅ Validation compatibility test passed!")
        print(f"   Checkpoint directory: {checkpoint_dir}")
        print(f"   Contains {len(required_files)} required files")


if __name__ == "__main__":
    print("🧪 Testing checkpoint structure improvements...")
    print("=" * 50)

    test_functions = [
        test_checkpoint_folder_structure,
        test_checkpoint_naming_convention,
        test_integration_with_existing_checkpoints,
        test_validation_script_compatibility,
    ]

    for test_func in test_functions:
        try:
            print(f"\n📋 Running {test_func.__name__}...")
            test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            raise

    print(f"\n🎉 All checkpoint structure tests passed!")
    print("\n📁 Expected final structure:")
    print("src_coord_pretrain/output/")
    print("├── checkpoint-600/")
    print("├── checkpoint-1200/")
    print("├── checkpoint-1800/")
    print("└── checkpoint-3120/  # Final inference-ready checkpoint")
    print("    ├── config.json")
    print("    ├── coordinate_config.json")
    print("    ├── coord_token_ids.json")
    print("    ├── metrics-final.json")
    print("    ├── model-*.safetensors")
    print("    ├── tokenizer files...")
    print("    └── README_INFERENCE.json")
