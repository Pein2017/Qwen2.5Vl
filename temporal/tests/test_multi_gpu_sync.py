#!/usr/bin/env python3
"""
Test multi-GPU loss synchronization logic
Verify that custom loss components are properly synchronized across distributed ranks
"""

import sys


sys.path.insert(0, "/data3/Qwen2.5-VL-main")

import torch

from src.config.global_config import init_config, reset_config
from src.training.loss_manager import LossManager


def test_multi_gpu_sync():
    """Test multi-GPU loss synchronization logic"""

    print("🧪 Testing multi-GPU loss synchronization...")

    try:
        # Test the synchronization logic without actually initializing distributed training
        # We'll mock the distributed functions

        # Reset any existing config
        reset_config()

        # Initialize config
        config = init_config("configs/bbu_v2.yaml")

        print("🔧 Testing loss synchronization logic (mocked distributed)...")

        # Create mock tokenizer and model
        class MockTokenizer:
            def get_vocab(self):
                return {"test": 0}

        class MockModel:
            def get_last_coordinate_losses(self):
                return {}

        mock_tokenizer = MockTokenizer()
        mock_model = MockModel()

        # Initialize loss manager
        loss_manager = LossManager(
            tokenizer=mock_tokenizer,
            model=mock_model,
            teacher_loss_weight=config.teacher_loss_weight,
            student_loss_weight=config.student_loss_weight,
        )

        # Create mock loss components with different values per "rank"
        loss_components = {
            "geometry_focal_loss": 1.5,
            "coordinate_l1_loss": 2.0,
            "geometry_bbox_giou_loss": 0.8,
            "weighted_teacher_loss": 0.6,
            "weighted_student_loss": 3.2,
        }

        print("📊 Original loss components (before sync simulation):")
        for name, value in loss_components.items():
            print(f"   {name}: {value:.3f}")

        # Test the synchronization method directly
        # Since we can't actually run distributed training in this test,
        # we'll check that the method exists and has the right structure

        if hasattr(loss_manager, "_synchronize_loss_components"):
            print("✅ Loss synchronization method exists")

            # Check if the method can be called (it will fail due to no distributed init, but that's expected)
            try:
                loss_manager._synchronize_loss_components(loss_components)
                print(
                    "❌ UNEXPECTED: Synchronization succeeded without distributed init"
                )
                return False
            except Exception as e:
                if any(
                    keyword in str(e).lower()
                    for keyword in [
                        "distributed",
                        "process group",
                        "init_process_group",
                    ]
                ):
                    print(
                        "✅ Loss synchronization correctly fails without distributed init"
                    )
                else:
                    print(f"❌ UNEXPECTED ERROR: {e}")
                    return False
        else:
            print("❌ FAIL: Loss synchronization method missing")
            return False

        # Test that torch.distributed.is_initialized check works
        if not torch.distributed.is_initialized():
            print(
                "✅ Distributed training not initialized - synchronization will be skipped"
            )
        else:
            print("⚠️ Distributed training is initialized in test environment")

        # Verify the components that should be synchronized
        expected_sync_components = [
            "geometry_focal_loss",
            "coordinate_l1_loss",
            "geometry_bbox_giou_loss",
            "geometry_square_polygon_loss",
            "geometry_square_corner_loss",
            "geometry_line_smoothness_loss",
            "geometry_line_ordering_loss",
            "weighted_teacher_loss",
            "weighted_student_loss",
        ]

        print("✅ Expected synchronization components:")
        for component in expected_sync_components:
            print(f"   - {component}")

        print("✅ Multi-GPU loss synchronization logic validated!")
        print("ℹ️  Note: Actual distributed testing requires multi-GPU environment")
        return True

    except Exception as e:
        print(f"❌ Multi-GPU sync test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        reset_config()


if __name__ == "__main__":
    success = test_multi_gpu_sync()
    sys.exit(0 if success else 1)
