#!/usr/bin/env python3
"""
Simple timeout test to detect if inference gets stuck.
Uses signal to timeout after a specified duration.
"""

import os
import signal
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def test_with_timeout(timeout_seconds=300):  # 5 minutes timeout
    """Test inference with timeout."""

    print(f"🕐 Setting timeout to {timeout_seconds} seconds")

    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        print(f"⏰ Starting test at {datetime.now()}")

        # Import and run the debug script
        from eval.debug_inference import debug_inference_step_by_step

        success = debug_inference_step_by_step()

        # Cancel the alarm if we complete successfully
        signal.alarm(0)

        if success:
            print("✅ Test completed successfully within timeout")
            return True
        else:
            print("❌ Test failed but didn't timeout")
            return False

    except TimeoutError:
        print(f"⏰ TIMEOUT: Operation took longer than {timeout_seconds} seconds")
        print("🚨 The model appears to be stuck during inference!")

        # Try to identify where it got stuck
        print("\n🔍 Possible causes:")
        print("1. Flash attention configuration issues")
        print("2. Memory allocation problems")
        print("3. Model generation loop stuck")
        print("4. CUDA synchronization issues")
        print("5. Infinite generation loop")

        return False
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        print(f"❌ Test failed with error: {e}")
        return False


def quick_gpu_test():
    """Quick test to verify GPU is working."""
    print("\n🚀 Quick GPU test...")

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False

        # Test basic GPU operations
        device = torch.device("cuda:0")

        # Create tensors
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)

        # Test computation
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time

        print(f"✅ GPU computation test passed in {gpu_time:.3f}s")
        print(
            f"💾 GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        # Clean up
        del a, b, c
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False


def check_model_files():
    """Check if model files exist and are accessible."""
    print("\n📁 Checking model files...")

    model_path = "output/qwen7B-lr_2e-7-tbs_8-epochs_20/checkpoint-660"

    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False

    # Check for essential files
    essential_files = [
        "config.json",
        "pytorch_model.bin",  # or model.safetensors
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]

    missing_files = []
    for file in essential_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            # Check for alternative names
            if file == "pytorch_model.bin":
                alt_path = os.path.join(model_path, "model.safetensors")
                if not os.path.exists(alt_path):
                    missing_files.append(file)
            else:
                missing_files.append(file)
        else:
            print(f"✅ Found: {file}")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All essential model files found")
    return True


if __name__ == "__main__":
    print("🕐 TIMEOUT TEST FOR INFERENCE")
    print("=" * 60)

    # Step 1: Quick GPU test
    if not quick_gpu_test():
        print("❌ GPU test failed, aborting")
        sys.exit(1)

    # Step 2: Check model files
    if not check_model_files():
        print("❌ Model files check failed, aborting")
        sys.exit(1)

    # Step 3: Run inference with timeout
    print("\n🧪 Running inference test with timeout...")
    success = test_with_timeout(timeout_seconds=300)  # 5 minutes

    if success:
        print("\n🎉 SUCCESS: Inference works correctly")
    else:
        print("\n💥 FAILURE: Inference has issues")
        print("\n🔧 Debugging suggestions:")
        print("1. Try reducing max_new_tokens to 128")
        print("2. Use eager attention instead of flash_attention_2")
        print("3. Check for CUDA version compatibility")
        print("4. Monitor GPU memory usage during inference")
        print("5. Try inference on CPU to isolate GPU issues")
