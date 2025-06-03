"""
Usage examples for the Qwen2.5VL BBU training framework.

This module demonstrates various ways to use the framework components
with direct imports as preferred.
"""

# Direct imports from specific modules
from src.config.config_manager import load_config
from src.data import BBUDataset, DataCollator
from src.inference import InferenceEngine
from src.logging import (
    configure_global_logging,
    get_data_logger,
    get_model_logger,
    get_training_logger,
)
from src.models.wrapper import ModelWrapper
from src.training.stability import StabilityMonitor
from src.training.trainer import create_trainer


def example_1_basic_training():
    """Example 1: Basic training setup with global logging."""

    # Step 1: Configure global logging
    configure_global_logging(
        log_dir="logs", log_level="INFO", verbose=True, is_training=True
    )

    # Step 2: Load configuration from YAML
    config = load_config("train_3b")  # Loads configs/train_3b.yaml

    # Step 3: Create and run trainer
    trainer = create_trainer(config)
    trainer.train()


def example_2_custom_configuration():
    """Example 2: Using custom configuration programmatically."""

    # Configure logging first
    configure_global_logging(log_level="DEBUG", verbose=True)

    # Create config programmatically
    config_dict = {
        "model_path": "/data4/swift/model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
        "model_size": "3B",
        "model_max_length": 8192,
        "train_data_path": "521_qwen_train.jsonl",
        "val_data_path": "521_qwen_val.jsonl",
        "output_dir": "output/custom_run",
        "num_train_epochs": 5,
        "per_device_train_batch_size": 1,
        "learning_rate": 8e-7,
        "loss_type": "object_detection",
        "bbox_weight": 0.8,
    }

    # Convert to config object
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    config = SimpleConfig(config_dict)

    # Create and run trainer
    trainer = create_trainer(config)
    trainer.train()


def example_3_custom_components():
    """Example 3: Using individual components separately."""

    # Setup logging
    configure_global_logging()
    logger = get_training_logger()

    # Create config
    config = load_config("debug")

    # Load individual components
    model_wrapper = ModelWrapper(config, get_model_logger())
    model, tokenizer, image_processor = model_wrapper.load_all()

    # Create dataset
    dataset = BBUDataset(config, tokenizer, image_processor, config.train_data_path)

    # Create data collator (simplified)
    collator = DataCollator(
        tokenizer=tokenizer,
        max_seq_length=config.model_max_length,
        use_dynamic_length=True,
    )

    # Test components
    logger.info(f"Model loaded: {type(model).__name__}")
    logger.info(f"Dataset size: {len(dataset)}")

    # Test data loading
    sample = dataset[0]
    batch = collator([sample])
    logger.info(f"Batch keys: {list(batch.keys())}")


def example_4_inference_usage():
    """Example 4: Using the inference engine."""

    # Setup
    configure_global_logging()

    # Create inference engine
    inference_engine = InferenceEngine(
        model_path="output/checkpoint-1000",
        device="auto",
        max_new_tokens=1024,
    )

    # Run inference
    image_path = "data/sample_image.jpg"
    prompt = "Please describe this image and identify any objects."

    result = inference_engine.inference(image_path=image_path, prompt=prompt)
    print(f"Inference result: {result}")


def example_5_stability_monitoring():
    """Example 5: Using stability monitoring separately."""

    # Setup
    configure_global_logging()
    logger = get_training_logger()

    # Create a mock config
    class MockConfig:
        max_consecutive_nan = 5
        max_nan_ratio = 0.3
        nan_recovery_enabled = True
        learning_rate_reduction_factor = 0.5
        gradient_clip_reduction_factor = 0.5

    config = MockConfig()

    # Create stability monitor
    stability_monitor = StabilityMonitor(config, logger)

    # Simulate checking losses
    import torch

    test_losses = [1.2, 0.8, float("nan"), 0.9, 1.1]

    for i, loss in enumerate(test_losses):
        loss_tensor = torch.tensor(loss)
        status = stability_monitor.check_loss_stability(loss_tensor, i)
        logger.info(f"Step {i}: Loss={loss}, Status={status['is_stable']}")


def example_6_global_logger_usage():
    """Example 6: Using the global logger system."""

    # Configure global logging with custom settings
    configure_global_logging(
        log_dir="custom_logs", log_level="DEBUG", verbose=True, console_level="INFO"
    )

    # Get different loggers
    training_logger = get_training_logger()
    model_logger = get_model_logger()
    data_logger = get_data_logger()

    # Use loggers
    training_logger.info("🚀 Training started")
    model_logger.debug("🔧 Model configuration loaded")
    data_logger.info("📊 Dataset loaded successfully")

    # All logs will be consistently formatted and saved


def example_7_distributed_training():
    """Example 7: Setting up distributed training."""

    # Setup for distributed training
    configure_global_logging(
        log_level="INFO", verbose=False
    )  # Less verbose for distributed

    # Load config with distributed settings
    config = load_config("qwen25vl_7b")  # Typically has distributed settings

    # The trainer will automatically handle distributed setup
    trainer = create_trainer(config)

    # Training will use distributed setup if environment variables are set
    # (RANK, WORLD_SIZE, etc.)
    trainer.train()


def example_8_development_workflow():
    """Example 8: Development and debugging workflow."""

    # Setup with debug logging
    configure_global_logging(log_level="DEBUG", verbose=True)
    logger = get_training_logger()

    # Load debug config
    config = load_config("debug")

    # Create trainer
    trainer = create_trainer(config)

    # Test components
    logger.info("🧪 Testing data loading...")
    trainer.test_data_loading(num_samples=2)

    logger.info("🧪 Testing model forward pass...")
    trainer.test_model_forward()

    # Start actual training (or just test setup)
    logger.info("🚀 Starting training...")
    trainer.train()


if __name__ == "__main__":
    print("🚀 BBU Training Framework Examples")
    print("=" * 50)

    # Run a simple example
    print("Running Example 1: Basic Training Setup")
    try:
        example_1_basic_training()
        print("✅ Example completed successfully!")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
