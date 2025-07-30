"""
Configuration System Test Suite

Comprehensive tests for the new Pydantic-based configuration system:
- BBUConfig validation and loading
- Domain config extraction and validation
- Configuration error handling
- YAML parsing and validation
- Computed properties and validators
"""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    BBUConfig,
    load_config,
    init_config,
    get_config,
    TrainingConfig,
    CoordinateConfig,
    ModelConfig,
    DataConfig,
    LoggingConfig,
    VisionConfig,
)
from src.logger_utils import get_logger
from tests.fixtures import ConfigFactory, TestUtils


logger = get_logger("test_config_system")


class TestBBUConfigCore(unittest.TestCase):
    """Test core BBUConfig functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_factory = ConfigFactory()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="bbu_config_test_"))
        
    def tearDown(self):
        """Clean up test environment."""
        TestUtils.cleanup_temp_directory(self.temp_dir)
    
    def _create_test_yaml(self, config_data: Dict[str, Any]) -> str:
        """Create a temporary YAML configuration file."""
        yaml_path = self.temp_dir / "test_config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        return str(yaml_path)
    
    def test_config_loading_from_yaml(self):
        """Test loading configuration from YAML file."""
        logger.info("Testing config loading from YAML")
        
        # Create minimal valid configuration
        config_data = {
            "model_path": "/test/model",
            "model_size": "3B",
            "model_max_length": 2048,
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "use_cache_inference": True,
            "model_hidden_size": 2048,
            "model_num_layers": 24,
            "model_num_attention_heads": 16,
            "model_vocab_size": 32000,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "vision_lr": 1e-5,
            "merger_lr": 1e-5,
            "llm_lr": 1e-5,
            "adapter_lr": 1e-4,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
            "gradient_checkpointing": True,
            "bf16": True,
            "fp16": False,
            "use_flash_attention": True,
            "mixed_precision": "bf16",
            "train_data_path": "/test/train.jsonl",
            "val_data_path": "/test/val.jsonl",
            "data_root": "/test/data",
            "max_total_length": 2048,
            "teacher_pool_file": "/test/teachers.jsonl",
            "num_teacher_samples": 100,
            "collator_type": "standard",
            "teacher_ratio": 0.3,
            "max_examples": 10000,
            "language": "english",
            "coordinate_tokens_enabled": True,
            "max_coord_value": 1000,
            "coordinate_loss_weight": 0.5,
            "regular_loss_weight": 1.0,
            "coordinate_lr": 5e-5,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "training_prompt_style": True,
            "use_consistent_prompts": True,
            "dataloader_num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            "remove_unused_columns": True,
            "output_dir": "/test/output",
            "run_name": "test_run",
            "tb_dir": "/test/tb",
            "teacher_loss_weight": 1.0,
            "student_loss_weight": 1.0,
            "eval_strategy": "steps",
            "eval_steps": 100,
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,
            "logging_steps": 50,
            "logging_dir": "/test/logs",
            "log_level": "INFO",
            "report_to": "tensorboard",
            "disable_tqdm": False,
            "verbose": True,
        }
        
        yaml_path = self._create_test_yaml(config_data)
        
        # Load configuration
        config = load_config(yaml_path)
        
        # Verify basic fields
        self.assertEqual(config.model_path, "/test/model")
        self.assertEqual(config.model_size, "3B")
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertTrue(config.coordinate_tokens_enabled)
        
        # Test computed properties
        self.assertIn("test_run", config.run_output_dir)
        self.assertIn("test_run", config.tensorboard_dir)
        
    def test_config_validation(self):
        """Test configuration validation."""
        logger.info("Testing config validation")
        
        # Test invalid learning rate
        invalid_config = {
            "model_path": "/test/model",
            "learning_rate": -1.0,  # Invalid: negative learning rate
            "per_device_train_batch_size": 2,
            "num_train_epochs": 3,
        }
        
        yaml_path = self._create_test_yaml(invalid_config)
        
        with self.assertRaises(Exception):
            load_config(yaml_path)
    
    def test_coordinate_config_validation(self):
        """Test coordinate configuration validation."""
        logger.info("Testing coordinate config validation")
        
        # Test coordinate tokens enabled but zero loss weight
        invalid_config = {
            "coordinate_tokens_enabled": True,
            "coordinate_loss_weight": 0.0,  # Invalid when coordinate tokens enabled
            "max_coord_value": 1000,
            "regular_loss_weight": 1.0,
            "coordinate_lr": 5e-5,
            # ... other required fields
        }
        
        # Create minimal valid config and update with invalid values
        valid_config = self.config_factory.create_minimal_config_dict()
        valid_config.update(invalid_config)
        
        yaml_path = self._create_test_yaml(valid_config)
        
        with self.assertRaises(Exception):
            load_config(yaml_path)


class TestDomainConfigExtraction(unittest.TestCase):
    """Test domain-specific configuration extraction."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_factory = ConfigFactory()
        self.config = self.config_factory.create_minimal_config()
    
    def test_training_config_extraction(self):
        """Test training configuration extraction."""
        logger.info("Testing training config extraction")
        
        training_config = TrainingConfig.from_bbu_config(self.config)
        
        # Verify key training parameters
        self.assertEqual(training_config.learning_rate, self.config.learning_rate)
        self.assertEqual(training_config.batch_size, self.config.per_device_train_batch_size)
        self.assertEqual(training_config.epochs, self.config.num_train_epochs)
        self.assertEqual(training_config.warmup_ratio, self.config.warmup_ratio)
        
        # Test validation
        self.assertGreater(training_config.learning_rate, 0)
        self.assertGreater(training_config.batch_size, 0)
        self.assertTrue(0 <= training_config.warmup_ratio <= 1)
    
    def test_coordinate_config_extraction(self):
        """Test coordinate configuration extraction."""
        logger.info("Testing coordinate config extraction")
        
        coordinate_config = CoordinateConfig.from_bbu_config(self.config)
        
        # Verify coordinate parameters
        self.assertEqual(coordinate_config.coordinate_tokens_enabled, 
                        self.config.coordinate_tokens_enabled)
        self.assertEqual(coordinate_config.max_coord_value, self.config.max_coord_value)
        self.assertEqual(coordinate_config.coordinate_loss_weight, 
                        self.config.coordinate_loss_weight)
        
        # Test validation when coordinate tokens enabled
        if coordinate_config.coordinate_tokens_enabled:
            self.assertGreater(coordinate_config.max_coord_value, 0)
            self.assertGreaterEqual(coordinate_config.coordinate_loss_weight, 0)
    
    def test_model_config_extraction(self):
        """Test model configuration extraction."""
        logger.info("Testing model config extraction")
        
        model_config = ModelConfig.from_bbu_config(self.config)
        
        # Verify model parameters
        self.assertEqual(model_config.model_path, self.config.model_path)
        self.assertEqual(model_config.model_size, self.config.model_size)
        self.assertEqual(model_config.model_max_length, self.config.model_max_length)
        
        # Test validation
        self.assertTrue(len(model_config.model_path) > 0)
        self.assertGreater(model_config.model_max_length, 0)
        self.assertGreater(model_config.model_hidden_size, 0)
    
    def test_data_config_extraction(self):
        """Test data configuration extraction."""
        logger.info("Testing data config extraction")
        
        data_config = DataConfig.from_bbu_config(self.config)
        
        # Verify data parameters
        self.assertEqual(data_config.train_data_path, self.config.train_data_path)
        self.assertEqual(data_config.max_total_length, self.config.max_total_length)
        self.assertEqual(data_config.teacher_ratio, self.config.teacher_ratio)
        
        # Test validation
        self.assertTrue(len(data_config.train_data_path) > 0)
        self.assertGreater(data_config.max_total_length, 0)
        self.assertTrue(0 <= data_config.teacher_ratio <= 1)
    
    def test_logging_config_extraction(self):
        """Test logging configuration extraction."""
        logger.info("Testing logging config extraction")
        
        logging_config = LoggingConfig.from_bbu_config(self.config)
        
        # Verify logging parameters
        self.assertEqual(logging_config.logging_steps, self.config.logging_steps)
        self.assertEqual(logging_config.log_level, self.config.log_level)
        self.assertEqual(logging_config.eval_strategy, self.config.eval_strategy)
        
        # Test validation
        self.assertGreater(logging_config.logging_steps, 0)
        self.assertIn(logging_config.log_level, ["DEBUG", "INFO", "WARNING", "ERROR"])
        self.assertIn(logging_config.eval_strategy, ["steps", "epoch", "no"])
    
    def test_vision_config_extraction(self):
        """Test vision configuration extraction."""
        logger.info("Testing vision config extraction")
        
        vision_config = VisionConfig.from_bbu_config(self.config)
        
        # Verify vision parameters
        self.assertEqual(vision_config.patch_size, self.config.patch_size)
        self.assertEqual(vision_config.merge_size, self.config.merge_size)
        self.assertEqual(vision_config.training_prompt_style, self.config.training_prompt_style)
        
        # Test validation
        self.assertGreater(vision_config.patch_size, 0)
        self.assertGreater(vision_config.merge_size, 0)


class TestConfigProperties(unittest.TestCase):
    """Test configuration computed properties and domain config access."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_factory = ConfigFactory()
        self.config = self.config_factory.create_minimal_config()
    
    def test_computed_properties(self):
        """Test computed configuration properties."""
        logger.info("Testing computed properties")
        
        # Test run output directory
        self.assertIn(self.config.run_name, self.config.run_output_dir)
        self.assertIn(self.config.output_dir, self.config.run_output_dir)
        
        # Test tensorboard directory
        self.assertIn(self.config.run_name, self.config.tensorboard_dir)
        self.assertIn(self.config.tb_dir, self.config.tensorboard_dir)
        
        # Test log file directory
        self.assertIn("logs", self.config.log_file_dir)
        self.assertIn(self.config.run_name, self.config.log_file_dir)
        
        # Test differential learning rate detection
        if any([self.config.vision_lr, self.config.merger_lr, self.config.llm_lr]) > 0:
            # If any component LR is set, check differential LR logic
            unique_lrs = set([lr for lr in [self.config.vision_lr, self.config.merger_lr, 
                                          self.config.llm_lr] if lr > 0])
            expected_differential = len(unique_lrs) > 1
            self.assertEqual(self.config.use_differential_lr, expected_differential)
    
    def test_domain_config_properties(self):
        """Test domain configuration property access."""
        logger.info("Testing domain config properties")
        
        # Test that domain config properties work
        training_config = self.config.training_config
        self.assertIsInstance(training_config, TrainingConfig)
        
        coordinate_config = self.config.coordinate_config
        self.assertIsInstance(coordinate_config, CoordinateConfig)
        
        model_config = self.config.model_config
        self.assertIsInstance(model_config, ModelConfig)
        
        data_config = self.config.data_config
        self.assertIsInstance(data_config, DataConfig)
        
        logging_config = self.config.logging_config
        self.assertIsInstance(logging_config, LoggingConfig)
        
        vision_config = self.config.vision_config
        self.assertIsInstance(vision_config, VisionConfig)


class TestConfigErrorHandling(unittest.TestCase):
    """Test configuration error handling and edge cases."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="bbu_config_error_test_"))
        
    def tearDown(self):
        """Clean up test environment."""
        TestUtils.cleanup_temp_directory(self.temp_dir)
    
    def test_missing_file_error(self):
        """Test error handling for missing configuration file."""
        logger.info("Testing missing file error handling")
        
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
    
    def test_invalid_yaml_error(self):
        """Test error handling for invalid YAML."""
        logger.info("Testing invalid YAML error handling")
        
        # Create invalid YAML file
        invalid_yaml_path = self.temp_dir / "invalid.yaml"
        with open(invalid_yaml_path, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        with self.assertRaises(Exception):
            load_config(str(invalid_yaml_path))
    
    def test_missing_required_fields(self):
        """Test error handling for missing required fields."""
        logger.info("Testing missing required fields")
        
        # Create config with missing required fields
        incomplete_config = {"model_path": "/test/model"}  # Missing many required fields
        
        yaml_path = self.temp_dir / "incomplete.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        with self.assertRaises(Exception):
            load_config(str(yaml_path))
    
    def test_invalid_field_values(self):
        """Test error handling for invalid field values."""
        logger.info("Testing invalid field values")
        
        # Test invalid enum value
        config_factory = ConfigFactory()
        config_data = config_factory.create_minimal_config_dict()
        config_data["model_size"] = "InvalidSize"  # Invalid enum value
        
        yaml_path = self.temp_dir / "invalid_enum.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with self.assertRaises(Exception):
            load_config(str(yaml_path))


class TestGlobalConfigManagement(unittest.TestCase):
    """Test global configuration management functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_factory = ConfigFactory()
        
        # Reset any existing global config
        import src.config
        src.config._global_config = None
    
    def test_init_and_get_config(self):
        """Test global config initialization and retrieval."""
        logger.info("Testing global config management")
        
        # Test that get_config fails before initialization
        with self.assertRaises(RuntimeError):
            get_config()
        
        # Create test configuration file
        config_file = self.config_factory.create_test_config_file()
        
        # Initialize global config
        config = init_config(str(config_file))
        self.assertIsInstance(config, BBUConfig)
        
        # Test that get_config now works
        retrieved_config = get_config()
        self.assertIs(config, retrieved_config)
        
        # Clean up
        TestUtils.cleanup_temp_file(config_file)


if __name__ == "__main__":
    unittest.main()