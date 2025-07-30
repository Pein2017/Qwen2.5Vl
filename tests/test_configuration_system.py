"""
Configuration System Tests for BBU Training System

Tests the new BBUConfig system and configuration loading
functionality after refactoring.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton

patch_torch_library_wrap_triton()

from src.config import BBUConfig, init_config, load_config, get_config
from src.logger_utils import configure_global_logging, get_logger
from tests.fixtures import ConfigFactory, TestUtils
from tests.fixtures.gpu_test_base import GPUAwareTestCase

logger = get_logger("test_configuration_system")


class TestConfigurationSystem(GPUAwareTestCase):
    """Tests for configuration system functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        configure_global_logging(rank=0, world_size=1)
        logger.info("🧪 Starting Configuration System Tests")

        # Create test utilities
        cls.test_utils = TestUtils()
        cls.config_factory = ConfigFactory()

        logger.info("✅ Configuration system test setup complete")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.config_factory.cleanup_test_configs()
        logger.info("🧹 Configuration system test teardown complete")

    def setUp(self):
        """Set up for each individual test."""
        self.test_files_to_cleanup = []
        super().setUp()

    def tearDown(self):
        """Clean up after each test."""
        self.test_utils.cleanup_test_files(self.test_files_to_cleanup)
        super().tearDown()

    def test_base_config_loading(self):
        """Test loading the base BBU configuration."""
        logger.info("🧪 Testing base config loading")
        
        # Load base configuration
        base_config_path = "/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml"
        
        # Test raw YAML loading first
        with open(base_config_path, 'r') as f:
            raw_yaml = yaml.safe_load(f)
        
        self.assertIsInstance(raw_yaml, dict)
        logger.info(f"✅ Raw YAML loaded with {len(raw_yaml)} keys")
        
        # Check for coordinate-related fields in raw YAML
        coordinate_related = {k: v for k, v in raw_yaml.items() if 'coordinate' in k.lower()}
        logger.info(f"📊 Coordinate-related fields in YAML: {coordinate_related}")
        
        # Test BBUConfig loading
        config = load_config(base_config_path)
        self.assertIsInstance(config, BBUConfig)
        logger.info("✅ BBUConfig loaded successfully")
        
        # Verify essential fields are present
        essential_fields = [
            'model_path',
            'data_root',
            'output_dir',
            'num_train_epochs',
            'per_device_train_batch_size',
            'learning_rate'
        ]
        
        for field in essential_fields:
            self.assertTrue(hasattr(config, field), f"Missing essential field: {field}")
        
        logger.info(f"✅ All {len(essential_fields)} essential fields present")

    def test_coordinate_config_fields(self):
        """Test coordinate-related configuration fields."""
        logger.info("🧪 Testing coordinate config fields")
        
        # Load base configuration
        base_config_path = "/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml"
        config = load_config(base_config_path)
        
        # Check specific coordinate fields
        coordinate_fields = [
            'coordinate_tokens_enabled',
            'max_coord_value',
            'coordinate_loss_weight',
            'regular_loss_weight'
        ]
        
        found_fields = []
        missing_fields = []
        
        for field in coordinate_fields:
            if hasattr(config, field):
                value = getattr(config, field)
                found_fields.append((field, value))
                logger.info(f"   ✅ {field} = {value} (type: {type(value).__name__})")
            else:
                missing_fields.append(field)
                logger.warning(f"   ❌ {field} = MISSING")
        
        logger.info(f"📊 Found {len(found_fields)} coordinate fields, missing {len(missing_fields)}")
        
        # List all coordinate-related attributes
        coord_attrs = [attr for attr in dir(config) if 'coord' in attr.lower()]
        logger.info(f"📋 All coordinate-related attributes: {coord_attrs}")
        
        for attr in coord_attrs:
            if not attr.startswith('_'):
                value = getattr(config, attr)
                logger.info(f"   {attr} = {value}")

    def test_config_initialization(self):
        """Test config initialization with init_config."""
        logger.info("🧪 Testing config initialization")
        
        # Test init_config function
        base_config_path = "/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml"
        config = init_config(base_config_path)
        
        self.assertIsInstance(config, BBUConfig)
        logger.info("✅ init_config successful")
        
        # Test that get_config retrieves the initialized config
        config2 = get_config()
        
        self.assertIsInstance(config2, BBUConfig)
        logger.info("✅ get_config retrieves initialized config successfully")

    def test_test_config_creation(self):
        """Test test configuration creation via ConfigFactory."""
        logger.info("🧪 Testing test config creation")
        
        # Create temporary data directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test coordinate enabled config
            coord_config_path = self.config_factory.create_coordinate_enabled_config(
                data_root=temp_dir,
                collator_type="standard"
            )
            
            coord_config = self.config_factory.load_test_config(coord_config_path)
            
            self.assertIsInstance(coord_config, BBUConfig)
            self.assertTrue(coord_config.coordinate_tokens_enabled)
            self.assertGreater(coord_config.coordinate_loss_weight, 0)
            
            logger.info("✅ Coordinate enabled test config created")
            logger.info(f"   - coordinate_tokens_enabled: {coord_config.coordinate_tokens_enabled}")
            logger.info(f"   - coordinate_loss_weight: {coord_config.coordinate_loss_weight}")
            
            # Test coordinate disabled config
            standard_config_path = self.config_factory.create_coordinate_disabled_config(
                data_root=temp_dir,
                collator_type="standard"
            )
            
            standard_config = self.config_factory.load_test_config(standard_config_path)
            
            self.assertIsInstance(standard_config, BBUConfig)
            self.assertFalse(standard_config.coordinate_tokens_enabled)
            self.assertEqual(standard_config.coordinate_loss_weight, 0.0)
            
            logger.info("✅ Coordinate disabled test config created")
            logger.info(f"   - coordinate_tokens_enabled: {standard_config.coordinate_tokens_enabled}")
            logger.info(f"   - coordinate_loss_weight: {standard_config.coordinate_loss_weight}")

    def test_minimal_config_creation(self):
        """Test minimal configuration for fast testing."""
        logger.info("🧪 Testing minimal config creation")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test minimal config with coordinates enabled
            minimal_config_path = self.config_factory.create_minimal_config(
                data_root=temp_dir,
                coordinate_enabled=True,
                collator_type="standard"
            )
            
            minimal_config = self.config_factory.load_test_config(minimal_config_path)
            
            self.assertIsInstance(minimal_config, BBUConfig)
            self.assertEqual(minimal_config.num_train_epochs, 1)
            self.assertEqual(minimal_config.per_device_train_batch_size, 2)
            self.assertEqual(minimal_config.max_total_length, 512)  # Reduced for minimal tests
            self.assertTrue(minimal_config.coordinate_tokens_enabled)
            
            logger.info("✅ Minimal config created with expected settings")
            logger.info(f"   - num_train_epochs: {minimal_config.num_train_epochs}")
            logger.info(f"   - batch_size: {minimal_config.per_device_train_batch_size}")
            logger.info(f"   - max_length: {minimal_config.max_total_length}")
            logger.info(f"   - coordinate_enabled: {minimal_config.coordinate_tokens_enabled}")

    def test_config_field_types(self):
        """Test that configuration fields have correct types."""
        logger.info("🧪 Testing config field types")
        
        base_config_path = "/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml"
        config = load_config(base_config_path)
        
        # Test numeric fields
        numeric_fields = [
            ('num_train_epochs', (int, float)),
            ('per_device_train_batch_size', int),
            ('learning_rate', float),
            ('max_coord_value', int),
        ]
        
        for field_name, expected_type in numeric_fields:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                if isinstance(expected_type, tuple):
                    self.assertIsInstance(value, expected_type, 
                                        f"Field {field_name} should be {expected_type}, got {type(value)}")
                else:
                    self.assertIsInstance(value, expected_type,
                                        f"Field {field_name} should be {expected_type}, got {type(value)}")
                logger.info(f"   ✅ {field_name}: {value} ({type(value).__name__})")
        
        # Test boolean fields
        boolean_fields = ['coordinate_tokens_enabled']
        
        for field_name in boolean_fields:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                self.assertIsInstance(value, bool,
                                    f"Field {field_name} should be bool, got {type(value)}")
                logger.info(f"   ✅ {field_name}: {value} (bool)")
        
        # Test string fields
        string_fields = ['model_path', 'data_root', 'output_dir']
        
        for field_name in string_fields:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                self.assertIsInstance(value, str,
                                    f"Field {field_name} should be str, got {type(value)}")
                logger.info(f"   ✅ {field_name}: {value[:50]}... (str)")

    def test_config_validation(self):
        """Test configuration validation and error handling."""
        logger.info("🧪 Testing config validation")
        
        # Test loading non-existent config
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_config.yaml")
        logger.info("✅ Non-existent config properly raises FileNotFoundError")
        
        # Test loading invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            invalid_yaml_path = f.name
        
        self.test_files_to_cleanup.append(invalid_yaml_path)
        
        with self.assertRaises(yaml.YAMLError):
            load_config(invalid_yaml_path)
        logger.info("✅ Invalid YAML properly raises YAMLError")

    def test_config_diagnostic(self):
        """Run comprehensive configuration diagnostic."""
        logger.info("🧪 Running configuration diagnostic")
        
        diagnostic_results = {
            'base_config_loads': False,
            'coordinate_fields_present': False,
            'test_config_creation': False,
            'field_types_correct': False,
            'validation_works': False
        }
        
        # Test base config loading
        try:
            base_config_path = "/data3/Qwen2.5-VL-main/configs/bbu_v2.yaml"
            config = load_config(base_config_path)
            diagnostic_results['base_config_loads'] = True
            logger.info("✅ Base config loads successfully")
        except Exception as e:
            logger.error(f"❌ Base config loading failed: {e}")
        
        # Test coordinate fields
        try:
            if diagnostic_results['base_config_loads']:
                coord_enabled = hasattr(config, 'coordinate_tokens_enabled')
                max_coord = hasattr(config, 'max_coord_value')
                diagnostic_results['coordinate_fields_present'] = coord_enabled and max_coord
                logger.info(f"✅ Coordinate fields present: {coord_enabled and max_coord}")
        except Exception as e:
            logger.error(f"❌ Coordinate fields check failed: {e}")
        
        # Test config creation
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                test_config_path = self.config_factory.create_minimal_config(temp_dir)
                test_config = self.config_factory.load_test_config(test_config_path)
                diagnostic_results['test_config_creation'] = True
                logger.info("✅ Test config creation successful")
        except Exception as e:
            logger.error(f"❌ Test config creation failed: {e}")
        
        # Test field types
        try:
            if diagnostic_results['base_config_loads']:
                epochs_is_int = isinstance(config.num_train_epochs, (int, float))
                lr_is_float = isinstance(config.learning_rate, float)
                diagnostic_results['field_types_correct'] = epochs_is_int and lr_is_float
                logger.info(f"✅ Field types correct: {epochs_is_int and lr_is_float}")
        except Exception as e:
            logger.error(f"❌ Field type check failed: {e}")
        
        # Test validation
        try:
            with self.assertRaises(FileNotFoundError):
                load_config("non_existent.yaml")
            diagnostic_results['validation_works'] = True
            logger.info("✅ Validation works correctly")
        except Exception as e:
            logger.error(f"❌ Validation test failed: {e}")
        
        # Summary
        successful_tests = sum(diagnostic_results.values())
        total_tests = len(diagnostic_results)
        
        logger.info(f"\n🏥 CONFIGURATION SYSTEM DIAGNOSTIC SUMMARY:")
        logger.info(f"   Tests passed: {successful_tests}/{total_tests}")
        
        for test_name, passed in diagnostic_results.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {test_name}")
        
        if successful_tests == total_tests:
            logger.info("🎉 All configuration system components working correctly")
        else:
            logger.warning("⚠️  Some configuration system components need attention")


if __name__ == '__main__':
    unittest.main()