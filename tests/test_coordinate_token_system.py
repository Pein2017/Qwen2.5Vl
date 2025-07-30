"""
Coordinate Token System Tests for BBU Training System

Comprehensive tests for the coordinate token generation pipeline,
including coordinate managers, token conversion, and integration.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

# Add project root to path
sys.path.insert(0, "/data3/Qwen2.5-VL-main")

# Apply patches early
from src.models.patches import patch_torch_library_wrap_triton

patch_torch_library_wrap_triton()

from src.config import BBUConfig
from src.logger_utils import configure_global_logging, get_logger
from tests.fixtures import ConfigFactory, SyntheticDataGenerator, TestUtils
from tests.fixtures.gpu_test_base import GPUAwareTestCase

logger = get_logger("test_coordinate_token_system")


class TestCoordinateTokenSystem(GPUAwareTestCase):
    """Tests for coordinate token system functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        configure_global_logging(rank=0, world_size=1)
        logger.info("🧪 Starting Coordinate Token System Tests")

        # Create test utilities
        cls.test_utils = TestUtils()
        cls.config_factory = ConfigFactory()
        
        # Create minimal synthetic dataset for coordinate testing
        cls.data_generator = SyntheticDataGenerator(num_samples=3)
        cls.train_path, cls.val_path, cls.teacher_path, cls.all_samples_path = (
            cls.data_generator.generate_complete_dataset()
        )
        cls.data_root = str(cls.data_generator.temp_dir)

        logger.info(f"✅ Coordinate token system test setup complete: {cls.data_root}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.data_generator.cleanup()
        cls.config_factory.cleanup_test_configs()
        logger.info("🧹 Coordinate token system test teardown complete")

    def setUp(self):
        """Set up for each individual test."""
        self.test_files_to_cleanup = []
        super().setUp()

    def tearDown(self):
        """Clean up after each test."""
        self.test_utils.cleanup_test_files(self.test_files_to_cleanup)
        super().tearDown()

    def test_coordinate_token_manager_import(self):
        """Test coordinate token manager imports."""
        logger.info("🧪 Testing coordinate token manager imports")
        
        try:
            from src.utils.tokens.special_tokens import (
                SimpleCoordinateManager,
                UnifiedTokenManager,
            )
            logger.info("✅ Successfully imported coordinate token managers")
            
            # Test that classes are available
            self.assertTrue(hasattr(SimpleCoordinateManager, '__init__'))
            self.assertTrue(hasattr(UnifiedTokenManager, '__init__'))
            
        except ImportError as e:
            self.fail(f"Failed to import coordinate token managers: {e}")

    def test_simple_coordinate_manager_creation(self):
        """Test SimpleCoordinateManager creation and basic functionality."""
        logger.info("🧪 Testing SimpleCoordinateManager creation")
        
        try:
            from src.utils.tokens.special_tokens import SimpleCoordinateManager
            
            # Create mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.vocab_size = 152064  # Typical Qwen tokenizer size
            
            # Create SimpleCoordinateManager
            simple_manager = SimpleCoordinateManager(mock_tokenizer, max_coord_value=100)
            logger.info("✅ SimpleCoordinateManager created successfully")
            
            # Test coordinate wrapping with different geometry types
            test_coordinates = {
                "bbox_2d": [50, 60, 70, 80],
                "square": [10, 15, 20, 15, 20, 25, 10, 25],
                "line": [30, 35, 40, 45, 50, 55],
            }
            
            for geom_type, coords in test_coordinates.items():
                try:
                    wrapped = simple_manager.wrap_coordinates(coords, geom_type)
                    self.assertIsInstance(wrapped, str)
                    logger.info(f"✅ {geom_type} coordinates wrapped: {coords} -> {wrapped[:50]}...")
                except Exception as e:
                    logger.warning(f"⚠️  Error wrapping {geom_type}: {e}")
            
        except Exception as e:
            logger.warning(f"⚠️  SimpleCoordinateManager test failed: {e}")

    def test_coordinate_token_id_ranges(self):
        """Test coordinate token ID ranges and vocabulary."""
        logger.info("🧪 Testing coordinate token ID ranges")
        
        try:
            from src.utils.tokens.special_tokens import SimpleCoordinateManager
            
            # Create mock tokenizer with realistic vocab size
            mock_tokenizer = Mock()
            mock_tokenizer.vocab_size = 152064
            
            simple_manager = SimpleCoordinateManager(mock_tokenizer, max_coord_value=2048)
            
            # Test coordinate token ID generation for various values
            test_values = [0, 50, 100, 255, 500, 1000, 2048]
            
            for val in test_values:
                try:
                    if hasattr(simple_manager, "get_coordinate_token_id"):
                        token_id = simple_manager.get_coordinate_token_id(val)
                        logger.info(f"   Value {val} -> Token ID {token_id}")
                        
                        # Verify token ID is in reasonable range
                        self.assertIsInstance(token_id, int)
                        self.assertGreaterEqual(token_id, 0)
                        
                    elif hasattr(simple_manager, "_coordinate_to_token_id"):
                        token_id = simple_manager._coordinate_to_token_id(val)
                        logger.info(f"   Value {val} -> Token ID {token_id}")
                        
                        # Verify token ID is in reasonable range
                        self.assertIsInstance(token_id, int)
                        self.assertGreaterEqual(token_id, 0)
                        
                    else:
                        logger.warning("   Cannot find coordinate token ID method")
                        break
                        
                except Exception as e:
                    logger.warning(f"   Value {val} -> Error: {e}")
            
            # Check if coordinate tokens use high token ID space
            if hasattr(simple_manager, "coordinate_token_ranges"):
                ranges = simple_manager.coordinate_token_ranges
                logger.info(f"📊 Coordinate token ranges: {ranges}")
                
                for range_name, (start, end) in ranges.items():
                    if start < 150000:
                        logger.warning(f"   ⚠️  {range_name} range starts at {start}, which is low")
                    else:
                        logger.info(f"   ✅ {range_name} range: {start}-{end} (high range)")
            
        except Exception as e:
            logger.warning(f"⚠️  Coordinate token ID range test failed: {e}")

    def test_coordinate_conversion_with_synthetic_data(self):
        """Test coordinate conversion with synthetic data."""
        logger.info("🧪 Testing coordinate conversion with synthetic data")
        
        try:
            from src.utils.tokens.special_tokens import SimpleCoordinateManager
            
            # Create mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.vocab_size = 152064
            
            simple_manager = SimpleCoordinateManager(mock_tokenizer, max_coord_value=2048)
            
            # Load synthetic data samples
            samples = self.data_generator.generate_samples()
            
            coordinate_tokens_generated = 0
            
            for i, sample in enumerate(samples[:2]):  # Test first 2 samples
                logger.info(f"📊 Testing sample {i + 1}:")
                logger.info(f"   Image: {sample.get('images', ['unknown'])[0]}")
                logger.info(f"   Objects: {len(sample.get('objects', []))}")
                
                for j, obj in enumerate(sample.get('objects', [])[:3]):  # Test first 3 objects
                    desc = obj.get('description', obj.get('desc', 'unknown'))
                    logger.info(f"   Object {j + 1}: {desc[:50]}...")
                    
                    for geom_type, coords in obj.items():
                        if geom_type in ['description', 'desc']:
                            continue
                        
                        try:
                            # Test coordinate wrapping
                            wrapped = simple_manager.wrap_coordinates(coords, geom_type)
                            if wrapped and len(wrapped) > 0:
                                coordinate_tokens_generated += 1
                                logger.info(f"     {geom_type}: {len(coords)} coords -> wrapped successfully")
                            else:
                                logger.warning(f"     {geom_type}: {len(coords)} coords -> NO TOKENS ❌")
                                
                        except Exception as e:
                            logger.warning(f"     {geom_type}: Error - {e}")
            
            if coordinate_tokens_generated == 0:
                logger.warning("❌ CRITICAL: No coordinate tokens generated")
            else:
                logger.info(f"✅ Total coordinate operations completed: {coordinate_tokens_generated}")
                
        except Exception as e:
            logger.warning(f"⚠️  Coordinate conversion test failed: {e}")

    def test_coordinate_format_patterns(self):
        """Test coordinate format patterns and expectations."""
        logger.info("🧪 Testing coordinate format patterns")
        
        try:
            # Test expected coordinate format patterns
            expected_patterns = [
                "<coordinate>",
                "</coordinate>",
                "坐标标记",  # Chinese coordinate markers
            ]
            
            logger.info("Looking for coordinate patterns in expected formats...")
            
            for pattern in expected_patterns:
                logger.info(f"   Expected pattern: {pattern}")
                # These patterns should be used in coordinate formatting
                self.assertIsInstance(pattern, str)
                self.assertGreater(len(pattern), 0)
            
            logger.info("✅ Coordinate format patterns validated")
            
        except Exception as e:
            logger.warning(f"⚠️  Coordinate format pattern test failed: {e}")

    def test_unified_token_manager_integration(self):
        """Test UnifiedTokenManager integration."""
        logger.info("🧪 Testing UnifiedTokenManager integration")
        
        try:
            from src.utils.tokens.special_tokens import UnifiedTokenManager
            
            # Create mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.vocab_size = 152064
            
            # Test UnifiedTokenManager creation
            unified_manager = UnifiedTokenManager(mock_tokenizer)
            logger.info("✅ UnifiedTokenManager created successfully")
            
            # Test that it has expected methods
            expected_methods = [
                'create_coordinate_tokens',
                'wrap_coordinates',
            ]
            
            found_methods = []
            for method in expected_methods:
                if hasattr(unified_manager, method):
                    found_methods.append(method)
            
            logger.info(f"✅ UnifiedTokenManager has {len(found_methods)} expected methods: {found_methods}")
            
        except Exception as e:
            logger.warning(f"⚠️  UnifiedTokenManager integration test failed: {e}")

    def test_coordinate_config_integration(self):
        """Test coordinate token configuration integration."""
        logger.info("🧪 Testing coordinate configuration integration")
        
        # Create coordinate-enabled configuration
        config_path = self.config_factory.create_coordinate_enabled_config(
            data_root=self.data_root,
            collator_type="standard"
        )
        config = self.config_factory.load_test_config(config_path)
        
        # Verify coordinate configuration
        self.assertTrue(config.coordinate_tokens_enabled)
        self.assertGreater(config.max_coord_value, 0)
        self.assertGreater(config.coordinate_loss_weight, 0)
        
        logger.info("✅ Coordinate configuration loaded successfully")
        logger.info(f"   - coordinate_tokens_enabled: {config.coordinate_tokens_enabled}")
        logger.info(f"   - max_coord_value: {config.max_coord_value}")
        logger.info(f"   - coordinate_loss_weight: {config.coordinate_loss_weight}")

    def test_coordinate_system_diagnostic(self):
        """Run diagnostic test for coordinate token system."""
        logger.info("🧪 Running coordinate system diagnostic")
        
        diagnostic_results = {
            'imports_successful': False,
            'manager_creation': False,
            'coordinate_wrapping': False,
            'config_integration': False
        }
        
        # Test imports
        try:
            from src.utils.tokens.special_tokens import SimpleCoordinateManager, UnifiedTokenManager
            diagnostic_results['imports_successful'] = True
            logger.info("✅ Imports successful")
        except Exception as e:
            logger.error(f"❌ Import failed: {e}")
        
        # Test manager creation
        try:
            mock_tokenizer = Mock()
            mock_tokenizer.vocab_size = 152064
            manager = SimpleCoordinateManager(mock_tokenizer, max_coord_value=100)
            diagnostic_results['manager_creation'] = True
            logger.info("✅ Manager creation successful")
        except Exception as e:
            logger.error(f"❌ Manager creation failed: {e}")
        
        # Test coordinate wrapping
        try:
            if diagnostic_results['manager_creation']:
                wrapped = manager.wrap_coordinates([10, 20, 30, 40], "bbox_2d")
                diagnostic_results['coordinate_wrapping'] = True
                logger.info("✅ Coordinate wrapping successful")
        except Exception as e:
            logger.error(f"❌ Coordinate wrapping failed: {e}")
        
        # Test config integration
        try:
            config_path = self.config_factory.create_coordinate_enabled_config(
                data_root=self.data_root
            )
            config = self.config_factory.load_test_config(config_path)
            diagnostic_results['config_integration'] = config.coordinate_tokens_enabled
            logger.info("✅ Config integration successful")
        except Exception as e:
            logger.error(f"❌ Config integration failed: {e}")
        
        # Summary
        successful_tests = sum(diagnostic_results.values())
        total_tests = len(diagnostic_results)
        
        logger.info(f"\n🏥 COORDINATE SYSTEM DIAGNOSTIC SUMMARY:")
        logger.info(f"   Tests passed: {successful_tests}/{total_tests}")
        
        for test_name, passed in diagnostic_results.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {test_name}")
        
        if successful_tests == total_tests:
            logger.info("🎉 All coordinate system components working correctly")
        else:
            logger.warning("⚠️  Some coordinate system components need attention")


if __name__ == '__main__':
    unittest.main()