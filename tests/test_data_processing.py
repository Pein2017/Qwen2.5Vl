"""
Data Processing Test Suite

Comprehensive tests for the unified data processing pipeline:
- BBUDataset functionality validation
- Data collator integration testing
- Multi-geometry data processing
- Teacher-student sample processing
- Error handling and edge cases
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import BBUDataset, create_data_collator
from src.logger_utils import get_logger
from tests.fixtures import ConfigFactory, SyntheticDataGenerator, TestUtils


logger = get_logger("test_data_processing")


class MockTokenizer:
    """Mock tokenizer for testing without external dependencies."""
    
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = 'left'
        self.vocab_size = 1000
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Mock chat template application."""
        formatted = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted += f"<{role}>{content}</{role}>"
        return formatted
    
    def __call__(self, text, padding=False, truncation=False, add_special_tokens=True, max_length=None):
        """Mock tokenization."""
        # Simple tokenization - convert to character codes
        text_str = str(text)
        ids = [ord(c) % self.vocab_size for c in text_str[:100]]  # Limit length
        
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        
        return {
            'input_ids': ids,
            'attention_mask': [1] * len(ids)
        }
    
    def decode(self, token_ids):
        """Mock decoding."""
        return ''.join(chr(token_id % 128 + 32) for token_id in token_ids)


class TestBBUDatasetCore(unittest.TestCase):
    """Test core BBUDataset functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="bbu_test_data_"))
        self.tokenizer = MockTokenizer()
        self.config_factory = ConfigFactory()
        
        # Create test configuration
        self.config = self.config_factory.create_minimal_config()
        self.config.data_root = str(self.test_data_dir)
        self.config.teacher_ratio = 0.0  # No teachers for basic tests
        self.config.num_teacher_samples = 0
        
    def tearDown(self):
        """Clean up test environment."""
        TestUtils.cleanup_temp_directory(self.test_data_dir)
    
    def _create_test_data_file(self, samples: List[Dict]) -> str:
        """Create test JSONL data file."""
        # Create dummy images
        dummy_img = Image.new('RGB', (640, 480), color='red')
        for i in range(3):
            dummy_img.save(self.test_data_dir / f"test_image_{i}.jpg")
        
        # Write test data
        test_data_path = self.test_data_dir / "test_samples.jsonl"
        with open(test_data_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        return str(test_data_path)
    
    def test_dataset_basic_creation(self):
        """Test basic BBUDataset creation."""
        logger.info("Testing basic dataset creation")
        
        # Create test data
        test_samples = [
            {
                "images": ["test_image_0.jpg"],
                "objects": [
                    {
                        "description": "Test equipment 1",
                        "bbox_2d": [100, 200, 150, 250]
                    }
                ]
            }
        ]
        
        test_data_path = self._create_test_data_file(test_samples)
        
        # Create dataset
        dataset = BBUDataset(
            data_path=test_data_path,
            tokenizer=self.tokenizer,
            image_processor=None,  # Mock processor
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=self.config
        )
        
        self.assertEqual(len(dataset), 1, "Dataset should have 1 sample")
        
    def test_dataset_sample_processing(self):
        """Test dataset sample processing."""
        logger.info("Testing dataset sample processing")
        
        # Create test data with multiple geometries
        test_samples = [
            {
                "images": ["test_image_0.jpg"],
                "objects": [
                    {
                        "description": "Test bbox",
                        "bbox_2d": [100, 200, 150, 250]
                    },
                    {
                        "description": "Test square", 
                        "square": [300, 400, 50]
                    },
                    {
                        "description": "Test line",
                        "line": [50, 60, 70, 80, 90, 100]
                    }
                ]
            }
        ]
        
        test_data_path = self._create_test_data_file(test_samples)
        
        # Create dataset
        dataset = BBUDataset(
            data_path=test_data_path,
            tokenizer=self.tokenizer,
            image_processor=None,
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=self.config
        )
        
        # Process sample
        sample = dataset[0]
        
        # Verify output structure
        required_keys = ['input_ids', 'labels', 'attention_mask']
        for key in required_keys:
            self.assertIn(key, sample, f"Sample missing required key: {key}")
            self.assertIsInstance(sample[key], torch.Tensor, 
                                f"Key {key} should be a tensor")
        
        # Check ground truth objects
        if 'ground_truth_objects' in sample:
            self.assertEqual(len(sample['ground_truth_objects']), 3,
                           "Should have 3 ground truth objects")


class TestDataCollatorIntegration(unittest.TestCase):
    """Test data collator integration with BBUDataset."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="bbu_collator_test_"))
        self.tokenizer = MockTokenizer()
        self.config_factory = ConfigFactory()
        self.config = self.config_factory.create_minimal_config()
        
    def tearDown(self):
        """Clean up test environment."""
        TestUtils.cleanup_temp_directory(self.test_data_dir)
    
    def test_standard_collator(self):
        """Test standard data collator functionality."""
        logger.info("Testing standard data collator")
        
        # Create mock samples that mimic BBUDataset output
        mock_samples = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'labels': torch.tensor([1, 2, 3, 4, 5]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
                'ground_truth_objects': [{'bbox_2d': [100, 200, 150, 250]}]
            },
            {
                'input_ids': torch.tensor([6, 7, 8]),
                'labels': torch.tensor([6, 7, 8]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'ground_truth_objects': [{'square': [300, 400, 50]}]
            }
        ]
        
        # Create collator
        collator = create_data_collator("standard", self.tokenizer)
        
        # Create batch
        batch = collator(mock_samples)
        
        # Verify batch structure
        self.assertIn('input_ids', batch, "Batch should have input_ids")
        self.assertIn('labels', batch, "Batch should have labels")
        self.assertIn('attention_mask', batch, "Batch should have attention_mask")
        
        # Check batch dimensions
        self.assertEqual(batch['input_ids'].shape[0], 2, "Batch size should be 2")
        self.assertEqual(batch['labels'].shape[0], 2, "Labels batch size should be 2")
        
    def test_packed_collator(self):
        """Test packed data collator functionality."""
        logger.info("Testing packed data collator")
        
        # Create mock samples
        mock_samples = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1])
            },
            {
                'input_ids': torch.tensor([4, 5, 6, 7]),
                'labels': torch.tensor([4, 5, 6, 7]),
                'attention_mask': torch.tensor([1, 1, 1, 1])
            }
        ]
        
        # Create packed collator
        collator = create_data_collator("packed", self.tokenizer)
        
        # Create batch
        batch = collator(mock_samples)
        
        # Verify batch has required keys
        required_keys = ['input_ids', 'labels', 'attention_mask']
        for key in required_keys:
            self.assertIn(key, batch, f"Packed batch missing key: {key}")


class TestTeacherStudentProcessing(unittest.TestCase):
    """Test teacher-student sample processing."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="bbu_teacher_test_"))
        self.tokenizer = MockTokenizer()
        
    def tearDown(self):
        """Clean up test environment."""
        TestUtils.cleanup_temp_directory(self.test_data_dir)
    
    @pytest.mark.skip(reason="Teacher pool functionality needs implementation")
    def test_teacher_pool_integration(self):
        """Test integration with teacher pool manager."""
        # This test would verify teacher pool functionality
        # Currently skipped as teacher pool may not be fully implemented
        pass
    
    def test_teacher_ratio_zero(self):
        """Test dataset with zero teacher ratio."""
        logger.info("Testing dataset with zero teacher ratio")
        
        # Create basic test data
        test_samples = [{
            "images": ["test_image_0.jpg"],
            "objects": [{"description": "Test", "bbox_2d": [100, 200, 150, 250]}]
        }]
        
        # Create dummy image
        dummy_img = Image.new('RGB', (640, 480), color='blue')
        dummy_img.save(self.test_data_dir / "test_image_0.jpg")
        
        # Write test data
        test_data_path = self.test_data_dir / "test_samples.jsonl"
        with open(test_data_path, 'w') as f:
            for sample in test_samples:
                f.write(json.dumps(sample) + '\n')
        
        # Create configuration with zero teacher ratio
        config = ConfigFactory().create_minimal_config()
        config.data_root = str(self.test_data_dir)
        config.teacher_ratio = 0.0
        config.num_teacher_samples = 0
        
        # Create dataset
        dataset = BBUDataset(
            data_path=str(test_data_path),
            tokenizer=self.tokenizer,
            image_processor=None,
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=config
        )
        
        # Verify dataset works
        self.assertEqual(len(dataset), 1, "Dataset should have 1 sample")
        sample = dataset[0]
        self.assertIn('input_ids', sample, "Sample should have input_ids")


class TestMultiGeometryProcessing(unittest.TestCase):
    """Test multi-geometry data processing."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="bbu_geometry_test_"))
        self.tokenizer = MockTokenizer()
        
    def tearDown(self):
        """Clean up test environment."""
        TestUtils.cleanup_temp_directory(self.test_data_dir)
    
    def test_mixed_geometry_processing(self):
        """Test processing of mixed geometry types."""
        logger.info("Testing mixed geometry processing")
        
        # Create test data with all geometry types
        test_samples = [
            {
                "images": ["test_image_0.jpg"],
                "objects": [
                    {
                        "description": "Rectangular component",
                        "bbox_2d": [100, 200, 150, 250]
                    },
                    {
                        "description": "Square component", 
                        "square": [300, 400, 50]
                    },
                    {
                        "description": "Linear component",
                        "line": [50, 60, 70, 80, 90, 100]
                    }
                ]
            }
        ]
        
        # Create dummy image
        dummy_img = Image.new('RGB', (800, 600), color='green')
        dummy_img.save(self.test_data_dir / "test_image_0.jpg")
        
        # Write test data
        test_data_path = self.test_data_dir / "test_samples.jsonl"
        with open(test_data_path, 'w') as f:
            for sample in test_samples:
                f.write(json.dumps(sample) + '\n')
        
        # Create configuration
        config = ConfigFactory().create_minimal_config()
        config.data_root = str(self.test_data_dir)
        
        # Create dataset
        dataset = BBUDataset(
            data_path=str(test_data_path),
            tokenizer=self.tokenizer,
            image_processor=None,
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=config
        )
        
        # Process sample
        sample = dataset[0]
        
        # Verify processing
        self.assertIn('input_ids', sample, "Sample should have input_ids")
        
        # Check ground truth preservation
        if 'ground_truth_objects' in sample:
            objects = sample['ground_truth_objects']
            geometry_types = set()
            for obj in objects:
                if 'bbox_2d' in obj:
                    geometry_types.add('bbox_2d')
                elif 'square' in obj:
                    geometry_types.add('square')
                elif 'line' in obj:
                    geometry_types.add('line')
            
            # Should have all three geometry types
            expected_types = {'bbox_2d', 'square', 'line'}
            self.assertTrue(expected_types.issubset(geometry_types),
                          f"Missing geometry types. Expected {expected_types}, got {geometry_types}")


class TestDataProcessingEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in data processing."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="bbu_edge_test_"))
        self.tokenizer = MockTokenizer()
        
    def tearDown(self):
        """Clean up test environment."""
        TestUtils.cleanup_temp_directory(self.test_data_dir)
    
    def test_empty_objects_list(self):
        """Test handling of samples with empty objects list."""
        logger.info("Testing empty objects list handling")
        
        test_samples = [
            {
                "images": ["test_image_0.jpg"],
                "objects": []  # Empty objects list
            }
        ]
        
        # Create dummy image
        dummy_img = Image.new('RGB', (640, 480), color='white')
        dummy_img.save(self.test_data_dir / "test_image_0.jpg")
        
        # Write test data
        test_data_path = self.test_data_dir / "test_samples.jsonl"
        with open(test_data_path, 'w') as f:
            for sample in test_samples:
                f.write(json.dumps(sample) + '\n')
        
        # Create configuration
        config = ConfigFactory().create_minimal_config()
        config.data_root = str(self.test_data_dir)
        
        # Create dataset - should handle empty objects gracefully
        dataset = BBUDataset(
            data_path=str(test_data_path),
            tokenizer=self.tokenizer,
            image_processor=None,
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=config
        )
        
        # Should create dataset without error
        self.assertEqual(len(dataset), 1, "Dataset should handle empty objects")
        
        # Should be able to process sample
        sample = dataset[0]
        self.assertIn('input_ids', sample, "Should process sample with empty objects")
    
    def test_missing_image_handling(self):
        """Test handling of missing image files."""
        logger.info("Testing missing image handling")
        
        test_samples = [
            {
                "images": ["missing_image.jpg"],  # This image doesn't exist
                "objects": [
                    {
                        "description": "Test component",
                        "bbox_2d": [100, 200, 150, 250]
                    }
                ]
            }
        ]
        
        # Write test data (but don't create the image)
        test_data_path = self.test_data_dir / "test_samples.jsonl"
        with open(test_data_path, 'w') as f:
            for sample in test_samples:
                f.write(json.dumps(sample) + '\n')
        
        # Create configuration
        config = ConfigFactory().create_minimal_config()
        config.data_root = str(self.test_data_dir)
        
        # Create dataset
        dataset = BBUDataset(
            data_path=str(test_data_path),
            tokenizer=self.tokenizer,
            image_processor=None,
            teacher_pool_manager=None,
            teacher_ratio=0.0,
            is_training=True,
            config=config
        )
        
        # Attempting to access the sample should handle missing image
        # This might raise an exception or return a placeholder - 
        # the important thing is it doesn't crash silently
        try:
            sample = dataset[0]
            # If it succeeds, verify basic structure
            self.assertIn('input_ids', sample, "Should have basic structure even with missing image")
        except Exception as e:
            # If it raises an exception, it should be a clear, informative error
            self.assertIsInstance(e, (FileNotFoundError, OSError, RuntimeError),
                                f"Should raise appropriate exception for missing image, got {type(e)}")


if __name__ == "__main__":
    unittest.main()