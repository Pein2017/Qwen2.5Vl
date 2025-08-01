#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete data flow pipeline tests for teacher-student training.

This module tests the complete data flow from raw data through collator
to model input, identifying where tensor flattening occurs and validating
the teacher-student processing pipeline.

Pipeline Flow:
1. Raw data (images + annotations) → Dataset
2. Dataset → Chat processor (tokenization + image processing)
3. Chat processor → Collator (batching + padding)
4. Collator → Model input (final tensors)

Focus: Identify where pixel_values tensor gets incorrectly flattened to 2D
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock, patch
from pathlib import Path

from src_new.data.collator import StandardDataCollator
from src_new.tests.fixtures.mock_objects import MockTokenizer, MockImageProcessor
from src_new.tests.fixtures.test_utils import create_mock_conversation


class TestPipelineDataFlow:
    """Test complete data flow through teacher-student pipeline."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer with proper geometry tokens."""
        tokenizer = MockTokenizer(vocab_size=151665)
        # Add the line tokens that would be added by token processor
        tokenizer.add_tokens(["<|line_start|>", "<|line_end|>"])
        return tokenizer

    @pytest.fixture
    def mock_image_processor(self):
        """Create mock image processor that simulates Qwen2.5-VL behavior."""
        return MockImageProcessor()

    @pytest.fixture
    def mock_config(self):
        """Create mock config for pipeline components."""
        config = Mock()
        config.max_total_length = 4096
        config.coordinate_tokens_enabled = True
        config.max_coord_value = 2048
        return config

    @pytest.fixture
    def standard_collator(self, mock_tokenizer, mock_config):
        """Create standard data collator."""
        return StandardDataCollator(
            tokenizer=mock_tokenizer,
            config=mock_config,
            pad_token_id=0,
            max_length=4096,
            label_pad_token_id=-100
        )

    def test_raw_data_to_processed_sample_flow(self, mock_image_processor):
        """
        Test the flow from raw data to processed sample.
        
        This simulates what happens in the dataset when processing
        teacher-student samples.
        """
        # Simulate raw teacher-student data
        raw_teacher_data = {
            "images": ["teacher_image.jpg"],
            "objects": [
                {"bbox_2d": [100, 100, 200, 200], "desc": "BBU设备教师示例"},
                {"quad": [150, 150, 250, 150, 250, 250, 150, 250], "desc": "连接器教师示例"}
            ]
        }
        
        raw_student_data = {
            "images": ["student_image.jpg"], 
            "objects": [
                {"bbox_2d": [120, 120, 220, 220], "desc": "BBU设备学生"},
                {"line": [50, 50, 100, 100, 150, 150], "desc": "线缆学生"}
            ]
        }
        
        print("🔍 Testing raw data to processed sample flow:")
        print(f"   - Teacher objects: {len(raw_teacher_data['objects'])}")
        print(f"   - Student objects: {len(raw_student_data['objects'])}")
        
        # Simulate image processing (what would happen in dataset)
        teacher_images = [Mock()]  # Mock PIL images
        student_images = [Mock()]
        
        # Process images through mock image processor
        teacher_processed = mock_image_processor(teacher_images)
        student_processed = mock_image_processor(student_images)
        
        print(f"   - Teacher pixel_values: {teacher_processed['pixel_values'].shape}")
        print(f"   - Student pixel_values: {student_processed['pixel_values'].shape}")
        print(f"   - Teacher image_grid_thw: {teacher_processed['image_grid_thw'].shape}")
        print(f"   - Student image_grid_thw: {student_processed['image_grid_thw'].shape}")
        
        # Verify image processing produces expected formats
        assert teacher_processed["pixel_values"].dim() == 4  # [N, C, H, W]
        assert student_processed["pixel_values"].dim() == 4
        assert teacher_processed["image_grid_thw"].dim() == 2  # [N, 3]
        assert student_processed["image_grid_thw"].dim() == 2
        
        print("   ✅ Image processing produces correct tensor formats")

    def test_chat_processor_simulation(self, mock_tokenizer):
        """
        Test chat processor simulation for teacher-student conversations.
        
        This simulates what happens in the chat processor when building
        teacher-student conversations.
        """
        # Create teacher-student conversation
        teacher_conversation = create_mock_conversation(include_teacher=False)
        student_conversation = create_mock_conversation(include_teacher=False)
        
        # Simulate combined teacher-student conversation
        combined_conversation = [
            {"role": "system", "content": "你是通信机房设备检测AI助手。"},
            {"role": "user", "content": "📚 参考示例: <image>"},
            {"role": "assistant", "content": '[{"bbox_2d":[100,100,200,200], "desc":"教师示例设备"}]'},
            {"role": "user", "content": "现在请检测以下图像: <image>"},
            {"role": "assistant", "content": '[{"bbox_2d":[120,120,220,220], "desc":"学生检测设备"}]'},
        ]
        
        print("🔍 Testing chat processor simulation:")
        print(f"   - Combined conversation length: {len(combined_conversation)}")
        
        # Simulate tokenization
        full_conversation_text = ""
        for msg in combined_conversation:
            full_conversation_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        
        # Mock tokenization result
        tokenized = mock_tokenizer(full_conversation_text, return_tensors="pt")
        
        print(f"   - Tokenized input_ids shape: {tokenized['input_ids'].shape}")
        print(f"   - Tokenized attention_mask shape: {tokenized['attention_mask'].shape}")
        
        # Verify tokenization produces reasonable results
        assert tokenized["input_ids"].dim() == 2  # [batch_size, seq_len]
        assert tokenized["attention_mask"].dim() == 2
        assert tokenized["input_ids"].shape[1] > 0  # Non-empty sequence
        
        print("   ✅ Chat processing produces correct token formats")

    def test_teacher_student_sample_creation(self, mock_tokenizer, mock_image_processor):
        """
        Test creation of teacher-student samples that would go to collator.
        
        This simulates the exact format that causes the bug.
        """
        print("🔍 Testing teacher-student sample creation:")
        
        # Create samples in the format that would be passed to collator
        # This simulates what the dataset produces
        
        # Teacher sample (first part of teacher-student conversation)
        teacher_sample = {
            "input_ids": torch.tensor([151644, 1587, 151645, 151644, 882, 100, 101, 151645]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
            "labels": torch.tensor([-100, -100, -100, -100, -100, 100, 101, -100]),
            # CRITICAL: Simulate Qwen2.5-VL patch format from image processor
            "pixel_values": torch.randn(1200, 1176),  # Patch format from teacher image
            "image_grid_thw": torch.tensor([[1, 30, 40]]),  # Teacher image dimensions
        }
        
        # Student sample (second part of teacher-student conversation)  
        student_sample = {
            "input_ids": torch.tensor([151644, 882, 102, 103, 151645, 151644, 77091, 104, 105, 151645]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            "labels": torch.tensor([-100, -100, -100, -100, -100, -100, 104, 105, -100, -100]),
            # CRITICAL: Another patch format from student image
            "pixel_values": torch.randn(1400, 1176),  # Different patch count
            "image_grid_thw": torch.tensor([[1, 35, 40]]),  # Student image dimensions
        }
        
        print(f"   - Teacher sample pixel_values: {teacher_sample['pixel_values'].shape}")
        print(f"   - Student sample pixel_values: {student_sample['pixel_values'].shape}")
        print(f"   - Teacher input_ids length: {teacher_sample['input_ids'].shape[0]}")
        print(f"   - Student input_ids length: {student_sample['input_ids'].shape[0]}")
        
        # Verify samples have correct structure
        for sample_name, sample in [("teacher", teacher_sample), ("student", student_sample)]:
            assert "input_ids" in sample
            assert "attention_mask" in sample
            assert "labels" in sample
            assert "pixel_values" in sample
            assert "image_grid_thw" in sample
            
            # Check tensor dimensions
            assert sample["pixel_values"].dim() == 2, f"{sample_name} pixel_values should be 2D patch format"
            assert sample["image_grid_thw"].dim() == 2, f"{sample_name} image_grid_thw should be 2D"
            
        print("   ✅ Teacher-student samples created with correct structure")
        
        return teacher_sample, student_sample

    def test_collator_processing_step_by_step(self, standard_collator, mock_tokenizer, mock_image_processor):
        """
        Test collator processing step by step to identify where flattening occurs.
        
        This is the critical test that shows exactly where the bug happens.
        """
        print("🔍 Testing collator processing step by step:")
        
        # Get teacher-student samples
        teacher_sample, student_sample = self.test_teacher_student_sample_creation(
            mock_tokenizer, mock_image_processor
        )
        
        # Step 1: Individual sample processing
        print("\n📋 Step 1: Individual sample analysis")
        print(f"   - Teacher pixel_values: {teacher_sample['pixel_values'].shape}")
        print(f"   - Student pixel_values: {student_sample['pixel_values'].shape}")
        
        # Step 2: Combine into batch
        print("\n📋 Step 2: Batch creation")
        features = [teacher_sample, student_sample]
        print(f"   - Batch size: {len(features)}")
        
        # Step 3: Collator processing
        print("\n📋 Step 3: Collator processing")
        batch = standard_collator(features)
        
        print(f"   - Output pixel_values shape: {batch['pixel_values'].shape}")
        print(f"   - Output image_grid_thw shape: {batch['image_grid_thw'].shape}")
        print(f"   - Output input_ids shape: {batch['input_ids'].shape}")
        
        # Step 4: Analysis of the result
        print("\n📋 Step 4: Result analysis")
        
        expected_total_patches = teacher_sample["pixel_values"].shape[0] + student_sample["pixel_values"].shape[0]
        actual_total_patches = batch["pixel_values"].shape[0]
        
        print(f"   - Expected total patches: {expected_total_patches}")
        print(f"   - Actual total patches: {actual_total_patches}")
        print(f"   - Patch feature dimension: {batch['pixel_values'].shape[1]}")
        
        # CRITICAL ANALYSIS: This is where the bug manifests
        if batch["pixel_values"].dim() == 2:
            print("   🐛 BUG CONFIRMED: pixel_values is 2D (concatenated patches)")
            print("   📍 ROOT CAUSE: Collator concatenates patch tensors along dim 0")
            print("   💡 EXPECTED: Should maintain batch structure for model input")
        else:
            print("   ✅ pixel_values has correct dimensionality")
        
        # Verify the concatenation behavior
        assert actual_total_patches == expected_total_patches
        assert batch["pixel_values"].shape[1] == 1176  # Patch feature dimension
        
        print("   ✅ Collator processing analysis complete")
        
        return batch

    def test_model_input_format_validation(self, standard_collator, mock_tokenizer, mock_image_processor):
        """
        Test that the collator output matches expected model input format.
        
        This validates whether the current output is compatible with Qwen2.5-VL.
        """
        print("🔍 Testing model input format validation:")
        
        # Get processed batch
        batch = self.test_collator_processing_step_by_step(
            standard_collator, mock_tokenizer, mock_image_processor
        )
        
        # Check required fields for Qwen2.5-VL model
        required_fields = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
        
        print("\n📋 Model input validation:")
        for field in required_fields:
            assert field in batch, f"Missing required field: {field}"
            print(f"   ✅ {field}: {batch[field].shape}")
        
        # Check tensor types
        for field in required_fields:
            assert isinstance(batch[field], torch.Tensor), f"{field} should be tensor"
        
        # Check batch consistency
        batch_size = batch["input_ids"].shape[0]
        print(f"\n📋 Batch consistency (batch_size={batch_size}):")
        
        # Text tensors should have same batch size
        text_fields = ["input_ids", "attention_mask", "labels"]
        for field in text_fields:
            assert batch[field].shape[0] == batch_size, f"{field} batch size mismatch"
            print(f"   ✅ {field} batch size: {batch[field].shape[0]}")
        
        # Image tensors - this is where the issue might be
        print(f"   📍 pixel_values shape: {batch['pixel_values'].shape}")
        print(f"   📍 image_grid_thw shape: {batch['image_grid_thw'].shape}")
        
        # For Qwen2.5-VL, pixel_values in 2D patch format might be correct
        # But we need to verify this matches what the model expects
        if batch["pixel_values"].dim() == 2:
            print("   ⚠️  pixel_values is 2D - verify this matches Qwen2.5-VL expectations")
        
        print("   ✅ Model input format validation complete")


if __name__ == "__main__":
    # Run tests directly for debugging
    import sys
    sys.path.append("/data3/Qwen2.5-VL-main")
    
    test_instance = TestPipelineDataFlow()
    
    # Create fixtures manually
    mock_tokenizer = MockTokenizer(vocab_size=151665)
    mock_tokenizer.add_tokens(["<|line_start|>", "<|line_end|>"])
    
    mock_image_processor = MockImageProcessor()
    
    mock_config = Mock()
    mock_config.max_total_length = 4096
    mock_config.coordinate_tokens_enabled = True
    mock_config.max_coord_value = 2048
    
    collator = StandardDataCollator(
        tokenizer=mock_tokenizer,
        config=mock_config,
        pad_token_id=0,
        max_length=4096,
        label_pad_token_id=-100
    )
    
    print("🚀 Running pipeline data flow tests...")
    
    try:
        test_instance.test_raw_data_to_processed_sample_flow(mock_image_processor)
        test_instance.test_chat_processor_simulation(mock_tokenizer)
        test_instance.test_teacher_student_sample_creation(mock_tokenizer, mock_image_processor)
        test_instance.test_collator_processing_step_by_step(collator, mock_tokenizer, mock_image_processor)
        test_instance.test_model_input_format_validation(collator, mock_tokenizer, mock_image_processor)
        
        print("🎉 All pipeline data flow tests completed!")
        
    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
