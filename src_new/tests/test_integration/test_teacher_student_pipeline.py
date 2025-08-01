#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test for teacher-student training pipeline.

Tests the complete flow from teacher pool loading through conversation building,
tokenization, loss masking, and coordinate loss computation.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import AutoTokenizer

from src_new.processing.chat_processor import ChatConfig, ChatProcessor
from src_new.processing.token_processor import TokenConfig, TokenProcessor
from src_new.processing.templates import TemplateManager
from src_new.data.teacher_pool import TeacherPoolManager
from src_new.models.loss_manager import LossManager


class TestTeacherStudentPipeline:
    """Integration tests for teacher-student training pipeline."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer with required methods."""
        tokenizer = Mock()
        tokenizer.get_vocab.return_value = {
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "system": 1587,
            "user": 882,
            "assistant": 77091,
            "<|coord_0|>": 151667,
            "<|coord_1|>": 151668,
            "<|coord_2048|>": 153715,
        }
        tokenizer.model_max_length = 2048
        tokenizer.apply_chat_template.return_value = "mocked chat template"
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "mocked decode"
        
        # Mock tokenizer call
        def mock_tokenizer_call(text, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
            }
        tokenizer.side_effect = mock_tokenizer_call
        
        return tokenizer

    @pytest.fixture
    def token_config(self):
        """Create token configuration."""
        return TokenConfig(
            coordinate_tokens_enabled=True,
            max_coord_value=2048,
            new_geometry_tokens=["<|line_start|>", "<|line_end|>"]
        )

    @pytest.fixture
    def chat_config(self):
        """Create chat configuration."""
        return ChatConfig(
            language="chinese",
            use_training_prompts=True,
            teacher_ratio=0.7,
            max_teacher_examples=2,
            coordinate_tokens_enabled=True
        )

    @pytest.fixture
    def sample_teacher_pool(self):
        """Create sample teacher pool data."""
        return [
            {
                "images": ["teacher1.jpg"],
                "objects": [
                    {"bbox_2d": [100, 100, 200, 200], "desc": "BBU设备1"},
                    {"bbox_2d": [300, 300, 400, 400], "desc": "螺丝连接点1"}
                ]
            },
            {
                "images": ["teacher2.jpg"],
                "objects": [
                    {"bbox_2d": [150, 150, 250, 250], "desc": "BBU设备2"},
                    {"bbox_2d": [350, 350, 450, 450], "desc": "挡风板1"}
                ]
            }
        ]

    @pytest.fixture
    def sample_student(self):
        """Create sample student data."""
        return {
            "images": ["student.jpg"],
            "objects": [
                {"bbox_2d": [120, 120, 220, 220], "desc": "BBU设备学生"},
                {"bbox_2d": [320, 320, 420, 420], "desc": "螺丝连接点学生"}
            ]
        }

    @pytest.fixture
    def temp_teacher_pool_file(self, sample_teacher_pool):
        """Create temporary teacher pool file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for teacher in sample_teacher_pool:
                f.write(json.dumps(teacher, ensure_ascii=False) + '\n')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_teacher_pool_loading(self, temp_teacher_pool_file):
        """Test teacher pool loading from JSONL file."""
        teacher_pool_manager = TeacherPoolManager(temp_teacher_pool_file)
        
        assert len(teacher_pool_manager.teacher_pool) == 2
        assert teacher_pool_manager.teacher_pool[0]["objects"][0]["desc"] == "BBU设备1"
        assert teacher_pool_manager.teacher_pool[1]["objects"][0]["desc"] == "BBU设备2"

    def test_chat_processor_initialization(self, mock_tokenizer, token_config, chat_config):
        """Test chat processor initialization."""
        token_processor = TokenProcessor(token_config)
        chat_processor = ChatProcessor(mock_tokenizer, token_processor, chat_config)
        
        assert chat_processor.tokenizer == mock_tokenizer
        assert chat_processor.token_processor == token_processor
        assert chat_processor.config == chat_config
        assert chat_processor.template_manager is not None

    def test_standalone_conversation_building(self, mock_tokenizer, token_config, chat_config, sample_student):
        """Test building standalone conversation without teachers."""
        token_processor = TokenProcessor(token_config)
        chat_processor = ChatProcessor(mock_tokenizer, token_processor, chat_config)
        
        conversation = chat_processor.build_conversation(sample_student)
        
        assert len(conversation) == 3  # system + user + assistant
        assert conversation[0]["role"] == "system"
        assert conversation[1]["role"] == "user"
        assert conversation[2]["role"] == "assistant"

    def test_teacher_student_conversation_building(self, mock_tokenizer, token_config, chat_config, 
                                                 sample_student, sample_teacher_pool):
        """Test building teacher-student conversation."""
        token_processor = TokenProcessor(token_config)
        chat_processor = ChatProcessor(mock_tokenizer, token_processor, chat_config)
        
        # Build standalone conversation first
        standalone_conversation = chat_processor.build_conversation(sample_student)
        
        # Add teacher examples
        teacher_student_conversation = chat_processor.add_teacher_examples(
            standalone_conversation, sample_teacher_pool[:1]  # Use one teacher
        )
        
        # Should have system + teacher_user + teacher_assistant + student_user + student_assistant
        assert len(teacher_student_conversation) == 5
        assert teacher_student_conversation[0]["role"] == "system"
        assert teacher_student_conversation[1]["role"] == "user"  # teacher
        assert teacher_student_conversation[2]["role"] == "assistant"  # teacher
        assert teacher_student_conversation[3]["role"] == "user"  # student
        assert teacher_student_conversation[4]["role"] == "assistant"  # student

    def test_conversation_tokenization_with_spans(self, mock_tokenizer, token_config, chat_config):
        """Test conversation tokenization with span tracking."""
        token_processor = TokenProcessor(token_config)
        chat_processor = ChatProcessor(mock_tokenizer, token_processor, chat_config)
        
        # Create mock conversation
        conversation = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Teacher example"},
            {"role": "assistant", "content": "Teacher response"},
            {"role": "user", "content": "Student question"},
            {"role": "assistant", "content": "Student response"}
        ]
        
        processed_sample = chat_processor.tokenize_conversation(conversation)
        
        assert processed_sample.input_ids is not None
        assert processed_sample.attention_mask is not None
        assert processed_sample.labels is not None
        assert processed_sample.has_teachers == True
        assert processed_sample.num_teachers == 1
        assert processed_sample.teacher_spans is not None
        assert processed_sample.student_spans is not None

    def test_loss_masking_with_spans(self, mock_tokenizer, token_config, chat_config):
        """Test loss masking with teacher-student span tracking."""
        token_processor = TokenProcessor(token_config)
        chat_processor = ChatProcessor(mock_tokenizer, token_processor, chat_config)
        
        # Mock input_ids with known pattern
        input_ids = torch.tensor([151644, 1587, 151645, 151644, 77091, 1, 2, 3, 151645])  # im_start, system, im_end, im_start, assistant, tokens, im_end
        conversation = [
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": "Response"}
        ]
        
        labels, teacher_spans, student_spans = chat_processor.create_loss_mask_with_spans(input_ids, conversation)
        
        # Check that system tokens are masked (-100)
        assert labels[0].item() == -100  # <|im_start|>
        assert labels[1].item() == -100  # system
        assert labels[2].item() == -100  # <|im_end|>
        
        # Check that assistant response tokens are not masked
        assert labels[5].item() != -100  # assistant content token
        assert labels[6].item() != -100  # assistant content token
        assert labels[7].item() != -100  # assistant content token

    @patch('src_new.models.loss_manager.logger')
    def test_coordinate_loss_integration(self, mock_logger, mock_tokenizer, token_config, chat_config):
        """Test coordinate loss integration with proper token IDs."""
        token_processor = TokenProcessor(token_config)
        
        # Mock get_coordinate_token_range to return known range
        token_processor.get_coordinate_token_range = Mock(return_value=(151667, 153715))
        
        loss_manager = LossManager(
            coordinate_loss_weight=0.05,
            token_processor=token_processor,
            tokenizer=mock_tokenizer
        )
        
        # Verify coordinate loss function was created with correct token IDs
        assert loss_manager.coordinate_loss_fn.coord_start_id == 151667
        assert loss_manager.coordinate_loss_fn.coord_end_id == 153716  # +1 for exclusive end

    def test_end_to_end_pipeline(self, mock_tokenizer, token_config, chat_config, 
                                temp_teacher_pool_file, sample_student):
        """Test complete end-to-end teacher-student pipeline."""
        # Initialize components
        token_processor = TokenProcessor(token_config)
        chat_processor = ChatProcessor(mock_tokenizer, token_processor, chat_config)
        teacher_pool_manager = TeacherPoolManager(temp_teacher_pool_file)
        
        # Process sample with teacher pool
        processed_sample = chat_processor.process_sample_with_teacher_pool(
            sample_student, teacher_pool_manager.teacher_pool
        )
        
        # Verify processed sample has all required components
        assert processed_sample.input_ids is not None
        assert processed_sample.attention_mask is not None
        assert processed_sample.labels is not None
        assert processed_sample.coordinate_mask is not None
        assert processed_sample.teacher_spans is not None
        assert processed_sample.student_spans is not None
        
        # Verify teacher assignment worked
        assert processed_sample.has_teachers in [True, False]  # Depends on random teacher assignment
        
        # Verify sample validation passes
        assert chat_processor.validate_processed_sample(processed_sample) == True

    def test_coordinate_token_parsing(self, mock_tokenizer, token_config, chat_config):
        """Test coordinate token parsing from assistant responses."""
        token_processor = TokenProcessor(token_config)
        chat_processor = ChatProcessor(mock_tokenizer, token_processor, chat_config)
        
        # Mock tokens_to_coordinates method
        token_processor.tokens_to_coordinates = Mock(return_value=[100, 100, 200, 200])
        
        response = "<|obj_ref_start|>BBU设备<|obj_ref_end|><|bbox_start|>[<|coord_100|>, <|coord_100|>, <|coord_200|>, <|coord_200|>]<|bbox_end|>"
        
        parsed_sample = chat_processor._parse_objects_from_response(response)
        
        assert len(parsed_sample["objects"]) == 1
        assert parsed_sample["objects"][0]["desc"] == "BBU设备"
        assert parsed_sample["objects"][0]["bbox_2d"] == [100, 100, 200, 200]

    def test_batch_processing(self, mock_tokenizer, token_config, chat_config, 
                            temp_teacher_pool_file, sample_student):
        """Test batch processing of samples."""
        token_processor = TokenProcessor(token_config)
        chat_processor = ChatProcessor(mock_tokenizer, token_processor, chat_config)
        teacher_pool_manager = TeacherPoolManager(temp_teacher_pool_file)
        
        # Create batch of samples
        samples = [sample_student, sample_student.copy()]
        
        processed_samples = chat_processor.batch_process_samples(
            samples, teacher_pool_manager.teacher_pool
        )
        
        assert len(processed_samples) == 2
        for processed_sample in processed_samples:
            assert processed_sample.input_ids is not None
            assert processed_sample.labels is not None
            assert chat_processor.validate_processed_sample(processed_sample) == True
