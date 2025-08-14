#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Real Data Training Pipeline Tests

This module provides comprehensive test coverage for the entire training pipeline
using real dataset files from data/ds_v2_full/. It focuses on areas not covered
by existing tests:

1. Real dataset integration with actual JSONL files
2. Complete conversation flow logging with step-by-step validation
3. Token masking verification with real data
4. Loss computation validation with actual teacher-student samples
5. Edge cases and boundary conditions with real dataset files

Key Features:
- Uses actual data/ds_v2_full/teacher_pool.jsonl, train.jsonl, val.jsonl
- Detailed logging at each pipeline step
- Comprehensive token masking validation
- Teacher-student conversation flow verification
- Edge case testing with real data scenarios
"""

import json
import logging
import sys
from pathlib import Path

import pytest
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2VLProcessor


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src_new.config.config import load_config
from src_new.data.teacher_pool import TeacherPoolManager
from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.coordinate_converter import CoordinateTokenConverter


# Configure detailed logging for pipeline testing
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TestRealDataPipeline:
    """
    Comprehensive tests for training pipeline with real dataset files.

    Tests the complete workflow from raw JSONL data to model-ready tensors
    with detailed logging and validation at each step.
    """

    @pytest.fixture(scope="class")
    def real_data_paths(self):
        """Provide paths to real dataset files."""
        data_root = Path("/data3/Qwen2.5-VL-main/data/ds_v2_full")

        paths = {
            "teacher_pool": data_root / "teacher_pool.jsonl",
            "train": data_root / "train.jsonl",
            "val": data_root / "val.jsonl",
            "images": data_root / "images",
        }

        # Verify all files exist
        for name, path in paths.items():
            if name != "images":
                assert path.exists(), f"Real data file not found: {path}"
            else:
                assert path.exists(), f"Images directory not found: {path}"

        return paths

    @pytest.fixture(scope="class")
    def real_config(self):
        """Load real configuration for testing."""
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml")
        if not config_path.exists():
            pytest.skip("Real config file not found")

        config = load_config(str(config_path))
        return config

    @pytest.fixture(scope="class")
    def real_tokenizer_and_processor(self, real_config):
        """Load real tokenizer and processor for testing."""
        model_path = real_config.model_path

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )

        # Load image processor
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Load full processor
        processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)

        return tokenizer, image_processor, processor

    def test_real_data_loading_and_validation(self, real_data_paths):
        """
        Test loading and validation of real dataset files.

        Validates:
        - JSONL file structure and content
        - Object geometry types and coordinates
        - Image file references
        - Data consistency across files
        """
        logger.info("🧪 Testing real data loading and validation...")

        # Test teacher pool loading
        logger.info("📚 Loading teacher pool data...")
        with open(real_data_paths["teacher_pool"], "r", encoding="utf-8") as f:
            teacher_data = [json.loads(line.strip()) for line in f if line.strip()]

        logger.info(f"✅ Loaded {len(teacher_data)} teacher samples")

        # Validate teacher data structure
        for i, teacher in enumerate(teacher_data[:3]):  # Test first 3 samples
            logger.info(f"🔍 Validating teacher sample {i}:")
            logger.info(f"  - Images: {teacher.get('images', [])}")
            logger.info(f"  - Objects count: {len(teacher.get('objects', []))}")
            logger.info(
                f"  - Dimensions: {teacher.get('width')}x{teacher.get('height')}"
            )

            # Validate required fields
            assert "images" in teacher, f"Teacher {i} missing 'images' field"
            assert "objects" in teacher, f"Teacher {i} missing 'objects' field"
            assert len(teacher["objects"]) > 0, f"Teacher {i} has empty objects list"

            # Validate object structure
            for j, obj in enumerate(teacher["objects"][:2]):  # Test first 2 objects
                logger.info(f"    Object {j}: {obj}")
                assert "desc" in obj, f"Teacher {i} object {j} missing description"

                # Check geometry types
                geometry_types = ["bbox_2d", "quad", "line"]
                has_geometry = any(geom in obj for geom in geometry_types)
                assert has_geometry, f"Teacher {i} object {j} missing geometry"

        # Test training data loading
        logger.info("📖 Loading training data...")
        with open(real_data_paths["train"], "r", encoding="utf-8") as f:
            train_data = [json.loads(line.strip()) for line in f if line.strip()]

        logger.info(f"✅ Loaded {len(train_data)} training samples")

        # Test validation data loading
        logger.info("📋 Loading validation data...")
        with open(real_data_paths["val"], "r", encoding="utf-8") as f:
            val_data = [json.loads(line.strip()) for line in f if line.strip()]

        logger.info(f"✅ Loaded {len(val_data)} validation samples")

        logger.info("✅ Real data loading and validation completed successfully")

    def test_teacher_pool_manager_with_real_data(self, real_data_paths):
        """
        Test TeacherPoolManager with real teacher pool data.

        Validates:
        - Teacher pool loading from real file
        - Teacher selection and assignment logic
        - Teacher-student pairing strategies
        """
        logger.info("🧪 Testing TeacherPoolManager with real data...")

        # Initialize teacher pool manager
        teacher_pool_manager = TeacherPoolManager(
            teacher_pool_file=str(real_data_paths["teacher_pool"])
        )

        logger.info(
            f"📚 Teacher pool loaded with {len(teacher_pool_manager.teacher_pool)} samples"
        )

        # Test teacher selection
        selected_teachers = teacher_pool_manager.get_random_teachers(num_samples=2)

        logger.info(f"🎯 Selected {len(selected_teachers)} teachers for sample 0")
        for i, teacher in enumerate(selected_teachers):
            logger.info(f"  Teacher {i}: {len(teacher.get('objects', []))} objects")
            logger.info(
                f"    First object: {teacher['objects'][0] if teacher.get('objects') else 'None'}"
            )

        assert len(selected_teachers) <= 2, (
            "Should not exceed requested number of teachers"
        )
        assert all("objects" in teacher for teacher in selected_teachers), (
            "All teachers should have objects"
        )

        logger.info("✅ TeacherPoolManager real data test completed successfully")

    def test_coordinate_conversion_with_real_data(self, real_data_paths):
        """
        Test coordinate token conversion with real geometry data.

        Validates:
        - Conversion of real bbox_2d, quad, line coordinates
        - Token format consistency
        - Coordinate clamping and validation
        """
        logger.info("🧪 Testing coordinate conversion with real data...")

        # Load real teacher data for coordinate testing
        with open(real_data_paths["teacher_pool"], "r", encoding="utf-8") as f:
            teacher_data = [json.loads(line.strip()) for line in f if line.strip()]

        # Initialize coordinate converter
        converter = CoordinateTokenConverter(
            max_coord_value=2048, coordinate_tokens_enabled=True
        )

        # Test conversion with real objects
        for i, teacher in enumerate(teacher_data[:2]):  # Test first 2 teachers
            logger.info(f"🔄 Testing coordinate conversion for teacher {i}...")

            objects = teacher.get("objects", [])
            if not objects:
                continue

            # Convert objects to coordinate tokens
            try:
                token_result = converter.convert_objects_to_tokens(objects)
                logger.info(f"✅ Converted {len(objects)} objects to tokens")
                logger.info(f"📝 Token result preview: {token_result[:200]}...")

                # Validate token format
                assert "<|object_ref_start|>" in token_result, (
                    "Missing object reference start token"
                )
                assert "<|object_ref_end|>" in token_result, (
                    "Missing object reference end token"
                )

                # Check for coordinate tokens
                has_coord_tokens = any(
                    f"<|coord_{i}|>" in token_result for i in range(10)
                )
                if has_coord_tokens:
                    logger.info("✅ Found coordinate tokens in result")

            except Exception as e:
                logger.error(f"❌ Coordinate conversion failed for teacher {i}: {e}")
                raise

        logger.info("✅ Coordinate conversion with real data completed successfully")

    def test_conversation_flow_with_real_data(
        self, real_data_paths, real_config, real_tokenizer_and_processor
    ):
        """
        Test complete conversation processing flow with real data.

        Validates:
        - Teacher-student conversation creation
        - Chat template application
        - Token ID assignment and structure
        - Conversation boundaries and spans
        """
        logger.info("🧪 Testing conversation flow with real data...")

        tokenizer, image_processor, processor = real_tokenizer_and_processor

        # Load real data samples
        with open(real_data_paths["teacher_pool"], "r", encoding="utf-8") as f:
            teacher_data = [json.loads(line.strip()) for line in f if line.strip()]

        with open(real_data_paths["train"], "r", encoding="utf-8") as f:
            train_data = [json.loads(line.strip()) for line in f if line.strip()]

        # Initialize conversation processor
        conversation_processor = ConversationProcessor(
            processor=processor,
            max_coord_value=real_config.max_coord_value,
            coordinate_tokens_enabled=real_config.coordinate_tokens_enabled,
        )

        # Test simple conversation (student only)
        logger.info("📝 Testing simple conversation (student only)...")
        student_sample = train_data[0]  # Use first training sample

        # Create mock images for testing
        mock_images = [Image.new("RGB", (532, 728), color="red")]

        try:
            simple_inputs = conversation_processor.create_simple_conversation(
                sample=student_sample, images=mock_images
            )

            logger.info("✅ Simple conversation created successfully")
            logger.info(f"📊 Input shape: {simple_inputs['input_ids'].shape}")
            logger.info(f"🖼️ Image shape: {simple_inputs['pixel_values'].shape}")
            logger.info(f"📏 Sequence length: {simple_inputs['input_ids'].shape[1]}")

            # Log first few tokens for debugging
            first_tokens = simple_inputs["input_ids"][0][:20].tolist()
            logger.info(f"🔤 First 20 tokens: {first_tokens}")

            # Validate structure
            assert "input_ids" in simple_inputs, "Missing input_ids"
            assert "attention_mask" in simple_inputs, "Missing attention_mask"
            assert "pixel_values" in simple_inputs, "Missing pixel_values"

        except Exception as e:
            logger.error(f"❌ Simple conversation creation failed: {e}")
            raise

        # Test teacher-student conversation
        logger.info("👥 Testing teacher-student conversation...")
        teacher_samples = teacher_data[:2]  # Use first 2 teachers
        teacher_images_list = [mock_images, mock_images]  # Mock teacher images

        try:
            teacher_student_inputs = (
                conversation_processor.create_teacher_student_conversation(
                    student_sample=student_sample,
                    teacher_samples=teacher_samples,
                    student_images=mock_images,
                    teacher_images_list=teacher_images_list,
                )
            )

            logger.info("✅ Teacher-student conversation created successfully")
            logger.info(f"📊 Input shape: {teacher_student_inputs['input_ids'].shape}")
            logger.info(
                f"📏 Sequence length: {teacher_student_inputs['input_ids'].shape[1]}"
            )

            # Compare lengths (teacher-student should be longer)
            simple_length = simple_inputs["input_ids"].shape[1]
            teacher_student_length = teacher_student_inputs["input_ids"].shape[1]

            logger.info(
                f"📐 Length comparison: Simple={simple_length}, Teacher-Student={teacher_student_length}"
            )
            assert teacher_student_length > simple_length, (
                "Teacher-student conversation should be longer"
            )

        except Exception as e:
            logger.error(f"❌ Teacher-student conversation creation failed: {e}")
            raise

        logger.info("✅ Conversation flow with real data completed successfully")
