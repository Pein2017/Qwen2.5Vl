#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Training Pipeline Integration Test

This module provides a comprehensive end-to-end test of the entire training
pipeline using real dataset files. It validates the complete workflow from
raw JSONL data to model-ready tensors with detailed step-by-step logging.

Key Features:
- End-to-end pipeline validation with real data
- Step-by-step logging of all major operations
- Complete conversation flow verification
- Token masking and loss computation validation
- Performance and memory usage monitoring
- Integration with actual BBU training components

This test serves as the ultimate validation that all pipeline components
work together correctly with real production data.
"""

import json
import logging
import sys
import time
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2VLProcessor


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src_new.config.config import load_config
from src_new.data.teacher_pool import TeacherPoolManager
from src_new.models.loss_manager import LossManager
from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.token_processor import TokenConfig, TokenProcessor


# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,  # Use INFO level to reduce noise in complete pipeline test
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TestCompletePipeline:
    """
    Complete end-to-end training pipeline test with real data.

    This test validates the entire workflow from raw dataset files
    to model-ready tensors, ensuring all components work together.
    """

    @pytest.fixture(scope="class")
    def pipeline_setup(self):
        """Set up complete pipeline components with real data."""
        logger.info("🚀 Setting up complete pipeline test environment...")

        # Load real configuration
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml")
        if not config_path.exists():
            pytest.skip("Real config file not found")

        config = load_config(str(config_path))
        logger.info(f"✅ Loaded config: {config.model_size} model")

        # Verify data files exist
        data_root = Path("/data3/Qwen2.5-VL-main/data/ds_v2_full")
        required_files = ["teacher_pool.jsonl", "train.jsonl", "val.jsonl"]

        for file_name in required_files:
            file_path = data_root / file_name
            assert file_path.exists(), f"Required data file not found: {file_path}"

        logger.info(f"✅ Verified data files in {data_root}")

        # Load tokenizer and processors
        logger.info("🔧 Loading tokenizer and processors...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True, use_fast=True
        )

        image_processor = Qwen2VLImageProcessor.from_pretrained(
            config.model_path, trust_remote_code=True
        )

        processor = Qwen2VLProcessor.from_pretrained(
            config.model_path, trust_remote_code=True
        )

        logger.info(f"✅ Loaded processors (vocab_size: {len(tokenizer)})")

        return config, tokenizer, image_processor, processor, data_root

    def test_complete_pipeline_workflow(self, pipeline_setup):
        """
        Test the complete training pipeline workflow.

        Steps:
        1. Load and validate real dataset files
        2. Initialize teacher pool manager
        3. Create dataset with conversation processor
        4. Process samples through complete pipeline
        5. Validate token masking and loss computation
        6. Monitor performance and memory usage
        """
        logger.info("🧪 Testing complete pipeline workflow...")

        config, tokenizer, image_processor, processor, data_root = pipeline_setup

        # Step 1: Load and validate dataset files
        logger.info("📚 Step 1: Loading and validating dataset files...")
        start_time = time.time()

        # Load teacher pool
        teacher_pool_manager = TeacherPoolManager(
            teacher_pool_file=str(data_root / "teacher_pool.jsonl")
        )

        logger.info(
            f"✅ Loaded teacher pool: {len(teacher_pool_manager.teacher_pool)} samples"
        )

        # Load training data sample for testing
        with open(data_root / "train.jsonl", "r", encoding="utf-8") as f:
            train_samples = [json.loads(line.strip()) for line in f if line.strip()][
                :5
            ]  # Test first 5

        logger.info(
            f"✅ Loaded training samples: {len(train_samples)} samples for testing"
        )
        load_time = time.time() - start_time
        logger.info(f"⏱️ Data loading time: {load_time:.2f}s")

        # Step 2: Initialize processing components
        logger.info("🔧 Step 2: Initializing processing components...")
        start_time = time.time()

        # Create token processor (no runtime extension; tokenizer already expanded)
        token_config = TokenConfig(
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
            max_coord_value=config.max_coord_value,
            new_geometry_tokens=[],
        )
        token_processor = TokenProcessor(token_config)

        # Update processor with pre-expanded tokenizer
        processor.tokenizer = tokenizer

        # Create conversation processor
        conversation_processor = ConversationProcessor(
            processor=processor, max_coord_value=config.max_coord_value
        )

        # Create loss manager
        loss_manager = LossManager(
            config=config, token_processor=token_processor, tokenizer=tokenizer
        )

        setup_time = time.time() - start_time
        logger.info(f"✅ Processing components initialized")
        logger.info(f"⏱️ Setup time: {setup_time:.2f}s")

        # Step 3: Process samples through complete pipeline
        logger.info("🔄 Step 3: Processing samples through complete pipeline...")

        processed_samples = []
        total_process_time = 0

        for i, sample in enumerate(train_samples):
            logger.info(f"📝 Processing sample {i + 1}/{len(train_samples)}...")
            start_time = time.time()

            try:
                # Create mock images (in real scenario, would load actual images)
                mock_images = [Image.new("RGB", (532, 728), color="blue")]

                # Decide whether to use teachers
                use_teachers = i % 2 == 0  # Use teachers for every other sample

                if use_teachers:
                    # Get teachers for this sample
                    teachers = teacher_pool_manager.get_random_teachers(num_samples=2)

                    if teachers:
                        # Create teacher-student conversation
                        teacher_images_list = [mock_images] * len(teachers)

                        inputs = (
                            conversation_processor.create_teacher_student_conversation(
                                student_sample=sample,
                                teacher_samples=teachers,
                                student_images=mock_images,
                                teacher_images_list=teacher_images_list,
                            )
                        )

                        logger.info(
                            f"  ✅ Created teacher-student conversation (teachers: {len(teachers)})"
                        )
                    else:
                        # Fallback to simple conversation
                        inputs = conversation_processor.create_simple_conversation(
                            sample=sample, images=mock_images
                        )
                        logger.info(
                            f"  ✅ Created simple conversation (no teachers available)"
                        )
                else:
                    # Create simple conversation
                    inputs = conversation_processor.create_simple_conversation(
                        sample=sample, images=mock_images
                    )
                    logger.info(f"  ✅ Created simple conversation")

                # Validate inputs structure
                required_keys = ["input_ids", "attention_mask", "pixel_values"]
                for key in required_keys:
                    assert key in inputs, f"Missing required key: {key}"

                # Log input statistics
                seq_len = inputs["input_ids"].shape[1]
                logger.info(f"  📊 Sequence length: {seq_len}")
                logger.info(f"  🖼️ Image shape: {inputs['pixel_values'].shape}")

                # Create coordinate mask for testing
                coord_mask = token_processor.create_coordinate_mask(
                    inputs["input_ids"][0], tokenizer
                )
                coord_count = coord_mask.sum().item()
                logger.info(f"  🎯 Coordinate tokens: {coord_count}")

                # Store processed sample
                processed_samples.append(
                    {
                        "inputs": inputs,
                        "coord_mask": coord_mask,
                        "sample_index": i,
                        "used_teachers": use_teachers,
                    }
                )

                process_time = time.time() - start_time
                total_process_time += process_time
                logger.info(f"  ⏱️ Processing time: {process_time:.2f}s")

            except Exception as e:
                logger.error(f"  ❌ Failed to process sample {i}: {e}")
                raise

        avg_process_time = total_process_time / len(train_samples)
        logger.info(f"✅ Processed {len(processed_samples)} samples successfully")
        logger.info(f"⏱️ Average processing time: {avg_process_time:.2f}s per sample")

        # Step 4: Validate loss computation
        logger.info("🧮 Step 4: Validating loss computation...")

        # Test loss computation with a processed sample
        test_sample = processed_samples[0]
        inputs = test_sample["inputs"]

        # Create mock model output for loss testing
        batch_size, seq_len = inputs["input_ids"].shape
        vocab_size = len(tokenizer)

        mock_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        mock_labels = inputs["input_ids"].clone()

        # Apply basic label masking (system/user prompts)
        mock_labels[:, :10] = -100  # Mask first 10 tokens

        try:
            # Compute loss
            loss_components = loss_manager.compute_loss_components(
                logits=mock_logits, labels=mock_labels
            )

            # Use the total loss from loss components
            total_loss = loss_components.loss

            logger.info(f"✅ Loss computation successful: {total_loss.item():.4f}")

            # Validate loss properties
            assert torch.isfinite(total_loss), "Loss should be finite"
            assert total_loss.item() >= 0, "Loss should be non-negative"

        except Exception as e:
            logger.error(f"❌ Loss computation failed: {e}")
            raise

        # Step 5: Performance summary
        logger.info("📊 Step 5: Performance summary...")

        total_tokens = sum(
            sample["inputs"]["input_ids"].shape[1] for sample in processed_samples
        )
        total_coord_tokens = sum(
            sample["coord_mask"].sum().item() for sample in processed_samples
        )

        logger.info(f"📈 Pipeline performance summary:")
        logger.info(f"  Samples processed: {len(processed_samples)}")
        logger.info(f"  Total tokens: {total_tokens}")
        logger.info(f"  Total coordinate tokens: {total_coord_tokens}")
        logger.info(
            f"  Average tokens per sample: {total_tokens / len(processed_samples):.1f}"
        )
        logger.info(
            f"  Coordinate token ratio: {total_coord_tokens / total_tokens * 100:.2f}%"
        )
        logger.info(f"  Total processing time: {total_process_time:.2f}s")
        logger.info(
            f"  Throughput: {len(processed_samples) / total_process_time:.2f} samples/s"
        )

        logger.info("✅ Complete pipeline workflow test completed successfully!")

        return processed_samples
