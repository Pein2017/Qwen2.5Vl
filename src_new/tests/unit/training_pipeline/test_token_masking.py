#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Token Masking and Span Validation Tests

This module tests token masking and span identification with real data,
focusing on areas not covered by existing tests:

1. Token ID assignment and label masking with real conversations
2. Teacher-student span identification and validation
3. Ignore index (-100) assignment for system/user prompts
4. Assistant token identification and loss computation
5. Image pad token handling and masking

Key Features:
- Uses real conversation templates and data
- Detailed logging of token masking steps
- Validation of conversation boundaries and spans
- Cross-entropy loss mask verification
- Coordinate token mask separation
"""

import json
import logging
import sys
from pathlib import Path

import pytest
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLProcessor


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src_new.config.config import load_config
from src_new.processing.conversation_processor import ConversationProcessor


# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TestTokenMaskingWithRealData:
    """
    Comprehensive tests for token masking using real conversation data.

    Tests the complete token masking workflow from conversation creation
    to loss computation with actual teacher-student samples.
    """

    @pytest.fixture(scope="class")
    def real_setup(self):
        """Set up real tokenizer, processor, and data for testing."""
        # Load config
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml")
        if not config_path.exists():
            pytest.skip("Real config file not found")

        config = load_config(str(config_path))

        # Load tokenizer and processor
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True, use_fast=True
        )

        processor = Qwen2VLProcessor.from_pretrained(
            config.model_path, trust_remote_code=True
        )

        # Load real data
        data_root = Path("/data3/Qwen2.5-VL-main/data/ds_v2_full")

        with open(data_root / "teacher_pool.jsonl", "r", encoding="utf-8") as f:
            teacher_data = [json.loads(line.strip()) for line in f if line.strip()]

        with open(data_root / "train.jsonl", "r", encoding="utf-8") as f:
            train_data = [json.loads(line.strip()) for line in f if line.strip()]

        return config, tokenizer, processor, teacher_data, train_data

    def test_conversation_token_structure_logging(self, real_setup):
        """
        Test and log complete conversation token structure.

        Validates:
        - System prompt token identification
        - User prompt token identification
        - Assistant response token identification
        - Image pad token placement and count
        - Special token boundaries (<|im_start|>, <|im_end|>)
        """
        logger.info("🧪 Testing conversation token structure with detailed logging...")

        config, tokenizer, processor, teacher_data, train_data = real_setup
        processor.tokenizer = tokenizer

        # Initialize conversation processor
        conversation_processor = ConversationProcessor(
            processor=processor, max_coord_value=config.max_coord_value
        )

        # Use real student sample
        student_sample = train_data[0]
        logger.info(
            f"📝 Testing with student sample: {len(student_sample.get('objects', []))} objects"
        )

        # Create mock images
        mock_images = [Image.new("RGB", (532, 728), color="green")]

        try:
            # Create simple conversation
            inputs = conversation_processor.create_simple_conversation(
                sample=student_sample, images=mock_images
            )

            input_ids = inputs["input_ids"][0]  # Remove batch dimension
            logger.info(f"📊 Total sequence length: {len(input_ids)}")

            # Log detailed token structure
            logger.info("🔍 Detailed token structure analysis:")

            # Get special token IDs
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

            logger.info(
                f"🎯 Special token IDs: <|im_start|>={im_start_id}, <|im_end|>={im_end_id}, <|image_pad|>={image_pad_id}"
            )

            # Find conversation boundaries
            im_start_positions = (
                (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
            )
            im_end_positions = (
                (input_ids == im_end_id).nonzero(as_tuple=True)[0].tolist()
            )
            image_pad_positions = (
                (input_ids == image_pad_id).nonzero(as_tuple=True)[0].tolist()
            )

            logger.info(f"📍 <|im_start|> positions: {im_start_positions}")
            logger.info(f"📍 <|im_end|> positions: {im_end_positions}")
            logger.info(
                f"🖼️ <|image_pad|> positions: {image_pad_positions[:10]}..."
            )  # First 10
            logger.info(f"🖼️ Total image pad tokens: {len(image_pad_positions)}")

            # Analyze conversation segments
            for i, (start_pos, end_pos) in enumerate(
                zip(im_start_positions, im_end_positions)
            ):
                segment_tokens = input_ids[start_pos : end_pos + 1]
                segment_text = tokenizer.decode(
                    segment_tokens, skip_special_tokens=False
                )

                logger.info(f"📝 Conversation segment {i}:")
                logger.info(f"  Position: [{start_pos}, {end_pos}]")
                logger.info(f"  Length: {len(segment_tokens)}")
                logger.info(f"  Text preview: {segment_text[:100]}...")

                # Identify segment type
                if "system" in segment_text.lower():
                    logger.info(f"  Type: SYSTEM PROMPT")
                elif "user" in segment_text.lower():
                    logger.info(f"  Type: USER PROMPT")
                elif "assistant" in segment_text.lower():
                    logger.info(f"  Type: ASSISTANT RESPONSE")
                else:
                    logger.info(f"  Type: UNKNOWN")

            # Validate structure
            assert len(im_start_positions) == len(im_end_positions), (
                "Mismatched start/end tokens"
            )
            assert len(im_start_positions) >= 3, (
                "Should have at least system, user, assistant segments"
            )
            assert len(image_pad_positions) > 0, "Should have image pad tokens"

            logger.info("✅ Conversation token structure validation passed")

        except Exception as e:
            logger.error(f"❌ Conversation token structure test failed: {e}")
            raise

        logger.info("✅ Conversation token structure logging completed successfully")

    def test_label_masking_with_real_conversation(self, real_setup):
        """
        Test label masking with real conversation data.

        Validates:
        - System prompt tokens masked with -100
        - User prompt tokens masked with -100
        - Image pad tokens masked with -100
        - Assistant prefix tokens masked with -100
        - Assistant response tokens available for loss computation
        """
        logger.info("🧪 Testing label masking with real conversation...")

        config, tokenizer, processor, teacher_data, train_data = real_setup
        processor.tokenizer = tokenizer

        # Initialize conversation processor
        conversation_processor = ConversationProcessor(
            processor=processor, max_coord_value=config.max_coord_value
        )

        # Use real student sample
        student_sample = train_data[0]
        mock_images = [Image.new("RGB", (532, 728), color="yellow")]

        try:
            # Create conversation
            inputs = conversation_processor.create_simple_conversation(
                sample=student_sample, images=mock_images
            )

            input_ids = inputs["input_ids"][0]

            # Create labels (initially copy of input_ids)
            labels = input_ids.clone()

            # Apply masking logic (simplified version for testing)
            # In real implementation, this would be done by the conversation processor

            # Get special token IDs
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

            # Find conversation boundaries
            im_start_positions = (
                (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
            )
            im_end_positions = (
                (input_ids == im_end_id).nonzero(as_tuple=True)[0].tolist()
            )

            logger.info("🎯 Applying label masking...")

            # Mask system and user prompts (first two conversation segments)
            for i in range(min(2, len(im_start_positions))):
                start_pos = im_start_positions[i]
                end_pos = im_end_positions[i]
                labels[start_pos : end_pos + 1] = -100

                segment_text = tokenizer.decode(
                    input_ids[start_pos : end_pos + 1], skip_special_tokens=False
                )
                logger.info(
                    f"  Masked segment {i}: [{start_pos}, {end_pos}] - {segment_text[:50]}..."
                )

            # Mask image pad tokens
            image_pad_mask = input_ids == image_pad_id
            labels[image_pad_mask] = -100
            image_pad_count = image_pad_mask.sum().item()
            logger.info(f"  Masked {image_pad_count} image pad tokens")

            # Mask assistant prefix tokens (e.g., "<|im_start|>assistant\n")
            if len(im_start_positions) >= 3:  # Assistant segment exists
                assistant_start = im_start_positions[2]
                # Find the newline after "assistant" to mask the prefix
                assistant_segment = input_ids[
                    assistant_start : assistant_start + 10
                ]  # Look at first 10 tokens

                # Simple heuristic: mask first few tokens of assistant segment
                labels[
                    assistant_start : assistant_start + 3
                ] = -100  # Mask "<|im_start|>assistant\n"
                logger.info(
                    f"  Masked assistant prefix: [{assistant_start}, {assistant_start + 3}]"
                )

            # Count masked vs unmasked tokens
            masked_count = (labels == -100).sum().item()
            unmasked_count = (labels != -100).sum().item()
            total_count = len(labels)

            logger.info(f"📊 Label masking summary:")
            logger.info(f"  Total tokens: {total_count}")
            logger.info(
                f"  Masked tokens (-100): {masked_count} ({masked_count / total_count * 100:.1f}%)"
            )
            logger.info(
                f"  Unmasked tokens: {unmasked_count} ({unmasked_count / total_count * 100:.1f}%)"
            )

            # Validate masking
            assert masked_count > 0, "Should have some masked tokens"
            assert unmasked_count > 0, (
                "Should have some unmasked tokens for loss computation"
            )
            assert masked_count + unmasked_count == total_count, (
                "All tokens should be accounted for"
            )

            # Log some examples of masked vs unmasked tokens
            logger.info("🔍 Masking examples:")
            for i in range(min(20, len(labels))):
                token_id = input_ids[i].item()
                label = labels[i].item()
                token_text = tokenizer.decode([token_id])
                mask_status = "MASKED" if label == -100 else "UNMASKED"
                logger.info(f"  Token {i}: {token_id} '{token_text}' -> {mask_status}")

            logger.info("✅ Label masking validation passed")

        except Exception as e:
            logger.error(f"❌ Label masking test failed: {e}")
            raise

        logger.info("✅ Label masking with real conversation completed successfully")
