#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Conversation Flow Logging Tests

This module provides detailed logging tests for the training pipeline conversation flow,
showing the complete processing stages from raw conversation templates to token indices.

Key Features:
- Stage 1: Raw conversation template logging (pre-tokenization)
- Stage 2: Token index analysis with teacher/student span identification
- Stage 3: Loss mask validation for dual-loss system
- Uses real data from data/ds_v2_full/ for realistic validation
- Detailed step-by-step logging with clear section headers

Test Coverage:
1. Simple conversation flow (student-only)
2. Teacher-student conversation flow (multi-turn)
3. Token span identification and validation
4. Cross-entropy loss mask verification
5. Coordinate token L1 loss mask verification
"""

import json
import logging
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLProcessor


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src_new.config.config import load_config
from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.token_processor import TokenConfig, TokenProcessor


# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TestConversationFlowLogging:
    """
    Comprehensive tests for conversation flow with detailed logging.

    Tests the complete workflow from conversation creation to token analysis
    with step-by-step validation and logging.
    """

    @pytest.fixture(scope="class")
    def conversation_setup(self):
        """Set up conversation processor and real data for testing."""
        # Load config
        config_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_debug.yaml")
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

        # Create token processor and extend tokenizer
        token_config = TokenConfig(
            coordinate_tokens_enabled=config.coordinate_tokens_enabled,
            max_coord_value=config.max_coord_value,
            new_geometry_tokens=[],
        )
        token_processor = TokenProcessor(token_config)

        if config.coordinate_tokens_enabled:
            original_vocab_size = len(tokenizer)
            token_processor.extend_tokenizer_vocabulary(tokenizer)
            logger.info(
                f"🔧 Extended tokenizer vocabulary: {original_vocab_size} -> {len(tokenizer)}"
            )

        # Update processor with extended tokenizer
        processor.tokenizer = tokenizer

        # Create conversation processor
        conversation_processor = ConversationProcessor(
            processor=processor, max_coord_value=config.max_coord_value
        )

        # Load real data
        data_root = Path("/data3/Qwen2.5-VL-main/data/ds_v2_full")

        with open(data_root / "teacher_pool.jsonl", "r", encoding="utf-8") as f:
            teacher_data = [json.loads(line.strip()) for line in f if line.strip()]

        with open(data_root / "train.jsonl", "r", encoding="utf-8") as f:
            train_data = [json.loads(line.strip()) for line in f if line.strip()]

        return (
            conversation_processor,
            tokenizer,
            token_processor,
            config,
            teacher_data,
            train_data,
        )

    def test_simple_conversation_flow_logging(self, conversation_setup):
        """
        Test simple conversation flow with comprehensive logging.

        Logs:
        - Stage 1: Raw conversation template (pre-tokenization)
        - Stage 2: Token index analysis
        - Stage 3: Loss mask validation
        """
        logger.info("🧪 Testing Simple Conversation Flow with Comprehensive Logging")
        logger.info("=" * 80)

        (
            conversation_processor,
            tokenizer,
            token_processor,
            config,
            teacher_data,
            train_data,
        ) = conversation_setup

        # Use real student sample
        student_sample = train_data[0]
        logger.info(
            f"📝 Using student sample with {len(student_sample.get('objects', []))} objects"
        )

        # Create mock images
        mock_images = [Image.new("RGB", (532, 728), color="blue")]

        # ===== STAGE 1: RAW CONVERSATION TEMPLATE LOGGING =====
        logger.info("\n🎯 STAGE 1: RAW CONVERSATION TEMPLATE (PRE-TOKENIZATION)")
        logger.info("-" * 60)

        # Get the raw conversation text before tokenization
        # We need to manually build the conversation to see the raw template
        from src_new.processing.templates import CONSTANTS

        system_prompt = CONSTANTS["SYSTEM_PROMPT"]
        student_prompt = CONSTANTS["STUDENT_USER_PROMPT"]

        # Convert objects to coordinate tokens
        objects = student_sample.get("objects", [])
        coordinate_response = (
            conversation_processor.coordinate_converter.convert_objects_to_tokens(
                objects
            )
        )

        # Build conversation messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": student_prompt},
                    {"type": "image"},
                ],
            },
            {"role": "assistant", "content": coordinate_response},
        ]

        # Get raw conversation text using apply_chat_template
        raw_conversation_text = conversation_processor.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        logger.info("📄 Raw Conversation Template:")
        logger.info("```")
        logger.info(raw_conversation_text)
        logger.info("```")

        # Log conversation structure analysis
        logger.info("\n🔍 Conversation Structure Analysis:")
        lines = raw_conversation_text.split("\n")
        for i, line in enumerate(lines):
            if "<|im_start|>" in line:
                logger.info(f"  Line {i:2d}: {line} ← CONVERSATION START")
            elif "<|im_end|>" in line:
                logger.info(f"  Line {i:2d}: {line} ← CONVERSATION END")
            elif "<|image_pad|>" in line:
                image_pad_count = line.count("<|image_pad|>")
                logger.info(
                    f"  Line {i:2d}: [IMAGE PADS: {image_pad_count}] ← IMAGE TOKENS"
                )
            elif "<|coord_" in line:
                coord_count = line.count("<|coord_")
                logger.info(
                    f"  Line {i:2d}: [COORD TOKENS: {coord_count}] ← COORDINATE TOKENS"
                )
            elif line.strip():
                logger.info(
                    f"  Line {i:2d}: {line[:50]}{'...' if len(line) > 50 else ''}"
                )

        # ===== STAGE 2: TOKEN INDEX ANALYSIS =====
        logger.info("\n🎯 STAGE 2: TOKEN INDEX ANALYSIS")
        logger.info("-" * 60)

        # Create the actual conversation inputs
        inputs = conversation_processor.create_simple_conversation(
            sample=student_sample, images=mock_images
        )

        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        logger.info(f"📊 Total sequence length: {len(input_ids)}")

        # Get special token IDs
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

        logger.info(f"🎯 Special token IDs:")
        logger.info(f"  <|im_start|> = {im_start_id}")
        logger.info(f"  <|im_end|> = {im_end_id}")
        logger.info(f"  <|image_pad|> = {image_pad_id}")

        # Find conversation boundaries
        im_start_positions = (
            (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        )
        im_end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0].tolist()
        image_pad_positions = (
            (input_ids == image_pad_id).nonzero(as_tuple=True)[0].tolist()
        )

        logger.info(f"\n📍 Conversation Boundaries:")
        logger.info(f"  <|im_start|> positions: {im_start_positions}")
        logger.info(f"  <|im_end|> positions: {im_end_positions}")
        logger.info(
            f"  <|image_pad|> positions: {image_pad_positions[:10]}{'...' if len(image_pad_positions) > 10 else ''}"
        )
        logger.info(f"  Total image pad tokens: {len(image_pad_positions)}")

        # Analyze conversation segments
        logger.info(f"\n🔍 Conversation Segment Analysis:")
        for i, (start_pos, end_pos) in enumerate(
            zip(im_start_positions, im_end_positions)
        ):
            segment_tokens = input_ids[start_pos : end_pos + 1]
            segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=False)

            # Identify segment type
            if "system" in segment_text.lower():
                segment_type = "SYSTEM PROMPT"
            elif "user" in segment_text.lower():
                segment_type = "USER PROMPT"
            elif "assistant" in segment_text.lower():
                segment_type = "ASSISTANT RESPONSE"
            else:
                segment_type = "UNKNOWN"

            logger.info(
                f"  Segment {i}: [{start_pos:3d}, {end_pos:3d}] ({end_pos - start_pos + 1:3d} tokens) - {segment_type}"
            )
            logger.info(
                f"    Preview: {segment_text[:80]}{'...' if len(segment_text) > 80 else ''}"
            )

        # ===== STAGE 3: LOSS MASK VALIDATION =====
        logger.info("\n🎯 STAGE 3: LOSS MASK VALIDATION")
        logger.info("-" * 60)

        # Create coordinate mask
        coord_mask = token_processor.create_coordinate_mask(input_ids, tokenizer)
        coord_count = coord_mask.sum().item()

        logger.info(f"🎯 Coordinate Token Analysis:")
        logger.info(f"  Total coordinate tokens found: {coord_count}")

        if coord_count > 0:
            coord_positions = torch.where(coord_mask)[0].tolist()
            logger.info(
                f"  Coordinate token positions: {coord_positions[:10]}{'...' if len(coord_positions) > 10 else ''}"
            )

            # Show coordinate token values
            coord_token_ids = input_ids[coord_mask].tolist()
            logger.info(
                f"  Coordinate token IDs: {coord_token_ids[:10]}{'...' if len(coord_token_ids) > 10 else ''}"
            )

        # Simulate label masking for cross-entropy loss
        labels = input_ids.clone()

        # Mask system and user prompts (first two conversation segments)
        for i in range(min(2, len(im_start_positions))):
            start_pos = im_start_positions[i]
            end_pos = im_end_positions[i]
            labels[start_pos : end_pos + 1] = -100

        # Mask image pad tokens
        image_pad_mask = input_ids == image_pad_id
        labels[image_pad_mask] = -100

        # Count masked vs unmasked tokens
        masked_count = (labels == -100).sum().item()
        unmasked_count = (labels != -100).sum().item()
        total_count = len(labels)

        logger.info(f"\n📊 Cross-Entropy Loss Mask Summary:")
        logger.info(f"  Total tokens: {total_count}")
        logger.info(
            f"  Masked tokens (-100): {masked_count} ({masked_count / total_count * 100:.1f}%)"
        )
        logger.info(
            f"  Unmasked tokens: {unmasked_count} ({unmasked_count / total_count * 100:.1f}%)"
        )

        # Show token-by-token analysis for first 30 tokens
        logger.info(f"\n🔍 Token-by-Token Analysis (first 30 tokens):")
        for i in range(min(30, len(input_ids))):
            token_id = input_ids[i].item()
            label = labels[i].item()
            token_text = tokenizer.decode([token_id])
            mask_status = "MASKED" if label == -100 else "UNMASKED"
            coord_status = "COORD" if coord_mask[i] else ""

            logger.info(
                f"  Token {i:2d}: {token_id:6d} '{token_text:15s}' -> {mask_status:8s} {coord_status}"
            )

        # Validate results
        assert len(im_start_positions) == len(im_end_positions), (
            "Mismatched start/end tokens"
        )
        assert len(im_start_positions) >= 3, (
            "Should have at least system, user, assistant segments"
        )
        assert len(image_pad_positions) > 0, "Should have image pad tokens"
        assert masked_count > 0, "Should have some masked tokens"
        assert unmasked_count > 0, (
            "Should have some unmasked tokens for loss computation"
        )

        logger.info("\n✅ Simple conversation flow logging completed successfully!")
        logger.info("=" * 80)

    def test_teacher_student_conversation_flow_logging(self, conversation_setup):
        """
        Test teacher-student conversation flow with comprehensive logging.

        Logs:
        - Stage 1: Multi-turn conversation template (pre-tokenization)
        - Stage 2: Teacher/student span identification
        - Stage 3: Dual-loss mask validation
        """
        logger.info(
            "🧪 Testing Teacher-Student Conversation Flow with Comprehensive Logging"
        )
        logger.info("=" * 80)

        (
            conversation_processor,
            tokenizer,
            token_processor,
            config,
            teacher_data,
            train_data,
        ) = conversation_setup

        # Use real data
        student_sample = train_data[0]
        teacher_samples = teacher_data[:2]  # Use first 2 teachers

        logger.info(
            f"📝 Using student sample with {len(student_sample.get('objects', []))} objects"
        )
        logger.info(f"📚 Using {len(teacher_samples)} teacher samples")

        # Create mock images
        mock_student_images = [Image.new("RGB", (532, 728), color="green")]
        mock_teacher_images_list = [
            [Image.new("RGB", (532, 728), color="red")] for _ in teacher_samples
        ]

        # ===== STAGE 1: MULTI-TURN CONVERSATION TEMPLATE LOGGING =====
        logger.info("\n🎯 STAGE 1: MULTI-TURN CONVERSATION TEMPLATE (PRE-TOKENIZATION)")
        logger.info("-" * 60)

        # Build multi-turn conversation manually to see the raw template
        from src_new.processing.templates import CONSTANTS

        system_prompt = CONSTANTS["SYSTEM_PROMPT"]
        teacher_prompt = CONSTANTS["TEACHER_USER_PROMPT"]
        student_prompt = CONSTANTS["STUDENT_USER_PROMPT"]

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add teacher examples
        for i, teacher_sample in enumerate(teacher_samples):
            teacher_objects = teacher_sample.get("objects", [])
            teacher_response = (
                conversation_processor.coordinate_converter.convert_objects_to_tokens(
                    teacher_objects
                )
            )

            # Teacher user message
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": teacher_prompt},
                        {"type": "image"},
                    ],
                }
            )

            # Teacher assistant response
            messages.append({"role": "assistant", "content": teacher_response})

        # Add student query
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": student_prompt},
                    {"type": "image"},
                ],
            }
        )

        # Get raw conversation text
        raw_conversation_text = conversation_processor.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logger.info("📄 Multi-Turn Conversation Template:")
        logger.info("```")
        logger.info(raw_conversation_text)
        logger.info("```")

        # Analyze conversation structure
        logger.info("\n🔍 Multi-Turn Conversation Structure Analysis:")
        lines = raw_conversation_text.split("\n")
        conversation_count = 0
        for i, line in enumerate(lines):
            if "<|im_start|>" in line:
                conversation_count += 1
                if "system" in line:
                    logger.info(f"  Line {i:2d}: {line} ← SYSTEM PROMPT")
                elif "user" in line:
                    logger.info(
                        f"  Line {i:2d}: {line} ← USER PROMPT #{conversation_count - 1}"
                    )
                elif "assistant" in line:
                    logger.info(
                        f"  Line {i:2d}: {line} ← ASSISTANT RESPONSE #{conversation_count - 1}"
                    )
            elif "<|im_end|>" in line:
                logger.info(f"  Line {i:2d}: {line} ← CONVERSATION END")
            elif "<|image_pad|>" in line:
                image_pad_count = line.count("<|image_pad|>")
                logger.info(
                    f"  Line {i:2d}: [IMAGE PADS: {image_pad_count}] ← IMAGE TOKENS"
                )
            elif "<|coord_" in line:
                coord_count = line.count("<|coord_")
                logger.info(
                    f"  Line {i:2d}: [COORD TOKENS: {coord_count}] ← COORDINATE TOKENS"
                )
            elif line.strip() and len(line.strip()) > 5:
                logger.info(
                    f"  Line {i:2d}: {line[:60]}{'...' if len(line) > 60 else ''}"
                )

        logger.info(f"\n📊 Conversation Statistics:")
        logger.info(f"  Total conversation turns: {conversation_count}")
        logger.info(
            f"  Expected structure: 1 system + {len(teacher_samples) * 2} teacher + 1 student = {1 + len(teacher_samples) * 2 + 1} turns"
        )

        # ===== STAGE 2: TEACHER/STUDENT SPAN IDENTIFICATION =====
        logger.info("\n🎯 STAGE 2: TEACHER/STUDENT SPAN IDENTIFICATION")
        logger.info("-" * 60)

        # Create the actual conversation inputs
        inputs = conversation_processor.create_teacher_student_conversation(
            student_sample=student_sample,
            teacher_samples=teacher_samples,
            student_images=mock_student_images,
            teacher_images_list=mock_teacher_images_list,
        )

        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        logger.info(f"📊 Total sequence length: {len(input_ids)}")

        # Get special token IDs
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # Find conversation boundaries
        im_start_positions = (
            (input_ids == im_start_id).nonzero(as_tuple=True)[0].tolist()
        )
        im_end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0].tolist()

        logger.info(f"\n📍 Conversation Boundaries:")
        logger.info(f"  <|im_start|> positions: {im_start_positions}")
        logger.info(f"  <|im_end|> positions: {im_end_positions}")

        # Identify teacher and student spans
        teacher_spans = []
        student_spans = []

        logger.info(f"\n🔍 Teacher/Student Span Analysis:")
        for i, (start_pos, end_pos) in enumerate(
            zip(im_start_positions, im_end_positions)
        ):
            segment_tokens = input_ids[start_pos : end_pos + 1]
            segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=False)

            # Identify segment type
            if "system" in segment_text.lower():
                segment_type = "SYSTEM PROMPT"
            elif "user" in segment_text.lower():
                segment_type = "USER PROMPT"
            elif "assistant" in segment_text.lower():
                # Determine if this is teacher or student assistant response
                if i <= len(teacher_samples) * 2:  # Teacher responses come first
                    segment_type = "TEACHER ASSISTANT RESPONSE"
                    teacher_spans.append((start_pos, end_pos))
                else:
                    segment_type = "STUDENT ASSISTANT RESPONSE"
                    student_spans.append((start_pos, end_pos))
            else:
                segment_type = "UNKNOWN"

            logger.info(
                f"  Segment {i}: [{start_pos:3d}, {end_pos:3d}] ({end_pos - start_pos + 1:3d} tokens) - {segment_type}"
            )
            logger.info(
                f"    Preview: {segment_text[:80]}{'...' if len(segment_text) > 80 else ''}"
            )

        logger.info(f"\n🎯 Span Summary:")
        logger.info(f"  Teacher spans: {teacher_spans}")
        logger.info(f"  Student spans: {student_spans}")

        # ===== STAGE 3: DUAL-LOSS MASK VALIDATION =====
        logger.info("\n🎯 STAGE 3: DUAL-LOSS MASK VALIDATION")
        logger.info("-" * 60)

        # Create coordinate mask
        coord_mask = token_processor.create_coordinate_mask(input_ids, tokenizer)
        coord_count = coord_mask.sum().item()

        logger.info(f"🎯 Coordinate Token Analysis:")
        logger.info(f"  Total coordinate tokens found: {coord_count}")

        # Create cross-entropy loss mask
        labels = input_ids.clone()

        # Mask system prompts and user prompts (keep only assistant responses)
        for i, (start_pos, end_pos) in enumerate(
            zip(im_start_positions, im_end_positions)
        ):
            segment_tokens = input_ids[start_pos : end_pos + 1]
            segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=False)

            if "assistant" not in segment_text.lower():
                # Mask non-assistant segments
                labels[start_pos : end_pos + 1] = -100

        # Mask image pad tokens
        image_pad_mask = input_ids == image_pad_id
        labels[image_pad_mask] = -100

        # Count tokens by type
        masked_count = (labels == -100).sum().item()
        unmasked_count = (labels != -100).sum().item()
        coord_and_unmasked = (coord_mask & (labels != -100)).sum().item()

        logger.info(f"\n📊 Dual-Loss Mask Summary:")
        logger.info(f"  Total tokens: {len(input_ids)}")
        logger.info(
            f"  Cross-entropy masked tokens (-100): {masked_count} ({masked_count / len(input_ids) * 100:.1f}%)"
        )
        logger.info(
            f"  Cross-entropy unmasked tokens: {unmasked_count} ({unmasked_count / len(input_ids) * 100:.1f}%)"
        )
        logger.info(
            f"  Coordinate tokens (L1 loss): {coord_count} ({coord_count / len(input_ids) * 100:.1f}%)"
        )
        logger.info(f"  Coordinate tokens in unmasked region: {coord_and_unmasked}")

        # Validate results
        assert len(teacher_spans) == len(teacher_samples), (
            f"Expected {len(teacher_samples)} teacher spans, got {len(teacher_spans)}"
        )
        assert len(student_spans) == 0, (
            "Student spans should be empty (generation prompt)"
        )
        assert masked_count > 0, "Should have some masked tokens"
        assert unmasked_count > 0, (
            "Should have some unmasked tokens for loss computation"
        )
        assert coord_count > 0, "Should have coordinate tokens"

        logger.info(
            "\n✅ Teacher-student conversation flow logging completed successfully!"
        )
        logger.info("=" * 80)
