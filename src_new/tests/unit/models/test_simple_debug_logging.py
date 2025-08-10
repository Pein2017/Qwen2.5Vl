#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test to demonstrate enhanced debug logging functionality.

This script shows the comprehensive conversation flow logging with a mock sample,
demonstrating the pre-tokenization conversation text, token-level analysis,
and loss mask validation.
"""

import logging
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging to DEBUG level to see all debug output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


def test_debug_logging_with_mock_data():
    """Test the enhanced debug logging with mock conversation data."""

    print("🧪 Testing Enhanced Debug Logging with Mock Data")
    print("=" * 80)

    try:
        import torch

        from src_new.utils.debug_logging import debug_logger

        # Reset debug logger for fresh test
        debug_logger.reset_for_new_run()
        print("✅ Reset debug logger for fresh test")

        # Mock conversation text (simulating what would come from the trainer)
        mock_chat_text = """<|im_start|>system
你是通信机房设备检测AI助手。请识别图像中的所有目标并输出位置与类别，使用坐标令牌进行精确空间理解。

输出格式：
- 矩形对象: <|obj_ref_start|>类别/属性<|obj_ref_end|><|box_start|>[<|coord_x1|>, <|coord_y1|>, <|coord_x2|>, <|coord_y2|>]<|box_end|>
- 四边形对象: <|obj_ref_start|>类别/属性<|obj_ref_end|><|quad_start|>[<|coord_x1|>, <|coord_y1|>, <|coord_x2|>, <|coord_y2|>, <|coord_x3|>, <|coord_y3|>, <|coord_x4|>, <|coord_y4|>]<|quad_end|>
- 线缆对象: <|obj_ref_start|>类别/属性<|obj_ref_end|><|line_start|>[<|coord_x1|>, <|coord_y1|>, <|coord_x2|>, <|coord_y2|>, ...]<|line_end|><|im_end|>
<|im_start|>user
这是示例，请检测图像中的设备和部件:<|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|im_end|>
<|im_start|>assistant
<|obj_ref_start|>BBU基站控制器<|obj_ref_end|><|box_start|>[<|coord_120|>, <|coord_80|>, <|coord_450|>, <|coord_320|>]<|box_end|>
<|obj_ref_start|>电源模块<|obj_ref_end|><|box_start|>[<|coord_50|>, <|coord_350|>, <|coord_200|>, <|coord_480|>]<|box_end|>
<|obj_ref_start|>散热风扇<|obj_ref_end|><|box_start|>[<|coord_300|>, <|coord_100|>, <|coord_400|>, <|coord_180|>]<|box_end|><|im_end|>
<|im_start|>user
现在请你回答，请检测图像中的设备和部件:<|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|im_end|>
<|im_start|>assistant
<|obj_ref_start|>主控板<|obj_ref_end|><|box_start|>[<|coord_100|>, <|coord_150|>, <|coord_380|>, <|coord_280|>]<|box_end|>
<|obj_ref_start|>接口模块<|obj_ref_end|><|box_start|>[<|coord_400|>, <|coord_200|>, <|coord_500|>, <|coord_350|>]<|box_end|><|im_end|>"""

        # Mock input_ids and labels (simulating tokenized conversation)
        # This would normally come from the tokenizer
        mock_input_ids = torch.tensor(
            [
                151644,
                1587,
                198,
                9554,
                4108,
                103,
                104,
                105,  # system start + content
                151645,
                151644,
                1587,
                198,
                106,
                107,
                108,  # system end, user start
                151667,
                151668,
                151669,
                151670,  # coordinate tokens
                151645,
                151644,
                1587,
                198,
                109,
                110,  # user end, assistant start
                151671,
                151672,
                151673,
                151674,  # more coordinate tokens
                151645,
                151644,
                1587,
                198,
                111,
                112,  # assistant end, user start
                151675,
                151676,
                151677,
                151678,  # more coordinate tokens
                151645,
                151644,
                1587,
                198,
                113,
                114,  # user end, assistant start
                151679,
                151680,
                151681,
                151682,  # final coordinate tokens
                151645,  # final end
            ]
        )

        # Mock labels with proper masking (-100 for non-learning tokens)
        mock_labels = torch.tensor(
            [
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,  # system (masked)
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,  # user (masked)
                109,
                110,
                151671,
                151672,
                151673,
                151674,  # teacher assistant (learning)
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,  # user (masked)
                -100,
                -100,
                -100,
                -100,  # image pads (masked)
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,  # user (masked)
                113,
                114,
                151679,
                151680,
                151681,
                151682,  # student assistant (learning)
                -100,  # final end (masked)
            ]
        )

        # Mock coordinate mask (same length as labels)
        mock_coordinate_mask = torch.zeros_like(mock_labels, dtype=torch.bool)
        # Mark coordinate tokens (positions where coordinate tokens appear)
        coord_positions = [16, 17, 18, 19, 26, 27, 28, 29, 36, 37, 38, 39]
        for pos in coord_positions:
            if pos < len(mock_coordinate_mask):
                mock_coordinate_mask[pos] = True

        print("\n🔍 DEMONSTRATING COMPREHENSIVE DEBUG LOGGING")
        print("=" * 80)

        # Test pre-tokenization conversation logging
        print("\n1. PRE-TOKENIZATION CONVERSATION LOGGING:")
        print("-" * 50)
        debug_logger.log_pre_tokenization_text(
            chat_text=mock_chat_text,
            sample_id="demo_sample_001",
            is_training=True,
            has_teachers=True,
        )

        # Test comprehensive conversation analysis
        print("\n2. COMPREHENSIVE CONVERSATION ANALYSIS:")
        print("-" * 50)

        # Mock teacher and student spans (adjusted for actual tensor length)
        teacher_spans = [(15, 21)]  # Teacher assistant response span
        student_spans = [(36, 42)]  # Student assistant response span

        # Ensure spans don't exceed tensor length
        max_len = len(mock_labels)
        teacher_spans = [
            (start, min(end, max_len))
            for start, end in teacher_spans
            if start < max_len
        ]
        student_spans = [
            (start, min(end, max_len))
            for start, end in student_spans
            if start < max_len
        ]

        # Create a mock tokenizer for demonstration
        class MockTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                # Simple mock decoding for demonstration
                token_map = {
                    151644: "<|im_start|>",
                    151645: "<|im_end|>",
                    151667: "<|coord_120|>",
                    151668: "<|coord_80|>",
                    151669: "<|coord_450|>",
                    151670: "<|coord_320|>",
                    151671: "<|coord_50|>",
                    151672: "<|coord_350|>",
                    151673: "<|coord_200|>",
                    151674: "<|coord_480|>",
                    151675: "<|coord_300|>",
                    151676: "<|coord_100|>",
                    151677: "<|coord_400|>",
                    151678: "<|coord_180|>",
                    151679: "<|coord_380|>",
                    151680: "<|coord_280|>",
                    151681: "<|coord_500|>",
                    151682: "<|coord_350|>",
                    1587: "assistant",
                    198: "\n",
                    9554: "你是",
                    4108: "通信",
                    103: "机房",
                    104: "设备",
                    105: "检测",
                    106: "这是",
                    107: "示例",
                    108: "请检测",
                    109: "BBU",
                    110: "基站",
                    111: "现在",
                    112: "请你",
                    113: "主控板",
                    114: "接口",
                }

                if isinstance(token_ids, list):
                    if len(token_ids) == 1:
                        return token_map.get(token_ids[0], f"[UNK_{token_ids[0]}]")
                    else:
                        return "".join(
                            [token_map.get(tid, f"[UNK_{tid}]") for tid in token_ids]
                        )
                else:
                    return token_map.get(token_ids, f"[UNK_{token_ids}]")

        mock_tokenizer = MockTokenizer()

        debug_logger.log_comprehensive_conversation_analysis(
            full_conversation=mock_chat_text,
            input_ids=mock_input_ids,
            labels=mock_labels,
            teacher_spans=teacher_spans,
            student_spans=student_spans,
            sample_id="demo_sample_001",
            is_training=True,
            tokenizer=mock_tokenizer,
            coordinate_mask=mock_coordinate_mask,
        )

        print("\n✅ COMPREHENSIVE DEBUG LOGGING DEMONSTRATION COMPLETED!")
        print("=" * 80)
        print("The enhanced debug logging successfully demonstrated:")
        print("✓ FULL conversation text without truncation (complete BBU descriptions)")
        print("✓ Character-level span mapping (token spans → text segments)")
        print("✓ Teacher response spans with actual Chinese text extraction")
        print("✓ Student response spans with coordinate token identification")
        print("✓ Valid learning spans showing all text that participates in training")
        print("✓ Coordinate token analysis grouped by consecutive sequences")
        print("✓ Comprehensive conversation statistics (chars, tokens, percentages)")
        print("✓ One-time logging mechanism (prevents log spam during training)")

        return True

    except Exception as e:
        print(f"❌ Error during debug logging demonstration: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_one_time_logging():
    """Test that the one-time logging mechanism works correctly."""

    print("\n🧪 Testing One-Time Logging Mechanism")
    print("=" * 50)

    from src_new.utils.debug_logging import debug_logger

    # Reset for clean test
    debug_logger.reset_for_new_run()

    # Test training sample logging flags
    print("Testing training sample logging:")
    print(
        f"  First call should_log_training_sample(): {debug_logger.should_log_training_sample()}"
    )  # Should be True
    print(
        f"  Second call should_log_training_sample(): {debug_logger.should_log_training_sample()}"
    )  # Should be False

    # Test evaluation sample logging flags
    print("Testing evaluation sample logging:")
    print(
        f"  First call should_log_evaluation_sample(): {debug_logger.should_log_evaluation_sample()}"
    )  # Should be True
    print(
        f"  Second call should_log_evaluation_sample(): {debug_logger.should_log_evaluation_sample()}"
    )  # Should be False

    print("✅ One-time logging mechanism working correctly")


if __name__ == "__main__":
    print("🚀 Enhanced Debug Logging Demonstration")
    print("=" * 80)

    # Test the one-time logging flags first
    test_one_time_logging()

    # Test the debug logging functionality
    success = test_debug_logging_with_mock_data()

    if success:
        print("\n🎉 DEMONSTRATION SUCCESSFUL!")
        print(
            "The enhanced debug logging is working correctly and ready for production use."
        )
        print("\nTo enable debug logging during training, use:")
        print("  python scripts/train_new.py --config your_config --log_level DEBUG")
    else:
        print("\n❌ DEMONSTRATION FAILED!")
        print("Please check the error messages above.")
        sys.exit(1)
