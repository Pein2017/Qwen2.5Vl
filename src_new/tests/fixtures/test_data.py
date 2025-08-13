"""
Test data generators and constants for comprehensive testing.
"""

from typing import Any, Dict, List

import torch


def create_sample_config(**overrides) -> Dict[str, Any]:
    """Create a sample configuration with optional overrides."""
    config = {
        # === REQUIRED FIELDS ===
        # Model settings
        "model_path": "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct-max_coord_1024",
        "model_size": "3B",
        "model_max_length": 120000,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": "bfloat16",
        # Optional model settings (included for compatibility)
        "use_cache": False,
        "model_hidden_size": 2048,
        "model_num_layers": 36,
        "model_num_attention_heads": 16,
        "model_vocab_size": 151665,
        # Training settings
        "num_train_epochs": 5,  # Reduced for testing
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-6,
        "vision_lr": 5e-7,
        "merger_lr": 5e-5,
        "llm_lr": 5e-6,
        "adapter_lr": 0.0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0001,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        "bf16": True,
        "fp16": False,
        "use_flash_attention": True,
        "mixed_precision": "bf16",
        # Data settings - only data_root needed, resolver will handle the rest
        "data_root": "test_data",
        "max_total_length": 12000,
        "num_teacher_samples": 1,
        "collator_type": "packed",
        "teacher_ratio": 0.5,
        "language": "chinese",
        # Output settings
        "output_dir": "test_output",
        "run_name": "test_run",
        "max_coord_value": 1024,
        # Coordinate token configuration (required)
        "coordinate_tokens_enabled": True,
        "coordinate_loss_weight": 0,  # Match bbu_v2_use_coord.yaml (int, not float)
        "regular_loss_weight": 1.0,
        "coordinate_temperature": 0.7,
        "coordinate_label_sigma": 16,  # Match YAML (int, not float)
        "coordinate_init_mode": "fourier_ramp",
        # Coordinate auxiliary losses (required)
        "coord_aux_enabled": False,
        "coord_aux_tau": 1.2,
        "coord_aux_sigma_bins": 8.0,
        "coord_aux_window_bins": 32,
        "coord_aux_topk": 100,
        "coord_aux_lambda_kce": 0.5,
        "coord_aux_lambda_unlike": 0.05,
        "coord_aux_lambda_lap1": 1e-4,
        "coord_aux_lambda_lap2": 1e-5,
        # Evaluation settings (required)
        "eval_strategy": "steps",
        "eval_steps": 10,
        "save_strategy": "steps",
        "save_steps": 20,
        "save_total_limit": 2,
        # Logging settings (required)
        "logging_steps": 5,
        "logging_dir": "test_logs",
        "report_to": "none",  # Disable wandb/tensorboard for testing
        "disable_tqdm": True,
        "verbose": False,
        # Essential settings (required)
        "remove_unused_columns": False,
        # Dataloader performance settings (required)
        "dataloader_num_workers": 0,  # Disable multiprocessing for testing
        "pin_memory": False,
        "prefetch_factor": 2,
        # Output settings (required)
        "tb_dir": "test_tb",
        # Teacher-student loss weights (required)
        "teacher_loss_weight": 0.3,
        "student_loss_weight": 1.0,
        # Vision processing parameters (required)
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "max_pixels": 401408,
        # Training control flags (required)
        "training_prompt_style": True,
        "use_consistent_prompts": True,
        # Model loading control flags (required)
        "skip_vocab_extension": False,
    }

    # Apply overrides
    config.update(overrides)
    return config


def create_sample_jsonl_data() -> List[Dict[str, Any]]:
    """Create sample JSONL data for testing."""
    return [
        {
            "images": ["test_images/sample_1.jpg"],
            "objects": [
                {"bbox_2d": [100, 150, 200, 250], "desc": "测试设备/基础检测目标"},
                {
                    "quad": [300, 400, 350, 410, 348, 425, 302, 415],
                    "desc": "标签贴纸/测试标识",
                },
            ],
            "width": 800,
            "height": 600,
        },
        {
            "images": ["test_images/sample_2.jpg"],
            "objects": [
                {
                    "line": [50, 100, 150, 120, 250, 140, 350, 160],
                    "desc": "线缆/测试连接线",
                },
                {"bbox_2d": [400, 500, 450, 550], "desc": "连接器/测试接口"},
            ],
            "width": 640,
            "height": 480,
        },
        {
            "images": ["test_images/sample_3.jpg"],
            "objects": [{"bbox_2d": [10, 20, 30, 40], "desc": "小型器件/精密部件"}],
            "width": 512,
            "height": 384,
        },
    ]


def create_sample_conversation() -> List[Dict[str, str]]:
    """Create a sample conversation for testing."""
    return [
        {
            "role": "system",
            "content": "你是通信机房设备检测AI助手。请识别图像中的所有目标并输出位置与类别。",
        },
        {"role": "user", "content": "请检测图像中的设备和部件: <image>"},
        {
            "role": "assistant",
            "content": '[{"bbox_2d":[100,150,200,250], "desc":"测试设备/基础检测目标"}]',
        },
    ]


def create_sample_batch(
    batch_size: int = 1, seq_len: int = 100, vocab_size: int = 151665
) -> Dict[str, torch.Tensor]:
    """Create a sample batch for testing."""
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "pixel_values": torch.randn(batch_size * 4, 1024),
        "image_grid_thw": torch.tensor([[1, 2, 2]] * batch_size),
    }


def create_coordinate_tokens_data() -> Dict[str, Any]:
    """Create test data specifically for coordinate token testing."""
    return {
        "raw_coordinates": [100, 200, 300, 400],
        "coordinate_tokens": [
            "<|coord_100|>",
            "<|coord_200|>",
            "<|coord_300|>",
            "<|coord_400|>",
        ],
        "token_ids": [151666, 151667, 151668, 151669],  # Extended vocab range
        "coordinate_ranges": {
            "bbox_2d": (0, 4),
            "quad": (0, 8),
            "line": (0, 16),  # Variable length
        },
    }


def create_teacher_student_data() -> Dict[str, Any]:
    """Create test data for teacher-student learning."""
    return {
        "teacher_sample": {
            "images": ["teacher_image.jpg"],
            "objects": [{"bbox_2d": [50, 60, 70, 80], "desc": "示例设备/参考目标"}],
            "width": 400,
            "height": 300,
        },
        "student_sample": {
            "images": ["student_image.jpg"],
            "objects": [
                {"bbox_2d": [150, 160, 170, 180], "desc": "目标设备/待检测对象"}
            ],
            "width": 500,
            "height": 400,
        },
        "teacher_ratio": 0.7,
        "multi_turn_conversation": [
            {"role": "system", "content": "学习参考示例，然后分析新图像。"},
            {"role": "user", "content": "📚 参考示例: <image>"},
            {
                "role": "assistant",
                "content": '[{"bbox_2d":[50,60,70,80], "desc":"示例设备/参考目标"}]',
            },
            {"role": "user", "content": "现在请根据参考示例检测以下图像: <image>"},
            {
                "role": "assistant",
                "content": '[{"bbox_2d":[150,160,170,180], "desc":"目标设备/待检测对象"}]',
            },
        ],
    }


def create_loss_components_data() -> Dict[str, torch.Tensor]:
    """Create test data for loss components."""
    return {
        "loss": torch.tensor(2.5),
        "llm_loss": torch.tensor([1.2, 1.8, 2.1]),
        "coordinate_loss": torch.tensor([0.3, 0.4, 0.2, 0.5]),
        "teacher_loss": torch.tensor([1.0, 1.5]),
        "student_loss": torch.tensor([2.0, 2.5, 1.8]),
    }


# Test constants
GEOMETRY_TYPES = ["bbox_2d", "quad", "line"]
SUPPORTED_LANGUAGES = ["chinese"]
MODEL_SIZES = ["3B", "7B"]
COLLATOR_TYPES = ["standard", "packed"]

# Special token constants for testing
SPECIAL_TOKENS = {
    "obj_ref_start": "<|object_ref_start|>",
    "obj_ref_end": "<|object_ref_end|>",
    "box_start": "<|box_start|>",
    "box_end": "<|box_end|>",
    "quad_start": "<|quad_start|>",
    "quad_end": "<|quad_end|>",
    "line_start": "<|line_start|>",
    "line_end": "<|line_end|>",
    "im_start": "<im_start>",
    "im_end": "<im_end>",
}

# Chat template constants
CHAT_TEMPLATE_ROLES = ["system", "user", "assistant"]

# Error test cases
ERROR_TEST_CASES = {
    "invalid_config": {
        "missing_model_path": {"model_size": "3B"},  # Missing required field
        "invalid_torch_dtype": {"torch_dtype": "invalid_dtype"},
        "negative_learning_rate": {"learning_rate": -1e-5},
    },
    "invalid_data": {
        "missing_objects": {"images": ["test.jpg"], "width": 100, "height": 100},
        "invalid_coordinates": {
            "objects": [{"bbox_2d": [100, 200], "desc": "test"}]
        },  # Wrong bbox format
        "missing_description": {"objects": [{"bbox_2d": [100, 200, 300, 400]}]},
    },
    "invalid_batch": {
        "mismatched_lengths": {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1]]),  # Wrong length
        },
        "wrong_dtype": {
            "input_ids": torch.tensor([[1.5, 2.5, 3.5]]),  # Should be int
        },
    },
}
