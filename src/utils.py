"""
Training utilities for Qwen2.5VL BBU training.

This module contains utilities specifically for training,
focusing on core functionality needed during the training process.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import transformers

# Import unified logging system

# Constants - shared between training and evaluation
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
DEFAULT_IMAGE_TOKEN = "<image>"

# Chat template that matches training setup exactly
CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# Default model paths
DEFAULT_BASE_MODEL_PATH = "/data4/swift/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_7B_MODEL_PATH = "/data4/swift/model_cache/Qwen/Qwen2.5-VL-7B-Instruct"


class UnifiedLogger:
    """
    Unified logger for BBU training with distributed training support.
    Environment variables are set by the launcher script.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_name: str = None,
        verbose: bool = True,
        log_level: int = logging.INFO,
        console_level: int = logging.INFO,
        is_training: bool = True,
    ):
        """
        Initialize unified logger.

        Args:
            log_dir: Directory for log files
            log_name: Name for log file (auto-generated if None)
            verbose: Enable verbose logging
            log_level: File logging level
            console_level: Console logging level
            is_training: Whether this is a training session
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.verbose = verbose
        self.is_training = is_training

        # Get distributed training info
        self.rank = self._get_rank()
        self.world_size = self._get_world_size()

        # Generate log name if not provided
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"training_{timestamp}"

        # Add rank suffix for distributed training
        if self.world_size > 1:
            log_name = f"{log_name}_rank{self.rank}"

        self.log_file = self.log_dir / f"{log_name}.log"

        # Setup logger
        self.logger = logging.getLogger(f"unified_logger_{log_name}")
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()

        # File handler (always enabled)
        file_handler = logging.FileHandler(self.log_file, mode="w")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            f"%(asctime)s - [Rank {self.rank}/{self.world_size}] - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler (rank 0 only)
        if self.rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            if verbose:
                console_formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
            else:
                console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def _get_rank(self) -> int:
        """Get distributed training rank."""
        # Try multiple environment variables for rank detection
        rank_vars = ["RANK", "LOCAL_RANK", "SLURM_PROCID"]
        for var in rank_vars:
            if var in os.environ:
                try:
                    return int(os.environ[var])
                except ValueError:
                    continue
        return 0

    def _get_world_size(self) -> int:
        """Get distributed training world size."""
        # Try multiple environment variables for world size detection
        world_size_vars = ["WORLD_SIZE", "SLURM_NTASKS"]
        for var in world_size_vars:
            if var in os.environ:
                try:
                    return int(os.environ[var])
                except ValueError:
                    continue
        return 1

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def info_all_ranks(self, msg: str):
        """Log info message from all ranks."""
        self.logger.info(f"[ALL_RANKS] {msg}")

    def debug_all_ranks(self, msg: str):
        """Log debug message from all ranks."""
        self.logger.debug(f"[ALL_RANKS] {msg}")

    def get_log_file(self) -> str:
        """Get log file path."""
        return str(self.log_file)

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1


def extract_prompts_from_conversation(
    conversations: List[Dict[str, str]],
) -> tuple[str, str]:
    """
    Extract system and user prompts from conversation format.

    Args:
        conversations: List of conversation turns

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = ""
    user_prompt = ""

    for conv in conversations:
        role = conv.get("role", conv.get("from", ""))
        content = conv.get("content", conv.get("value", ""))

        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt = content
            break  # Take the first user message

    return system_prompt, user_prompt


def find_ground_truth_response(conversations: List[Dict[str, str]]) -> Optional[str]:
    """
    Find the ground truth assistant response from conversations.

    Args:
        conversations: List of conversation turns

    Returns:
        Ground truth response or None if not found
    """
    # Find the last assistant response
    for conv in reversed(conversations):
        role = conv.get("role", conv.get("from", ""))
        content = conv.get("content", conv.get("value", ""))

        if role == "assistant" and content:
            return content

    return None


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:100]}... Error: {e}")
    return data


def preprocess_conversations(
    sources: List[List[Dict]],
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List[int] = None,
    logger: Optional[Any] = None,
) -> Dict:
    """
    Preprocess conversations for training.

    Args:
        sources: List of conversation sources
        tokenizer: Tokenizer to use
        grid_thw_image: Image grid dimensions
        logger: Optional logger

    Returns:
        Preprocessed data dictionary
    """
    if grid_thw_image is None:
        grid_thw_image = []

    if logger:
        logger.debug(f"Preprocessing {len(sources)} conversation sources")

    # Normalize conversation format
    normalized_sources = []
    for source in sources:
        normalized_source = _normalize_conversation_format(source)
        normalized_sources.append(normalized_source)

    # Process conversations
    input_ids_list = []
    labels_list = []

    for source in normalized_sources:
        # Build conversation text
        conversation_text = ""
        label_text = ""

        for i, conv in enumerate(source):
            role = conv.get("role", conv.get("from", ""))
            content = conv.get("content", conv.get("value", ""))

            if role == "system":
                conversation_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                # Replace image tokens if needed
                if grid_thw_image:
                    content = _replace_vision_tokens(content, grid_thw_image, i)
                conversation_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                conversation_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                label_text = content  # Only the last assistant response is the target

        # Tokenize
        input_ids = tokenizer.encode(conversation_text, add_special_tokens=False)

        # Create labels (mask everything except the assistant response)
        labels = [-100] * len(input_ids)
        if label_text:
            label_ids = tokenizer.encode(label_text, add_special_tokens=False)
            # Find where the assistant response starts in the input
            assistant_start = conversation_text.rfind("<|im_start|>assistant\n")
            if assistant_start != -1:
                assistant_prefix = conversation_text[:assistant_start]
                prefix_ids = tokenizer.encode(
                    assistant_prefix, add_special_tokens=False
                )
                start_idx = len(prefix_ids) + 2  # +2 for assistant tokens
                end_idx = start_idx + len(label_ids)
                if end_idx <= len(labels):
                    labels[start_idx:end_idx] = label_ids

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
    }


def _normalize_conversation_format(source: List[Dict]) -> List[Dict]:
    """
    Normalize conversation format to use consistent keys.

    Args:
        source: Raw conversation source

    Returns:
        Normalized conversation
    """
    normalized = []
    for conv in source:
        normalized_conv = {}

        # Normalize role field
        if "role" in conv:
            normalized_conv["role"] = conv["role"]
        elif "from" in conv:
            normalized_conv["role"] = conv["from"]
        else:
            normalized_conv["role"] = "unknown"

        # Normalize content field
        if "content" in conv:
            normalized_conv["content"] = conv["content"]
        elif "value" in conv:
            normalized_conv["content"] = conv["value"]
        else:
            normalized_conv["content"] = ""

        normalized.append(normalized_conv)

    return normalized


def _replace_vision_tokens(
    content: str, grid_thw_image: List[int], visual_replicate_index_image: int
) -> str:
    """
    Replace vision tokens in content.

    Args:
        content: Content string
        grid_thw_image: Image grid dimensions
        visual_replicate_index_image: Visual replication index

    Returns:
        Content with replaced vision tokens
    """
    if "<image>" in content and grid_thw_image:
        # Calculate number of vision tokens
        if visual_replicate_index_image < len(grid_thw_image):
            num_tokens = grid_thw_image[visual_replicate_index_image]
            vision_tokens = (
                "<|vision_start|>" + "<|image_pad|>" * num_tokens + "<|vision_end|>"
            )
            content = content.replace("<image>", vision_tokens, 1)

    return content


# Import torch here to avoid issues during module import


def validate_telecom_data_format(
    jsonl_path: str, max_samples: int = 10
) -> Dict[str, Any]:
    """
    Validate telecom data format.

    Args:
        jsonl_path: Path to JSONL file
        max_samples: Maximum samples to validate

    Returns:
        Validation results
    """
    results = {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "errors": [],
        "sample_analysis": [],
    }

    try:
        data = load_jsonl(jsonl_path)
        results["total_samples"] = len(data)

        for i, sample in enumerate(data[:max_samples]):
            sample_result = {
                "index": i,
                "valid": True,
                "issues": [],
            }

            # Check required fields
            if "conversations" not in sample:
                sample_result["valid"] = False
                sample_result["issues"].append("Missing 'conversations' field")

            if "image" not in sample and "images" not in sample:
                sample_result["valid"] = False
                sample_result["issues"].append("Missing image field")

            # Check conversation format
            if "conversations" in sample:
                conversations = sample["conversations"]
                if not isinstance(conversations, list):
                    sample_result["valid"] = False
                    sample_result["issues"].append("'conversations' is not a list")
                else:
                    # Check for required roles
                    roles = [
                        conv.get("role", conv.get("from", "")) for conv in conversations
                    ]
                    if "assistant" not in roles:
                        sample_result["valid"] = False
                        sample_result["issues"].append("No assistant response found")

            if sample_result["valid"]:
                results["valid_samples"] += 1
            else:
                results["invalid_samples"] += 1

            results["sample_analysis"].append(sample_result)

    except Exception as e:
        results["errors"].append(f"Failed to load file: {e}")

    return results
