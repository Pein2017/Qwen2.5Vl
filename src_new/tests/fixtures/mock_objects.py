"""
Mock objects for testing src_new components.

This module provides mock implementations of external dependencies
to enable isolated unit testing.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import torch


class MockTokenizer:
    """Mock tokenizer for testing chat processing and tokenization."""

    def __init__(self, vocab_size: int = 151665):
        self.vocab_size = vocab_size
        self.model_max_length = 120000
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3

        # Mock special tokens - reflecting actual Qwen2.5-VL pretrained vocabulary
        self.special_tokens = {
            # Chat formatting tokens (already in pretrained model)
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "<|endoftext|>": 151643,
            "<|image_pad|>": 151655,
            # Geometry tokens already in Qwen2.5-VL pretrained vocabulary
            "<|obj_ref_start|>": 151646,
            "<|obj_ref_end|>": 151647,
            "<|box_start|>": 151648,
            "<|box_end|>": 151649,
            "<|quad_start|>": 151650,
            "<|quad_end|>": 151651,
            # NEW: Line tokens that need to be added (only 2 tokens)
            # These will be added by token processor, starting at vocab_size (151665)
            "<|line_start|>": 151665,  # First new token
            "<|line_end|>": 151666,  # Second new token
        }

        # Mock coordinate tokens (2049 tokens: coord_0 to coord_2048)
        # These start after the 2 line tokens: 151665 + 2 = 151667
        self.coordinate_tokens = {}
        for i in range(2049):  # 0 to 2048 inclusive = 2049 tokens
            token = f"<|coord_{i}|>"
            self.coordinate_tokens[token] = 151667 + i  # Start after line tokens

        # Create a Mock for decode so tests can set side_effect
        self.decode = Mock(side_effect=self._decode_impl)

    def _decode_impl(self, token_ids: List[int], **kwargs) -> str:
        """Mock decoding implementation used by the decode Mock."""
        if not token_ids:
            return ""

        # Check for special tokens
        for token, token_id in self.special_tokens.items():
            if token_id in token_ids:
                return token

        # Check for coordinate tokens
        for token, token_id in self.coordinate_tokens.items():
            if token_id in token_ids:
                return token

        return f"decoded_text_{len(token_ids)}_tokens"

    def encode(self, text: str, **kwargs) -> List[int]:
        """Mock encoding that returns reasonable token ids."""
        if not text:
            return []

        # Simulate realistic tokenization based on text length
        # Roughly 1 token per 3-4 characters for mixed Chinese/English text
        estimated_tokens = max(1, len(text) // 3)

        # Add special token IDs for special tokens found in text
        token_ids = []

        # Check for special tokens and add their IDs
        for token, token_id in self.special_tokens.items():
            if token in text:
                token_ids.append(token_id)

        # Check for coordinate tokens and add their IDs
        for token, token_id in self.coordinate_tokens.items():
            if token in text:
                token_ids.append(token_id)

        # Fill remaining with regular token IDs
        remaining_tokens = max(0, estimated_tokens - len(token_ids))
        token_ids.extend(list(range(10, 10 + remaining_tokens)))

        return token_ids[:estimated_tokens]  # Cap at estimated length

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Mock decoding."""
        if not token_ids:
            return ""

        # Check for special tokens
        for token, token_id in self.special_tokens.items():
            if token_id in token_ids:
                return token

        # Check for coordinate tokens
        for token, token_id in self.coordinate_tokens.items():
            if token_id in token_ids:
                return token

        return f"decoded_text_{len(token_ids)}_tokens"

    def apply_chat_template(self, conversation: List[Dict[str, str]], **kwargs) -> str:
        """Mock chat template application."""
        formatted_parts = []
        for message in conversation:
            role = message["role"]
            content = message["content"]
            formatted_parts.append(f"<im_start>{role}\n{content}<im_end>")
        return "\n".join(formatted_parts)

    def add_tokens(self, new_tokens: List[str]) -> int:
        """Mock token addition - simulates extending tokenizer vocabulary."""
        added_count = 0
        for token in new_tokens:
            # Check if token already exists in any vocabulary
            if (
                token not in self.special_tokens
                and token not in self.coordinate_tokens
                and token not in self.get_vocab()
            ):
                # Add to appropriate dictionary
                if token.startswith("<|coord_"):
                    # Coordinate tokens start at 151667 (after line tokens)
                    coord_num = int(token.split("_")[1].split("|>")[0])
                    self.coordinate_tokens[token] = 151667 + coord_num
                elif token in ["<|line_start|>", "<|line_end|>"]:
                    # Line tokens are at 151665 and 151666
                    if token == "<|line_start|>":
                        self.special_tokens[token] = 151665
                    else:
                        self.special_tokens[token] = 151666
                else:
                    # Other new tokens start after coordinate tokens
                    self.special_tokens[token] = self.vocab_size + added_count
                added_count += 1

        self.vocab_size += added_count
        return added_count

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        vocab = {}
        vocab.update(self.special_tokens)
        vocab.update(self.coordinate_tokens)
        return vocab

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Make tokenizer callable for HuggingFace compatibility."""
        # Extract parameters
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length", 512)
        return_tensors = kwargs.get("return_tensors", None)

        # Encode text
        input_ids = self.encode(text)

        # Apply truncation
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Convert to tensors if requested
        if return_tensors == "pt":
            input_ids = torch.tensor([input_ids])
            attention_mask = torch.tensor([attention_mask])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class MockImageProcessor:
    """Mock image processor for testing vision components."""

    def __init__(self):
        self.do_resize = True
        self.size = {"height": 224, "width": 224}
        self.do_normalize = True
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def preprocess(self, images, **kwargs) -> Dict[str, torch.Tensor]:
        """Mock image preprocessing."""
        if isinstance(images, str):
            images = [images]

        batch_size = len(images)
        # Use a tiny grid per image [[1, 2, 2]] -> 4 patches per image
        grid_per_image = torch.tensor([1, 2, 2], dtype=torch.long)
        image_grid_thw = torch.stack([grid_per_image for _ in range(batch_size)])
        num_patches = int(
            (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
            .sum()
            .item()
        )
        hidden = 1024
        return {
            "pixel_values": torch.randn(num_patches, hidden),
            "image_grid_thw": image_grid_thw,
        }

    def __call__(self, images, **kwargs) -> Dict[str, torch.Tensor]:
        """Make the processor callable (same as preprocess)."""
        return self.preprocess(images, **kwargs)


class MockEmbeddings:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.num_embeddings = weight.shape[0]


class MockModel:
    """Mock model for testing model wrapper functionality."""

    def __init__(self, vocab_size: int = 151665, hidden_size: int = 2048):
        self.config = Mock()
        self.config.vocab_size = vocab_size
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = 16
        self.config.num_hidden_layers = 36

        # Mock model components
        self.embed_tokens = Mock()
        self.embed_tokens.weight = torch.randn(vocab_size, hidden_size)
        self.embed_tokens.num_embeddings = vocab_size

        self.lm_head = Mock()
        self.lm_head.weight = torch.randn(vocab_size, hidden_size)

        self.training = True
        self.device = torch.device("cpu")

    def get_input_embeddings(self):
        return MockEmbeddings(self.embed_tokens.weight)

    def forward(self, **kwargs) -> Mock:
        """Mock forward pass."""
        batch_size = kwargs.get("input_ids", torch.tensor([[1]])).size(0)
        seq_len = kwargs.get("input_ids", torch.tensor([[1]])).size(1)

        output = Mock()
        output.loss = torch.tensor(1.5) if "labels" in kwargs else None
        output.logits = torch.randn(batch_size, seq_len, self.config.vocab_size)
        output.hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        output.past_key_values = None

        return output

    def generate(self, **kwargs) -> torch.Tensor:
        """Mock generation."""
        batch_size = kwargs.get("input_ids", torch.tensor([[1]])).size(0)
        max_length = kwargs.get("max_length", 50)

        return torch.randint(0, self.config.vocab_size, (batch_size, max_length))

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Mock token embedding resize."""
        old_vocab_size = self.config.vocab_size
        self.config.vocab_size = new_vocab_size

        # Mock weight resizing
        old_weight = self.embed_tokens.weight
        new_weight = torch.randn(new_vocab_size, self.config.hidden_size)
        new_weight[:old_vocab_size] = old_weight
        self.embed_tokens.weight = new_weight
        self.embed_tokens.num_embeddings = new_vocab_size

    def train(self, mode: bool = True):
        """Mock training mode setting."""
        self.training = mode
        return self

    def eval(self):
        """Mock evaluation mode setting."""
        self.training = False
        return self

    def to(self, device):
        """Mock device movement."""
        self.device = device
        return self

    def parameters(self):
        """Mock parameters iteration."""
        yield self.embed_tokens.weight
        yield self.lm_head.weight


class MockTrainer:
    """Mock trainer for testing training components."""

    def __init__(self, model=None, args=None, train_dataset=None, eval_dataset=None):
        self.model = model or MockModel()
        self.args = args or Mock()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.state = Mock()
        self.state.global_step = 0
        self.state.epoch = 0
        self.control = Mock()

        # Mock training metrics
        self.training_history = []
        self.eval_history = []

    def train(self) -> Mock:
        """Mock training execution."""
        result = Mock()
        result.training_loss = 1.5
        result.global_step = 100
        result.metrics = {
            "train_loss": 1.5,
            "eval_loss": 1.3,
            "train_runtime": 60.0,
            "train_samples_per_second": 10.0,
        }
        return result

    def evaluate(self, eval_dataset=None) -> Dict[str, float]:
        """Mock evaluation."""
        return {
            "eval_loss": 1.3,
            "eval_accuracy": 0.85,
            "eval_runtime": 10.0,
            "eval_samples_per_second": 50.0,
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        """Mock loss computation."""
        loss = torch.tensor(1.5, requires_grad=True)

        if return_outputs:
            outputs = Mock()
            outputs.loss = loss
            outputs.logits = torch.randn(1, 100, 151665)
            return loss, outputs

        return loss

    def log(self, logs: Dict[str, float]) -> None:
        """Mock logging."""
        self.training_history.append(logs)


@dataclass
class MockLossComponents:
    """Mock loss components for testing."""

    loss: torch.Tensor
    llm_loss: Optional[torch.Tensor] = None
    coordinate_loss: Optional[torch.Tensor] = None
    teacher_loss: Optional[torch.Tensor] = None
    student_loss: Optional[torch.Tensor] = None


class MockDataset:
    """Mock dataset for testing data loading."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class MockDataCollator:
    """Mock data collator for testing batch creation."""

    def __init__(self, tokenizer: MockTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Mock collation."""
        batch_size = len(features)
        max_length = 100

        # Build minimal valid image tensors for Qwen2.5-VL
        grid_per_image = torch.tensor([1, 2, 2], dtype=torch.long)
        image_grid_thw = torch.stack([grid_per_image for _ in range(batch_size)])
        num_patches = int(
            (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
            .sum()
            .item()
        )
        hidden = 1024

        return {
            "input_ids": torch.randint(
                0, self.tokenizer.vocab_size, (batch_size, max_length)
            ),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
            "labels": torch.randint(
                0, self.tokenizer.vocab_size, (batch_size, max_length)
            ),
            "pixel_values": torch.randn(num_patches, hidden),
            "image_grid_thw": image_grid_thw,
        }


# Factory functions for easy mock creation
def create_mock_tokenizer(vocab_size: int = 151665) -> MockTokenizer:
    """Create a mock tokenizer with specified vocab size."""
    return MockTokenizer(vocab_size)


def create_mock_model(vocab_size: int = 151665, hidden_size: int = 2048) -> MockModel:
    """Create a mock model with specified parameters."""
    model = MockModel(vocab_size, hidden_size)
    # Override generate with an instance-level Mock to allow .side_effect in tests
    model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
    return model


def create_mock_trainer(model=None, **kwargs) -> MockTrainer:
    """Create a mock trainer with optional components."""
    return MockTrainer(model=model, **kwargs)


def create_mock_loss_components(**overrides) -> MockLossComponents:
    """Create mock loss components with optional overrides."""
    defaults = {
        "loss": torch.tensor(2.5),
        "llm_loss": torch.tensor(2.0),
        "coordinate_loss": torch.tensor(0.3),
        "teacher_loss": torch.tensor(1.8),
        "student_loss": torch.tensor(2.2),
    }
    defaults.update(overrides)
    return MockLossComponents(**defaults)


# Context managers for testing
class MockGPUContext:
    """Context manager for mocking GPU availability."""

    def __init__(self, available: bool = True):
        self.available = available
        self.original_is_available = None

    def __enter__(self):
        import torch.cuda

        self.original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: self.available
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch.cuda

        torch.cuda.is_available = self.original_is_available
