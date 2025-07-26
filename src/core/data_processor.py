"""
Data Processor for BBU Training System

Unified data processing and preparation pipeline.
Combines chat processing, teacher-student management, and dataset creation.

Key Features:
- Unified interface for data processing
- Teacher-student conversation handling
- Support for both legacy and new config systems
- Dataset creation with proper collation
- Integration with teacher pool management
"""

from pathlib import Path
from typing import Any, Tuple

from src.chat_processor import ChatProcessor
from src.data import BBUDataset, create_data_collator
from src.logger_utils import get_training_logger
from src.teacher_pool import create_teacher_pool_manager


class DataProcessor:
    """Unified data processing and dataset creation."""

    def __init__(
        self, tokenizer: Any, image_processor: Any, model: Any = None, config=None
    ):
        """
        Initialize data processor.

        Args:
            tokenizer: Model tokenizer
            image_processor: Image processor
            model: Model instance (optional, for simple token manager initialization)
            config: Explicit configuration object (new system)
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model = model
        self.logger = get_training_logger()

        # Use explicit configuration if provided, otherwise fall back to global config
        if config is None:
            from src.config import get_config

            self.config = get_config()
            self.logger.info(
                "📄 DataProcessor using global configuration system (fallback)"
            )
        else:
            self.config = config
            self.logger.info("📄 DataProcessor using explicit configuration system")

        # Initialize components
        self._init_chat_processor()
        self._init_teacher_pool_manager()

    def _init_chat_processor(self) -> None:
        """Initialize chat processor."""
        model_max_length = self.config.model_max_length

        # Get coordinate token configuration - strict validation
        if not hasattr(self.config, "coordinate_tokens_enabled"):
            raise ValueError(
                "coordinate_tokens_enabled must be explicitly configured in config"
            )
        coordinate_tokens_enabled = self.config.coordinate_tokens_enabled

        # Get max_coord_value from flattened config
        # BEFORE (with silent defaults):
        # max_coord_value = getattr(self.config, "coordinate_config_max_coord_value", 2048)
        # use_official_box_tokens = getattr(self.config, "coordinate_config_use_official_box_tokens", True)

        # AFTER (strict validation with simplified config):
        if not hasattr(self.config, "max_coord_value"):
            raise ValueError("max_coord_value must be explicitly configured in config")
        max_coord_value = self.config.max_coord_value

        # Use default for box tokens (always true with unified token manager)
        use_official_box_tokens = True

        self.logger.debug(
            f"🎯 CHAT PROCESSOR CONFIG: coordinate_enabled={coordinate_tokens_enabled}, max_coord={max_coord_value}, box_tokens={use_official_box_tokens}"
        )

        self.chat_processor = ChatProcessor(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            config=self.config,
            model_max_length=model_max_length,
            coordinate_tokens_enabled=coordinate_tokens_enabled,
            max_coord_value=max_coord_value,
        )

        # Update coordinate token ranges after tokenizer extension
        if coordinate_tokens_enabled:
            self.logger.info(
                "🎯 Updating coordinate token ranges in chat processor..."
            )
            self.chat_processor._update_coordinate_token_ranges()

        if coordinate_tokens_enabled:
            self.logger.info(
                f"✅ Chat processor initialized with coordinate tokens (geometry + object_ref wrapping)"
            )
        else:
            self.logger.info("✅ Chat processor initialized (JSON format only)")

    def _init_teacher_pool_manager(self) -> None:
        """Initialize teacher pool manager if available."""
        teacher_pool_file = self.config.teacher_pool_file

        self.teacher_pool_manager = None
        if teacher_pool_file and Path(teacher_pool_file).exists():
            self.teacher_pool_manager = create_teacher_pool_manager(self.config)
            self.logger.info(
                f"✅ Teacher pool manager initialized: {teacher_pool_file}"
            )
        else:
            self.logger.info("ℹ️  No teacher pool available")

    def create_datasets(self) -> Tuple[BBUDataset, BBUDataset]:
        """
        Create training and evaluation datasets.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        self.logger.info("📊 Creating datasets...")

        # Get data paths and configuration
        data_config = self._get_data_config()

        # Create training dataset
        train_dataset = BBUDataset(
            data_path=data_config["train_data_path"],
            chat_processor=self.chat_processor,
            teacher_pool_manager=self.teacher_pool_manager,
            teacher_ratio=data_config["teacher_ratio"],
            is_training=True,
            config=self.config,
        )

        # Create evaluation dataset (no teachers)
        eval_dataset = BBUDataset(
            data_path=data_config["val_data_path"],
            chat_processor=self.chat_processor,
            teacher_pool_manager=None,  # No teachers for evaluation
            teacher_ratio=data_config["val_teacher_ratio"],
            is_training=False,
            config=self.config,
        )

        self.logger.info(
            f"✅ Datasets created: Train={len(train_dataset)}, Eval={len(eval_dataset)}"
        )
        return train_dataset, eval_dataset

    def create_data_collator(self) -> Any:
        """
        Create data collator for batching.

        Returns:
            Configured data collator
        """
        self.logger.info("📦 Creating data collator...")

        collator_type = self.config.collator_type

        data_collator = create_data_collator(self.tokenizer, collator_type)
        self.logger.info(f"✅ Data collator created: {collator_type}")
        return data_collator

    def _get_data_config(self) -> dict:
        """Get data configuration parameters."""
        return {
            "train_data_path": self.config.train_data_path,
            "val_data_path": self.config.val_data_path,
            "teacher_ratio": self.config.teacher_ratio,
            "val_teacher_ratio": 0.0,  # Always use 0 teacher ratio for validation
        }

    def get_data_statistics(self) -> dict:
        """
        Get data processing statistics.

        Returns:
            Dictionary with data statistics
        """
        data_config = self._get_data_config()

        stats = {
            "train_data_path": data_config["train_data_path"],
            "val_data_path": data_config["val_data_path"],
            "teacher_ratio": data_config["teacher_ratio"],
            "teacher_pool_available": self.teacher_pool_manager is not None,
        }

        # Safely get model_max_length from chat_processor or config
        if hasattr(self, "chat_processor") and self.chat_processor is not None:
            if hasattr(self.chat_processor, "model_max_length"):
                stats["chat_processor_max_length"] = (
                    self.chat_processor.model_max_length
                )
            elif hasattr(self.chat_processor.tokenizer, "model_max_length"):
                stats["chat_processor_max_length"] = (
                    self.chat_processor.tokenizer.model_max_length
                )
            else:
                stats["chat_processor_max_length"] = None
        elif hasattr(self.config, "model_max_length") and self.config is not None:
            stats["chat_processor_max_length"] = self.config.model_max_length
        else:
            stats["chat_processor_max_length"] = None

        return stats

    @classmethod
    def create_datasets_and_collator(
        cls, tokenizer: Any, image_processor: Any
    ) -> Tuple[BBUDataset, BBUDataset, Any]:
        """
        Convenience method to create datasets and collator in one call.

        Args:
            tokenizer: Model tokenizer
            image_processor: Image processor

        Returns:
            Tuple of (train_dataset, eval_dataset, data_collator)
        """
        processor = cls(tokenizer, image_processor)
        train_dataset, eval_dataset = processor.create_datasets()
        data_collator = processor.create_data_collator()
        return train_dataset, eval_dataset, data_collator
