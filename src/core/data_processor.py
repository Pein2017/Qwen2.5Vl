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
from src.config import config, get_config_manager
from src.data import BBUDataset, create_data_collator
from src.logger_utils import get_training_logger
from src.teacher_pool import create_teacher_pool_manager


class DataProcessor:
    """Unified data processing and dataset creation."""

    def __init__(
        self, tokenizer: Any, image_processor: Any, use_new_config: bool = False
    ):
        """
        Initialize data processor.

        Args:
            tokenizer: Model tokenizer
            image_processor: Image processor
            use_new_config: Whether to use new domain-specific config system
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.use_new_config = use_new_config
        self.logger = get_training_logger()

        # Get configuration
        if use_new_config:
            try:
                self.config = get_config_manager()
                self.logger.info(
                    "✅ DataProcessor using new domain-specific configuration"
                )
            except RuntimeError:
                self.logger.warning(
                    "⚠️  New config system not available, falling back to legacy"
                )
                self.config = config
                self.use_new_config = False
        else:
            self.config = config
            self.logger.info("📄 DataProcessor using legacy configuration system")

        # Initialize components
        self._init_chat_processor()
        self._init_teacher_pool_manager()

    def _init_chat_processor(self) -> None:
        """Initialize chat processor."""
        model_max_length = (
            self.config.model.model_max_length
            if self.use_new_config
            else self.config.model_max_length
        )

        # Get coordinate token configuration - strict validation
        if self.use_new_config:
            coordinate_tokens_enabled = self.config.coordinate.enable_coordinate_tokens
            max_coord_value = self.config.coordinate.max_coord_value
        else:
            if not hasattr(self.config, "coordinate_tokens_enabled"):
                raise ValueError("coordinate_tokens_enabled must be explicitly configured in config")
            coordinate_tokens_enabled = self.config.coordinate_tokens_enabled
            
            # Try to get max_coord_value from coordinate_config section first, then fallback
            if hasattr(self.config, "coordinate_config") and hasattr(self.config.coordinate_config, "max_coord_value"):
                max_coord_value = self.config.coordinate_config.max_coord_value
            elif hasattr(self.config, "coordinate_config_max_coord_value"):
                max_coord_value = self.config.coordinate_config_max_coord_value
            else:
                self.logger.warning("⚠️ max_coord_value not found in config, using default 2048")
                max_coord_value = 2048

        # Get additional coordinate token configuration
        use_official_box_tokens = True  # Default
        if hasattr(self.config, "coordinate_config") and hasattr(self.config.coordinate_config, "use_official_box_tokens"):
            use_official_box_tokens = self.config.coordinate_config.use_official_box_tokens
        elif hasattr(self.config, "chat_processor") and hasattr(self.config.chat_processor, "use_official_box_tokens"):
            use_official_box_tokens = self.config.chat_processor.use_official_box_tokens

        self.logger.debug(f"🎯 CHAT PROCESSOR CONFIG: enabled={coordinate_tokens_enabled}, max_coord={max_coord_value}, box_tokens={use_official_box_tokens}")

        self.chat_processor = ChatProcessor(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            model_max_length=model_max_length,
            enable_coordinate_tokens=coordinate_tokens_enabled,
            max_coord_value=max_coord_value,
            use_official_box_tokens=use_official_box_tokens,
        )

        if coordinate_tokens_enabled:
            self.logger.info(
                f"✅ Chat processor initialized with coordinate tokens (max_coord_value: {max_coord_value})"
            )
        else:
            self.logger.info(
                "✅ Chat processor initialized (coordinate tokens disabled)"
            )

    def _init_teacher_pool_manager(self) -> None:
        """Initialize teacher pool manager if available."""
        teacher_pool_file = (
            self.config.data.teacher_pool_file
            if self.use_new_config
            else self.config.teacher_pool_file
        )

        self.teacher_pool_manager = None
        if teacher_pool_file and Path(teacher_pool_file).exists():
            self.teacher_pool_manager = create_teacher_pool_manager()
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
        )

        # Create evaluation dataset (no teachers)
        eval_dataset = BBUDataset(
            data_path=data_config["val_data_path"],
            chat_processor=self.chat_processor,
            teacher_pool_manager=None,  # No teachers for evaluation
            teacher_ratio=0.0,
            is_training=False,
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

        collator_type = (
            self.config.data.collator_type
            if self.use_new_config
            else self.config.collator_type
        )

        data_collator = create_data_collator(self.tokenizer, collator_type)
        self.logger.info(f"✅ Data collator created: {collator_type}")
        return data_collator

    def _get_data_config(self) -> dict:
        """Get data configuration parameters."""
        if self.use_new_config:
            return {
                "train_data_path": self.config.data.train_data_path,
                "val_data_path": self.config.data.val_data_path,
                "teacher_ratio": self.config.data.teacher_ratio,
            }
        else:
            return {
                "train_data_path": self.config.train_data_path,
                "val_data_path": self.config.val_data_path,
                "teacher_ratio": self.config.teacher_ratio,
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
            "chat_processor_max_length": self.chat_processor.model_max_length,
        }

        return stats

    @classmethod
    def create_datasets_and_collator(
        cls, tokenizer: Any, image_processor: Any, use_new_config: bool = False
    ) -> Tuple[BBUDataset, BBUDataset, Any]:
        """
        Convenience method to create datasets and collator in one call.

        Args:
            tokenizer: Model tokenizer
            image_processor: Image processor
            use_new_config: Whether to use new domain-specific config system

        Returns:
            Tuple of (train_dataset, eval_dataset, data_collator)
        """
        processor = cls(tokenizer, image_processor, use_new_config)
        train_dataset, eval_dataset = processor.create_datasets()
        data_collator = processor.create_data_collator()
        return train_dataset, eval_dataset, data_collator
