import os
from dataclasses import dataclass

from transformers import TrainingArguments


@dataclass
class Config:
    """
    Unified Configuration class - All values must be explicitly provided via YAML.
    Environment and GPU/distributed settings are handled by the launcher script.

    This ensures all configuration is explicit and prevents accidental reliance on hidden defaults.
    """

    def __init__(self, **kwargs):
        """
        Initialize config with explicit values from YAML.
        Environment and GPU settings are handled by launcher script.

        Raises:
            KeyError: If any required configuration is missing
            ValueError: If configuration values are invalid
        """

        # Validate that we have all required keys
        missing_keys = []

        # =====================================================================
        # MODEL SETTINGS - REQUIRED
        # =====================================================================
        try:
            self.model_path: str = kwargs["model_path"]
        except KeyError:
            missing_keys.append("model_path")

        try:
            self.model_size: str = kwargs["model_size"]
        except KeyError:
            missing_keys.append("model_size")

        try:
            self.model_max_length: int = kwargs["model_max_length"]
        except KeyError:
            missing_keys.append("model_max_length")

        try:
            self.cache_dir: str = kwargs["cache_dir"]
        except KeyError:
            missing_keys.append("cache_dir")

        try:
            self.attn_implementation: str = kwargs["attn_implementation"]
        except KeyError:
            missing_keys.append("attn_implementation")

        try:
            self.torch_dtype: str = kwargs["torch_dtype"]
        except KeyError:
            missing_keys.append("torch_dtype")

        try:
            self.use_cache: bool = kwargs["use_cache"]
        except KeyError:
            missing_keys.append("use_cache")

        # =====================================================================
        # DATA SETTINGS - REQUIRED
        # =====================================================================
        try:
            self.train_data_path: str = kwargs["train_data_path"]
        except KeyError:
            missing_keys.append("train_data_path")

        try:
            self.val_data_path: str = kwargs["val_data_path"]
        except KeyError:
            missing_keys.append("val_data_path")

        try:
            self.data_root: str = kwargs["data_root"]
        except KeyError:
            missing_keys.append("data_root")

        # REMOVED: max_pixels and min_pixels completely
        # Official QwenVL approach uses processor defaults without overrides

        # =====================================================================
        # LEARNING RATE SETTINGS - REQUIRED
        # =====================================================================
        try:
            self.vision_lr: float = kwargs["vision_lr"]
        except KeyError:
            missing_keys.append("vision_lr")

        try:
            self.mlp_lr: float = kwargs["mlp_lr"]
        except KeyError:
            missing_keys.append("mlp_lr")

        try:
            self.llm_lr: float = kwargs["llm_lr"]
        except KeyError:
            missing_keys.append("llm_lr")

        try:
            self.learning_rate: float = kwargs["learning_rate"]
        except KeyError:
            missing_keys.append("learning_rate")

        # =====================================================================
        # TRAINING SETTINGS - REQUIRED
        # =====================================================================
        try:
            self.num_train_epochs: int = kwargs["num_train_epochs"]
        except KeyError:
            missing_keys.append("num_train_epochs")

        try:
            self.per_device_train_batch_size: int = kwargs[
                "per_device_train_batch_size"
            ]
        except KeyError:
            missing_keys.append("per_device_train_batch_size")

        try:
            self.per_device_eval_batch_size: int = kwargs["per_device_eval_batch_size"]
        except KeyError:
            missing_keys.append("per_device_eval_batch_size")

        try:
            self.total_batch_size: int = kwargs["total_batch_size"]
        except KeyError:
            missing_keys.append("total_batch_size")

        try:
            self.gradient_accumulation_steps: int = kwargs[
                "gradient_accumulation_steps"
            ]
        except KeyError:
            missing_keys.append("gradient_accumulation_steps")

        # =====================================================================
        # TRAINING OPTIMIZATION - REQUIRED
        # =====================================================================
        try:
            self.warmup_ratio: float = kwargs["warmup_ratio"]
        except KeyError:
            missing_keys.append("warmup_ratio")

        try:
            self.lr_scheduler_type: str = kwargs["lr_scheduler_type"]
        except KeyError:
            missing_keys.append("lr_scheduler_type")

        try:
            self.max_grad_norm: float = kwargs["max_grad_norm"]
        except KeyError:
            missing_keys.append("max_grad_norm")

        try:
            self.weight_decay: float = kwargs["weight_decay"]
        except KeyError:
            missing_keys.append("weight_decay")

        try:
            self.gradient_checkpointing: bool = kwargs["gradient_checkpointing"]
        except KeyError:
            missing_keys.append("gradient_checkpointing")

        # =====================================================================
        # PRECISION SETTINGS - REQUIRED
        # =====================================================================
        try:
            self.bf16: bool = kwargs["bf16"]
        except KeyError:
            missing_keys.append("bf16")

        try:
            self.fp16: bool = kwargs["fp16"]
        except KeyError:
            missing_keys.append("fp16")

        # =====================================================================
        # EVALUATION AND SAVING - REQUIRED
        # =====================================================================
        try:
            self.eval_strategy: str = kwargs["eval_strategy"]
        except KeyError:
            missing_keys.append("eval_strategy")

        try:
            self.eval_steps: int = kwargs["eval_steps"]
        except KeyError:
            missing_keys.append("eval_steps")

        try:
            self.save_strategy: str = kwargs["save_strategy"]
        except KeyError:
            missing_keys.append("save_strategy")

        try:
            self.save_steps: int = kwargs["save_steps"]
        except KeyError:
            missing_keys.append("save_steps")

        try:
            self.save_total_limit: int = kwargs["save_total_limit"]
        except KeyError:
            missing_keys.append("save_total_limit")

        # =====================================================================
        # LOGGING - REQUIRED (except logging_dir which can be None)
        # =====================================================================
        try:
            self.logging_steps: int = kwargs["logging_steps"]
        except KeyError:
            missing_keys.append("logging_steps")

        # logging_dir can be None (will be auto-generated)
        self.logging_dir: str = kwargs.get("logging_dir")

        try:
            self.log_level: str = kwargs["log_level"]
        except KeyError:
            missing_keys.append("log_level")

        try:
            self.report_to: str = kwargs["report_to"]
        except KeyError:
            missing_keys.append("report_to")

        try:
            self.verbose: bool = kwargs["verbose"]
        except KeyError:
            missing_keys.append("verbose")

        # =====================================================================
        # LOSS CONFIGURATION - REQUIRED
        # =====================================================================
        try:
            self.loss_type: str = kwargs["loss_type"]
        except KeyError:
            missing_keys.append("loss_type")

        try:
            self.lm_weight: float = kwargs["lm_weight"]
        except KeyError:
            missing_keys.append("lm_weight")

        try:
            self.bbox_weight: float = kwargs["bbox_weight"]
        except KeyError:
            missing_keys.append("bbox_weight")

        try:
            self.giou_weight: float = kwargs["giou_weight"]
        except KeyError:
            missing_keys.append("giou_weight")

        try:
            self.class_weight: float = kwargs["class_weight"]
        except KeyError:
            missing_keys.append("class_weight")

        try:
            self.hungarian_matching: bool = kwargs["hungarian_matching"]
        except KeyError:
            missing_keys.append("hungarian_matching")

        try:
            self.detection_mode: str = kwargs["detection_mode"]
        except KeyError:
            missing_keys.append("detection_mode")

        try:
            self.inference_frequency: int = kwargs["inference_frequency"]
        except KeyError:
            missing_keys.append("inference_frequency")

        try:
            self.max_generation_length: int = kwargs["max_generation_length"]
        except KeyError:
            missing_keys.append("max_generation_length")

        try:
            self.use_semantic_similarity: bool = kwargs["use_semantic_similarity"]
        except KeyError:
            missing_keys.append("use_semantic_similarity")

        # =====================================================================
        # PERFORMANCE CONFIGURATION - REQUIRED
        # =====================================================================
        try:
            self.dataloader_num_workers: int = kwargs["dataloader_num_workers"]
        except KeyError:
            missing_keys.append("dataloader_num_workers")

        try:
            self.pin_memory: bool = kwargs["pin_memory"]
        except KeyError:
            missing_keys.append("pin_memory")

        try:
            self.prefetch_factor: int = kwargs["prefetch_factor"]
        except KeyError:
            missing_keys.append("prefetch_factor")

        try:
            self.data_flatten: bool = kwargs["data_flatten"]
        except KeyError:
            missing_keys.append("data_flatten")

        try:
            self.batching_strategy: str = kwargs["batching_strategy"]
        except KeyError:
            missing_keys.append("batching_strategy")

        try:
            self.remove_unused_columns: bool = kwargs["remove_unused_columns"]
        except KeyError:
            missing_keys.append("remove_unused_columns")

        try:
            self.collator_type: str = kwargs["collator_type"]
        except KeyError:
            missing_keys.append("collator_type")

        # =====================================================================
        # DEEPSPEED - HANDLED BY LAUNCHER SCRIPT
        # =====================================================================
        # DeepSpeed settings are now controlled by the launcher script via environment variables
        # These are no longer required in YAML files
        deepspeed_enabled_str = os.getenv("BBU_DEEPSPEED_ENABLED", "false").lower()
        self.deepspeed_enabled: bool = deepspeed_enabled_str == "true"
        self.deepspeed_config_file: str = os.getenv(
            "BBU_DEEPSPEED_CONFIG", "scripts/zero2.json"
        )

        # =====================================================================
        # OUTPUT - REQUIRED (except run_name which can be None)
        # =====================================================================
        try:
            self.output_base_dir: str = kwargs["output_base_dir"]
        except KeyError:
            missing_keys.append("output_base_dir")

        # run_name can be None (will be auto-generated)
        self.run_name: str = kwargs.get("run_name")

        try:
            self.tb_dir: str = kwargs["tb_dir"]
        except KeyError:
            missing_keys.append("tb_dir")

        # =====================================================================
        # STABILITY AND MONITORING - REQUIRED
        # =====================================================================
        try:
            self.max_consecutive_nan: int = kwargs["max_consecutive_nan"]
        except KeyError:
            missing_keys.append("max_consecutive_nan")

        try:
            self.max_consecutive_zero: int = kwargs["max_consecutive_zero"]
        except KeyError:
            missing_keys.append("max_consecutive_zero")

        try:
            self.max_nan_ratio: float = kwargs["max_nan_ratio"]
        except KeyError:
            missing_keys.append("max_nan_ratio")

        try:
            self.nan_monitoring_window: int = kwargs["nan_monitoring_window"]
        except KeyError:
            missing_keys.append("nan_monitoring_window")

        try:
            self.allow_occasional_nan: bool = kwargs["allow_occasional_nan"]
        except KeyError:
            missing_keys.append("allow_occasional_nan")

        try:
            self.nan_recovery_enabled: bool = kwargs["nan_recovery_enabled"]
        except KeyError:
            missing_keys.append("nan_recovery_enabled")

        try:
            self.learning_rate_reduction_factor: float = kwargs[
                "learning_rate_reduction_factor"
            ]
        except KeyError:
            missing_keys.append("learning_rate_reduction_factor")

        try:
            self.gradient_clip_reduction_factor: float = kwargs[
                "gradient_clip_reduction_factor"
            ]
        except KeyError:
            missing_keys.append("gradient_clip_reduction_factor")

        # =====================================================================
        # DEBUG MODE - REQUIRED
        # =====================================================================
        try:
            self.test_samples: int = kwargs["test_samples"]
        except KeyError:
            missing_keys.append("test_samples")

        try:
            self.test_forward_pass: bool = kwargs["test_forward_pass"]
        except KeyError:
            missing_keys.append("test_forward_pass")

        # =====================================================================
        # FAIL HARD if any required keys are missing
        # =====================================================================
        if missing_keys:
            raise KeyError(
                f"❌ MISSING REQUIRED CONFIGURATION KEYS: {missing_keys}\n"
                f"💡 All configuration must be explicitly provided in YAML files.\n"
                f"💡 No default values are allowed to ensure explicit configuration.\n"
                f"💡 Please add these keys to your YAML config file."
            )

        # =====================================================================
        # DERIVED/COMPUTED PROPERTIES
        # =====================================================================
        self.batch_size = self.per_device_train_batch_size  # Alias for compatibility
        self.num_epochs = self.num_train_epochs  # Alias for compatibility

        # Set output_dir from components
        if self.run_name:
            self.output_dir = f"{self.output_base_dir}/{self.run_name}"
        else:
            self.output_dir = self.output_base_dir

        # Set deepspeed path if enabled
        if self.deepspeed_enabled:
            self.deepspeed = self.deepspeed_config_file
        else:
            self.deepspeed = None

        # Set log_dir for compatibility
        self.log_dir = "logs"

        # Initialize GPU/distributed settings as None - will be set by launcher
        self.nproc_per_node = None
        self.cuda_visible_devices = None
        self.master_addr = None
        self.master_port = None

    @property
    def tune_vision(self) -> bool:
        """Auto-determine if vision encoder should be trained based on learning rate."""
        return self.vision_lr > 0

    @property
    def tune_mlp(self) -> bool:
        """Auto-determine if MLP connector should be trained based on learning rate."""
        return self.mlp_lr > 0

    @property
    def tune_llm(self) -> bool:
        """Auto-determine if LLM should be trained based on learning rate."""
        return self.llm_lr > 0

    @property
    def use_differential_lr(self) -> bool:
        """Auto-determine if differential learning rates should be used."""
        lrs = [self.vision_lr, self.mlp_lr, self.llm_lr]
        active_lrs = [lr for lr in lrs if lr > 0]
        return len(set(active_lrs)) > 1  # More than one unique non-zero learning rate

    def validate(self):
        """
        Validate configuration for missing or invalid values.
        Raises ValueError if any required configuration is missing or invalid.
        Environment and GPU settings are handled by launcher script.
        """
        # Check that at least one learning rate is non-zero
        if self.vision_lr == 0 and self.mlp_lr == 0 and self.llm_lr == 0:
            raise ValueError(
                "At least one learning rate (vision_lr, mlp_lr, llm_lr) must be > 0"
            )

        # Check use_cache and gradient_checkpointing compatibility
        if self.use_cache and self.gradient_checkpointing:
            raise ValueError(
                "use_cache=True is incompatible with gradient_checkpointing=True. "
                "Set either use_cache=False or gradient_checkpointing=False."
            )

        # Check paths exist
        from pathlib import Path

        if not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")

        data_root = Path(self.data_root)
        train_path = data_root / self.train_data_path
        val_path = data_root / self.val_data_path

        if not train_path.exists():
            raise ValueError(f"Training data not found: {train_path}")
        if not val_path.exists():
            raise ValueError(f"Validation data not found: {val_path}")

    def update_gpu_config(
        self,
        nproc_per_node: int,
        cuda_visible_devices: str,
        deepspeed_enabled: bool,
        deepspeed_config_file: str,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ):
        """
        Update GPU configuration from launcher script.
        Called by launcher to set distributed training parameters.

        Args:
            nproc_per_node: Number of processes per node (REQUIRED)
            cuda_visible_devices: CUDA visible devices string (REQUIRED)
            deepspeed_enabled: Whether to enable DeepSpeed (REQUIRED)
            deepspeed_config_file: Path to DeepSpeed config file (REQUIRED)
            master_addr: Master address for distributed training
            master_port: Master port for distributed training
        """
        self.nproc_per_node = nproc_per_node
        self.cuda_visible_devices = cuda_visible_devices
        self.deepspeed_enabled = deepspeed_enabled
        self.deepspeed_config_file = deepspeed_config_file
        self.master_addr = master_addr
        self.master_port = master_port

        # Update derived properties
        if self.deepspeed_enabled:
            self.deepspeed = self.deepspeed_config_file
        else:
            self.deepspeed = None

        # Auto-calculate gradient accumulation steps
        if self.nproc_per_node > 0 and self.per_device_train_batch_size > 0:
            expected_grad_accum = self.total_batch_size // (
                self.nproc_per_node * self.per_device_train_batch_size
            )
            if expected_grad_accum > 0:
                self.gradient_accumulation_steps = expected_grad_accum

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Create Config instance from dictionary (e.g., from YAML).

        Args:
            config_dict: Dictionary containing all configuration values

        Returns:
            Config: Configured instance
        """
        # Define required keys (GPU settings are optional - handled by launcher)
        required_keys = [
            # Model settings
            "model_path",
            "model_size",
            "model_max_length",
            "cache_dir",
            "attn_implementation",
            "torch_dtype",
            "use_cache",
            # Data settings
            "train_data_path",
            "val_data_path",
            "data_root",
            # Learning rates
            "vision_lr",
            "mlp_lr",
            "llm_lr",
            "learning_rate",
            # Training settings
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "total_batch_size",
            "gradient_accumulation_steps",
            # Training optimization
            "warmup_ratio",
            "lr_scheduler_type",
            "max_grad_norm",
            "weight_decay",
            "gradient_checkpointing",
            # Precision
            "bf16",
            "fp16",
            # Evaluation and saving
            "eval_strategy",
            "eval_steps",
            "save_strategy",
            "save_steps",
            "save_total_limit",
            # Logging
            "logging_steps",
            "log_level",
            "report_to",
            "verbose",
            # Loss configuration
            "loss_type",
            "lm_weight",
            "bbox_weight",
            "giou_weight",
            "class_weight",
            "hungarian_matching",
            "detection_mode",
            "inference_frequency",
            "max_generation_length",
            "use_semantic_similarity",
            # Performance
            "dataloader_num_workers",
            "pin_memory",
            "prefetch_factor",
            "data_flatten",
            "batching_strategy",
            "remove_unused_columns",
            # DeepSpeed - handled by launcher script, not required in YAML
            # Output
            "output_base_dir",
            "tb_dir",
            # Stability and monitoring
            "max_consecutive_nan",
            "max_consecutive_zero",
            "max_nan_ratio",
            "nan_monitoring_window",
            "allow_occasional_nan",
            "nan_recovery_enabled",
            "learning_rate_reduction_factor",
            "gradient_clip_reduction_factor",
            # Debug mode
            "test_samples",
            "test_forward_pass",
        ]

        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        # Create instance
        instance = cls(**config_dict)

        # Validate the configuration
        instance.validate()

        return instance


def create_training_arguments_with_deepspeed(config: Config) -> TrainingArguments:
    """
    Create TrainingArguments with proper DeepSpeed configuration.
    Matches the official qwen-vl-finetune approach exactly.
    """
    # Core training parameters - following official approach
    args = {
        "output_dir": config.output_dir,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "max_grad_norm": config.max_grad_norm,
        "weight_decay": config.weight_decay,
    }

    # Evaluation and saving
    args.update(
        {
            "eval_strategy": config.eval_strategy,
            "eval_steps": config.eval_steps,
            "save_strategy": config.save_strategy,
            "save_steps": config.save_steps,
            "save_total_limit": config.save_total_limit,
            "logging_steps": config.logging_steps,
        }
    )

    # Performance and optimization - following official approach
    args.update(
        {
            "bf16": config.bf16,
            "gradient_checkpointing": config.gradient_checkpointing,
            "remove_unused_columns": config.remove_unused_columns,
            "dataloader_num_workers": config.dataloader_num_workers,
            "dataloader_pin_memory": config.pin_memory,
            "report_to": [config.report_to]
            if isinstance(config.report_to, str)
            else config.report_to,
        }
    )

    # CRITICAL: Memory management - Enable automatic cache clearing
    args.update(
        {
            "torch_empty_cache_steps": 1,  # Clear cache every step to prevent memory accumulation
        }
    )

    # Optional parameters
    if config.logging_dir is not None:
        args["logging_dir"] = config.logging_dir
    if config.run_name is not None:
        args["run_name"] = config.run_name

    # CRITICAL: DeepSpeed configuration - SIMPLE approach like official
    if config.deepspeed_enabled:
        args["deepspeed"] = config.deepspeed_config_file
        print(f"🚀 DeepSpeed enabled: {config.deepspeed_config_file}")
    else:
        print("📱 Single GPU training (DeepSpeed disabled)")

    return TrainingArguments(**args)
