"""
HuggingFace-first dataset for Qwen2.5-VL training.

This is a clean refactored version that replaces the complex custom processing
with official HuggingFace components + focused coordinate token logic.

Key Features:
- Uses ConversationProcessor with official HuggingFace processor
- Eliminates 800+ lines of custom conversation processing
- Fail-fast validation with explicit errors
- Clean separation of concerns
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import Qwen2VLImageProcessor, Qwen2VLProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src_new_json.config.config import Config
from src_new_json.config.augmentation_config import SmartResizeConfig
from src_new_json.data.teacher_pool import TeacherPoolManager
from src_new_json.processing.conversation import ConversationBuilder
from src_new_json.processing.special_tokens import (
    ASSISTANT_SPAN_PATTERN,
    IM_END,
    IMAGE_PAD,
)
from src_new_json.types.arrays import (
    jaxtyped_beartype,
)
from src_new_json.utils.path_manager import create_path_manager
from src_new_json.utils.rank_aware_logging import get_rank_aware_logger


def get_data_logger() -> logging.Logger:
    """Get rank-aware logger for data module."""
    try:
        return get_rank_aware_logger("data")
    except Exception:
        # Fallback to standard logging
        logger = logging.getLogger("data")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_data_logger()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries.

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


class Dataset(TorchDataset):
    """HuggingFace-first dataset with clean conversation processing."""

    # Class attribute type annotations (non-trivial state)
    data_path: str
    tokenizer: PreTrainedTokenizerBase
    image_processor: Optional[Qwen2VLImageProcessor]
    teacher_pool_manager: Optional[TeacherPoolManager]
    config: Config

    data_root: str
    teacher_ratio: float
    num_teacher_samples: int

    hf_processor: Optional[Qwen2VLProcessor]
    conversation_processor: Optional[ConversationBuilder]

    raw_data: List[Dict[str, Any]]
    samples: List[Dict[str, Any]]

    teacher_assignments: Dict[str, Any]
    teacher_assignment_counts: Dict[str, int]

    # Augmentation curriculum
    augmentation_pipeline: Optional[object]
    _augmentation_schedule: Optional[List[Dict[str, Any]]] = None

    # Split flag
    is_eval: bool = False

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: Optional[Qwen2VLImageProcessor],
        teacher_pool_manager: Optional[TeacherPoolManager],
        config: Config,
    ):
        """
        Initialize HuggingFace-first dataset.

        Args:
            data_path: Path to JSONL file
            tokenizer: Tokenizer (not used in HF-first approach)
            image_processor: Image processor (not used in HF-first approach)
            teacher_pool_manager: Manager for teacher examples
            config: Configuration object

        Raises:
            ValueError: If required parameters are invalid
        """
        if not data_path:
            raise ValueError("data_path cannot be empty")

        if config is None:
            raise ValueError("config cannot be None")

        # Store configuration
        self.data_path = data_path
        self.tokenizer = tokenizer  # Kept for compatibility, not used
        self.image_processor = image_processor  # Kept for compatibility, not used
        self.teacher_pool_manager = teacher_pool_manager
        self.config = config

        # Mark split
        try:
            # Use normalized paths from config; load_config already normalizes
            self.is_eval = str(Path(self.data_path)) == str(Path(self.config.val_data_path))
        except Exception:
            self.is_eval = False

        # Initialize processing components
        self._initialize_processing_components()

        # Load and validate data
        self.raw_data = self._load_data()
        self.samples = self._validate_and_filter_samples(self.raw_data)
        # Initialize teacher assignment tracking
        self.teacher_assignments = {}
        self.teacher_assignment_counts = {}
        
        # Hardness EMA cache (student loss per sample)
        self._hardness_ema_alpha = float(self.config.hardness_alpha)
        self._hardness_warmup_epochs = int(self.config.hardness_warmup_epochs)
        # Size hardness EMA to dataset length (initialized to zeros)
        try:
            self._hardness_ema = [0.0 for _ in range(len(self.samples))]
        except Exception:
            self._hardness_ema = []
        self._hardness_weights = None

        # Initialize dynamic pairing structures and build epoch-0 map
        try:
            self._episode_map = None  # target_idx -> {"context_idx": int|None}
            self.set_epoch(0)
        except Exception as e:
            logger.warning(f"⚠️ set_epoch(0) during dataset init failed: {e}")

        logger.info(
            f"✅ HuggingFace-first dataset initialized with {len(self.samples)} samples"
        )

    def _initialize_processing_components(self):
        """Initialize HuggingFace-first processing components."""
        # Data processing settings
        self.data_root = self.config.data_root
        # Eval split must be single-turn; force teacher_ratio=0.0
        self.teacher_ratio = 0.0 if self.is_eval else self.config.teacher_ratio
        self.num_teacher_samples = self.config.num_teacher_samples

        # HuggingFace processor will be set by trainer
        self.hf_processor = None
        self.conversation_processor = None

        # Augmentation — fail-fast based on explicit configuration
        if not hasattr(self.config, "use_aug"):
            raise ValueError(
                "Missing required boolean 'use_aug' in config. Set use_aug: true|false explicitly in YAML."
            )

        self.augmentation_pipeline = None
        self._augmentation_schedule = self.config.augmentation_schedule
        self._smart_resize_defaults = _normalize_smart_resize_config(
            getattr(self.config, "augmentation_smart_resize_defaults", None)
        )
        if (
            self._smart_resize_defaults is None
            and getattr(self.config, "augmentation", None) is not None
            and hasattr(self.config.augmentation, "smart_resize")
        ):
            self._smart_resize_defaults = _normalize_smart_resize_config(
                getattr(self.config.augmentation, "smart_resize", None)
            )

        if bool(self.config.use_aug):
            # If a schedule is provided, set initial preset at epoch 0; else require augmentation block
            if self._augmentation_schedule:
                self.set_epoch(0)
                logger.info("✅ Augmentation enabled via schedule (epoch 0 applied)")
            else:
                if (
                    not hasattr(self.config, "augmentation")
                    or self.config.augmentation is None
                ):
                    raise ValueError(
                        "use_aug is true but 'augmentation' block is missing. Provide an explicit preset in YAML."
                    )
                # Accept either a ready AugmentationConfig or a preset wrapper
                from src_new_json.augmentation import ObjectAwareAugmentationPipeline
                from src_new_json.config.augmentation_config import (
                    AugmentationConfig as _AugCfg,
                )

                if isinstance(self.config.augmentation, _AugCfg):
                    self.augmentation_pipeline = (
                        ObjectAwareAugmentationPipeline.from_config(
                            self.config.augmentation
                        )
                    )
                else:
                    # Expect attributes: preset, rng_seed, apply_to_teachers
                    from src_new_json.augmentation.wrappers import get_preset_config

                    smart_resize_attr = _normalize_smart_resize_config(
                        getattr(self.config.augmentation, "smart_resize", None)
                    )
                    if smart_resize_attr is None:
                        smart_resize_attr = (
                            dict(self._smart_resize_defaults)
                            if self._smart_resize_defaults is not None
                            else None
                        )

                    cfg = get_preset_config(
                        self.config.augmentation.preset,
                        rng_seed=int(self.config.augmentation.rng_seed),
                        apply_to_teachers=bool(
                            self.config.augmentation.apply_to_teachers
                        ),
                        smart_resize=smart_resize_attr,
                    )
                    self.augmentation_pipeline = (
                        ObjectAwareAugmentationPipeline.from_config(cfg)
                    )
                logger.info("✅ Augmentation pipeline initialized (from config)")
        else:
            # No augmentation; ensure pipeline is None
            self.augmentation_pipeline = None
            logger.info("🔧 Augmentation disabled (use_aug=false)")

        logger.info("🎯 HuggingFace-first processing components initialized")

    def set_epoch(self, epoch_index: int) -> None:
        """Optional hook for trainer: update augmentation preset and dynamic pairing by epoch.

        - Applies augmentation schedule when provided
        - Builds dynamic contrastive pairing episode mapping per epoch when enabled
        """
        # Augmentation schedule (apply only when augmentation is enabled)
        if not bool(self.config.use_aug):
            # Ensure pipeline stays cleared when augmentation is disabled
            if getattr(self, "augmentation_pipeline", None) is not None:
                self.augmentation_pipeline = None
                logger.info("🔒 Augmentation disabled by config; pipeline cleared")
        elif self._augmentation_schedule:
            from src_new_json.augmentation import ObjectAwareAugmentationPipeline
            from src_new_json.augmentation.wrappers import get_preset_config

            active = None
            for entry in sorted(
                self._augmentation_schedule, key=lambda e: int(e["start_epoch"])
            ):
                if epoch_index >= int(entry["start_epoch"]):
                    active = entry
            if active is not None:
                preset_name = str(active["preset"])
                smart_resize_payload = _normalize_smart_resize_config(
                    active.get("smart_resize")
                )
                if (
                    smart_resize_payload is None
                    and preset_name != "off"
                ):
                    if self._smart_resize_defaults is None:
                        raise ValueError(
                            f"No smart_resize defaults available for preset '{preset_name}'."
                        )
                    smart_resize_payload = dict(self._smart_resize_defaults)
                cfg = get_preset_config(
                    preset_name,
                    rng_seed=self.config.seed,
                    smart_resize=smart_resize_payload,
                )
                self.augmentation_pipeline = ObjectAwareAugmentationPipeline.from_config(cfg)
                logger.info(
                    f"🔁 Augmentation preset switched at epoch {epoch_index}: {active['preset']}"
                )

        # Dynamic contrastive pairing mapping (training only)
        try:
            if (
                not self.is_eval
                and self.config.dynamic_pairing_enabled
                and hasattr(self, "samples")
                and isinstance(self.samples, list)
                and len(self.samples) > 0
            ):
                from src_new_json.sampling import BucketedSamplingEngine, SamplingConfig as _SampCfg
                base_seed = int(self.config.seed)
                # Reuse a persistent engine to leverage caches across epochs
                if not hasattr(self, "_pair_engine") or (self._pair_engine is None):
                    self._pair_engine = BucketedSamplingEngine()
                engine = self._pair_engine
                pair_cfg = _SampCfg(
                    temperature=float(self.config.dynamic_pair_temperature),
                    target_assignment=str(self.config.dynamic_pair_target_assignment),
                    cross_bucket_explore_prob=float(self.config.dynamic_pair_cross_bucket_explore_prob),
                    pool_fraction=float(self.config.pool_fraction),
                    pool_max=int(self.config.pool_max),
                )
                # Normalize hardness EMA to weights per epoch (after warmup only)
                sample_weights = None
                if epoch_index >= self._hardness_warmup_epochs and len(self._hardness_ema) == len(self.samples):
                    try:
                        arr = torch.tensor(self._hardness_ema, dtype=torch.float32)
                        if torch.isfinite(arr).any():
                            # Robust scaling via clipped percentiles
                            p10 = torch.quantile(arr, 0.10)
                            p90 = torch.quantile(arr, 0.90)
                            denom = max(1e-6, float((p90 - p10).item()))
                            norm = torch.clamp((arr - p10) / denom, 0.0, 1.0)
                            sample_weights = [float(x) for x in norm.tolist()]
                    except Exception as e:
                        logger.debug(f"hardness normalization skipped: {e}")

                # Force eval split single-turn by zeroing teacher_ratio in build
                teacher_ratio = 0.0 if self.is_eval else float(self.teacher_ratio)
                # Derive rank and worker_id for deterministic seeding across DDP/DataLoader workers
                try:
                    rank = 0
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        rank = int(torch.distributed.get_rank())
                except Exception:
                    rank = 0
                try:
                    # DataLoader worker id; default 0 if main process or dataset used outside workers
                    from torch.utils.data import get_worker_info
                    wi = get_worker_info()
                    worker_id = int(wi.id) if wi is not None else 0
                except Exception:
                    worker_id = 0

                episode_map, metrics = engine.build_epoch_map(
                    samples=self.samples,
                    cfg=pair_cfg,
                    base_seed=base_seed,
                    epoch_idx=int(epoch_index),
                    teacher_ratio=teacher_ratio,
                    is_eval=bool(self.is_eval),
                    sample_weights=sample_weights,
                    rank=rank,
                    worker_id=worker_id,
                )
                # Materialize: target_idx -> {context_idx}
                self._episode_map = {spec.target_idx: {"context_idx": spec.context_idx} for spec in episode_map.values()}
                logger.info(
                    f"🎯 Dynamic pairing @epoch {epoch_index}: pair_episodes={metrics.pair_episodes}, single_episodes~={metrics.single_episodes}, target_is_current_pct={metrics.target_is_current_pct:.1f}%"
                )
                logger.info(
                    f"📊 Context coverage≈{metrics.coverage_context_unique_pct:.1f}% hist={metrics.context_usage_histogram}"
                )
                # Log a few concrete pair examples to prove usage
                try:
                    example_pairs = []
                    for k in sorted(self._episode_map.keys())[:5]:
                        v = self._episode_map[k]
                        example_pairs.append((int(k), int(v["context_idx"]) if v["context_idx"] is not None else None))
                    if example_pairs:
                        logger.info(f"🔗 Pair examples (student→teacher): {example_pairs}")
                except Exception:
                    pass
            elif self.is_eval:
                # Eval split: enforce single-turn
                self._episode_map = {}
                logger.info("🔒 Eval split: forced single-turn episodes (no context)")
        except Exception as e:
            logger.warning(f"⚠️ Dynamic pairing set_epoch failed: {e}")

    def set_processor(self, hf_processor: Qwen2VLProcessor) -> None:
        """
        Set the HuggingFace processor and initialize conversation processor.

        This method should be called by the trainer after the processor is available.

        Args:
            hf_processor: Official HuggingFace Qwen2VLProcessor

        Raises:
            ValueError: If hf_processor is None
        """
        if hf_processor is None:
            raise ValueError("hf_processor cannot be None")

        self.hf_processor = hf_processor

        from src_new_json.processing.conversation import ConversationBuilder

        self.conversation_processor = ConversationBuilder(
            processor=hf_processor,
        )

        logger.info("✅ HuggingFace processor and conversation processor initialized")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        try:
            return read_jsonl(self.data_path)
        except Exception as e:
            raise ValueError(f"Failed to load data from {self.data_path}: {e}")

    def _validate_and_filter_samples(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and filter samples."""
        if not raw_data:
            raise ValueError("No data found in JSONL file")

        valid_samples = []
        for i, sample in enumerate(raw_data):
            if self._is_valid_sample(sample):
                valid_samples.append(sample)
            else:
                logger.warning(f"Skipping invalid sample {i}")

        if not valid_samples:
            raise ValueError("No valid samples found after filtering")

        # Apply max_dataset_size limit if specified in config
        # -1 means use all samples, None or 0 means no limit, positive values limit the dataset
        max_dataset_size = self.config.max_dataset_size
        if max_dataset_size is not None and max_dataset_size > 0:
            logger.debug(
                f"🔧 DEBUG MODE: Limiting dataset from {len(valid_samples)} to {max_dataset_size} samples"
            )
            valid_samples = valid_samples[:max_dataset_size]
        elif max_dataset_size == -1:
            logger.info(
                f"📊 Using all {len(valid_samples)} samples (max_dataset_size=-1)"
            )

        logger.info(
            f"Filtered {len(valid_samples)} valid samples from {len(raw_data)} total"
        )
        return valid_samples

    def _is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        """Check if sample is valid."""
        # Must have objects
        if "objects" not in sample or not sample["objects"]:
            return False

        # Must have images
        if "images" not in sample or not sample["images"]:
            return False

        from src_new_json.types.geometry import (
            is_valid_box_coords,
            is_valid_line_coords,
            is_valid_quad_coords,
        )

        # Strict: training input uses bbox_2d | quad | line only
        legacy_keys = ("bbox_2d", "quad", "line")

        for obj in sample["objects"]:
            if "desc" not in obj:
                return False

            present = [k for k in legacy_keys if k in obj]
            if len(present) != 1:
                return False

            g = present[0]
            coords = obj[g]
            if g == "bbox_2d":
                if not (isinstance(coords, list) and len(coords) == 4 and is_valid_box_coords(coords)):
                    return False
            elif g == "quad":
                if not (isinstance(coords, list) and len(coords) == 8 and is_valid_quad_coords(coords)):
                    return False
            elif g == "line":
                if not (isinstance(coords, list) and len(coords) >= 4 and is_valid_line_coords(coords)):
                    return False
            else:
                return False

        return True

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.samples)

    @jaxtyped_beartype
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get processed sample by index.

        Args:
            idx: Sample index

        Returns:
            Processed sample with tensors ready for model input

        Raises:
            ValueError: If conversation processor is not initialized
        """
        if self.conversation_processor is None:
            raise ValueError(
                "Conversation processor not initialized. Call set_processor() first."
            )

        # Get raw sample
        raw_sample = self.samples[idx]

        # Create structured sample with teacher assignments
        structured_sample = self._create_structured_sample(raw_sample, idx)

        # Process the sample through HuggingFace-first pipeline
        return self._process_sample_unified(structured_sample, idx)

    def _create_structured_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """Create structured sample with teacher assignments."""
        structured_sample = raw_sample.copy()
        # Attach index for downstream hardness updates
        structured_sample["sample_index"] = int(idx)

        # Prefer dynamic episode map when available (training split)
        teacher_samples: List[Dict[str, Any]] = []
        spec = getattr(self, "_episode_map", None)
        if (not self.is_eval) and isinstance(spec, dict):
            ep = spec[int(idx)] if int(idx) in spec else None
            if isinstance(ep, dict) and ("context_idx" in ep) and (ep["context_idx"] is not None):
                c_idx = int(ep["context_idx"])  # type: ignore[index]
                if 0 <= c_idx < len(self.samples):
                    teacher_samples = [self.samples[c_idx]]
        if teacher_samples:
            structured_sample["teacher_samples"] = teacher_samples
            logger.debug(
                f"Assigned {len(teacher_samples)} teacher(s) to sample {idx}"
            )

        return structured_sample

    # --- New helpers: unified variant sampling & image loading ---
    def _sample_variant(self) -> str:
        if self.is_eval:
            return "dense_caption"
        ratios = getattr(self, "_active_variant_ratios", None)
        if ratios is None:
            # No ratios configured => deterministic dense_caption
            return "dense_caption"
        keys = list(ratios.keys())
        weights = [float(ratios[k]) for k in keys]
        total = sum(weights)
        if total <= 0:
            return "dense_caption"
        r = random.random() * total
        acc = 0.0
        for k, w in zip(keys, weights):
            acc += w
            if r <= acc:
                logger.debug(f"🎛️ Variant sampled: '{k}' from ratios={ratios}")
                return k
        return keys[-1]

    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        path_manager = create_path_manager(self.data_root)
        images = []
        for img_path in image_paths:
            try:
                resolved = path_manager.resolve_path(img_path)
                image = Image.open(str(resolved)).convert("RGB")
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")
                raise
        return images

    @jaxtyped_beartype
    def _process_sample_unified(
        self, structured_sample: Dict[str, Any], idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Unified processing: centralized variant sampling and unified builder entry.
        """
        if self.conversation_processor is None:
            raise ValueError("Conversation processor not initialized")

        # Variant selection
        variant = self._sample_variant()

        # Teacher presence
        teacher_samples = structured_sample["teacher_samples"] if "teacher_samples" in structured_sample else []
        has_teachers = len(teacher_samples) > 0
        logger.debug(
            f"🧪 Sample idx={idx}: variant='{variant}', has_teachers={has_teachers}"
        )

        # Load images
        if has_teachers:
            teacher_images_list: List[List[Image.Image]] = []
            for t_sample in teacher_samples:
                if "images" not in t_sample or not isinstance(t_sample["images"], list) or len(t_sample["images"]) == 0:
                    raise ValueError("Teacher sample missing required 'images' list")
                t_paths = t_sample["images"]
                t_images = self._load_images(t_paths)
                teacher_images_list.append(t_images)
            if "images" not in structured_sample or not isinstance(structured_sample["images"], list) or len(structured_sample["images"]) == 0:
                raise ValueError("Student sample missing required 'images' list")
            student_images = self._load_images(structured_sample["images"])
        else:
            teacher_images_list = []
            if "images" not in structured_sample or not isinstance(structured_sample["images"], list) or len(structured_sample["images"]) == 0:
                raise ValueError("Student sample missing required 'images' list")
            student_images = self._load_images(structured_sample["images"])

        # Optional augmentation
        if self.augmentation_pipeline is not None:
            from torch.utils.data import get_worker_info

            wi = get_worker_info()
            worker_id = wi.id if wi is not None else 0

            # Student first
            student_seed_index = idx ^ (worker_id << 16)
            student_images, structured_sample = self.augmentation_pipeline.apply(
                sample=structured_sample,
                images=student_images,
                sample_index=student_seed_index,
            )
            logger.debug(
                f"✅ Applied augmentation (student) idx={idx}: image_size={student_images[0].size}, objects={len(structured_sample['objects'])}"
            )

            # Teachers optionally
            taug_cfg = self.config.teacher_augmentation
            aug_cfg = self.config.augmentation
            use_teacher_aug = False
            teacher_pipeline = None
            if taug_cfg is not None:
                from src_new_json.augmentation import ObjectAwareAugmentationPipeline

                teacher_pipeline = ObjectAwareAugmentationPipeline.from_config(taug_cfg)
                use_teacher_aug = True
            elif aug_cfg is not None and getattr(aug_cfg, "apply_to_teachers", False):
                teacher_pipeline = self.augmentation_pipeline
                use_teacher_aug = True

            if has_teachers and use_teacher_aug and teacher_pipeline is not None:
                new_teacher_images_list = []
                new_teacher_samples = []
                for t_i, (t_sample, t_images) in enumerate(
                    zip(teacher_samples, teacher_images_list)
                ):
                    t_seed_index = (idx * 997 + t_i) ^ (worker_id << 16)
                    t_images, t_sample = teacher_pipeline.apply(
                        sample=t_sample, images=t_images, sample_index=t_seed_index
                    )
                    logger.debug(
                        f"✅ Applied augmentation (teacher={t_i}) base_idx={idx}: image_size={t_images[0].size}, objects={len(t_sample['objects'])}"
                    )
                    new_teacher_images_list.append(t_images)
                    new_teacher_samples.append(t_sample)
                teacher_images_list = new_teacher_images_list
                teacher_samples = new_teacher_samples

        # Unified builder call
        if has_teachers:
            inputs = self.conversation_processor.create_conversation(
                sample=structured_sample,
                images=student_images,
                variant=variant,
                teacher_samples=teacher_samples,
                teacher_images_list=teacher_images_list,
            )
            logger.debug(f"✅ Created teacher-student conversation for sample {idx} (variant='{variant}')")
        else:
            inputs = self.conversation_processor.create_conversation(
                sample=structured_sample,
                images=student_images,
                variant=variant,
            )
            logger.debug(f"✅ Created student-only conversation for sample {idx} (variant='{variant}')")

        # Create labels with proper masking for training and extract spans (unchanged)
        labels, teacher_spans, student_spans = self._create_masked_labels_with_spans(
            input_ids=inputs["input_ids"],
            tokenizer=self.tokenizer,
            has_teachers=has_teachers,
            conversation_text=(inputs["conversation_text"] if "conversation_text" in inputs else None),
            offset_mapping=(inputs["offset_mapping"] if "offset_mapping" in inputs else None),
            num_teachers=len(teacher_samples) if has_teachers else 0,
        )
        inputs["labels"] = labels
        inputs["teacher_assistant_spans"] = teacher_spans
        inputs["student_assistant_spans"] = student_spans
        inputs.pop("assistant_spans", None)
        # Attach normalized variant for downstream strict grouping
        try:
            inputs["conversation_variant"] = str(variant)
        except Exception:
            inputs["conversation_variant"] = variant
        # Attach num_teachers for downstream debug/collator propagation
        try:
            inputs["num_teachers"] = int(len(teacher_samples) if has_teachers else 0)
        except Exception:
            inputs["num_teachers"] = 0

        # JSON mode: grouping is JSON-structure-based; wrapper tokens are not used.

        return inputs

    def _create_masked_labels_with_spans(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        has_teachers: bool = False,
        conversation_text: Optional[str] = None,
        offset_mapping: Optional[torch.Tensor] = None,
        num_teachers: int = 0,
    ) -> tuple[torch.Tensor, List[tuple[int, int]], List[tuple[int, int]]]:
        """
        Create properly masked labels using OFFSET MAPPING for accurate span detection.

        This method uses tokenizer's offset_mapping to get precise token-to-character
        alignment, ensuring spans correspond to actual token positions.
        """
        # Remove batch dimension for processing
        input_ids_1d = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
        labels = input_ids_1d.clone()
        teacher_spans = []
        student_spans = []

        try:
            # STEP 1: Derive the full conversation text by decoding input_ids to GUARANTEE alignment
            full_text = tokenizer.decode(
                input_ids_1d, skip_special_tokens=False
            )

            # STEP 2: Compute offset mapping from the decoded text to GUARANTEE token alignment
            tokenized_with_offsets = tokenizer(
                full_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            offset_mapping = tokenized_with_offsets["offset_mapping"][
                0
            ]  # Remove batch dim

            # STEP 3: Use provided num_teachers exactly (do not override)
            if not has_teachers:
                num_teachers = 0
            else:
                try:
                    num_teachers = int(num_teachers)
                except Exception:
                    raise RuntimeError(f"Invalid num_teachers value: {num_teachers}")
                if num_teachers < 0:
                    raise RuntimeError(f"num_teachers must be non-negative, got {num_teachers}")

            # STEP 4: Find assistant content spans using accurate text-to-token mapping
            from src_new_json.processing.span_extraction import find_assistant_spans
            assistant_spans = find_assistant_spans(
                full_text=full_text,
                offset_mapping=offset_mapping,
                tokenizer=tokenizer,
                include_eos=bool(self.config.span_include_im_end_in_labels),
                has_teachers=has_teachers,
                num_teachers=num_teachers,
            )

            # STEP 5: Mask all tokens initially
            labels.fill_(-100)

            # STEP 6: Unmask assistant content spans
            for span_info in assistant_spans:
                start_token, end_token, is_teacher = span_info

                # Validate span bounds
                if 0 <= start_token < end_token <= len(labels):
                    # Try to include the immediate <|im_end|> token (if present) in the span
                    try:
                        im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
                    except Exception:
                        im_end_id = None

                    final_end = end_token
                    if im_end_id is not None and im_end_id != -1:
                        # Prefer the token right at end_token, but allow a tiny lookahead window
                        # to be robust to tokenization boundary nuances.
                        for pos in range(end_token, min(len(labels), end_token + 5)):
                            if int(input_ids_1d[pos].item()) == int(im_end_id):
                                final_end = pos + 1  # include <|im_end|> in the span
                                logger.debug(
                                    f"✅ Extended span to include <|im_end|> at position {pos} for original span ({start_token}, {end_token})"
                                )
                                break

                    # Unmask this span (now possibly extended to include <|im_end|>)
                    labels[start_token:final_end] = input_ids_1d[start_token:final_end]

                    # Track spans for loss computation (use the extended end)
                    span = (start_token, final_end)
                    if is_teacher:
                        teacher_spans.append(span)
                        logger.debug(f"✅ Teacher span: {span}")
                    else:
                        student_spans.append(span)
                        logger.debug(f"✅ Student span: {span}")
                else:
                    logger.error(
                        f"Invalid span bounds: {start_token}:{end_token} for sequence length {len(labels)}"
                    )

            # STEP 7: Mask image pad tokens
            image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
            if image_pad_id is not None:
                image_pad_mask = input_ids_1d == image_pad_id
                labels[image_pad_mask] = -100

            # STEP 8 (debug): strict invariants for alignment and coverage
            if self.config.debug_alignment:
                # Re-decode labels where not -100 and check round-trip equals input_ids
                # (We check ids equality on unmasked region spans.)
                all_spans = teacher_spans + student_spans
                # Non-empty and sorted, non-overlapping
                if len(all_spans) == 0:
                    raise RuntimeError("debug_alignment: no assistant spans found")
                all_spans_sorted = sorted(all_spans)
                prev_end = -1
                for st, ed in all_spans_sorted:
                    if st <= prev_end:
                        raise RuntimeError(
                            f"debug_alignment: overlapping/unsorted spans detected: prev_end={prev_end}, curr=({st},{ed})"
                        )
                    prev_end = ed
                # Labels must equal input_ids inside spans
                for st, ed in all_spans_sorted:
                    if not torch.equal(labels[st:ed], input_ids_1d[st:ed]):
                        raise RuntimeError(
                            f"debug_alignment: labels do not match input_ids within span ({st},{ed})"
                        )
                # Outside spans must be -100
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for st, ed in all_spans_sorted:
                    mask[st:ed] = True
                if (labels[~mask] != -100).any():
                    raise RuntimeError(
                        "debug_alignment: found non -100 labels outside assistant spans"
                    )
                # Offset mapping consistency: token offsets must be non-decreasing and within text bounds
                if not (offset_mapping[:, 0] <= offset_mapping[:, 1]).all():
                    raise RuntimeError("debug_alignment: invalid offset ranges")
                if offset_mapping[-1, 1].item() > len(full_text):
                    raise RuntimeError(
                        "debug_alignment: offset mapping exceeds text length"
                    )

        except Exception as e:
            # Fail-fast: do not silently continue with invalid spans
            logger.error(
                f"Error in offset-based span detection: {type(e).__name__}: {e}"
            )
            raise

        # Log final span statistics
        logger.debug(
            f"📊 Final spans - Teachers: {len(teacher_spans)}, Students: {len(student_spans)}"
        )

        # Sanity check: ensure we unmasked some assistant tokens
        unmasked_count = int((labels != -100).sum().item())
        if unmasked_count == 0:
            raise RuntimeError(
                "Masking produced no learnable tokens; check assistant span detection and offset mapping."
            )

        return labels, teacher_spans, student_spans

    def _char_to_token_position(
        self, char_pos: int, offset_mapping: torch.Tensor
    ):
        """
        Convert character position to token position using offset mapping.

        Args:
            char_pos: Character position in text
            offset_mapping: Token offset mapping [(start_char, end_char), ...]

        Returns:
            Token position or None if not found
        """
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char <= char_pos < end_char:
                return token_idx
            elif char_pos == end_char and token_idx < len(offset_mapping) - 1:
                # Handle boundary case - character position at token boundary
                return token_idx + 1

        # If exact match not found, find closest token
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char >= char_pos:
                return token_idx

        return len(offset_mapping)  # End of sequence

    def _load_images_from_paths(self, image_paths: List[str]) -> List[Image.Image]:
        """Load images from paths."""
        path_manager = create_path_manager(self.data_root)
        images = []
        for img_path in image_paths:
            try:
                resolved = path_manager.resolve_path(img_path)
                resolved_str = str(resolved)
                image = Image.open(resolved_str).convert("RGB")
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")
                raise

        return images

    def update_hardness_ema(self, sample_indices: List[int], student_losses: List[float]) -> None:
        """Update per-sample EMA hardness from student loss values.

        Fail-fast on mismatched lengths or invalid indices; ignores updates if EMA not initialized.
        """
        if not isinstance(sample_indices, list) or not isinstance(student_losses, list):
            raise ValueError("update_hardness_ema expects lists for indices and losses")
        if len(sample_indices) != len(student_losses):
            raise ValueError(f"update_hardness_ema length mismatch: indices={len(sample_indices)} losses={len(student_losses)}")
        if not self._hardness_ema or len(self._hardness_ema) != len(self.samples):
            return
        a = float(self._hardness_ema_alpha)
        if a <= 0 or a > 1:
            raise ValueError(f"hardness_alpha must be in (0,1], got {a}")
        for idx, loss in zip(sample_indices, student_losses):
            if not (isinstance(idx, int) and 0 <= idx < len(self._hardness_ema)):
                continue
            try:
                loss_val = float(loss)
            except Exception:
                continue
            prev = float(self._hardness_ema[idx])
            self._hardness_ema[idx] = (1.0 - a) * prev + a * loss_val


# Export classes
__all__ = [
    "Dataset",
    "read_jsonl",
]
def _normalize_smart_resize_config(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, SmartResizeConfig):
        return {
            "enabled": bool(value.enabled),
            "factor": int(value.factor),
            "min_pixels": int(value.min_pixels),
            "max_pixels": int(value.max_pixels),
            "max_ratio": float(value.max_ratio),
        }
    if isinstance(value, dict):
        required = {"enabled", "factor", "min_pixels", "max_pixels", "max_ratio"}
        missing = required.difference(value.keys())
        if missing:
            raise ValueError(
                f"smart_resize mapping missing required keys: {sorted(missing)}"
            )
        return {
            "enabled": bool(value["enabled"]),
            "factor": int(value["factor"]),
            "min_pixels": int(value["min_pixels"]),
            "max_pixels": int(value["max_pixels"]),
            "max_ratio": float(value["max_ratio"]),
        }
    raise ValueError(f"Unsupported smart_resize configuration type: {type(value)}")
