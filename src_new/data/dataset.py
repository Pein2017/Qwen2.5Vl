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
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import Qwen2_5_VLProcessor, Qwen2VLImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src_new.config.augmentation_config import SmartResizeConfig
from src_new.config.config import Config
from src_new.data.teacher_pool import TeacherPoolManager
from src_new.processing.conversation import ConversationBuilder
from src_new.processing.span_mapping import map_spans_unexpanded_to_expanded
from src_new.processing.special_tokens import (
    GEOMETRY_TOKENS,
    IM_END,
    IMAGE_PAD,
)
from src_new.types.arrays import (
    jaxtyped_beartype,
)
from src_new.utils.path_manager import create_path_manager
from src_new.utils.rank_aware_logging import get_rank_aware_logger


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


DUMMY_IMAGE_VARIANTS = {"text_only"}


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

    hf_processor: Optional[Qwen2_5_VLProcessor]
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

    # Dynamic pairing state (per-epoch)
    _episode_map: Optional[Dict[int, Dict[str, Optional[int]]]] = None  # target_idx -> {context_idx:int|None}
    _context_use_count: Optional[Dict[int, int]] = None
    _type_to_indices: Optional[Dict[str, List[int]]] = None

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
        self._dummy_image = Image.new("RGB", (28, 28), color=(0, 0, 0))
        self._dummy_image_path = "__dummy_image__"

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

        # Initialize dynamic pairing structures
        self._episode_map = None
        self._context_use_count = None
        self._type_to_indices = None

        # Build initial epoch-0 mapping (aug preset and/or dynamic pairing)
        try:
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
        self._augmentation_schedule = getattr(
            self.config, "augmentation_schedule", None
        )
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
                from src_new.augmentation import ObjectAwareAugmentationPipeline
                from src_new.config.augmentation_config import (
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
                    from src_new.augmentation.wrappers import get_preset_config

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
                        getattr(self.config.augmentation, "preset"),
                        rng_seed=int(getattr(self.config.augmentation, "rng_seed")),
                        apply_to_teachers=bool(
                            getattr(self.config.augmentation, "apply_to_teachers")
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
        """Optional hook for trainer: update augmentation preset by epoch.

        If config.augmentation_schedule = [{start_epoch, preset}, ...] is defined,
        the first entry with start_epoch <= epoch_index and highest start_epoch wins.
        Also builds dynamic contrastive pairing episode mapping per epoch when enabled.
        """
        has_samples = hasattr(self, "samples") and isinstance(self.samples, list)

        if not self._augmentation_schedule and not getattr(self.config, "dynamic_pairing_enabled", False):
            # Nothing to do
            pass

        # Augmentation schedule handling (existing)
        if self._augmentation_schedule:
            from src_new.augmentation import ObjectAwareAugmentationPipeline
            from src_new.augmentation.wrappers import get_preset_config

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
                    rng_seed=getattr(self.config, "seed", 12345),
                    smart_resize=smart_resize_payload,
                )
                self.augmentation_pipeline = ObjectAwareAugmentationPipeline.from_config(cfg)
                logger.info(
                    f"🔁 Augmentation preset switched at epoch {epoch_index}: {active['preset']}"
                )

        # Dynamic contrastive pairing mapping (training only)
        if getattr(self.config, "dynamic_pairing_enabled", False) and not self.is_eval and has_samples:
            # Use modular pairing engine
            # Delayed import to avoid heavy dependencies at module import time
            from src_new.sampler.random_bucket import RandomBucketConfig as _SamplerCfg
            from src_new.sampler.random_bucket import RandomBucketSampler as _Sampler

            base_seed = int(getattr(self.config, "seed", 12345))

            # DEBUG: summarize bucket composition before sampling
            try:
                from collections import defaultdict
                type_counts = defaultdict(int)
                for s in self.samples:
                    try:
                        objs = s.get("objects", []) or []
                        if objs:
                            first = str(objs[0].get("type", "misc"))
                        else:
                            first = "misc"
                    except Exception:
                        first = "misc"
                    type_counts[first] += 1
                logger.debug(
                    f"[pairing] Epoch {epoch_index} bucket composition: "
                    + ", ".join(f"{k}:{v}" for k, v in list(type_counts.items())[:8])
                )
            except Exception:
                pass

            engine = _Sampler()
            pair_cfg = _SamplerCfg(
                cross_bucket_explore_prob=float(getattr(self.config, "dynamic_pair_cross_bucket_explore_prob", 0.0)),
            )

            episode_map, metrics = engine.build_epoch_map(
                samples=self.samples,
                cfg=pair_cfg,
                base_seed=base_seed,
                epoch_idx=int(epoch_index),
                teacher_ratio=float(self.teacher_ratio),
                is_eval=bool(self.is_eval),
            )

            # Materialize map and a simple usage count (hist only)
            self._episode_map = {spec.target_idx: {"context_idx": spec.context_idx} for spec in episode_map.values()}

            logger.info(
                f"🎯 Dynamic pairing @epoch {epoch_index}: pair_episodes={metrics.pair_episodes}, single_episodes~={metrics.single_episodes}, target_is_current_pct={metrics.target_is_current_pct:.1f}%"
            )
            try:
                # Additional DEBUG: context usage histogram top-k
                hist_items = list(metrics.context_usage_histogram.items())
                hist_items.sort(key=lambda x: -x[1])
                head = ", ".join([f"{i}:{c}" for i, c in hist_items[:5]])
                logger.debug(f"[pairing] Top context usage: {head}")
                logger.debug(f"[pairing] Unique contexts: {len(hist_items)}; coverage={metrics.coverage_context_unique_pct:.1f}%")
            except Exception:
                pass
            logger.info(
                f"📊 Context coverage: unique_context_used≈{int(metrics.coverage_context_unique_pct * len(self.samples) / 100.0)} ({metrics.coverage_context_unique_pct:.1f}%)"
            )
            logger.info(f"📈 context_usage_histogram={metrics.context_usage_histogram}")

        # For validation/eval split, enforce single-turn behavior
        if self.is_eval:
            self._episode_map = {}
            logger.info("🔒 Eval split: forced single-turn episodes (no context)")

    def set_processor(self, hf_processor: Qwen2_5_VLProcessor) -> None:
        """
        Set the HuggingFace processor and initialize conversation processor.

        This method should be called by the trainer after the processor is available.

        Args:
            hf_processor: Official HuggingFace Qwen2_5_VLProcessor

        Raises:
            ValueError: If hf_processor is None
        """
        if hf_processor is None:
            raise ValueError("hf_processor cannot be None")

        self.hf_processor = hf_processor

        # NEW: strict tokenizer special-token validations (fail-fast)
        try:
            tok = getattr(hf_processor, "tokenizer", None)
            if tok is None:
                raise ValueError("Processor is missing tokenizer; cannot validate special tokens")
            from src_new.processing.special_tokens import (
                require_core_special_tokens,
                require_geometry_tokens,
            )
            # Core chat/image tokens must exist
            require_core_special_tokens(tok)
            # Geometry/object-ref wrappers must exist; require line wrappers only when lines are used/configured
            # In our dataset all three shapes are supported; keep require_line=True unless explicitly disabled by config
            require_line_tokens: bool = True
            if hasattr(self.config, "require_line_tokens"):
                require_line_tokens = bool(getattr(self.config, "require_line_tokens"))
            require_geometry_tokens(tok, require_line=require_line_tokens)
        except Exception as e:
            raise ValueError(f"Tokenizer special-token validation failed: {e}")

        # Simplified conversation processor without coordinate token dependencies
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

        from src_new.types.geometry import (
            is_valid_box_coords,
            is_valid_line_coords,
            is_valid_quad_coords,
        )

        geometry_types = list(GEOMETRY_TOKENS.keys())
        for obj in sample["objects"]:
            if "desc" not in obj:
                return False

            present = [gt for gt in geometry_types if gt in obj]
            if len(present) != 1:
                return False

            g = present[0]
            coords = obj[g]
            if g == "bbox_2d" and not is_valid_box_coords(coords):
                return False
            if g == "quad" and not is_valid_quad_coords(coords):
                return False
            if g == "line" and not is_valid_line_coords(coords):
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

        raw_sample = self.samples[idx]

        structured_sample = self._create_structured_sample(raw_sample, idx)

        return self._process_sample_unified(structured_sample, idx)

    def _create_structured_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """Create structured sample with teacher assignments."""
        structured_sample = raw_sample.copy()

        # Prefer dynamic episode map if available and training split
        teacher_samples: List[Dict[str, Any]] = []
        if (not self.is_eval) and isinstance(self._episode_map, dict):
            spec = self._episode_map.get(int(idx)) if self._episode_map is not None else None
            if spec and spec.get("context_idx") is not None:
                c_idx = int(spec["context_idx"])  # type: ignore[index]
                if 0 <= c_idx < len(self.samples):
                    logger.debug(f"[pairing] __getitem__ idx={idx} -> context_idx={c_idx}")
                    teacher_samples.append(self.samples[c_idx])
                else:
                    logger.debug(f"[pairing] __getitem__ idx={idx} has invalid context_idx={c_idx}; skipping")
                    pass

        # Fallback to legacy teacher_pool_manager when no dynamic mapping
        if not teacher_samples:
            # No teacher fallback: simplified random pairing uses only episode_map
            pass

        if teacher_samples:
            structured_sample["teacher_samples"] = teacher_samples
            logger.debug(
                f"Assigned {len(teacher_samples)} teacher(s) to sample {idx}"
            )

        return structured_sample

    # --- New helpers: unified variant sampling & image loading ---
    def _sample_variant(self) -> str:
        if self.is_eval:
            # Eval: if configuration focuses exclusively on summary, use summary variant
            ratios = (
                getattr(self, "_active_variant_ratios", None)
                or getattr(self.config, "conversation_variant_ratios", None)
            )
            try:
                if isinstance(ratios, dict):
                    summary_weight = float(ratios.get("summary", 0.0))
                    total_weight = sum(float(v) for v in ratios.values()) if ratios else 0.0
                    # Summary-only when summary is the only non-zero weight (e.g., summary: 1)
                    if summary_weight > 0.0 and (total_weight - summary_weight) <= 0.0:
                        return "summary"
            except Exception:
                # Fall back to dense if any issue arises when interpreting ratios
                pass
            return "dense_caption"
        ratios = getattr(self, "_active_variant_ratios", None) or getattr(self.config, "conversation_variant_ratios", None) or {
            "dense_caption": 1.0,
            "coords_to_desc": 0.0,
            "desc_to_coords": 0.0,
            "text_only": 0.0,
        }
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
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("🎛️ Variant sampled: '%s' from ratios=%s", k, ratios)
                return k
        return keys[-1]

    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        path_manager = create_path_manager(self.data_root)
        images = []
        for img_path in image_paths:
            if img_path == self._dummy_image_path:
                images.append(self._dummy_image.copy())
                continue
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
        self,
        structured_sample: Dict[str, Any],
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified processing: centralized variant sampling and unified builder entry.
        """
        if self.conversation_processor is None:
            raise ValueError("Conversation processor not initialized")

        # Variant selection
        variant = self._sample_variant()
        variant_key = str(getattr(variant, "value", variant)).strip().lower()
        text_only_variant = variant_key in DUMMY_IMAGE_VARIANTS

        # Teacher presence (disabled for text-only variants)
        teacher_samples = [] if text_only_variant else structured_sample.get("teacher_samples", [])
        has_teachers = len(teacher_samples) > 0
        if text_only_variant and has_teachers:
            logger.debug(
                "Text-only variant '%s' requested; ignoring %d teacher samples",
                variant_key,
                len(teacher_samples),
            )
            teacher_samples = []
            has_teachers = False

        logger.debug(
            f"🧪 Sample idx={idx}: variant='{variant_key}', has_teachers={has_teachers}"
        )

        # Load images
        if text_only_variant:
            teacher_images_list = []
            student_images = [self._dummy_image]
            try:
                structured_sample = dict(structured_sample)
                structured_sample["images"] = [self._dummy_image_path]
            except Exception:
                structured_sample["images"] = [self._dummy_image_path]
        elif has_teachers:
            teacher_images_list: List[List[Image.Image]] = []
            for t_sample in teacher_samples:
                t_paths = t_sample.get("images", [])
                if not t_paths:
                    raise ValueError("Teacher sample missing required 'images' list")
                t_images = self._load_images(t_paths)
                teacher_images_list.append(t_images)
            student_images = self._load_images(structured_sample.get("images", []))
        else:
            teacher_images_list = []
            student_images = self._load_images(structured_sample.get("images", []))

        # Optional augmentation
        if self.augmentation_pipeline is not None and not text_only_variant:
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
            taug_cfg = getattr(self.config, "teacher_augmentation", None)
            aug_cfg = getattr(self.config, "augmentation", None)
            use_teacher_aug = False
            teacher_pipeline = None
            if taug_cfg is not None:
                from src_new.augmentation import ObjectAwareAugmentationPipeline

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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("✅ Created teacher-student conversation for sample %d (variant='%s')", idx, variant)
        else:
            inputs = self.conversation_processor.create_conversation(
                sample=structured_sample,
                images=student_images,
                variant=variant,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("✅ Created student-only conversation for sample %d (variant='%s')", idx, variant)

        # Create labels with proper masking for training and extract spans (unchanged)
        # Prefer precomputed spans from processor outputs when available
        pre_t = inputs.get("teacher_assistant_spans")
        pre_s = inputs.get("student_assistant_spans")
        if isinstance(pre_t, list) and isinstance(pre_s, list) and (pre_t or pre_s):
            # Build labels from provided spans without re-extraction
            labels = inputs["input_ids"].clone()
            labels.fill_(-100)
            teacher_spans = []
            student_spans = []
            # Normalize to list of tuples
            def _add_spans(sp_list, is_teacher: bool):
                nonlocal teacher_spans, student_spans
                for st, ed in sp_list:
                    st_i, ed_i = int(st), int(ed)
                    if 0 <= st_i < ed_i <= int(labels.shape[0]):
                        labels[st_i:ed_i] = inputs["input_ids"][st_i:ed_i]
                        if is_teacher:
                            teacher_spans.append((st_i, ed_i))
                        else:
                            student_spans.append((st_i, ed_i))
            _add_spans(pre_t, True)
            _add_spans(pre_s, False)
            # Mask image pads
            try:
                tokenizer = getattr(self.conversation_processor.processor, "tokenizer", None) if self.conversation_processor else None
                if tokenizer is not None:
                    image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
                    if image_pad_id is not None:
                        labels[inputs["input_ids"] == image_pad_id] = -100
            except Exception:
                pass
            # Fallback to local extraction if nothing got unmasked
            if int((labels != -100).sum().item()) == 0:
                labels, teacher_spans, student_spans = self._create_masked_labels_with_spans(
                    input_ids=inputs["input_ids"],
                    tokenizer=self.tokenizer,
                    has_teachers=has_teachers,
                    conversation_text=inputs.get("conversation_text"),
                    offset_mapping=inputs.get("offset_mapping"),
                    num_teachers=len(teacher_samples) if has_teachers else 0,
                )
        else:
            labels, teacher_spans, student_spans = self._create_masked_labels_with_spans(
                input_ids=inputs["input_ids"],
                tokenizer=self.tokenizer,
                has_teachers=has_teachers,
                conversation_text=inputs.get("conversation_text"),
                offset_mapping=inputs.get("offset_mapping"),
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

        # Removed: plain-text char-span grouping path (deprecated). Grouping now always relies on wrapper tokens.

        # Normalize text tensors to 1D per-sample for collator ([seq])
        try:
            if isinstance(inputs.get("input_ids"), torch.Tensor) and inputs["input_ids"].dim() == 2 and int(inputs["input_ids"].shape[0]) == 1:
                inputs["input_ids"] = inputs["input_ids"][0]
            if isinstance(inputs.get("attention_mask"), torch.Tensor) and inputs["attention_mask"].dim() == 2 and int(inputs["attention_mask"].shape[0]) == 1:
                inputs["attention_mask"] = inputs["attention_mask"][0]
        except Exception:
            pass

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
            # STEP 1: Prefer provided conversation_text / offset_mapping to avoid re-decode when available
            if isinstance(conversation_text, str) and len(conversation_text) > 0:
                full_text = conversation_text
            else:
                full_text = tokenizer.decode(input_ids_1d, skip_special_tokens=False)

            # STEP 2: Use provided offset_mapping when available; otherwise compute from full_text
            if isinstance(offset_mapping, torch.Tensor) and offset_mapping.ndim == 2:
                enc_ids = None
                # Use provided mapping directly from ConversationProcessor
                offset_mapping = offset_mapping
                # Ensure we also have unexpanded token ids for span alignment when needed
                try:
                    tokenized_for_ids = tokenizer(
                        full_text,
                        return_offsets_mapping=True,
                        add_special_tokens=False,
                        return_tensors="pt",
                        truncation=False,
                    )
                    enc_ids = tokenized_for_ids["input_ids"][0]
                except Exception:
                    enc_ids = None
            else:
                tokenized_with_offsets = tokenizer(
                    full_text,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                    truncation=False,
                )
                offset_mapping = tokenized_with_offsets["offset_mapping"][0]
                enc_ids = tokenized_with_offsets["input_ids"][0]

            # STEP 3: Respect provided num_teachers (dataset-level mapping)
            num_teachers = int(num_teachers) if has_teachers else 0

            # STEP 4: Find assistant content spans using accurate text-to-token mapping
            from src_new.processing.span_extraction import find_assistant_spans
            assistant_spans = find_assistant_spans(
                full_text=full_text,
                offset_mapping=offset_mapping,
                tokenizer=tokenizer,
                include_eos=bool(getattr(self.config, "span_include_im_end_in_labels", True)),
                has_teachers=has_teachers,
                input_ids_1d=input_ids_1d,
                num_teachers=num_teachers,
            )
            # Map spans from unexpanded tokenization to expanded input_ids when necessary
            try:
                if enc_ids is not None and int(enc_ids.shape[0]) != int(input_ids_1d.shape[0]):
                    image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
                    assistant_spans = map_spans_unexpanded_to_expanded(
                        ids_unexpanded=enc_ids,
                        ids_expanded=input_ids_1d,
                        image_pad_id=image_pad_id,
                        spans_unexpanded=assistant_spans,
                    )
            except Exception:
                # If mapping fails, keep original spans and continue; downstream debug_alignment will surface issues
                pass

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
            if getattr(self.config, "debug_alignment", False):
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
            # Fallback A: try recomputing offset mapping from full_text and re-extract spans
            try:
                tokenized_with_offsets_fb = tokenizer(
                    full_text,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                    truncation=False,
                )
                offset_fb = tokenized_with_offsets_fb["offset_mapping"][0]
                from src_new.processing.span_extraction import (
                    find_assistant_spans as _find,
                )
                spans_fb = _find(
                    full_text=full_text,
                    offset_mapping=offset_fb,
                    tokenizer=tokenizer,
                    include_eos=bool(getattr(self.config, "span_include_im_end_in_labels", True)),
                    has_teachers=has_teachers,
                    input_ids_1d=input_ids_1d,
                    num_teachers=num_teachers,
                )
                if spans_fb:
                    labels.fill_(-100)
                    teacher_spans.clear()
                    student_spans.clear()
                    for st, ed, is_teacher in spans_fb:
                        if 0 <= st < ed <= len(labels):
                            labels[st:ed] = input_ids_1d[st:ed]
                            span = (st, ed)
                            if is_teacher:
                                teacher_spans.append(span)
                            else:
                                student_spans.append(span)
                    # Re-mask image pads
                    image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
                    if image_pad_id is not None:
                        labels[input_ids_1d == image_pad_id] = -100
                    unmasked_count = int((labels != -100).sum().item())
            except Exception:
                pass

            # Fallback B: derive last assistant span by scanning markers in text
            if unmasked_count == 0:
                try:
                    last_hdr = full_text.rfind("<|im_start|>assistant")
                    if last_hdr >= 0:
                        end_pos = full_text.find("<|im_end|>", last_hdr)
                        if end_pos < 0:
                            end_pos = len(full_text)
                        st_tok = self._char_to_token_position(last_hdr, offset_mapping)
                        ed_tok = self._char_to_token_position(end_pos, offset_mapping)
                        if st_tok is not None and ed_tok is not None and 0 <= st_tok < ed_tok <= len(labels):
                            labels.fill_(-100)
                            labels[st_tok:ed_tok] = input_ids_1d[st_tok:ed_tok]
                            teacher_spans = []
                            student_spans = [(st_tok, ed_tok)]
                            image_pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
                            if image_pad_id is not None:
                                labels[input_ids_1d == image_pad_id] = -100
                            unmasked_count = int((labels != -100).sum().item())
                except Exception:
                    pass

        if unmasked_count == 0:
            raise RuntimeError(
                "Masking produced no learnable tokens; check assistant span detection and offset mapping."
            )

        return labels, teacher_spans, student_spans


    def _char_to_token_position(
        self, char_pos: int, offset_mapping: torch.Tensor
    ) -> Optional[int]:
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
        normalized = {
            "enabled": bool(value["enabled"]),
            "factor": int(value["factor"]),
            "min_pixels": int(value["min_pixels"]),
            "max_pixels": int(value["max_pixels"]),
            "max_ratio": float(value["max_ratio"]),
        }
        return normalized
    raise ValueError(f"Unsupported smart_resize configuration type: {type(value)}")
