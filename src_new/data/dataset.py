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

from src_new.config.config import Config
from src_new.data.teacher_pool import TeacherPoolManager
from src_new.processing.conversation import ConversationBuilder
from src_new.processing.special_tokens import (
    ASSISTANT_SPAN_PATTERN,
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

        logger.info(
            f"✅ HuggingFace-first dataset initialized with {len(self.samples)} samples"
        )

    def _initialize_processing_components(self):
        """Initialize HuggingFace-first processing components."""
        # Data processing settings
        self.data_root = self.config.data_root
        self.teacher_ratio = self.config.teacher_ratio
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

                    cfg = get_preset_config(
                        getattr(self.config.augmentation, "preset"),
                        rng_seed=int(getattr(self.config.augmentation, "rng_seed")),
                        apply_to_teachers=bool(
                            getattr(self.config.augmentation, "apply_to_teachers")
                        ),
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
        """
        if not self._augmentation_schedule:
            return
        from src_new.augmentation import ObjectAwareAugmentationPipeline
        from src_new.augmentation.wrappers import get_preset_config

        active = None
        for entry in sorted(
            self._augmentation_schedule, key=lambda e: int(e["start_epoch"])
        ):
            if epoch_index >= int(entry["start_epoch"]):
                active = entry
        if active is None:
            return
        cfg = get_preset_config(
            active["preset"], rng_seed=getattr(self.config, "seed", 12345)
        )
        self.augmentation_pipeline = ObjectAwareAugmentationPipeline.from_config(cfg)
        logger.info(
            f"🔁 Augmentation preset switched at epoch {epoch_index}: {active['preset']}"
        )
        # Variant schedule (optional)
        schedule = getattr(self.config, "conversation_variant_schedule", None)
        if isinstance(schedule, list) and schedule:
            chosen = None
            for entry in sorted(schedule, key=lambda e: int(e.get("start_epoch", 0))):
                if epoch_index >= int(entry.get("start_epoch", 0)):
                    chosen = entry
            if chosen and isinstance(chosen.get("ratios"), dict):
                self._active_variant_ratios = {
                    ("dense_caption" if k == "dense_captioning" else "coords_to_desc" if k == "coords_to_desc" else "desc_to_coords" if k == "desc_to_coords" else k): float(v)
                    for k, v in chosen["ratios"].items()
                }
                logger.info(
                    f"🔁 Variant ratios switched at epoch {epoch_index}: {self._active_variant_ratios}"
                )

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

        from src_new.processing.conversation import ConversationBuilder

        # Fail-fast: max_coord_value must come from YAML (no defaults allowed)
        if not hasattr(self.config, "max_coord_value"):
            raise ValueError(
                "max_coord_value is required in configuration (YAML) but was not found on Config."
            )
        max_coord_value = self.config.max_coord_value
        if not isinstance(max_coord_value, int) or max_coord_value <= 0:
            raise ValueError(
                f"max_coord_value must be a positive integer, got {max_coord_value!r}"
            )

        if not hasattr(self.config, "coordinate_tokens_enabled"):
            raise ValueError(
                "coordinate_tokens_enabled must be explicitly set in configuration (True/False)"
            )
        if not isinstance(self.config.coordinate_tokens_enabled, bool):
            raise ValueError(
                f"coordinate_tokens_enabled must be a bool, got {type(self.config.coordinate_tokens_enabled)}: {self.config.coordinate_tokens_enabled!r}"
            )

        self.conversation_processor = ConversationBuilder(
            processor=hf_processor,
            max_coord_value=max_coord_value,
            coordinate_tokens_enabled=self.config.coordinate_tokens_enabled,
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

        # Add teacher examples if conditions are met
        if (
            self.teacher_pool_manager
            and (not self.is_eval)
            and random.random() < self.teacher_ratio
            and len(self.teacher_pool_manager.teacher_pool) > 0
        ):
            # Select teachers dynamically based on current student sample
            # Note: Teacher samples serve as context input only during evaluation
            # (no teacher loss computation - only student response is trained)
            num_teachers = min(
                random.randint(1, self.num_teacher_samples),
                len(self.teacher_pool_manager.teacher_pool),
            )

            selected_teachers = self.teacher_pool_manager.select_teachers_for_student(
                structured_sample, num_samples=num_teachers
            )

            structured_sample["teacher_samples"] = selected_teachers
            logger.debug(
                f"Assigned {len(selected_teachers)} dynamic teacher(s) to sample {idx}"
            )

        return structured_sample

    # --- New helpers: unified variant sampling & image loading ---
    def _sample_variant(self) -> str:
        if self.is_eval:
            return "dense_caption"
        ratios = getattr(self, "_active_variant_ratios", None) or getattr(self.config, "conversation_variant_ratios", None) or {
            "dense_caption": 1.0,
            "coords_to_desc": 0.0,
            "desc_to_coords": 0.0,
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
        teacher_samples = structured_sample.get("teacher_samples", [])
        has_teachers = len(teacher_samples) > 0
        logger.debug(
            f"🧪 Sample idx={idx}: variant='{variant}', has_teachers={has_teachers}"
        )

        # Load images
        if has_teachers:
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

            # STEP 3: Set number of teachers (always 1 teacher + 1 student in current setup)
            num_teachers = 1 if has_teachers else 0

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
