import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import torch
from PIL import Image
from torchtyping import TensorType

# Runtime & shape-checking ----------------------------------------------
from typeguard import typechecked

from src.logger_utils import get_chat_logger

# Legacy coordinate processor removed - using unified coordinate manager
from src.utils.prompt import (
    CHINESE_FEW_SHOT_SECTION,
    ENGLISH_FEW_SHOT_SECTION,
    get_system_prompt,
)
from src.utils.schema import ChatMessage, ChatProcessorOutput, GroundTruthObject
from src.utils.tokens import SpecialTokens


logger = get_chat_logger()

# Dimensional symbols for torchtyping ----------------------------------
S = TypeVar("S")  # Sequence length
B = TypeVar("B")  # Batch size
C_TOK = TypeVar("C_TOK")  # Channels (avoid single ambiguous)
PT = TypeVar("PT")  # Flattened patch tokens count (replaces ambiguous I)
E = TypeVar("E")  # Embedding dimension for flattened vision tokens
N_IMG = TypeVar("N_IMG")  # Number of images in the sample
H = TypeVar("H")  # Height
W = TypeVar("W")  # Width


class ChatProcessor:
    """
    Single-purpose processor for BBU dataset chat template creation.

    Handles the complete pipeline from simplified JSONL to training-ready format.
    Uses pure JSON format for object detection output (Qwen2.5-VL compatible).
    """

    def __init__(self, tokenizer, image_processor, **kwargs):
        """
        Initialize the chat processor using global configuration.

        Args:
            tokenizer: Qwen2.5-VL tokenizer
            image_processor: Qwen2.5-VL image processor
        """
        # FAIL-FAST: Validate required parameters
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        if image_processor is None:
            raise ValueError("image_processor cannot be None")

        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # FAIL-FAST: Validate config access
        if "config" in kwargs and kwargs["config"] is not None:
            config = kwargs["config"]
        else:
            # Validate global config is initialized
            try:
                from src.config import get_config

                config = get_config()
            except RuntimeError as e:
                raise RuntimeError(
                    f"No valid configuration provided and global config not initialized: {e}"
                )

        # FAIL-FAST: Validate required config attributes
        if not hasattr(config, "data_root"):
            raise ValueError("Configuration missing required attribute: data_root")

        self.data_root = Path(config.data_root)

        # ---------------- Required parameters ----------------
        # FAIL-FAST: coordinate_tokens_enabled must be explicitly provided
        if "coordinate_tokens_enabled" not in kwargs:
            raise ValueError(
                "coordinate_tokens_enabled must be explicitly provided in kwargs"
            )
        coordinate_enabled = kwargs["coordinate_tokens_enabled"]

        # ---------------- Optional kwargs with explicit handling ----------------
        # Many call-sites (trainer / inference) pass extra kwargs such as
        # merge_size, max_length, use_training_prompts, language …
        # We keep only what is actually needed to stay compatible.

        # use_training_prompts with explicit default
        self.use_training_prompts = kwargs.get("use_training_prompts", False)

        # FAIL-FAST: Validate language configuration
        if "language" in kwargs:
            self.language = kwargs["language"]
        elif hasattr(config, "language"):
            self.language = config.language
        else:
            raise ValueError("language must be provided in kwargs or config")

        # Initialize special tokens (vision-only)
        self.tokens = SpecialTokens()

        # Initialize unified coordinate token manager
        if coordinate_enabled:
            # FAIL-FAST: max_coord_value must be provided when coordinate tokens are enabled
            if "max_coord_value" not in kwargs:
                raise ValueError(
                    "max_coord_value must be provided when coordinate tokens are enabled. "
                    "Ensure this field is explicitly set in your configuration."
                )

            # Store coordinate configuration for later initialization
            self.coordinate_config = {
                "enable_coordinate_tokens": True,
                "max_coord_value": kwargs["max_coord_value"],
                "coordinate_loss_weight": kwargs.get("coordinate_loss_weight", 1.0),
                "regular_loss_weight": kwargs.get("regular_loss_weight", 1.0),
                "soft_expectation_temperature": kwargs.get(
                    "soft_expectation_temperature", 1.0
                ),
            }

            # Initialize placeholders - will be set up later when model is available
            self.token_manager = None
            self.coordinate_manager = None

            logger.debug(
                f"🎯 Coordinate tokens configuration stored: max_coord={self.coordinate_config['max_coord_value']}"
            )
        else:
            self.coordinate_manager = None
            self.coordinate_config = None

        # Build system prompt using global config
        self.system_prompt = self._build_system_prompt()

    def _update_coordinate_token_ranges(self):
        """Initialize coordinate manager and update token ranges after tokenizer extension.

        Enhanced with comprehensive validation and error handling for coordinate token setup.
        """
        if not hasattr(self, "coordinate_config") or not self.coordinate_config:
            logger.debug(
                "🎯 No coordinate configuration - skipping coordinate manager setup"
            )
            return

        # Initialize UnifiedTokenManager now that we have the tokenizer set up
        # Note: We still don't have a model, but we can initialize without resizing embeddings
        logger.info("🎯 Initializing coordinate manager with current tokenizer...")

        try:
            # Create a simple coordinate manager without model resizing
            from src.utils.tokens.special_tokens import SimpleCoordinateManager

            self.coordinate_manager = SimpleCoordinateManager(
                tokenizer=self.tokenizer,
                max_coord_value=self.coordinate_config["max_coord_value"],
            )

            # Set token manager for backward compatibility
            self.token_manager = self.coordinate_manager

            # Enhanced validation: Verify coordinate manager initialization
            self._validate_coordinate_manager_initialization()

            logger.info("✅ Coordinate manager initialized successfully")
            logger.info(
                f"   max_coord_value: {self.coordinate_config['max_coord_value']}"
            )
            logger.info(f"   Language: {self.language}")
            logger.info(f"   Output format: Coordinate tokens + JSON fallback")

        except Exception as e:
            # FAIL-FAST: If coordinate manager initialization fails, raise detailed error
            logger.error(f"❌ Coordinate manager initialization failed: {e}")
            logger.error(f"   Config: {self.coordinate_config}")
            logger.error(f"   Tokenizer vocab size: {len(self.tokenizer.get_vocab())}")
            raise RuntimeError(
                f"Failed to initialize coordinate manager for coordinate token mode. "
                f"This is required when coordinate_tokens_enabled=True. Error: {e}"
            ) from e

        # Default context
        self._current_context: str = "training"

    def _validate_coordinate_manager_initialization(self) -> None:
        """
        Validate that coordinate manager was properly initialized with all required components.

        Raises:
            RuntimeError: If validation fails
        """
        # Basic existence check
        if not self.coordinate_manager:
            raise RuntimeError("Coordinate manager is None after initialization")

        # Check essential attributes are set
        required_attrs = ["tokenizer", "max_coord_value", "config"]
        for attr in required_attrs:
            if not hasattr(self.coordinate_manager, attr):
                raise RuntimeError(
                    f"Coordinate manager missing required attribute: {attr}"
                )

        # Check configuration is valid
        config = self.coordinate_manager.config
        if not hasattr(config, "enable_coordinate_tokens"):
            raise RuntimeError(
                "Coordinate manager config missing enable_coordinate_tokens"
            )

        # Use our helper for general coordinate manager validation
        try:
            self._validate_coordinate_manager("initialization validation")
        except RuntimeError as e:
            raise RuntimeError(f"Coordinate manager validation failed: {e}") from e

        # Check if coordinate tokens are available in vocabulary
        has_tokens = self.coordinate_manager.has_coordinate_tokens()
        vocab = self.tokenizer.get_vocab()
        coord_0_token = "<|coord_0|>"

        if coord_0_token in vocab:
            if not has_tokens:
                raise RuntimeError(
                    "Coordinate tokens found in vocabulary but coordinate manager reports they are not available. "
                    "This indicates a mismatch in token detection logic."
                )

            # Log successful validation
            logger.info(
                f"✅ Coordinate token validation passed: "
                f"range [{self.coordinate_manager.coord_start_id}, {self.coordinate_manager.coord_end_id}) "
                f"covers {self.coordinate_manager.coord_end_id - self.coordinate_manager.coord_start_id} tokens"
            )
        else:
            # Coordinate tokens not found in vocabulary
            if has_tokens:
                raise RuntimeError(
                    "Coordinate manager reports tokens are available but <|coord_0|> not found in vocabulary. "
                    "This indicates a critical inconsistency."
                )

            logger.warning(
                "⚠️ Coordinate tokens not found in vocabulary. "
                "Chat processor will fall back to JSON format. "
                "To use coordinate token mode, ensure tokens are properly added during model initialization."
            )

    def _build_system_prompt(self) -> str:
        """Build system prompt with pure JSON format for object detection."""

        # Use the proper prompt selection function
        task_type = "training" if self.use_training_prompts else "evaluation"
        base_prompt = get_system_prompt(language=self.language, task_type=task_type)

        # Get few shot section based on language
        if self.language == "chinese":
            few_shot_section = CHINESE_FEW_SHOT_SECTION
        else:  # English
            few_shot_section = ENGLISH_FEW_SHOT_SECTION

        # Candidates system removed - no longer needed

        return base_prompt + few_shot_section

    def process_sample(self, raw_sample: Dict[str, Any]) -> ChatProcessorOutput:
        """
        Process a single sample from teacher/student structured format.

        Args:
            raw_sample: Sample with 'teachers' (List[Sample]) and 'student' (Sample) structure

        Returns:
            ChatProcessorOutput containing input_ids, labels, attention_mask, pixel_values, image_grid_thw, and ground_truth_objects
        """
        # Debug: Log the sample structure
        teachers = raw_sample.get("teachers", [])
        logger.debug(f"📝 Processing sample: {len(teachers)} teachers + 1 student")

        # 1. Create conversation messages
        conversation_messages = self._create_conversation_messages(raw_sample)

        # 2. Process images, expand vision tokens, and get image dimensions
        (
            processed_conversation,
            images,
            image_dims,
        ) = self._process_images_and_tokens(
            conversation_messages, self._extract_all_image_paths(raw_sample)
        )

        # 3. Apply chat template and tokenize
        input_ids, labels, teacher_spans, student_spans = self._tokenize_conversation(
            processed_conversation
        )

        # 4. Process images for model input
        pixel_values, image_grid_thw = self._process_images_for_model(images)

        # 5. Extract and normalize ground truth objects for the student
        ground_truth_objects = self._extract_and_normalize_ground_truth(
            raw_sample, image_dims
        )

        return ChatProcessorOutput(
            input_ids=input_ids,
            labels=labels,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            ground_truth_objects=ground_truth_objects,
            teacher_assistant_spans=teacher_spans,
            student_assistant_spans=student_spans,
        )

    def _create_conversation_messages(
        self, sample: Dict[str, Any]
    ) -> List[ChatMessage]:
        """Return a validated list of :class:`ChatMessage` objects built from *sample*."""

        messages: list[ChatMessage] = []

        # 1) System prompt ----------------------------------------------------------------
        messages.append(ChatMessage(role="system", content=self.system_prompt))

        # 2) Learning instruction if teachers are present --------------------------------
        # FAIL-FAST: Validate sample structure
        if "teachers" not in sample:
            raise ValueError("Sample must contain 'teachers' field (can be empty list)")
        teachers: Sequence[Dict[str, Any]] = sample["teachers"]

        if teachers:
            from src.utils.prompt import get_learning_instruction

            learning_instruction = get_learning_instruction(
                language=self.language,
            )
            if learning_instruction.strip():
                messages.append(ChatMessage(role="user", content=learning_instruction))
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content="明白！我会仔细学习参考示例中的检测模式、标注风格和判断标准，然后应用到目标图像的检测中。"
                        if self.language == "chinese"
                        else "Understood! I will carefully study the detection patterns, annotation styles, and judgment criteria in the reference examples, then apply them to detect objects in the target image.",
                    )
                )

        # 3) Teacher examples --------------------------------------------------------------
        for i, teacher in enumerate(teachers):
            # FAIL-FAST: Validate teacher structure
            if "objects" not in teacher:
                raise ValueError(f"Teacher {i} must contain 'objects' field")

            # User uploads a teacher example image with clear context
            if self.language == "chinese":
                if len(teachers) == 1:
                    user_content = "参考示例:\n<image>"
                else:
                    user_content = f"参考示例 {i + 1}/{len(teachers)}:\n<image>"
            else:
                if len(teachers) == 1:
                    user_content = "Reference Example:\n<image>"
                else:
                    user_content = (
                        f"Reference Example {i + 1}/{len(teachers)}:\n<image>"
                    )

            messages.append(ChatMessage(role="user", content=user_content))

            # Assistant returns detection JSON with learning context
            objects = teacher["objects"]
            sorted_objects = self._sort_objects_by_position(objects)
            assistant_response = self._format_objects_response(sorted_objects)
            messages.append(ChatMessage(role="assistant", content=assistant_response))

        # 3) Student target ---------------------------------------------------------------
        # FAIL-FAST: Validate student structure
        if "student" not in sample:
            # If no explicit student field, the sample itself is the student
            student = sample
        else:
            student = sample["student"]

        # FAIL-FAST: Validate student structure
        if "objects" not in student:
            raise ValueError("Student must contain 'objects' field")

        # Add transitional instruction if teachers were provided
        if teachers:
            if self.language == "chinese":
                target_content = "现在请根据以上参考示例的检测模式和标注风格，检测以下目标图像:\n<image>"
            else:
                target_content = "Now apply the detection patterns and annotation style from the reference examples to detect objects in this target image:\n<image>"
        else:
            if self.language == "chinese":
                target_content = "请检测以下图像中的设备和部件:\n<image>"
            else:
                target_content = "Please detect all equipment and components in the following image:\n<image>"

        messages.append(ChatMessage(role="user", content=target_content))

        student_objects = student["objects"]
        sorted_student_objects = self._sort_objects_by_position(student_objects)
        student_response = self._format_objects_response(sorted_student_objects)
        messages.append(ChatMessage(role="assistant", content=student_response))

        return messages

    def _extract_all_image_paths(self, sample: Dict[str, Any]) -> List[str]:
        """Collect **all** image paths referenced in *sample* (teachers + student)."""

        image_paths: list[str] = []

        # FAIL-FAST: Validate sample structure
        if "teachers" not in sample:
            raise ValueError("Sample must contain 'teachers' field (can be empty list)")

        for teacher in sample["teachers"]:
            # FAIL-FAST: Validate teacher structure
            if "images" not in teacher:
                raise ValueError("Teacher must contain 'images' field")
            image_paths.extend(teacher["images"])

        # FAIL-FAST: Get student sample with validation
        if "student" not in sample:
            # If no explicit student field, the sample itself is the student
            student = sample
        else:
            student = sample["student"]

        # FAIL-FAST: Validate student structure
        if "images" not in student:
            raise ValueError("Student must contain 'images' field")

        image_paths.extend(student["images"])

        return image_paths

    def _sort_objects_by_position(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort objects by position (top-to-bottom, left-to-right)."""

        def sort_key(obj):
            box = obj.get("bbox_2d", [0, 0, 0, 0])
            return (box[1], box[0])  # Sort by y first, then x

        return sorted(objects, key=sort_key)

    def _create_json_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized JSON object from an object with geometry data.

        Args:
            obj: Object dictionary containing geometry data

        Returns:
            Standardized JSON object with geometry and label

        Raises:
            ValueError: If object structure is invalid
        """
        # FAIL-FAST: Validate object structure - must have at least one geometry type
        if not any(key in obj for key in ["bbox_2d", "square", "line"]):
            raise ValueError(
                f"Object must contain at least one geometry type (bbox_2d, square, or line): {obj}"
            )

        # Extract geometry data - support multiple formats
        if "bbox_2d" in obj:
            # FAIL-FAST: Validate description field
            if "desc" not in obj:
                raise ValueError(
                    f"Object with bbox_2d must contain 'desc' field: {obj}"
                )

            json_obj = {
                "bbox_2d": obj["bbox_2d"],
                "label": obj["desc"],
            }
        elif "square" in obj:
            # FAIL-FAST: Validate description field
            if "desc" not in obj:
                raise ValueError(f"Object with square must contain 'desc' field: {obj}")

            json_obj = {
                "square": obj["square"],
                "label": obj["desc"],
            }
        elif "line" in obj:
            # FAIL-FAST: Validate description field
            if "desc" not in obj:
                raise ValueError(f"Object with line must contain 'desc' field: {obj}")

            json_obj = {"line": obj["line"], "label": obj["desc"]}
        else:
            # This should never happen due to the initial validation
            raise ValueError(f"Object has no recognized geometry type: {obj}")

        return json_obj

    def _format_objects_response(self, objects: List[Dict[str, Any]]) -> str:
        """Format objects into appropriate format (JSON or coordinate tokens).

        Enhanced with robust coordinate token conversion and fallback mechanisms.
        """
        if not objects:
            return "[]"

        json_objects = []
        for obj in objects:
            json_objects.append(self._create_json_object(obj))

        # Format as JSON first
        json_response = json.dumps(
            json_objects, ensure_ascii=False, separators=(",", ": ")
        )

        # Convert to coordinate token format if enabled with enhanced reliability
        logger.debug(
            f"🔍 COORDINATE CHECK: coordinate_manager={self.coordinate_manager is not None}"
        )
        if self.coordinate_manager:
            logger.debug(
                f"🔍 COORDINATE CHECK: enable_coordinate_tokens={self.coordinate_manager.config.enable_coordinate_tokens}"
            )

        if (
            self.coordinate_manager
            and self.coordinate_manager.config.enable_coordinate_tokens
        ):
            try:
                # Enhanced coordinate conversion with validation
                coordinate_response = (
                    self._convert_to_coordinate_format_with_validation(
                        json_response, json_objects
                    )
                )

                logger.debug(
                    f"🎯 COORDINATE TOKENS: Enabled - converting JSON to coordinate format"
                )
                logger.debug(f"   📋 JSON response: {json_response}")
                logger.debug(f"   🎯 Coordinate response: {coordinate_response}")
                return coordinate_response

            except Exception as e:
                # FAIL-FAST: Log the error and fall back to JSON if coordinate conversion fails
                logger.error(f"❌ Coordinate token conversion failed: {e}")
                logger.error(f"   📋 Falling back to JSON format: {json_response}")
                # Instead of falling back silently, raise the error to surface issues
                raise RuntimeError(
                    f"Coordinate token conversion failed for objects: {json_objects}. "
                    f"Original error: {e}"
                ) from e
        else:
            logger.debug(f"🎯 COORDINATE TOKENS: Disabled - returning JSON format")
            logger.debug(f"   📋 JSON response: {json_response}")
            return json_response

    def _convert_to_coordinate_format_with_validation(
        self, json_response: str, json_objects: List[Dict[str, Any]]
    ) -> str:
        """
        Convert JSON to coordinate format with enhanced validation and error handling.

        Args:
            json_response: JSON string representation
            json_objects: Parsed JSON objects for validation

        Returns:
            Coordinate token formatted string

        Raises:
            RuntimeError: If coordinate token conversion fails validation
        """
        # Validate coordinate manager
        self._validate_coordinate_manager("coordinate token conversion")

        # Pre-validation: Check coordinate manager has valid token ranges
        if not self.coordinate_manager.has_coordinate_tokens():
            raise RuntimeError(
                "Coordinate tokens not available in vocabulary. "
                "Ensure coordinate tokens were properly added during model initialization."
            )

        # Enhanced pre-validation: Check objects can be converted to coordinates
        logger.debug(
            f"🎯 Pre-validating {len(json_objects)} objects for coordinate conversion"
        )

        for i, obj in enumerate(json_objects):
            try:
                self._validate_coordinate_ranges(obj, i)
            except Exception as e:
                logger.error(f"❌ Pre-validation failed for object {i}: {obj}")
                raise RuntimeError(f"Pre-validation failed for object {i}: {e}") from e

        logger.debug("✅ All objects passed pre-validation")

        # Perform the actual conversion using the coordinate manager
        try:
            logger.debug(
                f"🔄 Converting JSON to coordinate format: {json_response[:100]}..."
            )
            coordinate_response = (
                self.coordinate_manager.convert_json_to_coordinate_format(json_response)
            )
            logger.debug(
                f"✅ Conversion completed, result length: {len(coordinate_response)}"
            )

        except Exception as e:
            logger.error(f"❌ Coordinate manager conversion failed: {e}")
            logger.error(f"   📄 Input JSON: {json_response}")
            raise RuntimeError(f"Coordinate manager conversion failed: {e}") from e

        # Post-conversion validation: Ensure coordinate tokens were actually generated
        if coordinate_response == json_response:
            raise RuntimeError(
                "Coordinate token conversion produced identical output to JSON input. "
                "This indicates the conversion process failed silently."
            )

        # Enhanced validation: Check that we have expected number of coordinate tokens
        expected_coord_count = 0
        for obj in json_objects:
            # Count coordinates in each object
            for geom_key in ["bbox_2d", "line", "square"]:
                if geom_key in obj and isinstance(obj[geom_key], list):
                    expected_coord_count += len(obj[geom_key])
                    break

        actual_coord_count = coordinate_response.count("<|coord_")
        if actual_coord_count != expected_coord_count:
            logger.warning(
                f"⚠️ Coordinate token count mismatch: expected {expected_coord_count}, "
                f"found {actual_coord_count} in result"
            )
            # Don't fail for count mismatch, just warn

        # Validate the output contains expected coordinate token patterns
        try:
            self._validate_coordinate_token_output(coordinate_response)
        except Exception as e:
            logger.error(f"❌ Post-conversion validation failed: {e}")
            logger.error(f"   📄 Coordinate response: {coordinate_response[:200]}...")
            raise RuntimeError(f"Post-conversion validation failed: {e}") from e

        logger.debug(
            f"✅ Coordinate conversion validation complete: "
            f"{len(json_objects)} objects → {actual_coord_count} coordinate tokens"
        )

        return coordinate_response

    def _validate_geometry_type(
        self, obj: Dict[str, Any], obj_index: int
    ) -> Tuple[str, List]:
        """
        Validate and extract geometry type and coordinates from an object.

        Args:
            obj: Object dictionary containing geometry coordinates
            obj_index: Index of object for error reporting

        Returns:
            Tuple of (geometry_type, coordinates)

        Raises:
            ValueError: If geometry type is invalid or missing
        """
        # Check for available geometry types
        available_geom_types = []
        for geom_key in ["bbox_2d", "quad", "line"]:
            if geom_key in obj:
                available_geom_types.append(geom_key)

        if not available_geom_types:
            raise ValueError(
                f"Object {obj_index} has no valid geometry coordinates. "
                f"Expected one of: bbox_2d, square, line. Found keys: {list(obj.keys())}"
            )

        if len(available_geom_types) > 1:
            logger.warning(
                f"⚠️ Object {obj_index} has multiple geometry types: {available_geom_types}. "
                f"Using first available: {available_geom_types[0]}"
            )

        # Use the first available geometry type
        geometry_type = available_geom_types[0]
        coords = obj[geometry_type]

        # Enhanced coordinate structure validation
        if not isinstance(coords, list):
            raise ValueError(
                f"Object {obj_index} {geometry_type} coordinates must be a list, got {type(coords)}: {coords}"
            )

        if not coords:
            raise ValueError(
                f"Object {obj_index} {geometry_type} coordinate list is empty"
            )

        # Validate coordinate count for each geometry type
        if geometry_type == "bbox_2d":
            if len(coords) != 4:
                raise ValueError(
                    f"Object {obj_index} bbox_2d expected 4 coordinates [x1, y1, x2, y2], "
                    f"got {len(coords)}: {coords}"
                )
        elif geometry_type == "square":
            if len(coords) != 8:
                raise ValueError(
                    f"Object {obj_index} square expected 8 coordinates [x1, y1, x2, y2, x3, y3, x4, y4], "
                    f"got {len(coords)}: {coords}"
                )
        elif geometry_type == "line":
            if len(coords) < 4 or len(coords) % 2 != 0:
                raise ValueError(
                    f"Object {obj_index} line expected ≥4 coordinates with even count (x,y pairs), "
                    f"got {len(coords)}: {coords}"
                )

        return geometry_type, coords

    def _validate_coordinate_ranges(self, obj: Dict[str, Any], obj_index: int) -> None:
        """
        Validate that all coordinate values in an object are within valid range with enhanced checks.

        Args:
            obj: Object dictionary containing geometry coordinates
            obj_index: Index of object for error reporting

        Raises:
            ValueError: If coordinate values are invalid
        """
        # Validate coordinate manager
        self._validate_coordinate_manager("coordinate validation")
        max_coord = self.coordinate_manager.max_coord_value

        # Extract coordinates based on geometry type with enhanced validation
        geometry_type, coords = self._validate_geometry_type(obj, obj_index)

        # Enhanced coordinate value validation
        valid_coords = []
        for i, coord in enumerate(coords):
            try:
                # Enhanced type conversion with better error handling
                if isinstance(coord, str):
                    coord_str = coord.strip()
                    if not coord_str:
                        raise ValueError("Empty coordinate string")

                    # Handle decimals and negative numbers
                    try:
                        coord_float = float(coord_str)
                    except ValueError as e:
                        raise ValueError(
                            f"Cannot convert '{coord}' to number: {e}"
                        ) from e
                else:
                    coord_float = float(coord)

                coord_int = int(coord_float)

                # Enhanced range validation with specific error messages
                if coord_int < 0:
                    raise ValueError(
                        f"Object {obj_index} {geometry_type}[{i}] coordinate {coord_int} "
                        f"is negative. Coordinates must be non-negative integers >= 0."
                    )

                if coord_int >= max_coord:
                    raise ValueError(
                        f"Object {obj_index} {geometry_type}[{i}] coordinate {coord_int} "
                        f"exceeds maximum value {max_coord - 1}. "
                        f"Valid range: [0, {max_coord - 1}]. "
                        f"Check max_coord_value configuration ({max_coord})."
                    )

                # Additional validation for coordinate token mode
                if self.coordinate_manager.has_coordinate_tokens():
                    # Verify this coordinate can be mapped to a valid token
                    if (
                        self.coordinate_manager.coord_start_id is not None
                        and self.coordinate_manager.coord_end_id is not None
                    ):
                        expected_token_id = (
                            self.coordinate_manager.coord_start_id + coord_int
                        )
                        if expected_token_id >= self.coordinate_manager.coord_end_id:
                            raise ValueError(
                                f"Object {obj_index} {geometry_type}[{i}] coordinate {coord_int} "
                                f"maps to token ID {expected_token_id} which exceeds coordinate token range "
                                f"[{self.coordinate_manager.coord_start_id}, {self.coordinate_manager.coord_end_id}). "
                                f"This indicates a vocabulary-configuration mismatch."
                            )

                valid_coords.append(coord_int)

            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Object {obj_index} {geometry_type}[{i}] has invalid coordinate value '{coord}': {e}"
                ) from e

        # Log successful validation for debugging
        logger.debug(
            f"✅ Object {obj_index} coordinate validation passed: "
            f"{geometry_type}={valid_coords} (range: [0, {max_coord - 1}])"
        )

    def _validate_coordinate_token_output(self, coordinate_response: str) -> None:
        """
        Validate that coordinate token output contains expected patterns with enhanced checks.

        Args:
            coordinate_response: Coordinate token formatted response

        Raises:
            RuntimeError: If output validation fails
        """
        if not coordinate_response or not coordinate_response.strip():
            raise RuntimeError("Coordinate token output is empty or whitespace-only")

        import re

        # Enhanced pattern matching for coordinate tokens: <|coord_N|> where N is a number
        coord_token_pattern = r"<\|coord_(\d+)\|>"
        coord_token_matches = re.findall(coord_token_pattern, coordinate_response)

        if not coord_token_matches:
            raise RuntimeError(
                f"Coordinate token output validation failed: no coordinate tokens found in output. "
                f"Expected pattern: <|coord_N|> where N is a number. "
                f"Output: {coordinate_response[:200]}{'...' if len(coordinate_response) > 200 else ''}"
            )

        # Validate coordinate token values are within expected range
        max_coord = (
            self.coordinate_manager.max_coord_value if self.coordinate_manager else 2048
        )
        invalid_coords = []

        for coord_str in coord_token_matches:
            try:
                coord_value = int(coord_str)
                if coord_value < 0 or coord_value >= max_coord:
                    invalid_coords.append(coord_value)
            except ValueError:
                invalid_coords.append(coord_str)

        if invalid_coords:
            raise RuntimeError(
                f"Coordinate token output validation failed: invalid coordinate values found: {invalid_coords}. "
                f"Valid range: [0, {max_coord - 1}]"
            )

        # Enhanced geometry token validation
        geometry_patterns = {
            "box_start": r"<\|box_start\|>",
            "box_end": r"<\|box_end\|>",
            "line_start": r"<\|line_start\|>",
            "line_end": r"<\|line_end\|>",
            "square_start": r"<\|square_start\|>",
            "square_end": r"<\|square_end\|>",
        }

        found_geometry_tokens = {}
        for token_name, pattern in geometry_patterns.items():
            matches = re.findall(pattern, coordinate_response)
            if matches:
                found_geometry_tokens[token_name] = len(matches)

        if not found_geometry_tokens:
            raise RuntimeError(
                f"Coordinate token output validation failed: no geometry tokens found in output. "
                f"Expected one of: {list(geometry_patterns.keys())}. "
                f"Output: {coordinate_response[:200]}{'...' if len(coordinate_response) > 200 else ''}"
            )

        # Validate geometry token pairing (start/end tokens should match)
        geometry_pairs = [
            ("box_start", "box_end"),
            ("line_start", "line_end"),
            ("square_start", "square_end"),
        ]

        for start_token, end_token in geometry_pairs:
            start_count = found_geometry_tokens.get(start_token, 0)
            end_count = found_geometry_tokens.get(end_token, 0)

            if start_count != end_count and start_count > 0:
                logger.warning(
                    f"⚠️ Geometry token pairing mismatch: {start_token}={start_count}, {end_token}={end_count}"
                )

        # Enhanced object reference token validation
        object_ref_start_pattern = r"<\|object_ref_start\|>"
        object_ref_end_pattern = r"<\|object_ref_end\|>"

        start_refs = re.findall(object_ref_start_pattern, coordinate_response)
        end_refs = re.findall(object_ref_end_pattern, coordinate_response)

        if not start_refs:
            raise RuntimeError(
                f"Coordinate token output validation failed: missing <|object_ref_start|> tokens. "
                f"Output: {coordinate_response[:200]}{'...' if len(coordinate_response) > 200 else ''}"
            )

        if len(start_refs) != len(end_refs):
            logger.warning(
                f"⚠️ Object reference token count mismatch: "
                f"start={len(start_refs)}, end={len(end_refs)}"
            )

        # Additional structural validation: check for coordinate sequences within brackets
        bracket_coord_pattern = r"\[<\|coord_\d+\|>(?:,<\|coord_\d+\|>)*\]"
        coord_sequences = re.findall(bracket_coord_pattern, coordinate_response)

        if not coord_sequences:
            raise RuntimeError(
                f"Coordinate token output validation failed: no valid coordinate sequences found. "
                f"Expected pattern: [<|coord_N|>,<|coord_M|>,...]. "
                f"Output: {coordinate_response[:200]}{'...' if len(coordinate_response) > 200 else ''}"
            )

        # Log detailed validation results
        logger.debug(f"✅ Coordinate token output validation passed:")
        logger.debug(f"   📊 {len(coord_token_matches)} coordinate tokens found")
        logger.debug(f"   🔧 Geometry tokens: {found_geometry_tokens}")
        logger.debug(
            f"   📦 Object references: start={len(start_refs)}, end={len(end_refs)}"
        )
        logger.debug(f"   📐 Coordinate sequences: {len(coord_sequences)}")
        logger.debug(f"   📏 Response length: {len(coordinate_response)} characters")

    def _process_images_and_tokens(
        self, conversation: List[ChatMessage], image_paths: List[str]
    ) -> Tuple[List[ChatMessage], List[Image.Image], List[Tuple[int, int]]]:
        """
        Process images and expand vision tokens in conversation.

        Replaces <image> placeholders with proper vision token sequences.
        """
        processed_conversation: list[ChatMessage] = []
        images = []
        image_dims = []
        image_index = 0

        for message in conversation:
            content = message.content

            # Process image placeholders
            while "<image>" in content and image_index < len(image_paths):
                # Load image
                image_path = self.data_root / image_paths[image_index]
                # Fail-fast: raise explicit error if the image cannot be loaded.
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                image_dims.append(image.size)  # (width, height)

                # Calculate number of vision tokens required for this image
                num_vision_tokens = self._calculate_image_tokens(image)

                # Create vision token sequence using helper that inserts spaces between <|image_pad|> tokens
                # This prevents the tokenizer from returning `None` IDs for contiguous special tokens.
                vision_token_sequence = self.tokens.format_vision_tokens(
                    num_vision_tokens
                )

                # Replace placeholder with vision tokens
                content = content.replace("<image>", vision_token_sequence, 1)
                image_index += 1

            processed_conversation.append(
                ChatMessage(role=message.role, content=content)
            )

        # Final validation
        if image_index != len(image_paths):
            logger.warning(
                f"Mismatch between image placeholders and image paths. "
                f"Found {image_index} placeholders, but {len(image_paths)} images."
            )

        return processed_conversation, images, image_dims

    def _calculate_image_tokens(self, image: Image.Image) -> int:
        """Return the exact number of <|image_pad|> tokens the processor will emit for *image*.

        Older logic estimated this figure from a dummy tensor shaped like the
        image.  That breaks if the HF image-processor performs an internal
        resize or uses a different patch/merge configuration.  We now let the
        processor do its real preprocessing and read the `image_grid_thw`
        metadata that the model itself will consume during the forward pass.
        """
        # FAIL-FAST: Validate image processor has required attributes
        if not hasattr(self.image_processor, "preprocess"):
            raise AttributeError("Image processor must have 'preprocess' method")

        # Run the *actual* preprocessing pipeline for a single image.  This is
        # comparatively cheap (<1 ms for 896×1344) and guarantees the grid is
        # consistent with training/inference.
        processed = self.image_processor.preprocess([image], return_tensors="pt")

        # FAIL-FAST: Validate processed output contains required fields
        if "image_grid_thw" not in processed:
            raise ValueError(
                "Image processor did not return required 'image_grid_thw' field"
            )

        grid_thw = processed["image_grid_thw"][0]  # (t, h, w)

        # FAIL-FAST: Require merge_size to be explicitly defined
        if not hasattr(self.image_processor, "merge_size"):
            raise ValueError("Image processor must have 'merge_size' attribute defined")
        merge_size = self.image_processor.merge_size

        tokens_per_merge = merge_size**2

        # Number of flattened patch tokens after the spatial-merge step that
        # the vision tower applies internally.
        num_tokens: int = int(grid_thw.prod().item() // tokens_per_merge)

        return num_tokens

    def _extract_and_normalize_ground_truth(
        self, sample: Dict[str, Any], image_dims: List[Tuple[int, int]]
    ) -> List[GroundTruthObject]:
        r"""Return a list of :class:`src.schema.GroundTruthObject` instances.

        The bounding boxes are converted from absolute pixel coordinates to the
        *normalised* \[0,1] range expected by the detection loss.
        """
        # FAIL-FAST: Validate sample and image_dims
        if not isinstance(sample, dict):
            raise TypeError(f"Sample must be a dictionary, got {type(sample)}")

        if not image_dims:
            return []  # No images, no ground truth objects

        # FAIL-FAST: Validate student structure
        if "student" not in sample:
            # If no explicit student field, the sample itself is the student
            student = sample
        else:
            student = sample["student"]
            if not isinstance(student, dict):
                raise TypeError(f"Student must be a dictionary, got {type(student)}")

        # FAIL-FAST: Validate student structure
        if "objects" not in student:
            raise ValueError("Student must contain 'objects' field")

        student_objects = student["objects"]
        if not isinstance(student_objects, list):
            raise TypeError(f"Objects must be a list, got {type(student_objects)}")

        # The last image in the list corresponds to the student.
        if not student_objects:
            return []

        student_image_dims = image_dims[-1]
        width, height = student_image_dims

        normalized_objects: list[GroundTruthObject] = []
        for i, obj in enumerate(student_objects):
            # FAIL-FAST: Validate object is a dictionary
            if not isinstance(obj, dict):
                raise TypeError(f"Object {i} must be a dictionary, got {type(obj)}")

            # FAIL-FAST: Validate object structure - must have at least one geometry type
            if not any(key in obj for key in ["bbox_2d", "square", "line"]):
                raise ValueError(
                    f"Object {i} must contain at least one geometry type (bbox_2d, square, or line): {obj}"
                )

            # FAIL-FAST: Validate description field
            if "desc" not in obj:
                raise ValueError(f"Object {i} must contain 'desc' field: {obj}")
            desc = obj["desc"]
            if not isinstance(desc, str):
                raise TypeError(
                    f"Object {i} description must be a string, got {type(desc)}"
                )

            # Preserve native geometry format - no conversion to bounding box
            # Coordinates should already be normalized by the data conversion pipeline
            if "bbox_2d" in obj:
                coords = obj["bbox_2d"]
                geometry_type = "bbox_2d"
                # Validate bbox format
                if not (isinstance(coords, list) and len(coords) == 4):
                    raise ValueError(f"Invalid bbox_2d format for object {i}: {coords}")
                # Coordinates should already be normalized, so x1 < x2 and y1 < y2
                x1, y1, x2, y2 = coords
                if x1 >= x2 or y1 >= y2:
                    raise ValueError(
                        f"Invalid bbox_2d coordinates for object {i}: {coords}. "
                        f"Coordinates should be normalized during data preprocessing."
                    )
            elif "square" in obj:
                coords = obj["square"]
                geometry_type = "square"
                # Validate square format (8 coordinates)
                if not (isinstance(coords, list) and len(coords) == 8):
                    raise ValueError(f"Invalid square format for object {i}: {coords}")
            elif "line" in obj:
                coords = obj["line"]
                geometry_type = "line"
                # Validate line format (even number of coordinates >= 4)
                if not (
                    isinstance(coords, list)
                    and len(coords) >= 4
                    and len(coords) % 2 == 0
                ):
                    raise ValueError(f"Invalid line format for object {i}: {coords}")

            # Validate coordinates are within image bounds
            if geometry_type == "bbox_2d":
                x1, y1, x2, y2 = coords
                if not (
                    0 <= x1 < width
                    and 0 <= y1 < height
                    and 0 < x2 <= width
                    and 0 < y2 <= height
                ):
                    raise ValueError(
                        f"bbox_2d coordinates out of bounds for object {i}: {coords} for image size {width}x{height}"
                    )
            else:
                # For line and square, validate all coordinate pairs
                for j in range(0, len(coords), 2):
                    x, y = coords[j], coords[j + 1]
                    if not (0 <= x < width and 0 <= y < height):
                        raise ValueError(
                            f"{geometry_type} coordinates out of bounds for object {i}: point ({x}, {y}) for image size {width}x{height}"
                        )

            # Create normalized coordinates for the specific geometry type
            if geometry_type == "bbox_2d":
                # Normalize bbox coordinates to [0, 1] range
                normalized_coords = [
                    coords[0] / width,
                    coords[1] / height,
                    coords[2] / width,
                    coords[3] / height,
                ]
                # Build structured GT object with bbox_2d
                normalized_objects.append(
                    GroundTruthObject(
                        bbox=normalized_coords,
                        description=desc,
                        geometry_type="bbox_2d",
                    )
                )
            else:
                # For line and square objects, normalize all coordinate pairs
                normalized_coords = []
                for j in range(0, len(coords), 2):
                    x, y = coords[j], coords[j + 1]
                    normalized_coords.extend([x / width, y / height])

                # Build structured GT object preserving native geometry type
                if geometry_type == "line":
                    normalized_objects.append(
                        GroundTruthObject(
                            bbox=normalized_coords,
                            description=desc,
                            geometry_type="line",
                        )
                    )
                elif geometry_type == "square":
                    normalized_objects.append(
                        GroundTruthObject(
                            bbox=normalized_coords,
                            description=desc,
                            geometry_type="square",
                        )
                    )

        return normalized_objects

    @typechecked
    def _tokenize_conversation(
        self, conversation: List[ChatMessage]
    ) -> Tuple[
        TensorType["B", "S"],
        TensorType["B", "S"],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
    ]:
        """
        Tokenize conversation and create labels with proper masking.
        """
        # ``apply_chat_template`` expects a ``List[dict]`` – convert once here.
        formatted_text = self.tokenizer.apply_chat_template(
            [asdict(msg) for msg in conversation],
            tokenize=False,
            add_generation_prompt=False,
        )

        # Debug: Log the formatted text before adding endoftext
        logger.debug(
            f"📄 Formatted text before endoftext: {repr(formatted_text[-100:])}"
        )

        # Check if endoftext is already present
        if not formatted_text.endswith(self.tokens.ENDOFTEXT):
            # Add end of text token only if not already present
            formatted_text += self.tokens.ENDOFTEXT
            logger.debug(f"✅ Added ENDOFTEXT token")
        else:
            logger.debug(f"✅ ENDOFTEXT token already present")

        # Debug: Log the formatted text after adding endoftext
        logger.debug(
            f"📄 Formatted text after endoftext: {repr(formatted_text[-100:])}"
        )
        logger.debug(f"🔍 ENDOFTEXT token: {repr(self.tokens.ENDOFTEXT)}")

        # NOTE: Using `return_tensors="pt"` here leads to a hard failure when the tokenizer
        # encounters any `None` values in the produced python lists (typically caused by
        # special-token mis-alignment or exceedingly long inputs).  Instead we first obtain
        # the raw python lists from the tokenizer *without* tensor conversion and only then
        # convert to `torch.Tensor` once we are confident the data structure is correct.

        tokenized = self.tokenizer(
            formatted_text,
            padding=False,
            truncation=False,
            add_special_tokens=False,  # we explicitly bake all special tokens into the prompt
        )

        # Flatten tokenizer output using our helper
        flat_ids = self._flatten_tokenizer_output(tokenized["input_ids"])

        # Convert to tensor (1D)
        input_ids_1d = torch.tensor(flat_ids, dtype=torch.long)

        # Create labels (copy of input_ids)
        labels_1d = input_ids_1d.clone()

        # Mask non-assistant tokens → only assistant messages contribute to loss
        # Also extract teacher/student spans for loss splitting
        labels_1d, teacher_spans, student_spans = self._mask_non_assistant_tokens(
            labels_1d, conversation, formatted_text
        )

        # We **do not** unmask `<|endoftext|>` because in this project that token
        # serves the dual role of *padding* as well as a legacy data delimiter.
        # The actual generation stop token is `<|im_end|>` (tokenizer
        # `eos_token`).  Leaving `<|endoftext|>` masked ensures the language
        # model does not learn to emit padding tokens during generation.

        # ------------------------------------------------------------------
        # Collators (especially PackedDataCollator) expect an explicit batch
        # dimension (B=1) so that concatenation along *dim=1* works without
        # additional squeezing/unsqueezing steps.  We therefore add a leading
        # dimension **after** all 1-D processing is complete.
        # ------------------------------------------------------------------

        input_ids = input_ids_1d.unsqueeze(0)  # (1, S)
        labels = labels_1d.unsqueeze(0)  # (1, S)

        return input_ids, labels, teacher_spans, student_spans

    def _tokenize_message_part(self, text: str) -> List[int]:
        """
        Tokenize a message part without special tokens and flatten the result.

        Args:
            text: Text to tokenize

        Returns:
            Flattened list of token IDs
        """
        tokens = self.tokenizer(
            text,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]

        # Handle nested or flat lists
        return self._flatten_tokenizer_output(tokens)

    @typechecked
    def _mask_non_assistant_tokens(
        self,
        labels: TensorType["S"],
        conversation: List[ChatMessage],
        formatted_text: str,
    ) -> Tuple[TensorType["S"], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Return a version of ``labels`` where only tokens belonging to **assistant**
        messages remain; all others are replaced by ``-100`` so they do not
        contribute to the LM loss.

        This includes assistant messages coming from *teacher* examples **and** the
        final student answer.  The algorithm:

        1. Initialise a full mask (`-100`) the same shape as ``labels``.
        2. Iterate over the conversation sequentially.  For every assistant
           message, locate its byte-offset in ``formatted_text`` *after* the last
           match to avoid duplicates.
        3. Re-tokenise the prefix and the assistant content itself (with
           ``add_special_tokens=False``) to compute the span boundaries in token
           space.
        4. Copy the original token IDs from the untouched ``labels`` tensor back
           into the masked tensor for that assistant span.
        5. Advance the search cursor and continue until all assistant messages
           are processed.

        Returns:
            labels: Masked labels tensor
            teacher_spans: List of (start_idx, end_idx) for teacher assistant messages
            student_spans: List of (start_idx, end_idx) for student assistant message
        """

        # Preserve originals to restore assistant spans later
        original_ids = labels.clone()

        # Mask everything
        labels[:] = -100

        # We iteratively rebuild the template token-by-token, thereby knowing
        # the *exact* start/end token indices of every assistant span without
        # performing substring searches over ``formatted_text``.  This makes
        # the complexity strictly O(#tokens) instead of O(#messages × len(text)).

        token_offset: int = 0

        # Track teacher vs student spans for loss splitting
        teacher_spans: List[Tuple[int, int]] = []
        student_spans: List[Tuple[int, int]] = []

        # Count total assistant messages to identify teachers vs student
        assistant_messages = [msg for msg in conversation if msg.role == "assistant"]
        num_assistants = len(assistant_messages)
        current_assistant_idx = 0

        for msg in conversation:
            # ------------------------------------------------------------------
            # Re-tokenise *prefix*, *content* and *suffix* **exactly** as they
            # appear inside the global chat template so that `token_offset`
            # remains perfectly aligned with the flattened conversation.
            # ------------------------------------------------------------------

            prefix_str = f"{self.tokens.IM_START}{msg.role}\n"
            # The HF chat template appends a *newline* after <|im_end|> for every
            # message.  We must replicate that byte-for-byte to keep token
            # alignment in sync with the template; otherwise the running
            # offset drifts by one token per message which manifests as
            # leftover "assistant" prefixes in the label preview.
            suffix_str = f"{self.tokens.IM_END}\n"

            # Token counts ----------------------------------------------------
            prefix_tokens = self._tokenize_message_part(prefix_str)
            content_tokens = self._tokenize_message_part(msg.content)
            suffix_tokens = self._tokenize_message_part(suffix_str)

            # Un-mask assistant *content* (+ optional suffix) -----------------
            if msg.role == "assistant" and content_tokens:
                current_assistant_idx += 1

                start_idx = token_offset + len(prefix_tokens)
                end_idx = start_idx + len(content_tokens)

                labels[start_idx:end_idx] = original_ids[start_idx:end_idx]

                # STRICT VALIDATION: Check for coordinate tokens in assistant messages
                if (
                    hasattr(self, "coordinate_manager")
                    and self.coordinate_manager
                    and self.coordinate_manager.config.enable_coordinate_tokens
                ):
                    assistant_tokens = labels[start_idx:end_idx]
                    coord_token_count = 0
                    for token in assistant_tokens:
                        if (
                            self.coordinate_manager.coord_start_id
                            <= token.item()
                            < self.coordinate_manager.coord_end_id
                            or token.item()
                            == self.coordinate_manager.config.box_start_id
                            or token.item() == self.coordinate_manager.config.box_end_id
                        ):
                            coord_token_count += 1

                    if coord_token_count > 0:
                        logger.debug(
                            f"   🎯 Found {coord_token_count} coordinate tokens in assistant message"
                        )
                        logger.debug(f"   Assistant span: [{start_idx}:{end_idx}]")
                        logger.debug(
                            f"   Sample coordinate tokens: {assistant_tokens[: min(10, len(assistant_tokens))].tolist()}"
                        )

                        # Verify no coordinate tokens were set to -100
                        masked_coord_tokens = []
                        for i, token in enumerate(assistant_tokens):
                            if token.item() == -100:
                                orig_token = original_ids[start_idx + i]
                                if (
                                    self.coordinate_manager.coord_start_id
                                    <= orig_token.item()
                                    < self.coordinate_manager.coord_end_id
                                    or orig_token.item()
                                    == self.coordinate_manager.config.box_start_id
                                    or orig_token.item()
                                    == self.coordinate_manager.config.box_end_id
                                ):
                                    masked_coord_tokens.append((i, orig_token.item()))

                        if masked_coord_tokens:
                            logger.error(
                                f"❌ CRITICAL: Coordinate tokens set to -100 in assistant message!"
                            )
                            logger.error(
                                f"   Masked coordinate tokens: {masked_coord_tokens}"
                            )
                            logger.error(
                                f"   This will cause coordinate loss to be zero!"
                            )
                            raise RuntimeError(
                                f"Coordinate tokens detected in assistant message but {len(masked_coord_tokens)} "
                                f"tokens were set to -100. This will cause coordinate loss computation to fail."
                            )

                # Track spans for teacher-student loss splitting
                if current_assistant_idx < num_assistants:
                    # This is a teacher (all except the last)
                    teacher_spans.append((start_idx, end_idx))
                else:
                    # This is the student (the last assistant)
                    student_spans.append((start_idx, end_idx))

                # Optionally unmask the immediate `<|im_end|>` token (first
                # token of the suffix), preserving the rest as -100 so the
                # model explicitly learns to emit the terminator but not the
                # closing `<|im_start|>` of the next turn.
                if suffix_tokens:
                    im_end_token_id = suffix_tokens[0]
                    if (
                        end_idx < labels.size(0)
                        and original_ids[end_idx] == im_end_token_id
                    ):
                        labels[end_idx] = im_end_token_id

            # Advance offset by *full* message length ------------------------
            token_offset += (
                len(prefix_tokens) + len(content_tokens) + len(suffix_tokens)
            )

        return labels, teacher_spans, student_spans

    @typechecked
    def _process_images_for_model(
        self, images: List[Image.Image]
    ) -> Tuple[
        Optional[TensorType["PT", "E"]],  # (tokens, embed_dim)
        Optional[TensorType["N_IMG", 3]],  # (num_images, 3) grid spec
    ]:
        """Process images for model input using official approach with data_conversion pixel settings."""
        if not images:
            return None, None

        # Log original image sizes for debugging
        logger.debug(f"🖼️ PROCESSING {len(images)} IMAGES:")
        for i, img in enumerate(images):
            logger.debug(f"   Image {i}: {img.size} (W*H), mode={img.mode}")

        # Use processor directly like official QwenVL implementation
        # The processor is already configured with data_conversion/vision_process.py values
        processed = self.image_processor.preprocess(images, return_tensors="pt")

        pixel_values = processed["pixel_values"]
        image_grid_thw = processed.get("image_grid_thw")

        # CRITICAL: Log the vision token analysis
        logger.debug(f"🚨 VISION TOKEN ANALYSIS:")
        logger.debug(f"   Input images: {len(images)} images")
        logger.debug(f"   Original sizes: {[img.size for img in images]}")
        logger.debug(f"   pixel_values shape: {pixel_values.shape}")
        logger.debug(f"   Vision tokens generated (pre-merge): {pixel_values.shape[0]}")

        if image_grid_thw is not None:
            logger.debug(f"   image_grid_thw shape: {image_grid_thw.shape}")
            logger.debug(f"   image_grid_thw values: {image_grid_thw.tolist()}")

            # Calculate both pre-merge and post-merge token counts for clarity
            # EXPLICIT: Get merge_size from image processor - no defaults
            if hasattr(self.image_processor, "merge_size"):
                merge_size = self.image_processor.merge_size
            else:
                # Use Qwen2.5-VL default
                merge_size = 2
            merge_length = merge_size**2

            total_pre_merge = 0
            total_post_merge = 0

            for i, grid in enumerate(image_grid_thw):
                t, h, w = grid.tolist()
                pre_merge_tokens = t * h * w
                post_merge_tokens = pre_merge_tokens // merge_length
                total_pre_merge += pre_merge_tokens
                total_post_merge += post_merge_tokens

                logger.debug(
                    f"   Image {i}: grid=({t},{h},{w}) → {pre_merge_tokens} pre-merge → {post_merge_tokens} final tokens"
                )

            logger.debug(
                f"   TOTAL: {total_pre_merge} pre-merge → {total_post_merge} final tokens (merge_size={merge_size}²={merge_length})"
            )
        else:
            logger.debug(f"   No image_grid_thw available")

        # CRITICAL: Ensure bf16 precision for pixel_values
        if pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)
            logger.debug(
                f"🔧 Converted pixel_values from {processed['pixel_values'].dtype} to bf16: {pixel_values.dtype}"
            )
        else:
            logger.debug(f"✅ pixel_values already in bf16: {pixel_values.dtype}")

        # Debug: Log the shapes to understand the processing
        logger.debug(f"🖼️ OFFICIAL IMAGE PROCESSING:")
        logger.debug(f"   Number of images: {len(images)}")
        logger.debug(f"   pixel_values shape: {pixel_values.shape}")
        logger.debug(
            f"   image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else None}"
        )

        return pixel_values, image_grid_thw

    @typechecked
    def prepare_inputs_for_inference(
        self, images: List[Image.Image], text: str, is_first_step: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for inference following the official Qwen2.5-VL pattern.

        Args:
            images: List of PIL images
            text: Input text prompt
            is_first_step: Whether this is the first generation step (prefill)

        Returns:
            Dict containing properly formatted inputs for model.generate()
        """
        # Tokenize text without immediate tensor conversion for robustness
        raw_text_tokens = self.tokenizer(
            text, padding=False, truncation=False, add_special_tokens=False
        )

        # Debug: Log the tokenizer output structure
        logger.debug(f"📝 Tokenizer output structure:")
        logger.debug(f"   Type of raw_text_tokens: {type(raw_text_tokens)}")
        logger.debug(
            f"   Keys: {raw_text_tokens.keys() if hasattr(raw_text_tokens, 'keys') else 'N/A'}"
        )

        # Flatten tokenizer output using our helper
        flat_text_ids = self._flatten_tokenizer_output(raw_text_tokens["input_ids"])

        # Convert to tensor (batch dimension of 1)
        text_input_ids = torch.tensor(flat_text_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(text_input_ids, dtype=torch.bool)

        text_inputs = {"input_ids": text_input_ids, "attention_mask": attention_mask}

        # Handle vision inputs based on generation step
        if is_first_step and images:
            # First step: process images normally
            pixel_values, image_grid_thw = self._process_images_for_model(images)

            if pixel_values is not None:
                text_inputs["pixel_values"] = pixel_values

            if image_grid_thw is not None:
                text_inputs["image_grid_thw"] = image_grid_thw

            logger.debug(f"🔥 FIRST STEP: Added vision inputs")
            logger.debug(
                f"   pixel_values shape: {pixel_values.shape if pixel_values is not None else None}"
            )
            logger.debug(
                f"   image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else None}"
            )
        else:
            # Subsequent steps: don't add vision inputs at all
            # The model's prepare_inputs_for_generation will handle this
            logger.debug(f"🔥 SUBSEQUENT STEP: No vision inputs added")

        # Filter to only include valid generation parameters
        valid_generation_params = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        }

        filtered_inputs = {
            key: value
            for key, value in text_inputs.items()
            if key in valid_generation_params and value is not None
        }

        logger.debug(f"🔧 FILTERED INFERENCE INPUTS: {list(filtered_inputs.keys())}")

        return filtered_inputs

    # ------------------------------------------------------------------
    # Backwards-compat helpers expected by Dataset / Inference / Trainer
    # ------------------------------------------------------------------

    def set_context(self, context: str = "training") -> None:
        """No-op context setter kept for external compatibility."""
        self._current_context = context

    def get_current_system_prompt(self) -> str:
        """Return the system prompt currently in use (helper for logging)."""
        return self.system_prompt

    def _validate_coordinate_manager(self, for_operation: str = "general") -> None:
        """
        Validate that coordinate manager is properly initialized and configured.

        Args:
            for_operation: Description of operation requiring validation (for error messages)

        Raises:
            RuntimeError: If coordinate manager is not properly initialized or configured
        """
        # Check if coordinate manager exists
        if not self.coordinate_manager:
            raise RuntimeError(
                f"Coordinate manager not initialized for {for_operation}"
            )

        # Check token ranges are properly set if tokens should be available
        if self.coordinate_manager.has_coordinate_tokens():
            if (
                self.coordinate_manager.coord_start_id is None
                or self.coordinate_manager.coord_end_id is None
            ):
                raise RuntimeError(
                    f"Coordinate token ranges not properly initialized: "
                    f"start_id={self.coordinate_manager.coord_start_id}, "
                    f"end_id={self.coordinate_manager.coord_end_id}"
                )

            # Check range size matches configuration
            token_range_size = (
                self.coordinate_manager.coord_end_id
                - self.coordinate_manager.coord_start_id
            )
            if token_range_size != self.coordinate_manager.max_coord_value:
                logger.warning(
                    f"⚠️ Token range size ({token_range_size}) != max_coord_value "
                    f"({self.coordinate_manager.max_coord_value}). This may cause issues."
                )

        # Validate max_coord_value is reasonable
        max_coord = self.coordinate_manager.max_coord_value
        if max_coord <= 0:
            raise RuntimeError(
                f"Invalid max_coord_value: {max_coord}. Must be positive."
            )
        if max_coord > 10000:
            logger.warning(
                f"⚠️ Large max_coord_value detected: {max_coord}. "
                f"This will create {max_coord} coordinate tokens in vocabulary."
            )

    def _flatten_tokenizer_output(self, token_ids) -> List[int]:
        """
        Flatten tokenizer output to a simple list of token ids.
        Handles both nested lists [[int, int, ...]] and flat lists [int, int, ...].

        Args:
            token_ids: Token IDs from tokenizer, either as nested or flat list

        Returns:
            Flattened list of token IDs

        Raises:
            ValueError: If token_ids contains None values or is empty
        """
        if not token_ids:
            raise ValueError("Tokenizer returned empty token_ids list")

        # Handle nested list case
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            flat_ids = token_ids[0]
        else:
            # Already flat list case
            flat_ids = token_ids

        # Fail-fast if any element is None
        if any(tok is None for tok in flat_ids):
            raise ValueError(
                "Tokenizer produced `None` token IDs – check that all special tokens are "
                "present in the tokenizer vocabulary."
            )

        return flat_ids


def create_chat_processor(
    tokenizer,
    image_processor,
    data_root: str = "./",
    model_max_length: int = 8192,
) -> ChatProcessor:
    """
    Factory function to create ChatProcessor.

    Args:
        tokenizer: Qwen2.5-VL tokenizer
        image_processor: Qwen2.5-VL image processor
        data_root: Root directory for image paths
        model_max_length: Maximum sequence length

    Returns:
        Configured ChatProcessor instance
    """
    return ChatProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_root=data_root,
        model_max_length=model_max_length,
    )
