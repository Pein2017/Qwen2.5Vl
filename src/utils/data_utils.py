"""
Data Processing Utilities

This module contains utilities for data processing, parsing, and schema definitions
related to data handling in the BBU training pipeline.
"""

import json
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from src.logger_utils import get_logger


logger = get_logger("data_utils")

# Constants
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"

# Few-shot section constants (for compatibility with legacy imports)
ENGLISH_FEW_SHOT_SECTION = ""  # no built-in examples
CHINESE_FEW_SHOT_SECTION = ""  # 暂无内置示例

# Optional imports for semantic similarity
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Using rule-based similarity.")

# ==============================================================================
# Data Schema Definitions
# ==============================================================================


@dataclass
class ChatMessage:  # noqa: D401 – basic conversation unit
    """A single conversation message with role and content."""

    role: str  # 'user', 'assistant', 'system'
    content: str

    def __post_init__(self):
        if self.role not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {self.role}")

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ChatMessage":
        return cls(role=data["role"], content=data["content"])


@dataclass
class ImageSample:  # noqa: D401 – one image + objects pair (teacher OR student)
    """Single image with associated object annotations."""

    image_path: str
    objects: List[Dict[str, Any]]  # List of bbox/description pairs
    conversation: List[ChatMessage]

    def __post_init__(self):
        if not self.image_path:
            raise ValueError("image_path cannot be empty")
        if not isinstance(self.objects, list):
            raise ValueError("objects must be a list")


@dataclass
class MultiChatSample:  # noqa: D401 – full teacher/student bundle
    """Complete training sample with teacher examples and student query."""

    student: ImageSample
    teachers: List[ImageSample]  # Can be empty

    def __post_init__(self):
        if not isinstance(self.student, ImageSample):
            raise ValueError("student must be an ImageSample")
        if not isinstance(self.teachers, list):
            raise ValueError("teachers must be a list")


@dataclass
class GroundTruthObject:  # noqa: D401 – simple container
    """Ground truth object with bbox and description."""

    bbox: List[float]  # [x1, y1, x2, y2] in normalized coordinates
    description: str
    category: Optional[str] = None
    geometry_type: str = "bbox"  # bbox, square, line

    def __post_init__(self):
        # Validate coordinates based on geometry type
        if self.geometry_type in ["bbox", "bbox_2d"] and len(self.bbox) != 4:
            raise ValueError(f"bbox_2d must have 4 coordinates, got {len(self.bbox)}")
        elif self.geometry_type == "square" and len(self.bbox) != 8:
            raise ValueError(f"square must have 8 coordinates, got {len(self.bbox)}")
        elif self.geometry_type == "line" and (
            len(self.bbox) < 4 or len(self.bbox) % 2 != 0
        ):
            raise ValueError(
                f"line must have even number of coordinates (>=4), got {len(self.bbox)}"
            )

        # Validate description
        if not self.description.strip():
            raise ValueError("description cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "bbox": self.bbox,
            "description": self.description,
            "geometry_type": self.geometry_type,
        }
        if self.category:
            result["category"] = self.category
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthObject":
        return cls(
            bbox=data["bbox"],
            description=data["description"],
            category=data.get("category"),
            geometry_type=data.get("geometry_type", "bbox"),
        )

    def normalize_coordinates(
        self, image_width: int, image_height: int
    ) -> "GroundTruthObject":
        """Normalize absolute coordinates to [0, 1] range."""
        if self.geometry_type in ["bbox", "square"]:
            x1, y1, x2, y2 = self.bbox
            normalized_bbox = [
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height,
            ]
        elif self.geometry_type == "line":
            x1, y1, x2, y2 = self.bbox
            normalized_bbox = [
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height,
            ]
        else:
            normalized_bbox = self.bbox

        return GroundTruthObject(
            bbox=normalized_bbox,
            description=self.description,
            category=self.category,
            geometry_type=self.geometry_type,
        )


# ==============================================================================
# Data Loading and Formatting Functions
# ==============================================================================


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def format_object_description(obj: Dict[str, Any]) -> str:
    """Format object dictionary into description string."""
    # EXPLICIT: Check for required keys without fallbacks
    if "bbox" not in obj:
        raise ValueError("Object missing required 'bbox' key")
    if "description" not in obj:
        raise ValueError("Object missing required 'description' key")

    bbox = obj["bbox"]
    description = obj["description"]

    # Use simple JSON format without special tokens
    return json.dumps({"bbox": bbox, "description": description}, ensure_ascii=False)


def format_single_round_conversation(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Format single round conversation data."""
    # User message with image placeholder
    user_content = f"{DEFAULT_IMAGE_TOKEN}\n描述图像中的BBU设备。"

    # EXPLICIT: Check for objects without fallback
    if "objects" not in data:
        raise ValueError("Data missing required 'objects' key")

    objects = data["objects"]
    if not isinstance(objects, list):
        raise ValueError("Objects must be a list")

    # Format objects as JSON array
    formatted_objects = []
    for obj in objects:
        formatted_objects.append(
            {"bbox": obj["bbox"], "description": obj["description"]}
        )

    assistant_content = json.dumps(formatted_objects, ensure_ascii=False)

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def format_multi_round_conversation(data: Dict[str, Any]) -> str:
    """Format multi-round conversation with teacher examples."""
    conversation_parts = []

    # EXPLICIT: Check for teachers/examples without nested fallbacks
    teachers = []
    if "teachers" in data:
        teachers = data["teachers"]
    elif "examples" in data:
        teachers = data["examples"]
    # If neither exists, teachers remains empty list

    # Add teacher examples
    for teacher in teachers:
        # EXPLICIT: Check for objects in teacher
        if "objects" not in teacher:
            continue

        teacher_objects = teacher["objects"]
        if not teacher_objects:
            continue

        # Format as JSON array
        formatted_teacher_objects = json.dumps(teacher_objects, ensure_ascii=False)
        conversation_parts.append(f"示例：{formatted_teacher_objects}")

    # Add student query
    student_query = f"{DEFAULT_IMAGE_TOKEN}\n描述图像中的BBU设备。"
    if conversation_parts:
        conversation_parts.append(f"问题：{student_query}")
        return "\n\n".join(conversation_parts)
    else:
        return student_query


def format_conversation(
    data: Dict[str, Any],
    format_type: str = "single",
    include_system_prompt: bool = True,
) -> Union[List[Dict[str, str]], str]:
    """Format conversation data based on type."""
    if format_type == "single":
        return format_single_round_conversation(data)
    elif format_type == "multi":
        return format_multi_round_conversation(data)
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


# ==============================================================================
# Response Parser Class
# ==============================================================================


class ResponseParser:
    """
    Parser for Qwen2.5-VL responses with multiple format support.

    Optimized for clean JSON format:
    [{"bbox":[x1,y1,x2,y2],"description":"description"}]

    Also supports legacy formats and handles incomplete/truncated responses.
    """

    def __init__(self) -> None:
        # Initialize parser logger
        self.parser_logger = get_logger("response_parser")
        self._init_sentence_transformer()

    def _init_sentence_transformer(self):
        """Initialize SentenceTransformer for semantic similarity."""
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                # Try local cache first, fallback to downloading
                model_path = "model_cache/sentence-transformers/all-MiniLM-L6-v2/"
                try:
                    self.sentence_transformer = SentenceTransformer(model_path)
                except Exception:
                    # Fallback to downloading the model
                    self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

                self.parser_logger.info("SentenceTransformer initialized successfully")
            except Exception as e:
                self.parser_logger.warning(
                    f"Failed to initialize SentenceTransformer: {e}"
                )
                self.sentence_transformer = None
        else:
            self.sentence_transformer = None

    def parse_response(
        self,
        response_text: str,
        ground_truth_objects: Optional[List[Dict[str, Any]]] = None,
        similarity_threshold: float = 0.7,
        validate_coordinates: bool = True,
        allow_empty_descriptions: bool = False,
        expected_object_count: Optional[int] = None,
        coordinate_threshold: float = 0.05,
        coordinate_scale: str = "normalized",
        geometry_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse response text and extract object information.

        Args:
            response_text: Raw model response text
            ground_truth_objects: Optional ground truth for validation
            similarity_threshold: Minimum similarity for description matching
            validate_coordinates: Whether to validate coordinate ranges
            allow_empty_descriptions: Whether to allow empty descriptions
            expected_object_count: Expected number of objects (for validation)
            coordinate_threshold: Threshold for coordinate validation
            coordinate_scale: "normalized" or "absolute"
            geometry_types: List of allowed geometry types

        Returns:
            List of parsed objects with bbox and description
        """
        if not response_text or not response_text.strip():
            self.parser_logger.warning("Empty response text provided")
            return []

        # Clean the input text
        cleaned_text = self._clean_text(response_text)
        self.parser_logger.debug(f"Cleaned text length: {len(cleaned_text)}")

        # Try different parsing strategies in order of preference
        parsing_strategies = [
            ("clean_json", self._parse_clean_json_format),
            ("partial_json", self._parse_partial_json_format),
            ("legacy_json", self._parse_legacy_json_format),
            ("coordinate_tokens", self._parse_coordinate_tokens),
            ("special_tokens", self._parse_special_tokens),
            ("multi_geometry", self._parse_multi_geometry_tokens),
            ("unquoted", self._parse_unquoted_format),
            ("regex_extraction", self._parse_regex_extraction),
        ]

        parsed_objects = []

        for strategy_name, strategy_func in parsing_strategies:
            try:
                self.parser_logger.debug(f"Trying {strategy_name} parsing strategy")
                parsed_objects = strategy_func(cleaned_text)

                if parsed_objects:
                    _ = strategy_name
                    self.parser_logger.info(
                        f"Successfully parsed {len(parsed_objects)} objects using {strategy_name}"
                    )
                    break
                else:
                    self.parser_logger.debug(f"{strategy_name} returned no objects")

            except Exception as e:
                self.parser_logger.debug(f"{strategy_name} parsing failed: {str(e)}")
                continue

        if not parsed_objects:
            self.parser_logger.warning(
                "All parsing strategies failed, returning empty list"
            )
            return []

        # Validate and filter the parsed objects
        validated_objects = self._validate_and_filter_objects(
            parsed_objects,
            validate_coordinates=validate_coordinates,
            allow_empty_descriptions=allow_empty_descriptions,
            coordinate_threshold=coordinate_threshold,
            coordinate_scale=coordinate_scale,
            geometry_types=geometry_types,
        )

        self.parser_logger.info(
            f"Final validation: {len(validated_objects)}/{len(parsed_objects)} objects passed"
        )

        # Optional: Check against expected count
        if expected_object_count is not None:
            if len(validated_objects) != expected_object_count:
                self.parser_logger.warning(
                    f"Object count mismatch: expected {expected_object_count}, got {len(validated_objects)}"
                )

        return validated_objects

    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove common prefixes/suffixes
        text = re.sub(r"^(回答：|答案：|结果：|输出：)", "", text)
        text = re.sub(r"(谢谢！|感谢！)$", "", text)

        return text.strip()

    def _parse_clean_json_format(self, text: str) -> List[Dict[str, Any]]:
        """Parse clean JSON format: [{"bbox":[x1,y1,x2,y2],"description":"text"}]"""
        try:
            # Try direct JSON parsing first
            data = json.loads(text)
            if isinstance(data, list):
                return self._extract_valid_objects_from_list(data)
            elif isinstance(data, dict):
                return [data] if self._is_valid_object_structure(data) else []
            else:
                return []
        except json.JSONDecodeError:
            # Try to find JSON arrays in the text
            json_pattern = r"\[[\s\S]*?\]"
            matches = re.findall(json_pattern, text)

            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        objects = self._extract_valid_objects_from_list(parsed)
                        if objects:
                            return objects
                except json.JSONDecodeError:
                    continue

            return []

    def _parse_partial_json_format(self, text: str) -> List[Dict[str, Any]]:
        """Parse partially formatted JSON that might be incomplete."""
        # Attempt to repair common JSON issues
        repaired_text = self._attempt_json_repair(text)

        try:
            data = json.loads(repaired_text)
            if isinstance(data, list):
                return self._extract_valid_objects_from_list(data)
            elif isinstance(data, dict):
                return [data] if self._is_valid_object_structure(data) else []
        except json.JSONDecodeError:
            pass

        # Try to extract individual JSON objects
        objects = []
        object_pattern = r'\{[^{}]*"bbox"[^{}]*\}'
        matches = re.findall(object_pattern, text, re.IGNORECASE)

        for match in matches:
            try:
                obj = json.loads(match)
                if self._is_valid_object_structure(obj):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue

        return objects

    def _attempt_json_repair(self, text: str) -> str:
        """Attempt to repair common JSON formatting issues."""
        # Fix missing commas between objects
        text = re.sub(r"\}\s*\{", "},{", text)

        # Fix missing quotes around keys
        text = re.sub(r"(\w+):", r'"\1":', text)

        # Fix trailing commas
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Ensure proper array wrapping
        if not text.strip().startswith("[") and "{" in text:
            if text.count("{") > 1:
                text = "[" + text + "]"

        return text

    def _is_valid_object_structure(self, obj: Dict) -> bool:
        """Check if object has valid structure."""
        if not isinstance(obj, dict):
            return False

        # Must have bbox and description
        if "bbox" not in obj or "description" not in obj:
            return False

        # Bbox should be a list of numbers
        bbox = obj["bbox"]
        if not isinstance(bbox, list) or len(bbox) < 4:
            return False

        return True

    def _extract_valid_objects_from_list(self, parsed_list: List) -> List[Dict]:
        """Extract valid objects from a parsed list."""
        valid_objects = []
        for item in parsed_list:
            if self._is_valid_object_structure(item):
                valid_objects.append(item)
        return valid_objects

    def _parse_regex_extraction(self, text: str) -> List[Dict]:
        """Extract objects using regex patterns."""
        objects = []

        # Pattern for bbox coordinates and descriptions
        bbox_pattern = r'(?:bbox|坐标)["\']?\s*[:=]\s*\[([^\]]+)\].*?(?:description|描述|说明)["\']?\s*[:=]\s*["\']([^"\']+)["\']'
        matches = re.findall(bbox_pattern, text, re.IGNORECASE | re.DOTALL)

        for bbox_str, desc in matches:
            try:
                # Parse bbox coordinates
                coords = [float(x.strip()) for x in bbox_str.split(",")]
                if len(coords) >= 4:
                    objects.append({"bbox": coords[:4], "description": desc.strip()})
            except (ValueError, IndexError):
                continue

        return objects

    def _parse_legacy_json_format(self, text: str) -> List[Dict[str, Any]]:
        """Parse legacy JSON format with various structures."""
        # Legacy format might include extra fields
        try:
            data = json.loads(text)
            objects = []

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Extract bbox and description from various possible structures
                        bbox = None
                        description = None

                        # Try different key variations
                        for bbox_key in [
                            "bbox",
                            "bounding_box",
                            "coordinates",
                            "coords",
                        ]:
                            if bbox_key in item:
                                bbox = item[bbox_key]
                                break

                        for desc_key in [
                            "description",
                            "desc",
                            "text",
                            "label",
                            "caption",
                        ]:
                            if desc_key in item:
                                description = item[desc_key]
                                break

                        if bbox and description:
                            objects.append({"bbox": bbox, "description": description})

            return objects
        except json.JSONDecodeError:
            return []

    def _parse_coordinate_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Parse coordinate token format."""
        objects = []

        # Look for coordinate token patterns
        coord_pattern = r"<coord_(\d+)_(\d+)_(\d+)_(\d+)>([^<]+)"
        matches = re.findall(coord_pattern, text)

        for x1, y1, x2, y2, description in matches:
            try:
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                objects.append({"bbox": bbox, "description": description.strip()})
            except ValueError:
                continue

        # Also look for normalized coordinate tokens
        norm_pattern = r"<norm_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)>([^<]+)"
        matches = re.findall(norm_pattern, text)

        for x1, y1, x2, y2, description in matches:
            try:
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                objects.append({"bbox": bbox, "description": description.strip()})
            except ValueError:
                continue

        return objects

    def _parse_special_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Parse special token format."""
        objects = []

        # Look for special tokens like <bbox>, <square>, <line>
        special_patterns = [
            (r"<bbox>([^<]+)</bbox>([^<]+)", "bbox"),
            (r"<square>([^<]+)</square>([^<]+)", "square"),
            (r"<line>([^<]+)</line>([^<]+)", "line"),
        ]

        for pattern, geometry_type in special_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for coords_str, description in matches:
                try:
                    coords = [float(x.strip()) for x in coords_str.split(",")]
                    if len(coords) >= 4:
                        objects.append(
                            {
                                "bbox": coords[:4],
                                "description": description.strip(),
                                "geometry_type": geometry_type,
                            }
                        )
                except ValueError:
                    continue

        return objects

    def _parse_unquoted_format(self, text: str) -> List[Dict[str, Any]]:
        """Parse unquoted format that might not have proper JSON structure."""
        objects = []

        # Look for patterns like: bbox: [1,2,3,4] description: some text
        pattern = r"bbox\s*:\s*\[([^\]]+)\]\s*description\s*:\s*([^\n]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for coords_str, description in matches:
            try:
                coords = [float(x.strip()) for x in coords_str.split(",")]
                if len(coords) >= 4:
                    objects.append(
                        {"bbox": coords[:4], "description": description.strip()}
                    )
            except ValueError:
                continue

        return objects

    def _parse_multi_geometry_tokens(
        self, text: str, geometry_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Parse multi-geometry token format supporting bbox, square, and line."""
        if geometry_types is None:
            geometry_types = ["bbox", "square", "line"]

        objects = []

        for geometry_type in geometry_types:
            if geometry_type == "bbox":
                pattern = r"<bbox>([^<]+)</bbox>\s*([^<\n]+)"
            elif geometry_type == "square":
                pattern = r"<square>([^<]+)</square>\s*([^<\n]+)"
            elif geometry_type == "line":
                pattern = r"<line>([^<]+)</line>\s*([^<\n]+)"
            else:
                continue

            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)

            for coords_str, description in matches:
                try:
                    coords = self._parse_geometry_content(coords_str, geometry_type)
                    if coords and len(coords) >= 4:
                        objects.append(
                            {
                                "bbox": coords[:4],
                                "description": description.strip(),
                                "geometry_type": geometry_type,
                            }
                        )
                except Exception as e:
                    self.parser_logger.debug(
                        f"Failed to parse {geometry_type} coordinates: {e}"
                    )
                    continue

        return objects

    def _parse_geometry_content(
        self, coords_str: str, geometry_type: str
    ) -> Optional[List[float]]:
        """Parse coordinate content for different geometry types."""
        try:
            # Handle different coordinate formats
            coords_str = coords_str.strip()

            # Remove any surrounding brackets or parentheses
            coords_str = re.sub(r"^[\[\(]|[\]\)]$", "", coords_str)

            # Split by comma or space and convert to float
            coord_parts = re.split(r"[,\s]+", coords_str)
            coords = []

            for part in coord_parts:
                part = part.strip()
                if part:
                    try:
                        coords.append(float(part))
                    except ValueError:
                        continue

            # Validate coordinate count based on geometry type
            if geometry_type in ["bbox", "square", "line"] and len(coords) >= 4:
                return coords[:4]
            else:
                return None

        except Exception:
            return None

    def _parse_multi_geometry_tokens_alternative(
        self, text: str, geometry_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Alternative multi-geometry parsing with more flexible patterns."""
        if geometry_types is None:
            geometry_types = ["bbox", "square", "line"]

        objects = []

        # More flexible pattern that captures various formats
        for geometry_type in geometry_types:
            patterns = [
                rf"<{geometry_type}>([^<]+)</{geometry_type}>\s*([^<\n]+)",
                rf"<{geometry_type}>\s*([^<]+)\s*</{geometry_type}>\s*([^<\n]+)",
                rf"<{geometry_type}:([^>]+)>\s*([^<\n]+)",
                rf"{geometry_type}\s*:\s*\[([^\]]+)\]\s*([^<\n]+)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)

                for coords_str, description in matches:
                    try:
                        coords = self._parse_geometry_content(coords_str, geometry_type)
                        if coords and len(coords) >= 4:
                            objects.append(
                                {
                                    "bbox": coords[:4],
                                    "description": description.strip(),
                                    "geometry_type": geometry_type,
                                }
                            )
                    except Exception:
                        continue

        return objects

    def _validate_and_filter_objects(
        self,
        objects: List[Dict[str, Any]],
        validate_coordinates: bool = True,
        allow_empty_descriptions: bool = False,
        coordinate_threshold: float = 0.05,
        coordinate_scale: str = "normalized",
        geometry_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Validate and filter parsed objects."""
        if geometry_types is None:
            geometry_types = ["bbox", "square", "line"]

        validated_objects = []

        for obj in objects:
            try:
                # Get geometry type
                geometry_type = obj.get("geometry_type", "bbox")

                # Skip if geometry type not allowed
                if geometry_type not in geometry_types:
                    self.parser_logger.debug(
                        f"Skipping object with unsupported geometry type: {geometry_type}"
                    )
                    continue

                # Validate based on geometry type
                if geometry_type == "bbox":
                    if self._validate_bbox_object(
                        obj,
                        validate_coordinates,
                        allow_empty_descriptions,
                        coordinate_threshold,
                        coordinate_scale,
                    ):
                        validated_objects.append(obj)
                elif geometry_type == "square":
                    if self._validate_square_object(
                        obj,
                        validate_coordinates,
                        allow_empty_descriptions,
                        coordinate_threshold,
                        coordinate_scale,
                    ):
                        validated_objects.append(obj)
                elif geometry_type == "line":
                    if self._validate_line_object(
                        obj,
                        validate_coordinates,
                        allow_empty_descriptions,
                        coordinate_threshold,
                        coordinate_scale,
                    ):
                        validated_objects.append(obj)
                else:
                    self.parser_logger.debug(f"Unknown geometry type: {geometry_type}")

            except Exception as e:
                self.parser_logger.debug(f"Validation failed for object: {e}")
                continue

        return validated_objects

    def _validate_bbox_object(
        self,
        obj: Dict[str, Any],
        validate_coordinates: bool,
        allow_empty_descriptions: bool,
        coordinate_threshold: float,
        coordinate_scale: str,
    ) -> bool:
        """Validate bbox object."""
        # Check required fields
        if "bbox" not in obj or "description" not in obj:
            return False

        bbox = obj["bbox"]
        description = obj["description"]

        # Validate bbox structure
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False

        try:
            x1, y1, x2, y2 = [float(coord) for coord in bbox]
        except (ValueError, TypeError):
            return False

        # Validate coordinates if requested
        if validate_coordinates:
            if coordinate_scale == "normalized":
                # Normalized coordinates should be in [0, 1]
                if not all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
                    return False
            elif coordinate_scale == "absolute":
                # Absolute coordinates should be positive
                if not all(coord >= 0 for coord in [x1, y1, x2, y2]):
                    return False

            # Check coordinate ordering and minimum size
            if x2 <= x1 or y2 <= y1:
                return False

            # Check minimum object size
            width = x2 - x1
            height = y2 - y1
            if width < coordinate_threshold or height < coordinate_threshold:
                return False

        # Validate description
        if not allow_empty_descriptions and not description.strip():
            return False

        return True

    def _validate_square_object(
        self,
        obj: Dict[str, Any],
        validate_coordinates: bool,
        allow_empty_descriptions: bool,
        coordinate_threshold: float,
        coordinate_scale: str,
    ) -> bool:
        """Validate square object (same as bbox for now)."""
        # todo: implement square object validation, coordinates 8
        return self._validate_bbox_object(
            obj,
            validate_coordinates,
            allow_empty_descriptions,
            coordinate_threshold,
            coordinate_scale,
        )

    def _validate_line_object(
        self,
        obj: Dict[str, Any],
        validate_coordinates: bool,
        allow_empty_descriptions: bool,
        coordinate_threshold: float,
        coordinate_scale: str,
    ) -> bool:
        """Validate line object."""
        # Check required fields
        if "bbox" not in obj or "description" not in obj:
            return False

        bbox = obj["bbox"]
        description = obj["description"]

        # Validate bbox structure (line uses 4 coordinates: x1, y1, x2, y2)
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False

        try:
            x1, y1, x2, y2 = [float(coord) for coord in bbox]
        except (ValueError, TypeError):
            return False

        # Validate coordinates if requested
        if validate_coordinates:
            if coordinate_scale == "normalized":
                # Normalized coordinates should be in [0, 1]
                if not all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
                    return False
            elif coordinate_scale == "absolute":
                # Absolute coordinates should be positive
                if not all(coord >= 0 for coord in [x1, y1, x2, y2]):
                    return False

            # Check minimum line length
            line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if line_length < coordinate_threshold:
                return False

        # Validate description
        if not allow_empty_descriptions and not description.strip():
            return False

        return True

    def calculate_semantic_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate semantic similarity between two descriptions."""
        if self.sentence_transformer is None:
            # Fallback to simple string similarity
            desc1_clean = desc1.lower().strip()
            desc2_clean = desc2.lower().strip()

            if desc1_clean == desc2_clean:
                return 1.0

            # Simple word overlap similarity
            words1 = set(desc1_clean.split())
            words2 = set(desc2_clean.split())

            if not words1 or not words2:
                return 0.0

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            return intersection / union if union > 0 else 0.0

        try:
            embeddings = self.sentence_transformer.encode([desc1, desc2])
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0),
            ).item()
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            self.parser_logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0


# ==============================================================================
# Prompt Templates and Functions
# ==============================================================================

# System prompts
CHINESE_TRAINING_PROMPT = """你是一个专业的BBU设备检测助手。请根据图像识别并描述其中的BBU设备，以JSON格式返回结果。

格式要求：
[{"bbox":[x1,y1,x2,y2],"description":"设备描述"}]

其中：
- bbox为归一化坐标，取值范围[0,1]
- description为中文描述，简洁准确
- 支持多个设备检测

示例：
[{"bbox":[0.1,0.2,0.8,0.9],"description":"蓝色BBU主设备，正面视角"}]"""

CHINESE_EVALUATION_PROMPT = """请仔细观察图像中的BBU设备，准确识别其位置并提供详细描述。

要求：
1. 使用归一化坐标[0,1]
2. 提供准确的中文描述
3. 以JSON数组格式返回
4. 格式：[{"bbox":[x1,y1,x2,y2],"description":"描述"}]"""


def get_system_prompt(
    language: str = "chinese",
    task_type: str = "training",
    include_examples: bool = True,
    geometry_types: Optional[List[str]] = None,
) -> str:
    """Get system prompt based on language and task type."""
    if geometry_types is None:
        geometry_types = ["bbox"]

    if language.lower() == "chinese":
        if task_type == "training":
            base_prompt = CHINESE_TRAINING_PROMPT
        else:
            base_prompt = CHINESE_EVALUATION_PROMPT
    else:
        # English prompts
        base_prompt = """You are a professional BBU equipment detection assistant. Please identify and describe BBU equipment in the image, returning results in JSON format.

Format requirement:
[{"bbox":[x1,y1,x2,y2],"description":"equipment description"}]

Where:
- bbox uses normalized coordinates in range [0,1]
- description should be concise and accurate
- supports multiple equipment detection"""

    # Add geometry type instructions if needed
    if len(geometry_types) > 1:
        geometry_instruction = (
            f"\n\nSupported geometry types: {', '.join(geometry_types)}"
        )
        base_prompt += geometry_instruction

    return base_prompt


def get_user_prompt_prefix(
    language: str = "chinese", include_image_token: bool = True
) -> str:
    """Get user prompt prefix."""
    prefix = ""
    if include_image_token:
        prefix += DEFAULT_IMAGE_TOKEN + "\n"

    if language.lower() == "chinese":
        prefix += "描述图像中的BBU设备。"
    else:
        prefix += "Describe the BBU equipment in the image."

    return prefix


def get_learning_instruction(
    language: str = "chinese", examples: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Get learning instruction with optional examples."""
    if language.lower() == "chinese":
        instruction = "学习以下示例，然后回答问题：\n\n"
    else:
        instruction = "Learn from the following examples, then answer the question:\n\n"

    if examples:
        for i, example in enumerate(examples, 1):
            if language.lower() == "chinese":
                instruction += f"示例{i}：{json.dumps(example, ensure_ascii=False)}\n"
            else:
                instruction += (
                    f"Example {i}: {json.dumps(example, ensure_ascii=False)}\n"
                )
        instruction += "\n"

    return instruction


def format_few_shot_prompt(
    examples: List[Dict[str, Any]], query: str, language: str = "chinese"
) -> str:
    """Format few-shot prompt with examples and query."""
    prompt_parts = []

    # Add examples
    for i, example in enumerate(examples, 1):
        if language.lower() == "chinese":
            prompt_parts.append(f"示例{i}：{json.dumps(example, ensure_ascii=False)}")
        else:
            prompt_parts.append(
                f"Example {i}: {json.dumps(example, ensure_ascii=False)}"
            )

    # Add query
    if language.lower() == "chinese":
        prompt_parts.append(f"问题：{query}")
    else:
        prompt_parts.append(f"Question: {query}")

    return "\n\n".join(prompt_parts)


def validate_prompt_language(language: str) -> str:
    """Validate and normalize language specification."""
    language = language.lower().strip()
    if language in ["zh", "chinese", "中文"]:
        return "chinese"
    elif language in ["en", "english", "英文"]:
        return "english"
    else:
        return "chinese"  # Default to Chinese


def get_optimized_prompt_for_context(
    context_length: int, max_examples: int = 5, language: str = "chinese"
) -> str:
    """Get optimized prompt based on available context length."""
    base_prompt = get_system_prompt(language=language)

    # Estimate token usage (rough approximation)
    base_tokens = len(base_prompt.split()) * 1.3  # Account for tokenization

    available_tokens = context_length - base_tokens - 100  # Reserve for response

    if available_tokens < 200:
        # Minimal prompt for very short context
        if language.lower() == "chinese":
            return '识别BBU设备，返回JSON格式：[{"bbox":[x1,y1,x2,y2],"description":"描述"}]'
        else:
            return 'Identify BBU equipment, return JSON: [{"bbox":[x1,y1,x2,y2],"description":"description"}]'
    else:
        return base_prompt
