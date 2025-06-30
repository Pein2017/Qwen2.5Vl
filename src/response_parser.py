import json
import re
import warnings
from typing import Any, Dict, List

import torch
from sentence_transformers import SentenceTransformer

from src.logger_utils import get_logger

logger = get_logger("response_parser")


# Optional imports for semantic similarity
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Using rule-based similarity.")


class ResponseParser:
    """
    Parser for Qwen2.5-VL responses with multiple format support.

    Optimized for clean JSON format:
    [{"bbox":[x1,y1,x2,y2],"description":"description"}]

    Also supports legacy formats and handles incomplete/truncated responses.
    """

    def __init__(self) -> None:
        # Initialize parser logger
        from src.logger_utils import get_logger

        self.parser_logger = get_logger("response_parser")
        self._init_sentence_transformer()

    def _init_sentence_transformer(self):
        """Initialize SentenceTransformer for semantic similarity."""
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer(
                    "/data4/swift/model_cache/sentence-transformers/all-MiniLM-L6-v2/"
                )
                logger.info("✅ SentenceTransformer loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load SentenceTransformer: {e}")
                self.sentence_transformer = None
        else:
            self.sentence_transformer = None

    def parse_response(
        self, response_text: str, sample_index: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Parse response text into list of bbox objects.

        Prioritizes clean JSON format and gracefully handles incomplete responses.

        Args:
            response_text: Raw model response
            sample_index: Sample index for logging (optional)

        Returns:
            List of dictionaries with 'bbox' and 'description' keys
        """
        # Log raw prediction text
        sample_prefix = f"Sample {sample_index}: " if sample_index >= 0 else ""
        self.parser_logger.info(f"🔍 {sample_prefix}RAW PREDICTION TEXT:")
        self.parser_logger.info(f"   Length: {len(response_text)} chars")
        self.parser_logger.info(
            f"   Content: {repr(response_text[:200])}{'...' if len(response_text) > 200 else ''}"
        )

        if not response_text or not response_text.strip():
            self.parser_logger.warning(
                f"⚠️ {sample_prefix}Empty or whitespace-only response"
            )
            return []

        text = self._clean_text(response_text)
        self.parser_logger.debug(
            f"🧹 {sample_prefix}Cleaned text: {repr(text[:200])}{'...' if len(text) > 200 else ''}"
        )

        # Quick pattern analysis for debugging
        has_brackets = "[" in text and "]" in text
        has_braces = "{" in text and "}" in text
        has_special_tokens = "<|" in text and "|>" in text
        has_numbers = any(c.isdigit() for c in text)
        self.parser_logger.debug(
            f"🔍 {sample_prefix}Text patterns: brackets={has_brackets}, braces={has_braces}, special_tokens={has_special_tokens}, numbers={has_numbers}"
        )

        # Try parsing methods in order of preference (JSON first for new format)
        objects = None
        parsing_method = "none"
        parsing_attempts = []

        # Try clean JSON format first
        objects = self._parse_clean_json_format(text)
        if objects:
            parsing_method = "clean_json"
            parsing_attempts.append(f"clean_json: SUCCESS ({len(objects)} objects)")
        else:
            parsing_attempts.append("clean_json: FAILED")
            # Try partial JSON parsing for incomplete responses
            objects = self._parse_partial_json_format(text)
            if objects:
                parsing_method = "partial_json"
                parsing_attempts.append(
                    f"partial_json: SUCCESS ({len(objects)} objects)"
                )
            else:
                parsing_attempts.append("partial_json: FAILED")
                # Try legacy JSON format
                objects = self._parse_legacy_json_format(text)
                if objects:
                    parsing_method = "legacy_json"
                    parsing_attempts.append(
                        f"legacy_json: SUCCESS ({len(objects)} objects)"
                    )
                else:
                    parsing_attempts.append("legacy_json: FAILED")
                    # Try special tokens format
                    objects = self._parse_special_tokens(text)
                    if objects:
                        parsing_method = "special_tokens"
                        parsing_attempts.append(
                            f"special_tokens: SUCCESS ({len(objects)} objects)"
                        )
                    else:
                        parsing_attempts.append("special_tokens: FAILED")
                        # Try unquoted format
                        objects = self._parse_unquoted_format(text)
                        if objects:
                            parsing_method = "unquoted"
                            parsing_attempts.append(
                                f"unquoted: SUCCESS ({len(objects)} objects)"
                            )
                        else:
                            parsing_attempts.append("unquoted: FAILED")
                            # Try regex-based extraction as last resort
                            objects = self._parse_regex_extraction(text)
                            if objects:
                                parsing_method = "regex_extraction"
                                parsing_attempts.append(
                                    f"regex_extraction: SUCCESS ({len(objects)} objects)"
                                )
                            else:
                                parsing_attempts.append("regex_extraction: FAILED")
                                objects = []
                                parsing_method = "failed"

        self.parser_logger.info(f"📊 {sample_prefix}Parsing method: {parsing_method}")
        self.parser_logger.info(
            f"📊 {sample_prefix}Raw parsed objects: {len(objects)} items"
        )

        if objects:
            self.parser_logger.debug(f"📋 {sample_prefix}Raw objects: {objects}")

        validated_objects = self._validate_and_filter_objects(objects)
        self.parser_logger.info(
            f"✅ {sample_prefix}Final validated objects: {len(validated_objects)} items"
        )

        if validated_objects:
            self.parser_logger.info(
                f"📋 {sample_prefix}Final objects: {validated_objects}"
            )
        else:
            # Enhanced debugging when validation fails
            self.parser_logger.warning(
                f"⚠️ {sample_prefix}No valid objects after validation"
            )
            self.parser_logger.error(
                f"🚨 {sample_prefix}VALIDATION FAILURE DEBUG INFO:"
            )
            self.parser_logger.error(
                f"   📝 RAW PREDICTION TEXT (full): {repr(response_text)}"
            )
            self.parser_logger.error(f"   🧹 CLEANED TEXT: {repr(text)}")
            self.parser_logger.error(f"   🔍 PARSING METHOD USED: {parsing_method}")
            self.parser_logger.error(
                f"   🔄 ALL PARSING ATTEMPTS: {'; '.join(parsing_attempts)}"
            )
            self.parser_logger.error(f"   📊 RAW PARSED OBJECTS COUNT: {len(objects)}")
            if objects:
                self.parser_logger.error(f"   📋 RAW PARSED OBJECTS: {objects}")
            else:
                self.parser_logger.error(f"   ❌ NO OBJECTS WERE PARSED FROM TEXT")

        return validated_objects

    def _clean_text(self, text: str) -> str:
        """Clean and normalize response text."""
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```") and text.endswith("```"):
            lines = text.split("\n")
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])

        return text

    def _parse_clean_json_format(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse the new clean JSON format with robust error handling.

        Expected format: [{"bbox":[x1,y1,x2,y2],"description":"description"}]
        Skips individual items that can't be parsed.
        """
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                self.parser_logger.debug(
                    f"🔍 Clean JSON: Not a list, got {type(parsed)}"
                )
                return []

            objects = []
            skipped_items = []

            for i, item in enumerate(parsed):
                try:
                    # Validate item structure
                    if not isinstance(item, dict):
                        skipped_items.append(f"Item {i}: Not a dict, got {type(item)}")
                        continue

                    # Check for required keys (support 'bbox', 'bbox_2d' and description variants)
                    bbox = item.get("bbox_2d") or item.get("bbox")
                    description = (
                        item.get("description") or item.get("desc") or item.get("label")
                    )

                    if bbox is None:
                        skipped_items.append(f"Item {i}: Missing bbox key")
                        continue
                    if not description:
                        skipped_items.append(f"Item {i}: Missing description/desc key")
                        continue

                    # Validate bbox format
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        skipped_items.append(f"Item {i}: Invalid bbox format - {bbox}")
                        continue

                    # Convert coordinates to float
                    try:
                        coords = [float(x) for x in bbox]
                    except (ValueError, TypeError) as e:
                        skipped_items.append(
                            f"Item {i}: Non-numeric bbox - {bbox} (error: {e})"
                        )
                        continue

                    # Basic coordinate validation
                    x1, y1, x2, y2 = coords
                    if x1 >= x2 or y1 >= y2:
                        skipped_items.append(
                            f"Item {i}: Invalid geometry - [{x1}, {y1}, {x2}, {y2}]"
                        )
                        continue

                    # Add valid object – standardise with bbox_2d
                    objects.append(
                        {
                            "bbox_2d": coords,
                            "bbox": coords,  # backward-compatibility
                            "description": str(description).strip(),
                        }
                    )

                except Exception as e:
                    skipped_items.append(f"Item {i}: Exception - {e}")
                    continue

            if skipped_items:
                self.parser_logger.debug(
                    f"🔍 Clean JSON: Skipped {len(skipped_items)} items: {'; '.join(skipped_items)}"
                )

            return objects

        except json.JSONDecodeError as e:
            self.parser_logger.debug(f"🔍 Clean JSON: JSON decode error - {e}")
            return []
        except Exception as e:
            self.parser_logger.debug(f"🔍 Clean JSON: Unexpected error - {e}")
            return []

    def _parse_partial_json_format(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse incomplete/truncated JSON by extracting complete objects.

        Handles cases where JSON is cut off mid-object or mid-array.
        """
        self.parser_logger.debug(
            "🔍 Attempting partial JSON parsing for incomplete response"
        )

        # First, try to fix common truncation issues
        fixed_text = self._attempt_json_repair(text)
        if fixed_text != text:
            self.parser_logger.debug(
                f"🔧 Attempted JSON repair: {repr(fixed_text[:200])}"
            )
            try:
                parsed = json.loads(fixed_text)
                if isinstance(parsed, list):
                    return self._extract_valid_objects_from_list(parsed)
            except json.JSONDecodeError:
                pass

        # If repair didn't work, extract complete objects using regex
        objects = []

        # Pattern to match complete JSON objects with bbox and description
        object_pattern = (
            r'\{\s*"bbox"\s*:\s*\[[^\]]+\]\s*,\s*"description"\s*:\s*"[^"]*"\s*\}'
        )
        matches = re.findall(object_pattern, text)

        self.parser_logger.debug(
            f"🔍 Partial JSON: Found {len(matches)} complete object patterns"
        )

        for i, match in enumerate(matches):
            try:
                obj = json.loads(match)
                if self._is_valid_object_structure(obj):
                    objects.append(obj)
                    self.parser_logger.debug(
                        f"✅ Partial JSON: Extracted object {i}: {obj}"
                    )
                else:
                    self.parser_logger.debug(
                        f"❌ Partial JSON: Invalid object {i}: {obj}"
                    )
            except json.JSONDecodeError as e:
                self.parser_logger.debug(
                    f"❌ Partial JSON: Failed to parse object {i}: {e}"
                )
                continue

        return objects

    def _attempt_json_repair(self, text: str) -> str:
        """
        Attempt to repair truncated JSON by adding missing closing brackets.
        """
        # Count opening and closing brackets
        open_brackets = text.count("[")
        close_brackets = text.count("]")
        open_braces = text.count("{")
        close_braces = text.count("}")

        # If we have unmatched brackets, try to close them
        repaired = text

        # Close incomplete string if needed
        if repaired.count('"') % 2 == 1:
            repaired += '"'

        # Close incomplete objects
        missing_close_braces = open_braces - close_braces
        if missing_close_braces > 0:
            repaired += "}" * missing_close_braces

        # Close incomplete arrays
        missing_close_brackets = open_brackets - close_brackets
        if missing_close_brackets > 0:
            repaired += "]" * missing_close_brackets

        return repaired

    def _is_valid_object_structure(self, obj: Dict) -> bool:
        """Check if an object has the expected structure."""
        if not isinstance(obj, dict):
            return False

        bbox = obj.get("bbox_2d") or obj.get("bbox")
        description = obj.get("description") or obj.get("desc") or obj.get("label")

        return (
            bbox is not None
            and isinstance(bbox, list)
            and len(bbox) == 4
            and description is not None
            and isinstance(description, str)
        )

    def _extract_valid_objects_from_list(self, parsed_list: List) -> List[Dict]:
        """Extract valid objects from a parsed list, skipping invalid ones."""
        objects = []
        for i, item in enumerate(parsed_list):
            if self._is_valid_object_structure(item):
                objects.append(item)
            else:
                self.parser_logger.debug(f"🔍 Skipping invalid object {i}: {item}")
        return objects

    def _parse_regex_extraction(self, text: str) -> List[Dict]:
        """
        Last resort: extract bbox and description using regex patterns.

        Handles various formats and incomplete structures.
        """
        self.parser_logger.debug("🔍 Attempting regex extraction as last resort")

        objects = []

        # Pattern 1: Standard JSON-like format
        pattern1 = (
            r'"bbox"\s*:\s*\[\s*([0-9.,-\s]+)\s*\]\s*,\s*"description"\s*:\s*"([^"]*)"'
        )
        matches1 = re.findall(pattern1, text)

        for bbox_str, desc in matches1:
            try:
                coords = [float(x.strip()) for x in bbox_str.split(",")]
                if len(coords) == 4:
                    objects.append({"bbox": coords, "description": desc.strip()})
            except (ValueError, IndexError):
                continue

        # Pattern 2: Alternative format with 'desc' key
        pattern2 = r'"bbox"\s*:\s*\[\s*([0-9.,-\s]+)\s*\]\s*,\s*"desc"\s*:\s*"([^"]*)"'
        matches2 = re.findall(pattern2, text)

        for bbox_str, desc in matches2:
            try:
                coords = [float(x.strip()) for x in bbox_str.split(",")]
                if len(coords) == 4:
                    objects.append({"bbox": coords, "description": desc.strip()})
            except (ValueError, IndexError):
                continue

        # Pattern 3: Loose format without quotes around keys
        pattern3 = r'bbox\s*:\s*\[\s*([0-9.,-\s]+)\s*\]\s*,\s*description\s*:\s*["\']([^"\']*)["\']'
        matches3 = re.findall(pattern3, text)

        for bbox_str, desc in matches3:
            try:
                coords = [float(x.strip()) for x in bbox_str.split(",")]
                if len(coords) == 4:
                    objects.append({"bbox": coords, "description": desc.strip()})
            except (ValueError, IndexError):
                continue

        self.parser_logger.debug(f"🔍 Regex extraction: Found {len(objects)} objects")
        return objects

    def _parse_legacy_json_format(self, text: str) -> List[Dict[str, Any]]:
        """Parse legacy JSON format with 'desc' key."""
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                return []

            objects = []
            for item in parsed:
                try:
                    if (
                        isinstance(item, dict)
                        and "bbox" in item
                        and "desc" in item
                        and isinstance(item["bbox"], list)
                        and len(item["bbox"]) == 4
                    ):
                        coords = [float(x) for x in item["bbox"]]
                        objects.append({"bbox": coords, "description": item["desc"]})
                except (ValueError, TypeError):
                    continue

            return objects
        except (json.JSONDecodeError, TypeError):
            return []

    def _parse_special_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Parse Qwen2.5-VL special token format."""
        pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\(([^)]+)\),\s*\(([^)]+)\)<\|box_end\|>"
        matches = re.findall(pattern, text, re.DOTALL)

        self.parser_logger.debug(
            f"🔍 Special tokens: Found {len(matches)} regex matches"
        )
        if len(matches) == 0:
            # Check if text contains any special tokens at all
            has_ref_start = "<|object_ref_start|>" in text
            has_ref_end = "<|object_ref_end|>" in text
            has_box_start = "<|box_start|>" in text
            has_box_end = "<|box_end|>" in text
            self.parser_logger.debug(
                f"🔍 Special tokens presence: ref_start={has_ref_start}, ref_end={has_ref_end}, box_start={has_box_start}, box_end={has_box_end}"
            )

        objects = []
        skipped_matches = []

        for i, (desc, coords1, coords2) in enumerate(matches):
            try:
                x1, y1 = map(float, coords1.split(", "))
                x2, y2 = map(float, coords2.split(", "))
                objects.append({"bbox": [x1, y1, x2, y2], "description": desc.strip()})
            except (ValueError, IndexError) as e:
                skipped_matches.append(
                    f"Match {i}: coords1='{coords1}', coords2='{coords2}', error={e}"
                )
                continue

        if skipped_matches:
            self.parser_logger.debug(
                f"🔍 Special tokens: Skipped {len(skipped_matches)} matches: {'; '.join(skipped_matches)}"
            )

        return objects

    def _parse_unquoted_format(self, text: str) -> List[Dict[str, Any]]:
        """Parse unquoted format using regex."""
        pattern = r'\{bbox:\s*\[([^\]]+)\]\s*,\s*desc:\s*[\'"]([^\'"]+)[\'"]\s*\}'
        matches = re.findall(pattern, text)

        objects = []
        for bbox_str, desc in matches:
            try:
                coords = [float(x.strip()) for x in bbox_str.split(",")]
                if len(coords) == 4:
                    objects.append({"bbox": coords, "description": desc.strip()})
            except (ValueError, IndexError):
                continue

        return objects

    def _validate_and_filter_objects(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate and filter parsed objects with enhanced error handling.

        Skips objects that don't meet validation criteria instead of failing.
        """
        if not objects:
            self.parser_logger.debug("🔍 No objects to validate")
            return []

        validated = []
        skipped_reasons = []

        for i, obj in enumerate(objects):
            try:
                # Log the raw object being validated for debugging
                self.parser_logger.debug(f"🔍 Validating object {i}: {obj}")

                bbox = obj.get("bbox_2d") or obj.get("bbox") or []
                if not isinstance(bbox, list) or len(bbox) != 4:
                    skipped_reasons.append(
                        f"Object {i}: Invalid bbox format - got {type(bbox)} with value {bbox}"
                    )
                    continue

                # Ensure coordinates are numeric and valid
                try:
                    x1, y1, x2, y2 = [float(coord) for coord in bbox]
                except (ValueError, TypeError) as e:
                    skipped_reasons.append(
                        f"Object {i}: Non-numeric coordinates - {bbox} (error: {e})"
                    )
                    continue

                # Skip invalid geometries
                if x1 >= x2 or y1 >= y2:
                    skipped_reasons.append(
                        f"Object {i}: Invalid geometry (x1={x1}, y1={y1}, x2={x2}, y2={y2})"
                    )
                    continue

                # Skip negative coordinates (optional - depends on your coordinate system)
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    skipped_reasons.append(
                        f"Object {i}: Negative coordinates - [{x1}, {y1}, {x2}, {y2}]"
                    )
                    continue

                desc = (
                    obj.get("description") or obj.get("desc") or obj.get("label") or ""
                )
                desc = str(desc).strip()
                if not desc:
                    skipped_reasons.append(
                        f"Object {i}: Empty description - got '{obj.get('description', 'MISSING_KEY')}'"
                    )
                    continue

                validated.append(
                    {
                        "bbox_2d": [x1, y1, x2, y2],
                        "bbox": [x1, y1, x2, y2],  # legacy key
                        "description": desc,
                    }
                )
                self.parser_logger.debug(
                    f"✅ Object {i}: Valid - bbox=[{x1}, {y1}, {x2}, {y2}], desc='{desc}'"
                )

            except (ValueError, TypeError) as e:
                skipped_reasons.append(
                    f"Object {i}: Exception during validation - {e} (object: {obj})"
                )
                continue

        # Log validation summary
        if skipped_reasons:
            self.parser_logger.warning(
                f"⚠️ Validation skipped {len(skipped_reasons)} objects:"
            )
            for reason in skipped_reasons:
                self.parser_logger.warning(f"   {reason}")

            # If all objects were skipped, log as error for better visibility
            if len(validated) == 0 and len(objects) > 0:
                self.parser_logger.error(
                    f"🚨 ALL {len(objects)} OBJECTS FAILED VALIDATION:"
                )
                for i, reason in enumerate(skipped_reasons):
                    self.parser_logger.error(f"   Object {i}: {reason}")

        self.parser_logger.debug(
            f"🔍 Validation complete: {len(validated)}/{len(objects)} objects passed"
        )
        return validated

    def calculate_semantic_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate semantic similarity between descriptions."""
        if self.sentence_transformer is not None:
            try:
                embeddings = self.sentence_transformer.encode([desc1, desc2])
                similarity = torch.cosine_similarity(
                    torch.tensor(embeddings[0]).unsqueeze(0),
                    torch.tensor(embeddings[1]).unsqueeze(0),
                ).item()
                return max(0.0, similarity)
            except Exception as e:
                raise ValueError(f"Semantic similarity model not available: {e}")
