"""
Chat Template Manager for Qwen2.5-VL

Handles Jinja2-based conversation formatting with clean separation
between raw data and model-specific tokens.
"""

import logging
from typing import Any, Dict, List, Optional

from jinja2 import Template

logger = logging.getLogger(__name__)


class ChatTemplateManager:
    """Manages Jinja2 templates for different conversation formats."""

    # Base system prompt template
    SYSTEM_PROMPT_TEMPLATE = """You are Q-Vision-QC, an expert assistant specialized in telecom-equipment inspection.
Your task: detect and describe objects in the image using the special token format.

OUTPUT FORMAT:
For each detected object, use this exact format:
<|object_ref_start|>description<|object_ref_end|><|box_start|>(x1, y1), (x2, y2)<|box_end|>

Where:
- description: comma-separated object type and details
- (x1, y1): top-left corner coordinates  
- (x2, y2): bottom-right corner coordinates
- coordinates are absolute pixel values

Multiple objects should be separated by newlines.
Sort by top-to-bottom (increasing y), then left-to-right (increasing x).
Always respond in English only.
Output only the special token format (no extra text or explanations)."""

    # Single-round conversation template
    SINGLE_ROUND_TEMPLATE = """<|im_start|>system
{{ system_prompt }}
<|im_end|>
<|im_start|>user
<image>
<|im_end|>
<|im_start|>assistant
{% for obj in objects %}
<|object_ref_start|>{{ obj.formatted_desc }}<|object_ref_end|><|box_start|>{{ obj.formatted_box }}<|box_end|>
{% endfor %}<|endoftext|>"""

    # Multi-round conversation template with examples
    MULTI_ROUND_TEMPLATE = """<|im_start|>system
{{ system_prompt }}
<|im_end|>
{% for example in examples %}
<|im_start|>user
<image>
<|im_end|>
<|im_start|>assistant
{% for obj in example.objects %}
<|object_ref_start|>{{ obj.formatted_desc }}<|object_ref_end|><|box_start|>{{ obj.formatted_box }}<|box_end|>
{% endfor %}
<|im_end|>
{% endfor %}
<|im_start|>user
It's your turn to analyze this image <image> What can you see?
<|im_end|>
<|im_start|>assistant
{% for obj in objects %}
<|object_ref_start|>{{ obj.formatted_desc }}<|object_ref_end|><|box_start|>{{ obj.formatted_box }}<|box_end|>
{% endfor %}<|endoftext|>"""

    def __init__(self, use_candidates: bool = True, candidates_content: str = ""):
        """Initialize template manager."""
        self.use_candidates = use_candidates
        self.candidates_content = candidates_content

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

        # Compile templates
        self.single_template = Template(self.SINGLE_ROUND_TEMPLATE)
        self.multi_template = Template(self.MULTI_ROUND_TEMPLATE)

    def _build_system_prompt(self) -> str:
        """Build system prompt with optional candidates."""
        base_prompt = self.SYSTEM_PROMPT_TEMPLATE

        if self.use_candidates and self.candidates_content:
            candidates_section = f"""

CANDIDATE PHRASES:
The following phrases represent common objects and properties in telecom equipment:
{self.candidates_content}

Use these phrases when applicable to maintain consistency."""
            return base_prompt + candidates_section

        return base_prompt

    def format_single_round(self, objects: List[Dict[str, Any]]) -> str:
        """Format single-round conversation."""
        # Sort objects by position
        sorted_objects = self._sort_objects_by_position(objects)

        # Format objects for template
        formatted_objects = [self._format_object(obj) for obj in sorted_objects]

        return self.single_template.render(
            system_prompt=self.system_prompt, objects=formatted_objects
        )

    def format_multi_round(
        self, objects: List[Dict[str, Any]], examples: List[Dict[str, Any]] = None
    ) -> str:
        """Format multi-round conversation with examples."""
        # Sort main objects
        sorted_objects = self._sort_objects_by_position(objects)
        formatted_objects = [self._format_object(obj) for obj in sorted_objects]

        # Format examples
        formatted_examples = []
        if examples:
            for example in examples:
                example_objects = self._sort_objects_by_position(
                    example.get("objects", [])
                )
                formatted_example_objects = [
                    self._format_object(obj) for obj in example_objects
                ]
                formatted_examples.append(
                    {
                        "objects": formatted_example_objects,
                        "image": example.get("image", ""),
                    }
                )

        return self.multi_template.render(
            system_prompt=self.system_prompt,
            objects=formatted_objects,
            examples=formatted_examples,
        )

    def _sort_objects_by_position(
        self, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort objects by position (top-to-bottom, left-to-right)."""

        def sort_key(obj):
            box = obj.get("box", [0, 0, 0, 0])
            return (box[1], box[0])  # y1, x1

        return sorted(objects, key=sort_key)

    def _format_object(self, obj: Dict[str, Any]) -> Dict[str, str]:
        """Format object for template rendering."""
        # Format description
        desc_parts = []
        if "type" in obj:
            desc_parts.append(f"object_type:{obj['type']}")
        if "property" in obj:
            desc_parts.append(f"property:{obj['property']}")
        if "extra_info" in obj:
            desc_parts.append(f"extra_info:{obj['extra_info']}")

        # Fallback to raw description
        if not desc_parts and "desc" in obj:
            desc_parts.append(obj["desc"])

        formatted_desc = ";".join(desc_parts) if desc_parts else "unknown"

        # Format bounding box
        box = obj.get("box", [0, 0, 0, 0])
        formatted_box = f"({box[0]}, {box[1]}), ({box[2]}, {box[3]})"

        return {
            "formatted_desc": formatted_desc,
            "formatted_box": formatted_box,
            "raw_obj": obj,
        }


def create_chat_template_manager(
    use_candidates: bool = True, candidates_file: Optional[str] = None
) -> ChatTemplateManager:
    """Factory function to create chat template manager."""
    candidates_content = ""

    if use_candidates and candidates_file:
        try:
            with open(candidates_file, "r", encoding="utf-8") as f:
                candidates_content = f.read().strip()
            logger.info(f"Loaded candidates from {candidates_file}")
        except Exception as e:
            logger.warning(f"Failed to load candidates from {candidates_file}: {e}")

    return ChatTemplateManager(use_candidates, candidates_content)
