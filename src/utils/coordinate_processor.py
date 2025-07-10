"""
Coordinate Token Processor for Soft Expectation Regression

Converts between JSON bbox format and coordinate token sequences.
Integrates with existing ChatProcessor pipeline.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from src.logger_utils import get_detection_logger

logger = get_detection_logger()


@dataclass
class CoordinateTokenConfig:
    """Configuration for coordinate token processing."""
    enable_coordinate_tokens: bool = False
    max_coord_value: int = 2048
    use_official_box_tokens: bool = True
    box_start_id: int = 151648  # <|box_start|>
    box_end_id: int = 151649    # <|box_end|>
    normalize_coordinates: bool = True


class CoordinateTokenProcessor:
    """
    Processor to convert between JSON bbox format and coordinate token sequences.
    
    Supports both formats:
    - JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "description"}]
    - Tokens: "Object: <|box_start|><coord_100><coord_200><coord_1500><coord_1800><|box_end|>"
    """
    
    def __init__(self, config: CoordinateTokenConfig):
        self.config = config
        self.enabled = config.enable_coordinate_tokens
        
        if self.enabled:
            logger.info(f"🎯 Coordinate token processor enabled with {config.max_coord_value} resolution")
        else:
            logger.info("📝 Using standard JSON format (coordinate tokens disabled)")
    
    def convert_json_to_coordinate_format(self, json_response: str) -> str:
        """
        Convert JSON bbox response to coordinate token format.
        
        Args:
            json_response: "[{'bbox_2d': [x1, y1, x2, y2], 'label': 'desc'}]"
            
        Returns:
            "Object 1: <|box_start|><coord_x1><coord_y1><coord_x2><coord_y2><|box_end|>"
        """
        if not self.enabled:
            return json_response
            
        try:
            # Parse JSON response
            objects = json.loads(json_response)
            if not objects:
                return "[]"
                
            # Convert each object to coordinate token format
            coordinate_responses = []
            for i, obj in enumerate(objects):
                bbox = obj.get("bbox_2d", [0, 0, 0, 0])
                label = obj.get("label", "object")
                
                # Convert bbox to coordinate tokens
                coord_tokens = self._bbox_to_coordinate_tokens(bbox)
                
                # Format with description
                coord_response = f"{label}: {coord_tokens}"
                coordinate_responses.append(coord_response)
                
            return "\n".join(coordinate_responses)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to convert JSON to coordinate format: {e}")
            return json_response  # Fallback to original format
    
    def convert_coordinate_to_json_format(self, coordinate_response: str) -> str:
        """
        Convert coordinate token response back to JSON format.
        
        Args:
            coordinate_response: "Object: <|box_start|><coord_100><coord_200><coord_1500><coord_1800><|box_end|>"
            
        Returns:
            "[{'bbox_2d': [x1, y1, x2, y2], 'label': 'Object'}]"
        """
        if not self.enabled:
            return coordinate_response
            
        try:
            # Pattern to match coordinate sequences
            coord_pattern = r'(.*?):\s*<\|box_start\|>(<coord_\d+>){4}<\|box_end\|>'
            matches = re.findall(coord_pattern, coordinate_response)
            
            if not matches:
                # Try to parse as JSON (fallback)
                try:
                    json.loads(coordinate_response)
                    return coordinate_response
                except:
                    return "[]"
            
            # Convert back to JSON format
            json_objects = []
            for match in matches:
                label = match[0].strip()
                
                # Extract coordinate tokens from the full text
                coord_match = re.search(
                    rf'{re.escape(label)}:\s*<\|box_start\|>(<coord_\d+>){{4}}<\|box_end\|>',
                    coordinate_response
                )
                
                if coord_match:
                    # Extract individual coordinate tokens
                    coord_tokens = re.findall(r'<coord_(\d+)>', coord_match.group(0))
                    
                    if len(coord_tokens) == 4:
                        # Convert back to normalized coordinates
                        bbox = self._coordinate_tokens_to_bbox([int(t) for t in coord_tokens])
                        json_objects.append({"bbox_2d": bbox, "label": label})
            
            return json.dumps(json_objects, ensure_ascii=False, separators=(",", ": "))
            
        except Exception as e:
            logger.warning(f"Failed to convert coordinate format to JSON: {e}")
            return "[]"  # Fallback to empty array
    
    def _bbox_to_coordinate_tokens(self, bbox: List[float]) -> str:
        """Convert normalized bbox to coordinate token sequence."""
        if len(bbox) != 4:
            logger.warning(f"Invalid bbox format: {bbox}")
            bbox = [0.0, 0.0, 1.0, 1.0]  # Default bbox
        
        # Normalize if needed
        if self.config.normalize_coordinates:
            # Clamp to [0, 1] range
            bbox = [max(0.0, min(1.0, coord)) for coord in bbox]
        
        # Convert to coordinate indices
        coord_indices = []
        for coord in bbox:
            coord_idx = int(coord * (self.config.max_coord_value - 1))
            coord_idx = max(0, min(coord_idx, self.config.max_coord_value - 1))
            coord_indices.append(coord_idx)
        
        # Format as token sequence
        coord_tokens = [f"<coord_{idx}>" for idx in coord_indices]
        
        return f"<|box_start|>{''.join(coord_tokens)}<|box_end|>"
    
    def _coordinate_tokens_to_bbox(self, coord_indices: List[int]) -> List[float]:
        """Convert coordinate token indices back to normalized bbox."""
        if len(coord_indices) != 4:
            logger.warning(f"Invalid coordinate indices: {coord_indices}")
            return [0.0, 0.0, 1.0, 1.0]  # Default bbox
        
        # Convert back to normalized coordinates
        bbox = []
        for coord_idx in coord_indices:
            # Clamp to valid range
            coord_idx = max(0, min(coord_idx, self.config.max_coord_value - 1))
            normalized_coord = coord_idx / (self.config.max_coord_value - 1)
            bbox.append(normalized_coord)
        
        return bbox
    
    def get_coordinate_token_ids(self, bbox: List[float], tokenizer) -> List[int]:
        """
        Get actual token IDs for coordinate sequence.
        
        Args:
            bbox: Normalized bbox coordinates [x1, y1, x2, y2]
            tokenizer: Tokenizer with coordinate tokens
            
        Returns:
            List of token IDs: [box_start_id, coord_id1, coord_id2, coord_id3, coord_id4, box_end_id]
        """
        if not self.enabled:
            return []
        
        # Convert bbox to coordinate indices
        coord_indices = []
        for coord in bbox:
            coord_idx = int(coord * (self.config.max_coord_value - 1))
            coord_idx = max(0, min(coord_idx, self.config.max_coord_value - 1))
            coord_indices.append(coord_idx)
        
        # Get token IDs
        token_ids = [self.config.box_start_id]  # <|box_start|>
        
        for coord_idx in coord_indices:
            # Coordinate tokens start after original vocab
            # This assumes coordinate tokens were added starting from original_vocab_size
            coord_token_text = f"<coord_{coord_idx}>"
            coord_token_id = tokenizer.convert_tokens_to_ids(coord_token_text)
            
            if coord_token_id == tokenizer.unk_token_id:
                logger.warning(f"Unknown coordinate token: {coord_token_text}")
                # Use a fallback coordinate token ID
                coord_token_id = 151936 + coord_idx  # Approximate fallback
            
            token_ids.append(coord_token_id)
        
        token_ids.append(self.config.box_end_id)  # <|box_end|>
        
        return token_ids
    
    def validate_coordinate_format(self, text: str) -> bool:
        """
        Validate that text contains properly formatted coordinate tokens.
        
        Args:
            text: Text to validate
            
        Returns:
            True if coordinate format is valid
        """
        if not self.enabled:
            return True  # Always valid when disabled
        
        # Check for coordinate token pattern
        coord_pattern = r'<\|box_start\|>(<coord_\d+>){4}<\|box_end\|>'
        matches = re.findall(coord_pattern, text)
        
        # Validate coordinate ranges
        for match in re.finditer(coord_pattern, text):
            coord_tokens = re.findall(r'<coord_(\d+)>', match.group(0))
            for coord_str in coord_tokens:
                coord_val = int(coord_str)
                if coord_val >= self.config.max_coord_value:
                    logger.warning(f"Coordinate value {coord_val} exceeds max {self.config.max_coord_value}")
                    return False
        
        return True


def create_coordinate_processor(enable_coordinate_tokens: bool = False) -> CoordinateTokenProcessor:
    """Factory function to create coordinate processor with default config."""
    config = CoordinateTokenConfig(
        enable_coordinate_tokens=enable_coordinate_tokens,
        max_coord_value=2048,
        use_official_box_tokens=True,
    )
    return CoordinateTokenProcessor(config)


# Example usage and testing functions
def demo_coordinate_conversion():
    """Demonstrate coordinate token conversion."""
    processor = create_coordinate_processor(enable_coordinate_tokens=True)
    
    # Example JSON input
    json_input = '[{"bbox_2d": [0.1, 0.2, 0.8, 0.9], "label": "screw connector"}]'
    
    print("JSON Input:")
    print(json_input)
    
    # Convert to coordinate format
    coord_format = processor.convert_json_to_coordinate_format(json_input)
    print("\nCoordinate Format:")
    print(coord_format)
    
    # Convert back to JSON
    json_output = processor.convert_coordinate_to_json_format(coord_format)
    print("\nJSON Output:")
    print(json_output)
    
    # Validate roundtrip
    print(f"\nRoundtrip successful: {json_input == json_output}")


if __name__ == "__main__":
    demo_coordinate_conversion()