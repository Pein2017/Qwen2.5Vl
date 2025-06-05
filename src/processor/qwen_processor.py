"""
Qwen2.5-VL Processor

Main processor that combines chat templates and vision token expansion
for clean, modular processing pipeline.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer

from ..templates.chat_templates import ChatTemplateManager
from .vision_utils import VisionTokenExpander

logger = logging.getLogger(__name__)


class Qwen25VLProcessor:
    """
    Main processor for Qwen2.5-VL that handles the complete pipeline:
    1. Raw data → Chat template formatting
    2. Chat template → Vision token expansion  
    3. Vision tokens → Model inputs
    """
    
    def __init__(
        self,
        tokenizer,
        image_processor,
        chat_template_manager: ChatTemplateManager,
        data_root: str = "./"
    ):
        """Initialize processor components."""
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.chat_template_manager = chat_template_manager
        self.data_root = data_root
        
        # Initialize vision token expander
        self.vision_expander = VisionTokenExpander(image_processor)
        
        logger.info("Qwen25VLProcessor initialized")
    
    def process_raw_sample(
        self,
        raw_sample: Dict[str, Any],
        multi_round: bool = False,
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process raw sample through complete pipeline.
        
        Args:
            raw_sample: Raw data with 'objects' and 'images' keys
            multi_round: Whether to use multi-round template
            examples: Optional examples for few-shot learning
            
        Returns:
            Model inputs ready for training/inference
        """
        # Step 1: Extract data
        objects = raw_sample.get('objects', [])
        image_paths = raw_sample.get('images', [])
        
        if not image_paths:
            raise ValueError("No images provided in sample")
        
        # Step 2: Generate conversation text using chat template
        if multi_round and examples:
            conversation_text = self.chat_template_manager.format_multi_round(
                objects=objects,
                examples=examples
            )
        else:
            conversation_text = self.chat_template_manager.format_single_round(
                objects=objects
            )
        
        logger.debug(f"Generated conversation: {len(conversation_text)} chars")
        
        # Step 3: Expand vision tokens
        expanded_text, token_counts = self.vision_expander.process_conversation_images(
            conversation_text=conversation_text,
            image_paths=image_paths
        )
        
        logger.debug(f"Expanded to {len(expanded_text)} chars, token_counts: {token_counts}")
        
        # Step 4: Process images
        pixel_values, image_grid_thw = self._process_images(image_paths)
        
        # Step 5: Tokenize
        model_inputs = self._tokenize_conversation(expanded_text)
        
        # Step 6: Combine everything
        result = {
            **model_inputs,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
        }
        
        # Add metadata
        result['metadata'] = {
            'num_images': len(image_paths),
            'token_counts': token_counts,
            'conversation_length': len(conversation_text),
            'expanded_length': len(expanded_text)
        }
        
        return result
    
    def process_for_generation(
        self,
        raw_sample: Dict[str, Any],
        multi_round: bool = False,
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Process sample for generation (inference).
        
        Returns:
            Tuple of (model_inputs, ground_truth_text)
        """
        # Process normally but split for generation
        full_result = self.process_raw_sample(raw_sample, multi_round, examples)
        
        # Extract ground truth (everything after last <|im_start|>assistant)
        # This would need to be implemented based on your specific needs
        ground_truth = self._extract_ground_truth(raw_sample)
        
        # Modify inputs for generation (remove ground truth part)
        generation_inputs = self._prepare_for_generation(full_result)
        
        return generation_inputs, ground_truth
    
    def _process_images(self, image_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process images and return pixel values and grid_thw."""
        import os

        from PIL import Image
        
        images = []
        for path in image_paths:
            full_path = os.path.join(self.data_root, path) if not os.path.isabs(path) else path
            image = Image.open(full_path).convert("RGB")
            images.append(image)
        
        # Process all images
        processed = self.image_processor(images=images, return_tensors="pt")
        
        return processed['pixel_values'], processed['image_grid_thw']
    
    def _tokenize_conversation(self, conversation_text: str) -> Dict[str, torch.Tensor]:
        """Tokenize conversation text."""
        # Use tokenizer directly since we've already formatted the conversation
        inputs = self.tokenizer(
            conversation_text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        
        return inputs
    
    def _extract_ground_truth(self, raw_sample: Dict[str, Any]) -> str:
        """Extract ground truth from raw sample."""
        # This is a placeholder - implement based on your needs
        objects = raw_sample.get('objects', [])
        
        # Format objects as ground truth
        ground_truth_parts = []
        for obj in objects:
            desc_parts = []
            if 'type' in obj:
                desc_parts.append(f"object_type:{obj['type']}")
            if 'property' in obj:
                desc_parts.append(f"property:{obj['property']}")
            if 'extra_info' in obj:
                desc_parts.append(f"extra_info:{obj['extra_info']}")
            
            desc = ';'.join(desc_parts) if desc_parts else obj.get('desc', 'unknown')
            box = obj.get('box', [0, 0, 0, 0])
            
            ground_truth_parts.append(
                f"<|object_ref_start|>{desc}<|object_ref_end|>"
                f"<|box_start|>({box[0]}, {box[1]}), ({box[2]}, {box[3]})<|box_end|>"
            )
        
        return '\n'.join(ground_truth_parts) + '<|endoftext|>'
    
    def _prepare_for_generation(self, full_result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for generation by removing ground truth part."""
        # This is a placeholder - implement based on your needs
        # You'd need to truncate the input_ids at the right point
        return full_result


def create_qwen25vl_processor(
    model_path: str,
    use_candidates: bool = True,
    candidates_file: Optional[str] = None,
    data_root: str = "./"
) -> Qwen25VLProcessor:
    """Factory function to create complete processor."""
    
    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True).image_processor
    
    # Create chat template manager
    from ..templates.chat_templates import create_chat_template_manager
    chat_manager = create_chat_template_manager(use_candidates, candidates_file)
    
    # Create processor
    processor = Qwen25VLProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        chat_template_manager=chat_manager,
        data_root=data_root
    )
    
    return processor 