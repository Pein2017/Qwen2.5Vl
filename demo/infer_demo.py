#!/usr/bin/env python3
"""
Simple single-image demo for Qwen2.5-VL generation before training.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.patches import apply_comprehensive_qwen25_fixes


def main():
    # Apply all Qwen2.5-VL fixes before loading the model.
    # This addresses mRoPE and other dimension mismatch issues.
    apply_comprehensive_qwen25_fixes()

    # Load processor and model
    model_path = "output_detection/qwen_3B_detection_new_objecttype/checkpoint-200"
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # Load and prepare image
    image_path = "ds_rescaled/QC-20230114-0000212_98292.jpeg"
    image = Image.open(image_path).convert("RGB")

    # Define prompt using the official chat template.
    # This ensures the <|image_pad|> token is correctly placed.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    # "text": "Please describe the image in detail",
                    "text": r"""
Detect all objects like cabel, device and label, and output JSON like:
[
  {"bbox_2d": [x1, y1, x2, y2],"label": "..."},
  ...
]
""",
                },
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Prepare inputs for generation
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    # Move inputs to the model's device
    inputs = inputs.to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    # Decode and clean the output to get only the assistant's response
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    try:
        # The response includes the prompt, so we extract the assistant's part
        assistant_response = response.split("assistant\n")[1].strip()
    except IndexError:
        # Fallback if the template changes or output is unexpected
        assistant_response = response

    print(assistant_response)


if __name__ == "__main__":
    main()
