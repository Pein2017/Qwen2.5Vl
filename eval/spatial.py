import ast
import json
import xml.etree.ElementTree as ET

import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Simple chat template used during training (consistent with data_qwen.py)
DEFAULT_CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

additional_colors = [
    colorname for (colorname, colorcode) in ImageColor.colormap.items()
]


def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f"x{i + 1}")
            y = root.attrib.get(f"y{i + 1}")
            points.append([x, y])
        alt = root.attrib.get("alt")
        phrase = root.text.strip() if root.text else None
        return {"points": points, "alt": alt, "phrase": phrase}
    except Exception as e:
        print(e)
        return None


def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """
    print("\n=== PLOTTING DEBUG ===")
    print(f"Raw bounding_boxes input: {repr(bounding_boxes)}")
    print(f"Input dimensions: {input_width} x {input_height}")

    # Load the image
    img = im
    width, height = img.size
    print(f"Image size: {img.size}")
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)
    print(f"After parse_json: {repr(bounding_boxes)}")

    # Try to load font, fallback if not available
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except OSError:
        print("Warning: NotoSansCJK-Regular.ttc not found, using default font")
        font = ImageFont.load_default()

    # Parse JSON with comprehensive error handling
    json_output = None
    try:
        json_output = json.loads(bounding_boxes)
        print(f"Successfully parsed JSON with json.loads: {json_output}")
        print(f"Type of parsed JSON: {type(json_output)}")
    except Exception as e:
        print(f"Error parsing with json.loads: {e}")
        try:
            # Try with ast.literal_eval instead
            json_output = ast.literal_eval(bounding_boxes)
            print(f"Successfully parsed JSON with ast.literal_eval: {json_output}")
            print(f"Type of parsed JSON: {type(json_output)}")
        except Exception as e2:
            print(f"Error parsing with ast.literal_eval: {e2}")
            print("Cannot parse JSON, returning without plotting")
            return

    # Validate JSON structure
    if json_output is None:
        print("ERROR: json_output is None")
        return

    if not isinstance(json_output, list):
        print(f"ERROR: Expected list, got {type(json_output)}")
        print(f"Content: {json_output}")
        return

    print(f"Number of bounding boxes to plot: {len(json_output)}")

    # Iterate over the bounding boxes with comprehensive validation
    for i, bounding_box in enumerate(json_output):
        print(f"\nProcessing bounding bbox_2d {i}: {bounding_box}")
        print(f"Type of bounding_box: {type(bounding_box)}")

        # Validate bounding bbox_2d structure
        if not isinstance(bounding_box, dict):
            print(
                f"ERROR: Expected dict, got {type(bounding_box)} for bounding bbox_2d {i}"
            )
            continue

        if "bbox_2d" not in bounding_box:
            print(f"ERROR: 'bbox_2d' key not found in bounding bbox_2d {i}")
            print(f"Available keys: {list(bounding_box.keys())}")
            continue

        bbox_2d = bounding_box["bbox_2d"]
        print(f"bbox_2d: {bbox_2d}, type: {type(bbox_2d)}")

        # Validate bbox_2d structure
        if not isinstance(bbox_2d, (list, tuple)):
            print(f"ERROR: Expected list/tuple for bbox_2d, got {type(bbox_2d)}")
            continue

        if len(bbox_2d) != 4:
            print(f"ERROR: Expected 4 coordinates, got {len(bbox_2d)}")
            continue

        # Validate all coordinates are numbers
        try:
            coords = [float(x) for x in bbox_2d]
            print(f"Validated coordinates: {coords}")
        except (ValueError, TypeError) as e:
            print(f"ERROR: Invalid coordinates in bbox_2d: {e}")
            continue

        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        try:
            # Handle both [x1,y1,x2,y2] and [y1,x1,y2,x2] formats
            # Based on your training data, it appears to be [x1,y1,x2,y2] format
            abs_x1 = int(coords[0] / input_width * width)
            abs_y1 = int(coords[1] / input_height * height)
            abs_x2 = int(coords[2] / input_width * width)
            abs_y2 = int(coords[3] / input_height * height)

            # Ensure coordinates are in correct order
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1

            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            print(f"Absolute coordinates: ({abs_x1}, {abs_y1}) to ({abs_x2}, {abs_y2})")

            # Validate coordinates are within image bounds
            if abs_x1 < 0 or abs_y1 < 0 or abs_x2 > width or abs_y2 > height:
                print(
                    f"WARNING: Coordinates outside image bounds. Image: {width}x{height}"
                )

            # Draw the bounding bbox_2d
            draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

            # Draw the text - handle string format labels
            if "label" in bounding_box:
                label = str(bounding_box["label"])
                # Extract just the main label part if it's in string format
                if "label:" in label:
                    # Extract the part after "label:" and before the next ";"
                    label_parts = label.split(";")
                    for part in label_parts:
                        if part.startswith("label:"):
                            label = part.replace("label:", "").strip()
                            break
                draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color, font=font)
                print(f"Drew label: {label}")
            else:
                print("No label found in bounding bbox_2d")

        except Exception as e:
            print(f"ERROR drawing bounding bbox_2d {i}: {e}")
            continue

    print("=== PLOTTING COMPLETE ===")
    # Save the image
    img.save("test_plot.png")
    print("Image saved as test_plot.png")


def plot_points(im, text, input_width, input_height):
    img = im
    width, height = img.size
    draw = ImageDraw.Draw(img)
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ] + additional_colors
    xml_text = text.replace("```xml", "")
    xml_text = xml_text.replace("```", "")
    data = decode_xml_points(xml_text)
    if data is None:
        img.save("test_plot.png")
        return
    points = data["points"]
    description = data["phrase"]

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    for i, point in enumerate(points):
        color = colors[i % len(colors)]
        abs_x1 = int(point[0]) / input_width * width
        abs_y1 = int(point[1]) / input_height * height
        radius = 2
        draw.ellipse(
            [(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)],
            fill=color,
        )
        draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color, font=font)

    img.save("test_plot.png")


# @title Parsing JSON output
def parse_json(json_output):
    print("\n=== PARSE_JSON DEBUG ===")
    print(f"Input type: {type(json_output)}")
    print(f"Input: {repr(json_output)}")

    # If input is already a valid JSON structure, return it
    if isinstance(json_output, (list, dict)):
        print("Input is already parsed JSON, returning as-is")
        import json

        return json.dumps(json_output)  # Convert to string for consistency

    # Convert to string if needed
    if not isinstance(json_output, str):
        json_output = str(json_output)
        print(f"Converted to string: {repr(json_output)}")

    # Try to parse as JSON directly first
    try:
        import json

        parsed = json.loads(json_output)
        print(f"Successfully parsed as direct JSON: {parsed}")
        return json.dumps(parsed)  # Return as string for consistency
    except json.JSONDecodeError as e:
        print(f"Not valid direct JSON: {e}")

    # Look for JSON patterns in the text
    import re

    # First try to find JSON arrays with bbox_2d pattern (most specific)
    bbox_pattern = r'\[[\s\S]*?"bbox_2d"[\s\S]*?\]'
    bbox_matches = re.findall(bbox_pattern, json_output, re.DOTALL)
    if bbox_matches:
        print(f"Found bbox JSON pattern: {bbox_matches[0][:200]}...")
        try:
            parsed = json.loads(bbox_matches[0])
            print(f"Successfully parsed bbox pattern: {parsed}")
            return json.dumps(parsed)
        except json.JSONDecodeError as e:
            print(f"Failed to parse bbox pattern: {e}")

    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    print(f"Number of lines: {len(lines)}")

    original_output = json_output
    found_json_block = False

    for i, line in enumerate(lines):
        if line.strip() == "```json":
            print(f"Found ```json at line {i}")
            json_output = "\n".join(
                lines[i + 1 :]
            )  # Remove everything before "```json"
            json_output = json_output.split("```")[
                0
            ]  # Remove everything after the closing "```"
            print(f"Extracted JSON: {repr(json_output)}")
            found_json_block = True
            break  # Exit the loop once "```json" is found

    if not found_json_block:
        print("No ```json block found, using original input")
        json_output = original_output

    # Try to find JSON array or object patterns if no markdown block found
    if not found_json_block:
        print("Looking for JSON patterns in text...")

        # Look for JSON array (more greedy pattern)
        array_pattern = r"\[[\s\S]*?\]"
        array_matches = re.findall(array_pattern, json_output, re.DOTALL)
        if array_matches:
            # Try each match to see if it's valid JSON
            for match in array_matches:
                try:
                    parsed = json.loads(match)
                    print(f"Found valid JSON array: {match[:100]}...")
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    continue

        # Look for JSON object
        object_pattern = r"\{[\s\S]*?\}"
        object_matches = re.findall(object_pattern, json_output, re.DOTALL)
        if object_matches:
            for match in object_matches:
                try:
                    parsed = json.loads(match)
                    print(f"Found valid JSON object: {match[:100]}...")
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    continue

    print(f"Final fallback output: {repr(json_output)}")
    print("=== PARSE_JSON COMPLETE ===")
    return json_output


model_path = "/data4/Qwen2.5-VL-main/output/"
print(f"Loading model from: {model_path}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
print("Model loaded successfully!")

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_path)

# Set the chat template from tokenizer config
if not hasattr(processor, "chat_template") or processor.chat_template is None:
    print(
        "WARNING: Chat template not found in processor, setting template from tokenizer config"
    )
    processor.chat_template = DEFAULT_CHAT_TEMPLATE
    # Also set it on the tokenizer if it exists
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
else:
    print("Chat template found in processor")

print("Processor loaded successfully!")


def inference(
    img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024
):
    print("\n=== INFERENCE DEBUG ===")
    print(f"Image path: {img_url}")
    print(f"Prompt: {prompt}")
    print(f"System prompt: {system_prompt}")

    image = Image.open(img_url)
    print(f"Image size: {image.size}")

    # Use the list format (should work with proper template)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": img_url},
            ],
        },
    ]
    print(f"Messages: {messages}")

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("Chat template applied successfully!")

    print("Input text:")
    print(text)
    print("=" * 50)

    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to("cuda")
    print(f"Input tensor shapes: {inputs.input_ids.shape}")
    print(
        f"Image features shape: {inputs.pixel_values.shape if 'pixel_values' in inputs else 'No pixel_values'}"
    )

    print("Generating response...")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    print("\n=== MODEL OUTPUT ===")
    print("Raw output:")
    print(repr(output_text))
    print("\nFormatted output:")
    print(output_text[0])
    print("=" * 50)

    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14
    print(f"Input dimensions: {input_width} x {input_height}")

    return output_text[0], input_height, input_width


image_path = "/data4/Qwen2.5-VL-main/ds_rescaled/QC-20230216-0000240_116147.jpeg"


## Use a local HuggingFace model to inference.
# Updated prompt to match training data format exactly
prompt = "Please first output bbox coordinates and names of every item in this image in JSON format"
response, input_height, input_width = inference(image_path, prompt)

print("\n" + "=" * 80)
print("FINAL MODEL RESPONSE:")
print("=" * 80)
print(response)
print("=" * 80)
print(f"Response type: {type(response)}")
print(f"Response length: {len(response) if isinstance(response, str) else 'N/A'}")
print("=" * 80)

# Ask user if they want to proceed with plotting
try:
    proceed = input("\nDo you want to proceed with plotting? (y/n): ").lower().strip()
    if proceed != "y":
        print("Skipping plotting.")
        exit(0)
except KeyboardInterrupt:
    print("\nSkipping plotting.")
    exit(0)

image = Image.open(image_path)
print(f"\nOriginal image size: {image.size}")
image.thumbnail([640, 640], Image.Resampling.LANCZOS)
print(f"Thumbnail image size: {image.size}")

try:
    plot_bounding_boxes(image, response, input_width, input_height)
    print("Plotting completed successfully!")
except Exception as e:
    print(f"ERROR during plotting: {e}")
    import traceback

    traceback.print_exc()
