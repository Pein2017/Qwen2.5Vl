#!/usr/bin/env python3
"""
🔍 Qwen2.5-VL Comprehensive Demo Script - Latest Architecture

Purpose:
- Mimic the complete training pipeline: config loading → data preparation → model initialization → forward pass
- Use the actual yaml config and src modules for realistic testing
- Debug and validate the full pipeline in an easy-to-use script

Key Features:
- ✅ Uses ConfigManager to load yaml configs
- ✅ Uses ChatProcessor for data preparation
- ✅ Uses ModelWrapper for model initialization
- ✅ Tests the complete forward pass with loss calculation
- ✅ Validates special token handling and object detection outputs

Architecture Alignment:
- Follows the simplified JSONL format (examples + target)
- Uses the unified ChatProcessor for data preparation
- Applies comprehensive Qwen2.5-VL fixes automatically
- Tests both training and inference modes

NO TRY-EXCEPT: All errors will surface immediately for precise debugging
"""

import os
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_manager import ConfigManager
from src.data import BBUDataset, create_data_collator
from src.models import ModelWrapper, apply_comprehensive_qwen25_fixes


def setup_environment():
    """Setup environment and apply comprehensive fixes"""
    print("🔍 Setting up environment...")
    print(f"📁 Working directory: {os.getcwd()}")

    # Apply comprehensive Qwen2.5-VL fixes BEFORE any model loading
    print("🔧 Applying comprehensive Qwen2.5-VL fixes...")
    apply_comprehensive_qwen25_fixes()
    print("✅ All Qwen2.5-VL fixes applied successfully")


def load_config(config_name: str = "base", overrides: list = None):
    """Load configuration using ConfigManager"""
    print(f"\n📋 Loading configuration: {config_name}")

    config_manager = ConfigManager()
    config = config_manager.load_config(config_name, overrides)

    print(f"✅ Configuration loaded successfully")
    print(f"   Model: {config.model_path}")
    print(f"   Model size: {config.model_size}")
    print(f"   Max length: {config.model_max_length}")
    print(f"   Train data: {config.train_data_path}")
    print(f"   Val data: {config.val_data_path}")
    print(f"   Use candidates: {config.use_candidates}")

    return config


def load_tokenizer_and_processor(config):
    """Load tokenizer and processor"""
    print(f"\n🔄 Loading tokenizer and processor...")

    model_path = config.model_path
    assert Path(model_path).exists(), f"Model path does not exist: {model_path}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"✅ Processor loaded")

    # Validate special tokens
    from src.tokens import SpecialTokens

    tokens = SpecialTokens()

    print(f"\n🎯 SPECIAL TOKEN VALIDATION:")
    special_token_checks = [
        ("Vision Start", tokens.VISION_START),
        ("Vision End", tokens.VISION_END),
        ("Image Pad", tokens.IMAGE_PAD),
        ("Object Ref Start", tokens.OBJECT_REF_START),
        ("Object Ref End", tokens.OBJECT_REF_END),
        ("Box Start", tokens.BOX_START),
        ("Box End", tokens.BOX_END),
    ]

    for name, token in special_token_checks:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            print(f"   ✅ {name}: {token} → {token_id}")
        else:
            print(f"   ❌ {name}: {token} → NOT FOUND")

    return tokenizer, processor


def create_dataset(config, tokenizer, processor):
    """Create dataset using BBUDataset"""
    print(f"\n📊 Creating dataset...")

    # Check data paths
    data_root = Path(config.data_root)
    train_path = data_root / config.train_data_path
    val_path = data_root / config.val_data_path

    print(f"   Data root: {data_root}")
    print(f"   Train path: {train_path}")
    print(f"   Val path: {val_path}")

    # Use validation data for demo (smaller and faster)
    demo_data_path = val_path if val_path.exists() else train_path
    assert demo_data_path.exists(), f"Demo data path does not exist: {demo_data_path}"

    print(f"   Using demo data: {demo_data_path}")

    # Create dataset
    dataset = BBUDataset(
        config=config,
        tokenizer=tokenizer,
        image_processor=processor.image_processor,
        data_path=str(demo_data_path),
    )

    print(f"✅ Dataset created with {len(dataset)} samples")

    return dataset


def analyze_sample_structure(dataset, sample_idx: int = 0):
    """Analyze a sample from the dataset"""
    print(f"\n🔍 Analyzing sample {sample_idx}...")

    # Get raw sample for analysis
    raw_sample = dataset.data[sample_idx]

    print(f"📊 RAW SAMPLE STRUCTURE:")
    print(f"   Examples: {len(raw_sample.get('examples', []))}")
    print(f"   Target images: {len(raw_sample.get('target', {}).get('images', []))}")
    print(f"   Target objects: {len(raw_sample.get('target', {}).get('objects', []))}")

    # Show examples breakdown
    examples = raw_sample.get("examples", [])
    for i, example in enumerate(examples):
        images = example.get("images", [])
        objects = example.get("objects", [])
        print(f"   Example {i + 1}: {len(images)} images, {len(objects)} objects")

    # Show target structure
    target = raw_sample.get("target", {})
    target_objects = target.get("objects", [])
    if target_objects:
        print(f"\n📄 TARGET OBJECTS PREVIEW:")
        for i, obj in enumerate(target_objects[:3]):  # Show first 3
            desc = obj.get("desc", "")
            box = obj.get("box", [])
            print(f"   {i + 1}. {desc} → {box}")
        if len(target_objects) > 3:
            print(f"   ... and {len(target_objects) - 3} more objects")

    return raw_sample


def test_data_processing(dataset, sample_idx: int = 0):
    """Test data processing pipeline"""
    print(f"\n🔧 Testing data processing pipeline...")

    # Process sample through ChatProcessor
    processed_sample = dataset[sample_idx]

    print(f"📊 PROCESSED SAMPLE STRUCTURE:")
    for key, value in processed_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {type(value)}")

    # Analyze input sequence
    input_ids = processed_sample["input_ids"]
    labels = processed_sample["labels"]

    print(f"\n🔍 SEQUENCE ANALYSIS:")
    print(f"   Input length: {len(input_ids)}")
    print(f"   Labels length: {len(labels)}")

    # Count special tokens
    from src.tokens import SpecialTokens

    tokens = SpecialTokens()

    # Convert tokens to IDs for counting
    tokenizer = dataset.tokenizer
    vision_start_id = tokenizer.convert_tokens_to_ids(tokens.VISION_START)
    vision_end_id = tokenizer.convert_tokens_to_ids(tokens.VISION_END)
    image_pad_id = tokenizer.convert_tokens_to_ids(tokens.IMAGE_PAD)
    object_ref_start_id = tokenizer.convert_tokens_to_ids(tokens.OBJECT_REF_START)

    vision_start_count = (input_ids == vision_start_id).sum().item()
    vision_end_count = (input_ids == vision_end_id).sum().item()
    image_pad_count = (input_ids == image_pad_id).sum().item()
    object_ref_count = (input_ids == object_ref_start_id).sum().item()

    print(f"   Vision tokens: start={vision_start_count}, end={vision_end_count}")
    print(f"   Image pad tokens: {image_pad_count}")
    print(f"   Object references: {object_ref_count}")

    # Check image data
    if "pixel_values" in processed_sample:
        pixel_values = processed_sample["pixel_values"]
        image_grid_thw = processed_sample["image_grid_thw"]
        print(f"   Pixel values: {pixel_values.shape}")
        print(f"   Image grid THW: {image_grid_thw.shape}")
        print(f"   Image grid values: {image_grid_thw}")

    # Show input text preview
    print(f"\n📝 INPUT TEXT PREVIEW (last 200 chars):")
    input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"   {repr(input_text[-200:])}")

    # Check training labels
    non_ignore_labels = labels[labels != -100]
    print(f"\n🎯 TRAINING LABELS:")
    print(f"   Total tokens: {len(labels)}")
    print(f"   Training tokens: {len(non_ignore_labels)}")
    print(f"   Ignored tokens: {len(labels) - len(non_ignore_labels)}")

    if len(non_ignore_labels) > 0:
        label_text = tokenizer.decode(non_ignore_labels, skip_special_tokens=False)
        print(f"   Label text preview: {repr(label_text[:100])}...")

    return processed_sample


def create_data_collator_and_batch(
    config, tokenizer, dataset, sample_indices: list = [0]
):
    """Create data collator and test batch creation"""
    print(f"\n📦 Creating data collator and test batch...")

    # Create data collator
    collator = create_data_collator(
        tokenizer=tokenizer,
        max_total_length=getattr(config, "max_total_length", None),
        collator_type=getattr(config, "collator_type", "standard"),
    )

    print(f"✅ Data collator created: {type(collator).__name__}")

    # Create batch
    samples = [dataset[i] for i in sample_indices]
    batch = collator(samples)

    print(f"\n📊 BATCH STRUCTURE:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {type(value)}")

    return collator, batch


def load_model(config, tokenizer):
    """Load model using ModelWrapper"""
    print(f"\n🤖 Loading model...")

    # Convert DictConfig to Config object
    from omegaconf import OmegaConf

    from src.config.base import Config

    config_dict = OmegaConf.to_container(config, resolve=True)
    config_obj = Config.from_dict(config_dict)

    # Create ModelWrapper
    model_wrapper = ModelWrapper(config=config_obj)

    # Load all components
    model, tokenizer_loaded, image_processor = model_wrapper.load_all()

    # Move model to GPU if available
    if torch.cuda.is_available():
        print(f"🔄 Moving model to GPU...")
        model = model.cuda()
        print(f"✅ Model moved to GPU")

    print(f"✅ Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable_params:,}")

    return model_wrapper, model


def test_forward_pass(model, batch, config):
    """Test forward pass with loss calculation"""
    print(f"\n🚀 Testing forward pass...")

    # Move batch to device
    device = model.device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    print(f"📊 FORWARD PASS INPUT:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} on {value.device}")

    # Forward pass
    print(f"\n🔄 Running forward pass...")

    with torch.no_grad():
        outputs = model(**batch)

    print(f"✅ Forward pass completed successfully!")

    # Analyze outputs
    print(f"\n📊 MODEL OUTPUTS:")
    if hasattr(outputs, "loss") and outputs.loss is not None:
        print(f"   Loss: {outputs.loss.item():.6f}")

    if hasattr(outputs, "logits"):
        logits = outputs.logits
        print(f"   Logits: {logits.shape} ({logits.dtype})")
        print(
            f"   Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]"
        )

    # Check for additional loss components if using object detection loss
    if hasattr(outputs, "loss_dict"):
        print(f"   Loss components: {outputs.loss_dict}")

    return outputs


def test_generation(model, tokenizer, batch, config):
    """Test generation capabilities"""
    print(f"\n🎯 Testing generation...")

    # Prepare generation inputs (remove labels)
    generation_inputs = {k: v for k, v in batch.items() if k != "labels"}

    # Move generation inputs to same device as model
    device = next(model.parameters()).device
    generation_inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in generation_inputs.items()
    }

    # Generation parameters
    gen_params = {
        "max_new_tokens": 50,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": False,
    }

    print(f"🎯 GENERATION PARAMETERS:")
    for key, value in gen_params.items():
        print(f"   {key}: {value}")

    # Generate
    print(f"\n🔄 Running generation...")

    with torch.no_grad():
        outputs = model.generate(**generation_inputs, **gen_params)

    print(f"✅ Generation completed successfully!")

    # Analyze generated output
    input_length = generation_inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_length:]

    print(f"\n📄 GENERATION ANALYSIS:")
    print(f"   Input length: {input_length} tokens")
    print(f"   Generated: {generated_ids.shape[1]} tokens")

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    generated_clean = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\n📝 GENERATED OUTPUT (with special tokens):")
    print(f"   {repr(generated_text)}")

    print(f"\n📝 GENERATED OUTPUT (clean):")
    print(f"   {repr(generated_clean)}")

    # Check for object detection format
    from src.tokens import SpecialTokens

    tokens = SpecialTokens()

    has_object_refs = tokens.OBJECT_REF_START in generated_text
    has_boxes = tokens.BOX_START in generated_text

    print(f"\n🎯 OBJECT DETECTION FORMAT CHECK:")
    print(f"   Has object references: {'✅' if has_object_refs else '❌'}")
    print(f"   Has bounding boxes: {'✅' if has_boxes else '❌'}")

    return outputs, generated_text


def main():
    """Main execution function"""
    print("🚀 QWEN2.5-VL COMPREHENSIVE DEMO - LATEST ARCHITECTURE")
    print("=" * 70)

    # Step 1: Setup environment
    setup_environment()

    # Step 2: Load configuration
    config = load_config("base", overrides=["test_samples=2"])

    # Step 3: Load tokenizer and processor
    tokenizer, processor = load_tokenizer_and_processor(config)

    # Step 4: Create dataset
    dataset = create_dataset(config, tokenizer, processor)

    # Step 5: Analyze sample structure
    raw_sample = analyze_sample_structure(dataset, sample_idx=0)

    # Step 6: Test data processing
    processed_sample = test_data_processing(dataset, sample_idx=0)

    # Step 7: Create data collator and batch
    collator, batch = create_data_collator_and_batch(config, tokenizer, dataset, [0])

    # Step 8: Load model
    model_wrapper, model = load_model(config, tokenizer)

    # Step 9: Test forward pass
    forward_outputs = test_forward_pass(model, batch, config)

    # Step 10: Test generation
    generation_outputs, generated_text = test_generation(
        model, tokenizer, batch, config
    )

    print(f"\n" + "=" * 70)
    print("🎉 COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
    print("\n📋 PIPELINE VALIDATION:")
    print("   ✅ Configuration loading with ConfigManager")
    print("   ✅ Data preparation with ChatProcessor")
    print("   ✅ Model initialization with ModelWrapper")
    print("   ✅ Forward pass with loss calculation")
    print("   ✅ Generation with special token handling")
    print("   ✅ Object detection format validation")
    print("\n📊 KEY METRICS:")
    print(
        f"   • Model: {config.model_size} ({sum(p.numel() for p in model.parameters()):,} parameters)"
    )
    print(f"   • Dataset: {len(dataset)} samples")
    print(f"   • Sequence length: {processed_sample['input_ids'].shape[0]} tokens")
    print(
        f"   • Training tokens: {len(processed_sample['labels'][processed_sample['labels'] != -100])}"
    )
    print(
        f"   • Vision tokens: {(processed_sample['input_ids'] == tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum().item()}"
    )
    print(f"   • Forward pass loss: {forward_outputs.loss.item():.6f}")
    print("\n🔧 NEXT STEPS:")
    print("   1. Run full training with: python scripts/train.py --config base")
    print("   2. Monitor training with: tensorboard --logdir tb_detection")
    print("   3. Evaluate results with: python scripts/evaluate.py")
    print("\n   🚀 Ready for full training pipeline!")


if __name__ == "__main__":
    main()
