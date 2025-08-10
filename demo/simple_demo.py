#!/usr/bin/env python3
"""
Simple BBU Training Demo - Manual Step-by-Step Debugging

This script manually walks through every step of the training process without using
the BBUTrainer or Transformers Trainer, making it easy to debug with breakpoints.

Usage:
    cd /data4/Qwen2.5-VL-main
    python notebook/simple_demo.py
"""

import sys
from pathlib import Path

import torch


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from transformers import AutoProcessor, AutoTokenizer

from src.config import config, init_config
from src.data import BBUDataset, StandardDataCollator
from src.detection_loss import DetectionLoss
from src.logger_utils import configure_global_logging, get_training_logger
from src.models.patches import apply_comprehensive_qwen25_fixes, verify_qwen25_patches
from src.models.wrapper import Qwen25VLWithDetection


def setup_environment():
    """Setup environment for single GPU debugging."""
    print("🌍 Setting up environment for debugging...")

    # Set single GPU
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    # Disable distributed training
    import os

    os.environ.pop("RANK", None)
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("LOCAL_RANK", None)

    print(f"✅ Using device: {device}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(
        f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
    )

    return device


def load_configuration():
    """Load configuration from base_flat.yaml."""
    print("\n📄 Loading configuration...")

    # Initialize config
    config_path = "configs/base_flat.yaml"
    init_config(config_path, overrides={})

    print(f"✅ Config loaded from: {config_path}")
    print(f"   Model path: {config.model_path}")
    print(
        f"   Learning rates: vision={config.vision_lr}, mlp={config.mlp_lr}, llm={config.llm_lr}, detection={config.detection_lr}"
    )
    print(f"   Detection enabled: {config.detection_enabled}")
    print(f"   Detection loss weight: {config.detection_loss_weight}")
    print(f"   Train data: {config.train_data_path}")
    print(f"   Val data: {config.val_data_path}")

    return config


def setup_logging():
    """Setup logging for debugging."""
    print("\n📊 Setting up logging...")

    configure_global_logging(
        log_dir="logs/debug",
        log_level="DEBUG",
        verbose=True,
        is_training=True,
        console_level="INFO",
    )

    logger = get_training_logger()
    logger.info("🔧 Debug logging initialized")

    return logger


def load_model_and_tokenizer(config, device):
    """Load model, tokenizer, and image processor using unified loading mechanism."""
    print("\n🔧 Loading model and tokenizer...")

    # Apply patches first
    print("   Applying Qwen2.5-VL patches...")
    if not apply_comprehensive_qwen25_fixes():
        raise RuntimeError("Failed to apply Qwen2.5-VL fixes")

    # Load tokenizer from the model path
    print(f"   Loading tokenizer from: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    # Load processor from the model path
    print(f"   Loading processor from: {config.model_path}")
    processor = AutoProcessor.from_pretrained(config.model_path)
    image_processor = processor.image_processor

    # Configure image processor with data conversion constraints
    try:
        from data_conversion.vision_process import MAX_PIXELS, MIN_PIXELS

        image_processor.min_pixels = MIN_PIXELS
        image_processor.max_pixels = MAX_PIXELS
        print(
            f"   ✅ Image processor configured: min_pixels={MIN_PIXELS}, max_pixels={MAX_PIXELS}"
        )
    except ImportError:
        print("   ⚠️  Using default image processor pixel constraints")

    # Use unified loading mechanism - automatically detects checkpoint type
    print(f"   Creating model with unified loading mechanism...")
    model = Qwen25VLWithDetection.from_pretrained(
        model_path=config.model_path,
        num_queries=config.detection_num_queries,
        max_caption_length=config.detection_max_caption_length,
        tokenizer=tokenizer,
    )

    # Move to device
    print(f"   Moving model to device: {device}")
    model = model.to(device)

    # Verify patches
    print("   Verifying patches...")
    if not verify_qwen25_patches():
        raise RuntimeError("Patch verification failed")

    # Disable cache for training
    model.base_model.config.use_cache = False

    # Set training mode
    model.train()

    print("✅ Model and tokenizer loaded successfully")
    print(
        f"   Base model parameters: {sum(p.numel() for p in model.base_model.parameters()):,}"
    )
    print(
        f"   Detection head parameters: {sum(p.numel() for p in model.detection_head.parameters()):,}"
    )

    return model, tokenizer, image_processor


def setup_datasets(config, tokenizer, image_processor):
    """Setup train and validation datasets."""
    print("\n📚 Setting up datasets...")

    # Create train dataset
    print(f"   Loading train dataset: {config.train_data_path}")
    train_dataset = BBUDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_path=config.train_data_path,
    )

    # Create validation dataset
    print(f"   Loading validation dataset: {config.val_data_path}")
    val_dataset = BBUDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_path=config.val_data_path,
    )

    # Create data collator
    print("   Creating data collator...")
    data_collator = StandardDataCollator(tokenizer=tokenizer)

    print("✅ Datasets created successfully")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset, data_collator


def setup_optimizer(config, model):
    """Setup optimizer - simplified for debugging."""
    print("\n⚙️ Setting up optimizer...")

    # SIMPLIFIED: Just train detection head for debugging
    trainable_params = []

    # Set all parameters to not require grad first
    for param in model.parameters():
        param.requires_grad = False

        # Only train detection head for debugging
    for param in model.detection_head.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    print(f"   Training only detection head: {len(trainable_params)} parameters")
    print(f"   Learning rate: {config.detection_lr}")

    # Debug: Check parameter devices
    devices = set()
    dtypes = set()
    for param in trainable_params:
        devices.add(param.device)
        dtypes.add(param.dtype)
    print(f"   Parameter devices: {devices}")
    print(f"   Parameter dtypes: {dtypes}")

    # Create optimizer with specific settings for mixed precision
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.detection_lr,
        weight_decay=config.weight_decay,
        eps=1e-8,
        amsgrad=False,
    )

    print(f"✅ Optimizer created successfully")

    return optimizer


def setup_detection_loss(config, tokenizer):
    """Setup detection loss function."""
    print("\n🎯 Setting up detection loss...")

    detection_loss = DetectionLoss(
        bbox_weight=config.detection_bbox_weight,
        giou_weight=config.detection_giou_weight,
        objectness_weight=config.detection_objectness_weight,
        caption_weight=config.detection_caption_weight,
        tokenizer=tokenizer,
    )

    print("✅ Detection loss initialized")
    print(f"   Bbox weight: {config.detection_bbox_weight}")
    print(f"   GIoU weight: {config.detection_giou_weight}")
    print(f"   Objectness weight: {config.detection_objectness_weight}")
    print(f"   Caption weight: {config.detection_caption_weight}")

    return detection_loss


def get_single_batch(dataset, data_collator, batch_size=1):
    """Get a single batch for debugging."""
    print(f"\n📦 Creating single batch (size={batch_size})...")

    # Get samples
    samples = []
    for i in range(min(batch_size, len(dataset))):
        sample = dataset[i]
        samples.append(sample)
        print(f"   Sample {i}: input_ids shape={sample['input_ids'].shape}")
        if "pixel_values" in sample and sample["pixel_values"] is not None:
            print(f"   Sample {i}: pixel_values shape={sample['pixel_values'].shape}")
        if "ground_truth_objects" in sample:
            print(
                f"   Sample {i}: ground_truth objects={len(sample['ground_truth_objects'])}"
            )

    # Collate batch
    batch = data_collator(samples)

    print("✅ Batch created successfully")
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    if "pixel_values" in batch:
        print(f"   pixel_values shape: {batch['pixel_values'].shape}")
    if "ground_truth_objects" in batch:
        print(f"   ground_truth_objects: {len(batch['ground_truth_objects'])} samples")
        for i, gt_objects in enumerate(batch["ground_truth_objects"]):
            print(f"     Sample {i}: {len(gt_objects)} objects")

    return batch


def manual_forward_pass(model, batch, device):
    """Manually perform forward pass step by step."""
    print("\n🔄 Manual Forward Pass...")

    # Move batch to device
    print("   Moving batch to device...")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    # Extract ground truth objects
    ground_truth_objects = batch.pop("ground_truth_objects", [])

    print(f"   Ground truth objects extracted: {len(ground_truth_objects)} samples")

    # Prepare inputs for base model (clean inputs)
    model_inputs = {
        "input_ids": batch["input_ids"],
        "labels": batch["labels"],
        "attention_mask": batch["attention_mask"],
        "output_hidden_states": True,
    }

    # Add image inputs if present
    if "pixel_values" in batch:
        model_inputs["pixel_values"] = batch["pixel_values"]
    if "image_grid_thw" in batch:
        model_inputs["image_grid_thw"] = batch["image_grid_thw"]

    print("   Performing base model forward pass...")
    print(f"   Input shapes: input_ids={model_inputs['input_ids'].shape}")
    if "pixel_values" in model_inputs:
        print(f"                 pixel_values={model_inputs['pixel_values'].shape}")

    # Base model forward pass (disable mixed precision for debugging)
    with torch.cuda.amp.autocast(enabled=False):
        outputs = model.base_model(**model_inputs)

    print("   ✅ Base model forward completed")
    print(f"   LM loss: {outputs.loss.item():.6f}")
    print(f"   Hidden states shape: {outputs.hidden_states[-1].shape}")

    # Detection head forward pass
    detection_outputs = None

    if config.detection_enabled and ground_truth_objects:
        print("   Performing detection head forward pass...")

        hidden_states = outputs.hidden_states[-1]  # Final layer
        attention_mask = model_inputs.get("attention_mask")

        # Detection head forward
        detection_outputs = model.detection_head(
            hidden_states,
            attention_mask,
            ground_truth_objects,
            training=model.training,
        )

        print("   ✅ Detection head forward completed")
        print(f"   Prediction shapes:")
        print(f"     pred_boxes: {detection_outputs['pred_boxes'].shape}")
        print(f"     pred_objectness: {detection_outputs['pred_objectness'].shape}")
        print(f"     caption_logits: {detection_outputs['caption_logits'].shape}")

        # Show sample predictions
        sample_boxes = detection_outputs["pred_boxes"][0, :3].detach().cpu()
        sample_objectness = (
            torch.sigmoid(detection_outputs["pred_objectness"][0, :3]).detach().cpu()
        )
        print(f"   Sample predictions (first 3 queries):")
        for i in range(3):
            print(
                f"     Query {i}: box={sample_boxes[i].tolist()}, objectness={sample_objectness[i].item():.3f}"
            )

    return outputs, detection_outputs, ground_truth_objects


def generate_captions_autoregressive(
    model,
    hidden_states,
    attention_mask,
    tokenizer,
    num_queries=10,
    max_length=32,
    temperature=1.0,
):
    """
    Generate captions autoregressively using the detection head.

    Args:
        model: The Qwen25VLWithDetection model
        hidden_states: (B, seq_len, hidden_size) - Hidden states from base model
        attention_mask: (B, seq_len) - Attention mask
        tokenizer: Tokenizer for decoding
        num_queries: Number of detection queries to generate captions for
        max_length: Maximum caption length
        temperature: Sampling temperature (1.0 = no change, <1.0 = more focused)

    Returns:
        List of generated captions with metadata
    """
    model.eval()
    batch_size = hidden_states.shape[0]
    device = hidden_states.device

    generated_captions = []

    with torch.no_grad():
        # Get detection predictions (boxes and objectness)
        detection_outputs = model.detection_head(
            hidden_states, attention_mask, ground_truth_objects=None, training=False
        )

        pred_boxes = detection_outputs["pred_boxes"]  # (B, N, 4)
        pred_objectness = detection_outputs["pred_objectness"]  # (B, N)

        # Get objectness scores and sort by confidence
        objectness_scores = torch.sigmoid(pred_objectness)  # (B, N)

        for b in range(batch_size):
            batch_captions = []

            # Sort queries by confidence
            sorted_indices = torch.argsort(objectness_scores[b], descending=True)
            top_indices = sorted_indices[:num_queries]

            for rank, query_idx in enumerate(top_indices):
                query_idx_val = query_idx.item()
                confidence = objectness_scores[b, query_idx].item()
                box = pred_boxes[b, query_idx].detach().cpu().tolist()

                # Generate caption autoregressively for this query
                generated_caption = generate_single_caption_autoregressive(
                    model,
                    hidden_states[b : b + 1],
                    attention_mask[b : b + 1],
                    query_idx_val,
                    tokenizer,
                    max_length,
                    temperature,
                )

                batch_captions.append(
                    {
                        "rank": rank + 1,
                        "query_idx": query_idx_val,
                        "confidence": confidence,
                        "box": box,
                        "caption": generated_caption["text"],
                        "token_count": generated_caption["token_count"],
                        "generation_steps": generated_caption["steps"],
                    }
                )

            generated_captions.append(batch_captions)

    return generated_captions


def generate_single_caption_autoregressive(
    model,
    hidden_states,
    attention_mask,
    query_idx,
    tokenizer,
    max_length=32,
    temperature=1.0,
):
    """
    Generate a single caption autoregressively for a specific query using the detection head's method.

    Args:
        model: The Qwen25VLWithDetection model
        hidden_states: (1, seq_len, hidden_size) - Single sample hidden states
        attention_mask: (1, seq_len) - Single sample attention mask
        query_idx: Index of the query to generate caption for
        tokenizer: Tokenizer for decoding
        max_length: Maximum caption length
        temperature: Sampling temperature

    Returns:
        Dict with generated caption and metadata
    """
    device = hidden_states.device

    # Get object features from detection head
    with torch.no_grad():
        # Get detection predictions to extract object features
        detection_outputs = model.detection_head(
            hidden_states, attention_mask, ground_truth_objects=None, training=False
        )

        object_features = detection_outputs["object_features"]  # (1, N, D)

        # Use the detection head's new method for proper autoregressive generation
        generation_result = model.detection_head.generate_single_query_caption(
            object_features, query_idx, max_length=max_length
        )

        generated_tokens = generation_result["tokens"]
        generation_logits = generation_result["logits"]

    # Create detailed generation steps
    generation_steps = []

    for step, (token_id, logits) in enumerate(
        zip(generated_tokens[1:], generation_logits)
    ):  # Skip start token
        # Apply temperature if specified
        if temperature != 1.0:
            logits = logits / temperature

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get top 5 alternatives
        top_5_probs, top_5_indices = torch.topk(probs, 5)

        step_info = {
            "step": step,
            "next_token_id": token_id,
            "next_token_prob": probs[token_id].item(),
            "top_5_tokens": [],
        }

        # Decode top 5 tokens
        for idx, prob in zip(top_5_indices, top_5_probs):
            try:
                token_text = tokenizer.decode([idx.item()], skip_special_tokens=False)
                step_info["top_5_tokens"].append((token_text, prob.item()))
            except:
                step_info["top_5_tokens"].append((f"[ID:{idx.item()}]", prob.item()))

        generation_steps.append(step_info)

    # Decode the generated sequence
    try:
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_text = generated_text.strip()
        if not generated_text:
            generated_text = "[empty]"
    except Exception as e:
        generated_text = f"[decode_error: {str(e)[:50]}]"

    return {
        "text": generated_text,
        "token_count": len(generated_tokens),
        "tokens": generated_tokens,
        "steps": generation_steps,
    }


def detokenize_captions(caption_logits, tokenizer, top_k=5):
    """
    DEPRECATED: Simple detokenizer that takes argmax at each position.
    Use generate_captions_autoregressive() instead for proper generation.

    This function is kept for comparison purposes only.
    """
    print(
        "⚠️  WARNING: Using deprecated detokenize_captions. Consider using generate_captions_autoregressive() instead."
    )

    batch_size, num_queries, max_len, vocab_size = caption_logits.shape

    # Get predicted token IDs (greedy decoding)
    pred_token_ids = torch.argmax(caption_logits, dim=-1)  # (B, N, max_len)

    decoded_captions = []

    for b in range(batch_size):
        batch_captions = []

        for q in range(min(top_k, num_queries)):
            # Get token sequence for this query
            token_sequence = pred_token_ids[b, q]  # (max_len,)

            # Convert to CPU and remove padding/special tokens
            token_list = token_sequence.detach().cpu().tolist()

            # Find end token or truncate at reasonable length
            if tokenizer.eos_token_id in token_list:
                end_idx = token_list.index(tokenizer.eos_token_id)
                token_list = token_list[:end_idx]

            # Remove padding tokens
            if tokenizer.pad_token_id is not None:
                token_list = [t for t in token_list if t != tokenizer.pad_token_id]

            # Decode to text
            try:
                decoded_text = tokenizer.decode(token_list, skip_special_tokens=True)
                # Clean up the text
                decoded_text = decoded_text.strip()
                if not decoded_text:
                    decoded_text = "[empty]"
            except Exception as e:
                decoded_text = f"[decode_error: {str(e)[:50]}]"

            batch_captions.append(
                {
                    "query_idx": q,
                    "caption": decoded_text,
                    "token_count": len(token_list),
                }
            )

        decoded_captions.append(batch_captions)

    return decoded_captions


def show_autoregressive_generation_analysis(
    model, hidden_states, attention_mask, tokenizer, top_k=5
):
    """
    Show detailed autoregressive generation analysis for top queries.

    Args:
        model: The Qwen25VLWithDetection model
        hidden_states: (B, seq_len, hidden_size) - Hidden states from base model
        attention_mask: (B, seq_len) - Attention mask
        tokenizer: Tokenizer for decoding
        top_k: Number of top queries to analyze
    """
    print(f"\n🤖 AUTOREGRESSIVE GENERATION ANALYSIS (Top {top_k} queries)")
    print("=" * 80)

    # Generate captions autoregressively
    generated_captions = generate_captions_autoregressive(
        model,
        hidden_states,
        attention_mask,
        tokenizer,
        num_queries=top_k,
        max_length=20,
    )

    for batch_idx, batch_captions in enumerate(generated_captions):
        print(f"\n📦 Batch {batch_idx + 1}:")

        for caption_info in batch_captions:
            print(
                f"\n   🎯 Query {caption_info['query_idx']} (Rank #{caption_info['rank']}):"
            )
            print(f"      Confidence: {caption_info['confidence']:.3f}")
            print(f"      Box: {caption_info['box']}")
            print(
                f'      Generated: "{caption_info["caption"]}" ({caption_info["token_count"]} tokens)'
            )

            # Show generation steps for top 3 queries
            if caption_info["rank"] <= 3:
                print(f"      🔄 Generation Steps:")
                for step_info in caption_info["generation_steps"][
                    :5
                ]:  # Show first 5 steps
                    step = step_info["step"]
                    token_id = step_info["next_token_id"]
                    prob = step_info["next_token_prob"]
                    try:
                        token_text = tokenizer.decode([token_id])
                        print(
                            f"         Step {step}: '{token_text}' (ID:{token_id}, P:{prob:.3f})"
                        )
                    except:
                        print(f"         Step {step}: ID:{token_id} (P:{prob:.3f})")

                    # Show top alternatives
                    print(
                        f"           Top alternatives: {step_info['top_5_tokens'][:3]}"
                    )
            print()


def show_top_predictions_with_captions(detection_outputs, tokenizer, top_k=10):
    """
    DEPRECATED: Show top predictions with simple argmax decoding.
    Use show_autoregressive_generation_analysis() instead.
    """
    print("⚠️  WARNING: Using deprecated show_top_predictions_with_captions.")
    print("    Consider using show_autoregressive_generation_analysis() instead.")

    pred_boxes = detection_outputs["pred_boxes"]  # (B, N, 4)
    pred_objectness = detection_outputs["pred_objectness"]  # (B, N)
    caption_logits = detection_outputs["caption_logits"]  # (B, N, max_len, vocab_size)

    batch_size = pred_boxes.shape[0]

    print(f"\n🏆 TOP {top_k} PREDICTIONS (ranked by confidence):")
    print("=" * 80)

    for b in range(batch_size):
        print(f"\n📦 Batch {b + 1}:")

        # Get objectness scores and sort by confidence
        objectness_scores = torch.sigmoid(pred_objectness[b]).detach().cpu()  # (N,)
        sorted_indices = torch.argsort(objectness_scores, descending=True)

        # Get top-k indices
        top_indices = sorted_indices[:top_k]

        # Detokenize captions for top predictions
        top_caption_logits = caption_logits[b][top_indices].unsqueeze(
            0
        )  # (1, top_k, max_len, vocab_size)
        decoded_captions = detokenize_captions(
            top_caption_logits, tokenizer, top_k=top_k
        )

        # Display results
        for rank, idx in enumerate(top_indices):
            idx_val = idx.item()
            confidence = objectness_scores[idx].item()
            box = pred_boxes[b, idx].detach().cpu().tolist()
            caption_info = decoded_captions[0][rank]

            print(
                f"   #{rank + 1:2d} | Query {idx_val:2d} | Conf: {confidence:.3f} | Box: {box}"
            )
            print(
                f'        Caption: "{caption_info["caption"]}" ({caption_info["token_count"]} tokens)'
            )

            # Show some raw logits for debugging
            if rank < 3:  # Only for top 3
                raw_logits = caption_logits[b, idx, :5].detach().cpu()  # First 5 tokens
                top_tokens = torch.argmax(raw_logits, dim=-1).tolist()
                try:
                    raw_text = tokenizer.decode(top_tokens, skip_special_tokens=True)
                    print(f'        Raw (first 5): "{raw_text}"')
                except:
                    print(f"        Raw tokens: {top_tokens}")
            print()


def analyze_caption_predictions(detection_outputs, ground_truth_objects, tokenizer):
    """
    Analyze caption predictions vs ground truth using both methods.
    """
    print(f"\n🔍 CAPTION ANALYSIS:")
    print("=" * 50)

    # Show ground truth captions
    if ground_truth_objects and len(ground_truth_objects[0]) > 0:
        print(f"📋 Ground Truth Objects ({len(ground_truth_objects[0])} objects):")
        for i, gt_obj in enumerate(ground_truth_objects[0][:5]):  # Show first 5
            gt_caption = gt_obj["desc"]
            gt_box = gt_obj["box"]
            print(f'   GT {i + 1}: "{gt_caption}" | Box: {gt_box}')
        print()

    # Show DEPRECATED argmax predictions for comparison
    print("📊 ARGMAX DECODING (for comparison):")
    show_top_predictions_with_captions(detection_outputs, tokenizer, top_k=5)

    # Caption quality analysis
    caption_logits = detection_outputs["caption_logits"]
    pred_objectness = detection_outputs["pred_objectness"]

    # Get confidence statistics
    objectness_scores = torch.sigmoid(pred_objectness[0]).detach().cpu()
    max_conf = objectness_scores.max().item()
    mean_conf = objectness_scores.mean().item()
    min_conf = objectness_scores.min().item()

    print(f"📊 Confidence Statistics:")
    print(f"   Max confidence: {max_conf:.3f}")
    print(f"   Mean confidence: {mean_conf:.3f}")
    print(f"   Min confidence: {min_conf:.3f}")

    # Caption length analysis
    pred_token_ids = torch.argmax(caption_logits[0], dim=-1)  # (N, max_len)
    caption_lengths = []

    for q in range(min(10, pred_token_ids.shape[0])):
        tokens = pred_token_ids[q].detach().cpu().tolist()
        # Count non-padding tokens
        if tokenizer.pad_token_id is not None:
            length = len([t for t in tokens if t != tokenizer.pad_token_id])
        else:
            length = len(tokens)
        caption_lengths.append(length)

    if caption_lengths:
        avg_length = sum(caption_lengths) / len(caption_lengths)
        print(f"   Average caption length: {avg_length:.1f} tokens")
        print(
            f"   Caption length range: {min(caption_lengths)}-{max(caption_lengths)} tokens"
        )


def manual_loss_computation(
    outputs, detection_outputs, ground_truth_objects, detection_loss_fn
):
    """Manually compute loss step by step."""
    print("\n📊 Manual Loss Computation...")

    # Language modeling loss
    lm_loss = outputs.loss
    print(f"   Language modeling loss: {lm_loss.item():.6f}")

    # Detection loss
    detection_loss_value = 0.0
    if (
        config.detection_enabled
        and detection_outputs is not None
        and ground_truth_objects
    ):
        print("   Computing detection loss...")

        # Check if we have valid ground truth
        valid_gt = any(len(gt) > 0 for gt in ground_truth_objects)
        if valid_gt:
            detection_loss_value = detection_loss_fn(
                detection_outputs, ground_truth_objects
            )
            print(f"   Detection loss: {detection_loss_value.item():.6f}")
        else:
            print("   No valid ground truth objects - skipping detection loss")

    # Combined loss
    total_loss = lm_loss + config.detection_loss_weight * detection_loss_value
    print(f"   Detection loss weight: {config.detection_loss_weight}")
    print(
        f"   Weighted detection loss: {(config.detection_loss_weight * detection_loss_value):.6f}"
    )
    print(f"   Total loss: {total_loss.item():.6f}")

    return total_loss, lm_loss, detection_loss_value


def manual_loss_computation_detailed(
    outputs, detection_outputs, ground_truth_objects, detection_loss_fn
):
    """Manually compute loss with detailed breakdown of all components."""
    print("\n📊 Detailed Loss Computation...")

    # Language modeling loss
    lm_loss = outputs.loss
    print(f"   Language modeling loss: {lm_loss.item():.6f}")

    # Initialize detection loss components
    detection_loss_value = 0.0
    loss_breakdown = {
        "bbox_loss": 0.0,
        "caption_loss": 0.0,
        "objectness_loss": 0.0,
    }

    # Detection loss with detailed breakdown
    if (
        config.detection_enabled
        and detection_outputs is not None
        and ground_truth_objects
    ):
        print("   Computing detection loss with breakdown...")

        # Check if we have valid ground truth
        valid_gt = any(len(gt) > 0 for gt in ground_truth_objects)
        if valid_gt:
            # Get detailed loss breakdown from detection loss function
            detection_loss_value = detection_loss_fn(
                detection_outputs, ground_truth_objects
            )

            # Access the detailed loss components from the detection loss function
            # Note: This assumes the detection loss function stores these internally
            if hasattr(detection_loss_fn, "last_bbox_loss"):
                loss_breakdown["bbox_loss"] = detection_loss_fn.last_bbox_loss
            if hasattr(detection_loss_fn, "last_caption_loss"):
                loss_breakdown["caption_loss"] = detection_loss_fn.last_caption_loss
            if hasattr(detection_loss_fn, "last_objectness_loss"):
                loss_breakdown["objectness_loss"] = (
                    detection_loss_fn.last_objectness_loss
                )

            print(f"   Detection loss breakdown:")
            print(f"     Total detection: {detection_loss_value.item():.6f}")
            print(f"     - Bbox loss: {loss_breakdown['bbox_loss']:.6f}")
            print(f"     - Caption loss: {loss_breakdown['caption_loss']:.6f}")
            print(f"     - Objectness loss: {loss_breakdown['objectness_loss']:.6f}")
        else:
            print("   No valid ground truth objects - skipping detection loss")

    # Combined loss
    total_loss = lm_loss + config.detection_loss_weight * detection_loss_value
    print(f"   Detection loss weight: {config.detection_loss_weight}")
    print(
        f"   Weighted detection loss: {(config.detection_loss_weight * detection_loss_value):.6f}"
    )
    print(f"   Total loss: {total_loss.item():.6f}")

    return total_loss, lm_loss, detection_loss_value, loss_breakdown


def manual_backward_pass(total_loss, optimizer):
    """Manually perform backward pass step by step."""
    print("\n⬅️ Manual Backward Pass...")

    # Zero gradients
    print("   Zeroing gradients...")
    optimizer.zero_grad()

    # Backward pass
    print("   Computing gradients...")
    total_loss.backward()

    # Check gradients
    print("   Checking gradients...")
    total_grad_norm = 0.0
    param_count = 0

    # Get all parameters from optimizer
    all_params = []
    for group in optimizer.param_groups:
        all_params.extend(group["params"])

    for param in all_params:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
            param_count += 1

    if param_count > 0:
        total_grad_norm = total_grad_norm ** (1.0 / 2)
        print(f"   Total gradient norm: {total_grad_norm:.6f}")

    # Gradient clipping
    if config.max_grad_norm > 0:
        print(f"   Applying gradient clipping (max_norm={config.max_grad_norm})...")
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, config.max_grad_norm)
        print(f"   Gradient norm before clipping: {grad_norm:.6f}")

    # Optimizer step
    # Debug: Check parameter devices before optimizer step
    print("   Checking parameter devices before optimizer step...")
    devices = set()
    dtypes = set()
    for param in all_params:
        devices.add(param.device)
        dtypes.add(param.dtype)
        if param.grad is not None:
            devices.add(param.grad.device)
            dtypes.add(param.grad.dtype)
    print(f"   Parameter/gradient devices: {devices}")
    print(f"   Parameter/gradient dtypes: {dtypes}")

    print("   Performing optimizer step...")
    optimizer.step()

    print("   ✅ Backward pass completed")


def debug_ground_truth_objects(ground_truth_objects):
    """Debug ground truth objects structure."""
    print("\n🔍 Debugging Ground Truth Objects...")

    for batch_idx, gt_objects in enumerate(ground_truth_objects):
        print(f"   Batch {batch_idx}: {len(gt_objects)} objects")
        for obj_idx, obj in enumerate(gt_objects[:3]):  # Show first 3 objects
            print(f"     Object {obj_idx}: {obj}")


def create_dataloader(dataset, data_collator, batch_size=1, shuffle=True):
    """Create a simple dataloader for training."""
    print(f"\n📦 Creating dataloader (batch_size={batch_size}, shuffle={shuffle})...")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
        num_workers=0,  # Single process for debugging
        pin_memory=True,
    )

    print(f"✅ Dataloader created: {len(dataloader)} batches")
    return dataloader


def train_single_epoch(
    model, dataloader, optimizer, detection_loss_fn, device, epoch, config, tokenizer
):
    """Train for one epoch and track all metrics."""
    print(f"\n🏃 EPOCH {epoch + 1} TRAINING")
    print("=" * 50)

    model.train()

    # Metrics tracking
    epoch_metrics = {
        "total_loss": 0.0,
        "lm_loss": 0.0,
        "detection_loss": 0.0,
        "bbox_loss": 0.0,
        "caption_loss": 0.0,
        "objectness_loss": 0.0,
        "num_batches": 0,
        "num_samples": 0,
    }

    for batch_idx, batch in enumerate(dataloader):
        print(f"\n📦 Batch {batch_idx + 1}/{len(dataloader)}")

        # Forward pass
        outputs, detection_outputs, ground_truth_objects = manual_forward_pass(
            model, batch, device
        )

        # Loss computation with detailed breakdown
        total_loss, lm_loss, detection_loss_value, loss_breakdown = (
            manual_loss_computation_detailed(
                outputs, detection_outputs, ground_truth_objects, detection_loss_fn
            )
        )

        # Caption analysis for first batch of each epoch
        if batch_idx == 0:
            # Show autoregressive generation analysis
            hidden_states = outputs.hidden_states[-1]  # Final layer
            attention_mask = batch.get("attention_mask")
            show_autoregressive_generation_analysis(
                model, hidden_states, attention_mask, tokenizer, top_k=5
            )

            # Also show traditional analysis for comparison
            analyze_caption_predictions(
                detection_outputs, ground_truth_objects, tokenizer
            )

        # Backward pass
        manual_backward_pass(total_loss, optimizer)

        # Update metrics
        batch_size = batch["input_ids"].size(0)
        epoch_metrics["total_loss"] += total_loss.item() * batch_size
        epoch_metrics["lm_loss"] += lm_loss.item() * batch_size
        epoch_metrics["detection_loss"] += detection_loss_value * batch_size
        epoch_metrics["bbox_loss"] += loss_breakdown["bbox_loss"] * batch_size
        epoch_metrics["caption_loss"] += loss_breakdown["caption_loss"] * batch_size
        epoch_metrics["objectness_loss"] += (
            loss_breakdown["objectness_loss"] * batch_size
        )
        epoch_metrics["num_batches"] += 1
        epoch_metrics["num_samples"] += batch_size

        # Print batch metrics
        print(f"   Batch metrics:")
        print(f"     Total loss: {total_loss.item():.6f}")
        print(f"     LM loss: {lm_loss.item():.6f}")
        print(f"     Detection loss: {detection_loss_value:.6f}")
        print(f"     - Bbox: {loss_breakdown['bbox_loss']:.6f}")
        print(f"     - Caption: {loss_breakdown['caption_loss']:.6f}")
        print(f"     - Objectness: {loss_breakdown['objectness_loss']:.6f}")

        # Break after a few batches for debugging
        if batch_idx >= 4:  # Train on 5 batches per epoch for debugging
            print(f"   🛑 Stopping after {batch_idx + 1} batches for debugging")
            break

    # Calculate average metrics
    if epoch_metrics["num_samples"] > 0:
        for key in [
            "total_loss",
            "lm_loss",
            "detection_loss",
            "bbox_loss",
            "caption_loss",
            "objectness_loss",
        ]:
            epoch_metrics[key] /= epoch_metrics["num_samples"]

    return epoch_metrics


def evaluate_model(model, dataloader, detection_loss_fn, device, config, tokenizer):
    """Evaluate model on validation set."""
    print(f"\n🔍 VALIDATION")
    print("=" * 30)

    model.eval()

    # Metrics tracking
    val_metrics = {
        "total_loss": 0.0,
        "lm_loss": 0.0,
        "detection_loss": 0.0,
        "bbox_loss": 0.0,
        "caption_loss": 0.0,
        "objectness_loss": 0.0,
        "num_batches": 0,
        "num_samples": 0,
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"   Val batch {batch_idx + 1}/{len(dataloader)}")

            # Forward pass
            outputs, detection_outputs, ground_truth_objects = manual_forward_pass(
                model, batch, device
            )

            # Loss computation
            total_loss, lm_loss, detection_loss_value, loss_breakdown = (
                manual_loss_computation_detailed(
                    outputs, detection_outputs, ground_truth_objects, detection_loss_fn
                )
            )

            # Caption analysis for first validation batch
            if batch_idx == 0:
                print(f"\n🔍 VALIDATION CAPTION ANALYSIS:")
                # Show autoregressive generation analysis
                hidden_states = outputs.hidden_states[-1]  # Final layer
                attention_mask = batch.get("attention_mask")
                show_autoregressive_generation_analysis(
                    model, hidden_states, attention_mask, tokenizer, top_k=3
                )

                # Also show traditional analysis for comparison
                analyze_caption_predictions(
                    detection_outputs, ground_truth_objects, tokenizer
                )

            # Update metrics
            batch_size = batch["input_ids"].size(0)
            val_metrics["total_loss"] += total_loss.item() * batch_size
            val_metrics["lm_loss"] += lm_loss.item() * batch_size
            val_metrics["detection_loss"] += detection_loss_value * batch_size
            val_metrics["bbox_loss"] += loss_breakdown["bbox_loss"] * batch_size
            val_metrics["caption_loss"] += loss_breakdown["caption_loss"] * batch_size
            val_metrics["objectness_loss"] += (
                loss_breakdown["objectness_loss"] * batch_size
            )
            val_metrics["num_batches"] += 1
            val_metrics["num_samples"] += batch_size

            # Break after a few batches for debugging
            if batch_idx >= 2:  # Validate on 3 batches for debugging
                print(f"   🛑 Stopping after {batch_idx + 1} validation batches")
                break

    # Calculate average metrics
    if val_metrics["num_samples"] > 0:
        for key in [
            "total_loss",
            "lm_loss",
            "detection_loss",
            "bbox_loss",
            "caption_loss",
            "objectness_loss",
        ]:
            val_metrics[key] /= val_metrics["num_samples"]

    return val_metrics


def print_epoch_summary(epoch, train_metrics, val_metrics):
    """Print comprehensive epoch summary."""
    print(f"\n📊 EPOCH {epoch + 1} SUMMARY")
    print("=" * 50)

    print("🏃 TRAINING METRICS:")
    print(f"   Total Loss:     {train_metrics['total_loss']:.6f}")
    print(f"   LM Loss:        {train_metrics['lm_loss']:.6f}")
    print(f"   Detection Loss: {train_metrics['detection_loss']:.6f}")
    print(f"     - Bbox:       {train_metrics['bbox_loss']:.6f}")
    print(f"     - Caption:    {train_metrics['caption_loss']:.6f}")
    print(f"     - Objectness: {train_metrics['objectness_loss']:.6f}")
    print(f"   Samples:        {train_metrics['num_samples']}")

    print("\n🔍 VALIDATION METRICS:")
    print(f"   Total Loss:     {val_metrics['total_loss']:.6f}")
    print(f"   LM Loss:        {val_metrics['lm_loss']:.6f}")
    print(f"   Detection Loss: {val_metrics['detection_loss']:.6f}")
    print(f"     - Bbox:       {val_metrics['bbox_loss']:.6f}")
    print(f"     - Caption:    {val_metrics['caption_loss']:.6f}")
    print(f"     - Objectness: {val_metrics['objectness_loss']:.6f}")
    print(f"   Samples:        {val_metrics['num_samples']}")


def test_autoregressive_generation():
    """
    Simple test function to demonstrate autoregressive caption generation
    without running the full training loop.
    """
    print("🧪 TESTING AUTOREGRESSIVE CAPTION GENERATION")
    print("=" * 60)

    # Setup
    device = setup_environment()
    config = load_configuration()
    logger = setup_logging()

    # Load model and data
    model, tokenizer, image_processor = load_model_and_tokenizer(config, device)
    train_dataset, val_dataset, data_collator = setup_datasets(
        config, tokenizer, image_processor
    )

    # Get a single batch
    batch = get_single_batch(train_dataset, data_collator, batch_size=1)

    # Forward pass
    outputs, detection_outputs, ground_truth_objects = manual_forward_pass(
        model, batch, device
    )

    # Show ground truth for context
    if ground_truth_objects and len(ground_truth_objects[0]) > 0:
        print(f"\n📋 Ground Truth Objects ({len(ground_truth_objects[0])} objects):")
        for i, gt_obj in enumerate(ground_truth_objects[0][:3]):  # Show first 3
            gt_caption = gt_obj["desc"]
            gt_box = gt_obj["box"]
            print(f'   GT {i + 1}: "{gt_caption}" | Box: {gt_box}')

    # Test autoregressive generation
    hidden_states = outputs.hidden_states[-1]  # Final layer
    attention_mask = batch.get("attention_mask")

    print(f"\n🤖 AUTOREGRESSIVE GENERATION TEST:")
    show_autoregressive_generation_analysis(
        model, hidden_states, attention_mask, tokenizer, top_k=8
    )

    # Compare with argmax decoding
    print(f"\n📊 COMPARISON WITH ARGMAX DECODING:")
    show_top_predictions_with_captions(detection_outputs, tokenizer, top_k=5)

    print(f"\n✅ Autoregressive generation test completed!")


def test_with_fresh_model():
    """
    Test autoregressive generation with a fresh, untrained model to compare.
    This will show the difference between trained and untrained detection heads.
    """
    print("🆕 TESTING WITH FRESH (UNTRAINED) MODEL")
    print("=" * 60)

    # Setup
    device = setup_environment()
    config = load_configuration()

    # Override to use base model instead of checkpoint
    original_model_path = config.model_path
    config.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"  # Use base model

    # Load fresh model
    model, tokenizer, image_processor = load_model_and_tokenizer(config, device)
    train_dataset, val_dataset, data_collator = setup_datasets(
        config, tokenizer, image_processor
    )

    # Get a single batch
    batch = get_single_batch(train_dataset, data_collator, batch_size=1)

    # Forward pass
    outputs, detection_outputs, ground_truth_objects = manual_forward_pass(
        model, batch, device
    )

    # Test autoregressive generation
    hidden_states = outputs.hidden_states[-1]
    attention_mask = batch.get("attention_mask")

    print(f"\n🤖 FRESH MODEL AUTOREGRESSIVE GENERATION:")
    show_autoregressive_generation_analysis(
        model, hidden_states, attention_mask, tokenizer, top_k=3
    )

    print(f"\n✅ Fresh model test completed!")

    # Restore original config
    config.model_path = original_model_path


def initialize_detection_head_with_better_weights(model, tokenizer):
    """
    Initialize detection head with better weights for meaningful generation.
    """
    print("🔧 Initializing detection head with better weights...")

    # Get some common tokens for initialization
    common_tokens = [
        "cable",
        "fiber",
        "label",
        "screw",
        "connection",
        "correct",
        "incorrect",
        "matches",
        "proper",
        "protection",
        "radius",
        "bend",
        "odf",
        "huawei",
        "bbu",
        "cabinet",
        "occupied",
        "aligned",
        "horizontally",
        "vertically",
    ]

    # Get token IDs
    token_ids = []
    for token in common_tokens:
        try:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                token_ids.extend(ids)
        except:
            pass

    if token_ids:
        print(f"   Found {len(token_ids)} relevant token IDs")

        # Initialize caption head bias to favor these tokens
        with torch.no_grad():
            caption_head = model.detection_head.caption_head
            if hasattr(caption_head, "__iter__"):
                # Sequential model - get the last linear layer
                for layer in reversed(caption_head):
                    if isinstance(layer, torch.nn.Linear):
                        # Set small positive bias for relevant tokens
                        for token_id in token_ids[:100]:  # Limit to first 100
                            if token_id < layer.bias.shape[0]:
                                layer.bias[token_id] += 0.1
                        break
            else:
                # Single linear layer
                if isinstance(caption_head, torch.nn.Linear):
                    for token_id in token_ids[:100]:
                        if token_id < caption_head.bias.shape[0]:
                            caption_head.bias[token_id] += 0.1

        print("   ✅ Detection head initialized with domain-specific token bias")
    else:
        print("   ⚠️  No relevant tokens found for initialization")


def test_with_better_initialization():
    """
    Test autoregressive generation with better initialization.
    """
    print("🎯 TESTING WITH BETTER INITIALIZATION")
    print("=" * 60)

    # Setup
    device = setup_environment()
    config = load_configuration()

    # Use base model for fresh start
    original_model_path = config.model_path
    config.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Load model and initialize better
    model, tokenizer, image_processor = load_model_and_tokenizer(config, device)
    initialize_detection_head_with_better_weights(model, tokenizer)

    train_dataset, val_dataset, data_collator = setup_datasets(
        config, tokenizer, image_processor
    )

    # Get a single batch
    batch = get_single_batch(train_dataset, data_collator, batch_size=1)

    # Forward pass
    outputs, detection_outputs, ground_truth_objects = manual_forward_pass(
        model, batch, device
    )

    # Test autoregressive generation
    hidden_states = outputs.hidden_states[-1]
    attention_mask = batch.get("attention_mask")

    print(f"\n🤖 BETTER INITIALIZED AUTOREGRESSIVE GENERATION:")
    show_autoregressive_generation_analysis(
        model, hidden_states, attention_mask, tokenizer, top_k=5
    )

    print(f"\n✅ Better initialization test completed!")

    # Restore original config
    config.model_path = original_model_path


def test_with_trained_model(checkpoint_path: str):
    """
    Test autoregressive generation with a trained model checkpoint.

    Args:
        checkpoint_path: Path to the trained model checkpoint directory
    """
    print("🎓 TESTING WITH TRAINED MODEL")
    print("=" * 60)
    print(f"Loading from: {checkpoint_path}")

    # Setup
    device = setup_environment()
    config = load_configuration()
    logger = setup_logging()

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(config.model_path)
    image_processor = processor.image_processor

    # Configure image processor
    try:
        from data_conversion.vision_process import MAX_PIXELS, MIN_PIXELS

        image_processor.min_pixels = MIN_PIXELS
        image_processor.max_pixels = MAX_PIXELS
        print(
            f"✅ Image processor configured: min_pixels={MIN_PIXELS}, max_pixels={MAX_PIXELS}"
        )
    except ImportError:
        print("⚠️  Using default image processor pixel constraints")

    # Load model with trained detection head
    import os

    from src.models.wrapper import Qwen25VLWithDetection

    detection_head_path = os.path.join(checkpoint_path, "detection_head.pth")

    if os.path.exists(detection_head_path):
        print(f"✅ Found detection head weights: {detection_head_path}")
        model = Qwen25VLWithDetection.from_pretrained_with_detection(
            base_model_path=checkpoint_path,  # Use checkpoint for base model too
            detection_head_path=detection_head_path,
            num_queries=config.detection_num_queries,
            max_caption_length=config.detection_max_caption_length,
            tokenizer=tokenizer,
        )
    else:
        print(f"⚠️  No detection head weights found, using base model only")
        model = Qwen25VLWithDetection(
            base_model_path=checkpoint_path,
            num_queries=config.detection_num_queries,
            max_caption_length=config.detection_max_caption_length,
            tokenizer=tokenizer,
        )

    # Move to device
    model = model.to(device)
    model.eval()

    # Setup datasets
    train_dataset, val_dataset, data_collator = setup_datasets(
        config, tokenizer, image_processor
    )

    # Get a single batch
    batch = get_single_batch(train_dataset, data_collator, batch_size=1)

    # Forward pass
    outputs, detection_outputs, ground_truth_objects = manual_forward_pass(
        model, batch, device
    )

    # Show ground truth for context
    if ground_truth_objects and len(ground_truth_objects[0]) > 0:
        print(f"\n📋 Ground Truth Objects ({len(ground_truth_objects[0])} objects):")
        for i, gt_obj in enumerate(ground_truth_objects[0][:3]):
            gt_caption = gt_obj["desc"]
            gt_box = gt_obj["box"]
            print(f'   GT {i + 1}: "{gt_caption}" | Box: {gt_box}')

    # Test autoregressive generation
    hidden_states = outputs.hidden_states[-1]
    attention_mask = batch.get("attention_mask")

    print(f"\n🤖 TRAINED MODEL AUTOREGRESSIVE GENERATION:")
    show_autoregressive_generation_analysis(
        model, hidden_states, attention_mask, tokenizer, top_k=8
    )

    print(f"\n✅ Trained model test completed!")


def main():
    """Main multi-epoch training function."""
    print("🚀 BBU Multi-Epoch Training Demo")
    print("=" * 50)

    # Setup
    device = setup_environment()
    config = load_configuration()
    logger = setup_logging()

    # Load model and data
    model, tokenizer, image_processor = load_model_and_tokenizer(config, device)
    train_dataset, val_dataset, data_collator = setup_datasets(
        config, tokenizer, image_processor
    )

    # Setup training components
    optimizer = setup_optimizer(config, model)
    detection_loss_fn = setup_detection_loss(config, tokenizer)

    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset, data_collator, batch_size=1, shuffle=True
    )
    val_dataloader = create_dataloader(
        val_dataset, data_collator, batch_size=1, shuffle=False
    )

    # Training configuration
    num_epochs = 3  # Small number for debugging

    print(f"\n🎯 TRAINING CONFIGURATION")
    print(f"   Epochs: {num_epochs}")
    print(f"   Train batches per epoch: 5 (for debugging)")
    print(f"   Val batches per epoch: 3 (for debugging)")
    print(f"   Total train samples: {len(train_dataset)}")
    print(f"   Total val samples: {len(val_dataset)}")

    # Training history
    training_history = {
        "train_total_loss": [],
        "train_lm_loss": [],
        "train_detection_loss": [],
        "val_total_loss": [],
        "val_lm_loss": [],
        "val_detection_loss": [],
    }

    # Training loop
    for epoch in range(num_epochs):
        # Training
        train_metrics = train_single_epoch(
            model,
            train_dataloader,
            optimizer,
            detection_loss_fn,
            device,
            epoch,
            config,
            tokenizer,
        )

        # Validation
        val_metrics = evaluate_model(
            model, val_dataloader, detection_loss_fn, device, config, tokenizer
        )

        # Print summary
        print_epoch_summary(epoch, train_metrics, val_metrics)

        # Update history
        training_history["train_total_loss"].append(train_metrics["total_loss"])
        training_history["train_lm_loss"].append(train_metrics["lm_loss"])
        training_history["train_detection_loss"].append(train_metrics["detection_loss"])
        training_history["val_total_loss"].append(val_metrics["total_loss"])
        training_history["val_lm_loss"].append(val_metrics["lm_loss"])
        training_history["val_detection_loss"].append(val_metrics["detection_loss"])

    # Final summary
    print("\n" + "=" * 50)
    print("🎉 MULTI-EPOCH TRAINING COMPLETED!")
    print("=" * 50)

    print("\n📈 TRAINING HISTORY:")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}:")
        print(
            f"  Train - Total: {training_history['train_total_loss'][epoch]:.6f}, "
            f"LM: {training_history['train_lm_loss'][epoch]:.6f}, "
            f"Detection: {training_history['train_detection_loss'][epoch]:.6f}"
        )
        print(
            f"  Val   - Total: {training_history['val_total_loss'][epoch]:.6f}, "
            f"LM: {training_history['val_lm_loss'][epoch]:.6f}, "
            f"Detection: {training_history['val_detection_loss'][epoch]:.6f}"
        )

    print("\n🎯 Key debugging points:")
    print("  - train_single_epoch(): Debug training loop")
    print("  - evaluate_model(): Debug validation")
    print("  - manual_forward_pass(): Debug model forward")
    print("  - manual_loss_computation_detailed(): Debug loss calculation")
    print("  - manual_backward_pass(): Debug gradient computation")


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Run autoregressive generation test with checkpoint
            test_autoregressive_generation()
        elif sys.argv[1] == "fresh":
            # Test with fresh untrained model
            test_with_fresh_model()
        elif sys.argv[1] == "init":
            # Test with better initialization
            test_with_better_initialization()
        elif sys.argv[1] == "trained":
            # Test with trained model checkpoint
            if len(sys.argv) > 2:
                checkpoint_path = sys.argv[2]
                test_with_trained_model(checkpoint_path)
            else:
                print("Usage: python simple_demo.py trained <checkpoint_path>")
                print("Example: python simple_demo.py trained /path/to/checkpoint")
        elif sys.argv[1] == "train":
            # Run full training
            main()
        else:
            print("Usage:")
            print(
                "  python simple_demo.py test                    # Test with checkpoint (current)"
            )
            print(
                "  python simple_demo.py fresh                   # Test with fresh untrained model"
            )
            print(
                "  python simple_demo.py init                    # Test with better initialization"
            )
            print(
                "  python simple_demo.py trained <checkpoint>    # Test with trained model"
            )
            print("  python simple_demo.py train                   # Run full training")
    else:
        # Default: run full training
        main()
