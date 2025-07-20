# Core Components API Reference

Quick reference for essential APIs in `src/core/`, `src/models/`, `src/training/`, and `src/utils/`.

## 🏭 Core Factories & Processors

### DataProcessor (`src/core/data_processor.py`)
```python
# Quick Setup
processor = DataProcessor(tokenizer, image_processor)
train_ds, eval_ds = processor.create_datasets()
collator = processor.create_data_collator()

# One-liner
train_ds, eval_ds, collator = DataProcessor.create_datasets_and_collator(tokenizer, image_processor)
```

**Key Config**: `train_data_path`, `val_data_path`, `coordinate_tokens_enabled`

### CheckpointManager (`src/core/checkpoint_manager.py`)
```python
# Save model safely
manager = CheckpointManager()
success = manager.save_model_safely(trainer, "/path/to/checkpoint")

# Validate checkpoint
is_valid = manager.validate_checkpoint("/path/to/checkpoint")
info = manager.get_checkpoint_info("/path/to/checkpoint")
```

**Key Features**: HF compatibility, coordinate token extensions, integrity validation

## 🤖 Model Components

### Qwen25VLWithDetection (`src/models/wrapper.py`)
```python
# Load with coordinate support
model = Qwen25VLWithDetection.from_pretrained(
    model_path="/path/to/model",
    tokenizer=tokenizer,
    coordinate_config=coord_config
)

# Forward pass
outputs = model(**inputs)  # Auto-detects coordinate vs standard mode
text = model.generate(**kwargs)
```

**Key Features**: Auto vocabulary extension, coordinate loss computation, pretrained weight preservation

## 🎯 Training Orchestration

### TrainingCoordinator (`src/training/training_coordinator.py`)
```python
# Setup training
coordinator = TrainingCoordinator(model, tokenizer)
setup_info = coordinator.setup_training()

# Compute loss with multi-task support
total_loss, loss_components = coordinator.compute_loss(outputs, inputs)
coordinator.step_update(step, epoch)
```

**Key Features**: Detection freeze scheduling, parameter grouping, enhanced loss logging

### LossManager (`src/training/loss_manager.py`)
```python
# Simple loss extraction
loss_manager = LossManager(tokenizer)
total_loss, components = loss_manager.compute_total_loss(outputs, inputs)
averaged_losses = loss_manager.get_averaged_losses()
```

**Key Features**: Simplified design, delegates to model wrapper, coordinate loss extraction

## 🎯 Coordinate Token System

### CoordinateTokenManager (`src/utils/coordinate_token_manager.py`)
```python
# Create manager
manager = create_coordinate_token_manager(tokenizer, vocab_size, config_dict)

# Format conversion
coord_text = manager.convert_json_to_coordinate_format(json_str)
json_str = manager.convert_coordinate_to_json_format(coord_text)

# Training utilities
bbox_spans = manager.detect_bbox_spans(input_ids)
losses = manager.compute_coordinate_losses(logits, labels, bbox_spans)
mask = manager.create_coordinate_mask(token_ids)
```

**Key Features**: Automatic JSON↔coordinate conversion, bbox span detection, comprehensive loss computation

## ⚙️ Essential Configuration

### Coordinate Tokens
```yaml
coordinate_tokens_enabled: true
coordinate_config_max_coord_value: 2048
coordinate_lr: 1e-4
coordinate_loss_weight: 1.0
soft_expectation_temperature: 1.0
```

### Data & Model
```yaml
train_data_path: "data/train.jsonl"
val_data_path: "data/val.jsonl"
model_path: "/path/to/qwen2.5-vl"
model_max_length: 8192
torch_dtype: "bfloat16"
```

## 🔗 Common Usage Patterns

### Complete Training Setup
```python
# 1. Create components
processor = DataProcessor(tokenizer, image_processor)
train_ds, eval_ds, collator = processor.create_datasets_and_collator()

# 2. Load model with coordinate support
model = Qwen25VLWithDetection.from_pretrained(
    model_path, tokenizer, coordinate_config
)

# 3. Setup training coordination
coordinator = TrainingCoordinator(model, tokenizer)
setup_info = coordinator.setup_training()

# 4. Training loop integration
total_loss, components = coordinator.compute_loss(outputs, inputs)
```

### Coordinate Token Operations
```python
# Setup coordinate manager
manager = create_coordinate_token_manager(tokenizer, vocab_size, config)

# Data conversion in pipeline
json_data = '{"bbox_2d": [10, 20, 100, 200], "desc": "bbu设备"}'
coord_text = manager.convert_json_to_coordinate_format(json_data)
# Result: "bbu设备: <|box_start|><coord_10><coord_20><coord_100><coord_200><|box_end|>"
```

### Model Saving & Loading
```python
# Safe model saving
checkpoint_manager = CheckpointManager()
success = checkpoint_manager.save_model_safely(trainer, output_dir)

# Validation
is_valid = checkpoint_manager.validate_checkpoint(output_dir)
```

---
**💡 Quick Tips:**
- All components auto-detect coordinate vs standard modes
- Loss computation handles multi-task scenarios automatically  
- Use factory functions for component creation
- Configuration drives behavior - check config dependencies for each component