# Core Components API Reference (2025 Modular Architecture)

Quick reference for essential APIs in the current modular structure: `src/training/`, `src/models/`, `src/core/`, `src/config/`, and `src/utils/`.

## 🏋️ Training System (`src/training/`)

### BBUTrainer (`src/training/trainer.py`)
```python
# Enhanced HuggingFace Trainer with multi-component loss logging
from src.training.trainer import BBUTrainer

trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    cfg=config,  # DirectConfig instance
    image_processor=image_processor,
    training_coordinator=coordinator  # Optional
)

# Train with automatic loss component tracking
trainer.train()
```

**Key Features**: Multi-component loss logging, teacher-student support, robust validation

### TrainingCoordinator (`src/training/training_coordinator.py`)
```python
# Training orchestration for multi-task learning
from src.training.training_coordinator import TrainingCoordinator

coordinator = TrainingCoordinator(
    model=model,
    tokenizer=tokenizer,
    config_obj=config
)
coordinator.setup_training()
```

**Key Features**: Multi-task coordination, state management, component delegation

### LossManager (`src/training/loss_manager.py`)
```python
# Multi-task loss computation
from src.training.loss_manager import LossManager

loss_manager = LossManager(
    tokenizer=tokenizer,
    model=model,
    teacher_loss_weight=0.3,
    student_loss_weight=1.0
)

# Compute losses
total_loss, loss_components = loss_manager.compute_total_loss(inputs, model_outputs)
```

**Key Features**: LLM + coordinate L1 loss, teacher-student splitting

### TrainerFactory (`src/training/trainer_factory.py`)
```python
# Factory pattern for trainer creation
from src.training.trainer_factory import create_trainer_with_coordinator

trainer = create_trainer_with_coordinator(training_args)
```

**Key Features**: Unified trainer creation, automatic component setup

## 🤖 Model System (`src/models/`)

### ModelLoader (`src/models/model_loader.py`)
```python
# Unified model loading for training and inference
from src.models.model_loader import load_model_and_processor_unified

model, tokenizer, image_processor = load_model_and_processor_unified(
    model_path=config.model_path,
    for_inference=False,
    attn_implementation=config.attn_implementation
)
```

**Key Features**: Training-inference consistency, automatic token initialization, patch application

### Qwen25VLWithDetection (`src/models/wrapper.py`)
```python
# Main model wrapper with detection capabilities
# (Usually created automatically by ModelLoader)

# Forward pass with coordinate token support
outputs = model(**inputs)  # Auto-detects coordinate vs standard mode
```

**Key Features**: Extended vocabulary, coordinate token support, multi-geometry handling

### Patches (`src/models/patches.py`)
```python
# Applied automatically by ModelLoader
# Includes: mRoPE fix, Flash Attention 2, memory optimization
```

## 🏭 Core System (`src/core/`)

### DataProcessor (`src/core/data_processor.py`)
```python
# Unified data processing pipeline
from src.core.data_processor import DataProcessor

processor = DataProcessor(tokenizer, image_processor, model)
train_dataset, eval_dataset = processor.create_datasets()
data_collator = processor.create_data_collator()
```

**Key Features**: Dataset creation, collator setup, teacher pool integration

### CheckpointManager (`src/core/checkpoint_manager.py`)
```python
# Model saving and loading utilities
from src.core.checkpoint_manager import CheckpointManager

manager = CheckpointManager()
success = manager.save_model_safely(trainer, "/path/to/checkpoint")
```

**Key Features**: Checkpoint management, recovery utilities

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