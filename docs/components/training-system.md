# Training System Components

**Detailed documentation for the modular training system (`src/training/`)**

## Overview

The training system has been refactored into modular components with clear separation of concerns. Each component has a specific responsibility and can be tested, debugged, and extended independently.

## Component Architecture

```
src/training/
├── trainer.py              # BBUTrainer - Enhanced HuggingFace trainer
├── training_coordinator.py # TrainingCoordinator - Training orchestration
├── loss_manager.py         # LossManager - Multi-task loss computation
├── parameter_manager.py    # ParameterManager - Learning rate management
├── trainer_factory.py      # TrainerFactory - Factory pattern for trainer creation
├── callbacks.py            # Training callbacks and monitoring
└── stability.py            # Training stability utilities
```

## BBUTrainer (`trainer.py`)

### Component Contract

**Input**:
- `model`: Qwen25VLWithDetection instance
- `args`: HuggingFace TrainingArguments
- `train_dataset`: BBUDataset for training
- `eval_dataset`: BBUDataset for evaluation
- `data_collator`: BBU-specific data collator
- `cfg`: DirectConfig instance
- `image_processor`: Image processor for VLM
- `training_coordinator`: Optional TrainingCoordinator

**Output**:
- Trained model with coordinate token support
- Training logs with component-wise loss breakdown
- Checkpoints with proper coordinate token handling

**Dependencies**:
- HuggingFace Transformers Trainer (base class)
- LossManager (created internally or via coordinator)
- DirectConfig for configuration access

**Side Effects**:
- Saves checkpoints to `output_dir`
- Logs to wandb (if configured)
- Modifies model weights during training

### Key Features

#### Multi-Component Loss Logging
```python
# Automatic loss component tracking
logs = {
    'loss': total_loss.item(),
    'llm_loss': llm_loss.item(),
    'coordinate_l1_loss': coord_loss.item(),
    'teacher_loss': teacher_loss.item(),
    'student_loss': student_loss.item()
}
```

#### Teacher-Student Learning Support
- Automatic detection of teacher vs student samples
- Span-based loss splitting based on token counts
- Configurable teacher/student ratios

#### Enhanced Validation
- Coordinate token validation during training
- Model output format checking
- Training stability monitoring

### Usage Example

```python
from src.training.trainer import BBUTrainer
from src.training.training_coordinator import TrainingCoordinator

# Create coordinator (optional but recommended)
coordinator = TrainingCoordinator(
    model=model,
    tokenizer=tokenizer,
    config_obj=config
)
coordinator.setup_training()

# Create trainer
trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    cfg=config,
    image_processor=image_processor,
    training_coordinator=coordinator
)

# Train
trainer.train()
```

## TrainingCoordinator (`training_coordinator.py`)

### Component Contract

**Input**:
- `model`: Qwen25VLWithDetection instance
- `tokenizer`: Tokenizer with coordinate tokens
- `config_obj`: DirectConfig instance

**Output**:
- Configured training environment
- Initialized loss manager
- Training state management

**Dependencies**:
- LossManager for loss computation
- Model and tokenizer for coordinate token setup
- DirectConfig for training parameters

**Side Effects**:
- Modifies model configuration
- Sets up loss computation strategy
- Initializes training monitoring

### Key Features

#### Multi-Task Training Coordination
- Coordinates between LLM and coordinate learning tasks
- Manages training state across components
- Handles component-specific configurations

#### Component Integration
- Automatic LossManager creation and configuration
- Integration with BBUTrainer for enhanced training
- Centralized training state management

### Usage Example

```python
from src.training.training_coordinator import TrainingCoordinator

coordinator = TrainingCoordinator(
    model=model,
    tokenizer=tokenizer,
    config_obj=config
)

# Setup training environment
coordinator.setup_training()

# Access configured components
loss_manager = coordinator.loss_manager
training_state = coordinator.get_training_state()
```

## LossManager (`loss_manager.py`)

### Component Contract

**Input**:
- `tokenizer`: Tokenizer with coordinate tokens
- `model`: Model for loss computation
- `teacher_loss_weight`: Weight for teacher samples (default: 0.3)
- `student_loss_weight`: Weight for student samples (default: 1.0)

**Output**:
- `total_loss`: Combined loss for backpropagation
- `loss_components`: Dictionary with individual loss components

**Dependencies**:
- Tokenizer for coordinate token identification
- Model for forward pass and loss computation
- PyTorch for loss computation functions

**Side Effects**:
- None (pure computation component)

### Key Features

#### Multi-Component Loss Computation
```python
# Two main loss components
total_loss = llm_loss + coordinate_l1_loss

# Component breakdown
loss_components = {
    'llm_loss': llm_loss.item(),
    'coordinate_l1_loss': coord_loss.item(),
    'teacher_loss': teacher_loss.item(),
    'student_loss': student_loss.item()
}
```

#### Teacher-Student Loss Splitting
- **Teachers**: Get LLM loss only (high-quality demonstrations)
- **Students**: Get LLM + coordinate loss (model's own predictions)
- **Span-Based**: Proportional allocation based on token counts

#### Coordinate Token Handling
- Automatic identification of coordinate tokens in sequences
- L1 loss computation for coordinate accuracy
- Robust handling of variable-length sequences

### Usage Example

```python
from src.training.loss_manager import LossManager

loss_manager = LossManager(
    tokenizer=tokenizer,
    model=model,
    teacher_loss_weight=0.3,
    student_loss_weight=1.0
)

# Compute losses during training
total_loss, loss_components = loss_manager.compute_total_loss(
    inputs=batch,
    model_outputs=model_outputs
)
```

## TrainerFactory (`trainer_factory.py`)

### Component Contract

**Input**:
- `training_args`: HuggingFace TrainingArguments

**Output**:
- Fully configured BBUTrainer instance
- All necessary components automatically created

**Dependencies**:
- DataProcessor for dataset creation
- ModelLoader for model loading
- TrainingCoordinator for training setup
- DirectConfig for configuration

**Side Effects**:
- Creates datasets and data collator
- Loads and configures model
- Sets up training environment

### Key Features

#### Factory Pattern Implementation
- Single entry point for trainer creation
- Automatic component dependency resolution
- Configuration-driven setup

#### Component Integration
- Automatic DataProcessor creation
- Model loading with coordinate token support
- TrainingCoordinator setup and integration

### Usage Example

```python
from src.training.trainer_factory import create_trainer_with_coordinator

# One-line trainer creation
trainer = create_trainer_with_coordinator(training_args)

# Start training immediately
trainer.train()
```

## ParameterManager (`parameter_manager.py`)

### Component Contract

**Input**:
- `model`: Model with coordinate tokens
- `config`: Configuration with learning rate settings

**Output**:
- Parameter groups with differential learning rates
- Optimizer configuration

**Dependencies**:
- Model parameters for grouping
- Configuration for learning rate settings

**Side Effects**:
- None (parameter analysis only)

### Key Features

#### Differential Learning Rates
- Higher learning rates for coordinate tokens
- Standard rates for base model parameters
- Configurable rate multipliers

#### Parameter Grouping
- Automatic identification of coordinate token parameters
- Separate parameter groups for different learning rates
- Integration with HuggingFace optimizers

## Training Callbacks (`callbacks.py`)

### Available Callbacks

#### ProgressCallback
- Training progress monitoring
- Step-by-step loss tracking
- Component-wise metrics logging

#### StabilityCallback
- Training stability monitoring
- Gradient norm tracking
- Loss spike detection

### Usage Example

```python
from src.training.callbacks import ProgressCallback, StabilityCallback

trainer = BBUTrainer(
    # ... other arguments
    callbacks=[
        ProgressCallback(),
        StabilityCallback(gradient_threshold=10.0)
    ]
)
```

## Training Stability (`stability.py`)

### Features

#### Gradient Monitoring
- Gradient norm computation
- Spike detection and logging
- Training stability metrics

#### Loss Validation
- Loss component validation
- NaN/Inf detection
- Training health checks

### Usage Example

```python
from src.training.stability import check_training_stability

# During training loop
stability_metrics = check_training_stability(
    model=model,
    loss=total_loss,
    gradients=gradients
)
```

## Integration Patterns

### Standard Training Setup
```python
# 1. Factory pattern (recommended)
trainer = create_trainer_with_coordinator(training_args)
trainer.train()

# 2. Manual setup (for customization)
coordinator = TrainingCoordinator(model, tokenizer, config)
coordinator.setup_training()

trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    cfg=config,
    image_processor=image_processor,
    training_coordinator=coordinator
)
trainer.train()
```

### Custom Loss Manager
```python
# Custom loss weights
loss_manager = LossManager(
    tokenizer=tokenizer,
    model=model,
    teacher_loss_weight=0.5,  # Custom weight
    student_loss_weight=0.8   # Custom weight
)

coordinator = TrainingCoordinator(
    model=model,
    tokenizer=tokenizer,
    config_obj=config,
    loss_manager=loss_manager  # Use custom loss manager
)
```

## Debugging and Monitoring

### Component Health Checks
```python
# Check trainer setup
assert trainer.model is not None
assert trainer.training_coordinator is not None
assert hasattr(trainer.training_coordinator, 'loss_manager')

# Check loss manager
loss_manager = trainer.training_coordinator.loss_manager
assert loss_manager.tokenizer is not None
assert loss_manager.model is not None
```

### Training Metrics
- Monitor `loss`, `llm_loss`, `coordinate_l1_loss` in logs
- Check teacher/student loss balance
- Validate coordinate token learning progress

---

**Next Steps**:
- **Model System**: [model-system.md](model-system.md)
- **Data Pipeline**: [data-pipeline.md](data-pipeline.md)
- **Configuration**: [configuration.md](configuration.md)
