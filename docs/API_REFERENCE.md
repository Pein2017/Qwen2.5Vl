# API Reference and Migration Guide

**Complete API documentation and migration guide for the BBU training pipeline**

## 🏗️ **Core API Components**

### **Configuration System**

#### **Configuration Loading**
```python
from src_new.config.config import load_config

# Load configuration from YAML file with comprehensive validation
config = load_config('configs/bbu_v2.yaml')

# Access configuration parameters
print(f"Model path: {config.model_path}")
print(f"Coordinate tokens: {config.coordinate_tokens_enabled}")
print(f"Max coord value: {config.max_coord_value}")

# Configuration is automatically validated during loading
# Raises FileNotFoundError, ValueError, or TypeError if invalid
```

#### **Configuration Parameters**
```python
@dataclass
class Config:
    # === REQUIRED FIELDS (no defaults) ===
    # Model settings
    model_path: str
    model_size: str
    model_max_length: int
    attn_implementation: str
    torch_dtype: str

    # Training settings
    num_train_epochs: int
    per_device_train_batch_size: int
    learning_rate: float
    vision_lr: float
    merger_lr: float
    llm_lr: float

    # Data paths
    train_data_path: str
    val_data_path: str
    teacher_pool_file: str

    # === OPTIONAL FIELDS (with defaults) ===
    # Coordinate Token System
    coordinate_tokens_enabled: bool = False
    max_coord_value: int = 1024
    coordinate_loss_weight: float = 0.05
    coordinate_loss_temperature: float = 1.0

    # Teacher-Student Training
    teacher_ratio: float = 0.5
    teacher_loss_weight: float = 0.3
    student_loss_weight: float = 1.0
```

### **Model System**

#### **Model Loading and Initialization**
```python
from src_new.models.wrapper import DetectionModel
from src_new.processing.token_processor import TokenProcessor, TokenConfig

# Load base model and tokenizer
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
base_model = Qwen2VLForConditionalGeneration.from_pretrained(config.model_path)
processor = Qwen2VLProcessor.from_pretrained(config.model_path)
tokenizer = processor.tokenizer

# Initialize token processor for coordinate tokens
token_config = TokenConfig(
    max_coord_value=config.max_coord_value,
    coordinate_tokens_enabled=config.coordinate_tokens_enabled
)
token_processor = TokenProcessor(token_config)

# Extend tokenizer and model (if coordinate tokens enabled)
if config.coordinate_tokens_enabled:
    tokenizer = token_processor.extend_tokenizer_vocabulary(tokenizer)
    base_model = token_processor.extend_model_embeddings(base_model, tokenizer)

# Create detection model wrapper
model = DetectionModel(
    base_model=base_model,
    config=config,
    tokenizer=tokenizer,
    skip_expansion=True  # Already expanded above
)
```

#### **Model Training**
```python
from src_new.training.bbu_trainer import BBUTrainer
from src_new.data.dataset import BBUDataset
from src_new.data.collator import create_data_collator

# Create datasets
train_dataset = BBUDataset(config, tokenizer, split='train')
eval_dataset = BBUDataset(config, tokenizer, split='val')

# Create data collator
data_collator = create_data_collator(tokenizer, processor)

# Initialize trainer
trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```

### **Data Processing**

#### **Dataset Creation**
```python
from src_new.data.dataset import BBUDataset
from src_new.data.teacher_pool import TeacherPoolManager

# Initialize teacher pool (for teacher-student training)
teacher_pool = TeacherPoolManager(
    teacher_data_path=config.teacher_data_path,
    max_teachers_per_sample=1,
)

# Create dataset
dataset = BBUDataset(
    config=config,
    tokenizer=tokenizer,
    teacher_pool_manager=teacher_pool,
    teacher_ratio=config.teacher_ratio,
)

# Access samples
sample = dataset[0]
print(f"Sample keys: {list(sample.keys())}")
```

#### **Data Collation**
```python
from src_new.data.collator import PackedDataCollator

# Create data collator
collator = PackedDataCollator(
    tokenizer=tokenizer,
    max_length=config.model_max_length,
)

# Collate batch
batch = collator([dataset[i] for i in range(4)])
```

### **Coordinate Token System**

#### **Coordinate Token Processing (TokenProcessor)**
```python
from src_new.processing.token_processor import TokenProcessor, TokenConfig

proc = TokenProcessor(TokenConfig(max_coord_value=2048, coordinate_tokens_enabled=True))

# Convert numeric coords to coord tokens
print(proc.coordinates_to_tokens([150, 10, 211, 35]))
# Parse coord tokens back to ints
print(proc.tokens_to_coordinates(["<|coord_150|>", "<|coord_10|>"]))
```

#### **Coordinate Conversion for Objects (CoordinateTokenConverter)**
```python
from src_new.processing.coordinate_converter import CoordinateTokenConverter

converter = CoordinateTokenConverter(max_coord_value=2048)
obj = {"bbox_2d": [100, 200, 300, 400], "desc": "BBU设备"}
print(converter.convert_objects_to_tokens([obj]))
```

### **Teacher-Student Training**

#### **Teacher Pool Management**
```python
from src_new.data.teacher_pool import TeacherPoolManager

# Initialize teacher pool
teacher_pool = TeacherPoolManager(
    teacher_data_path="data/teacher_pool.jsonl",
    max_teachers_per_sample=1,
)

# Get random teachers
teachers = teacher_pool.get_random_teachers(num_samples=2)
```

#### **Loss Computation**
```python
from src_new.models.loss_manager import LossManager

# Initialize loss manager
loss_manager = LossManager(
    config=config,
    tokenizer=tokenizer,
    teacher_loss_weight=0.5,
    student_loss_weight=0.5,
)

# Compute loss with teacher-student separation
loss_dict = loss_manager.compute_loss(
    logits=model_output.logits,
    labels=batch["labels"],
    teacher_spans=batch.get("teacher_spans"),
    student_spans=batch.get("student_spans"),
)

# Access individual loss components
print(f"Teacher LLM Loss: {loss_dict['teacher_llm_loss']}")
print(f"Student LLM Loss: {loss_dict['student_llm_loss']}")
print(f"Teacher L1 Loss: {loss_dict['teacher_l1_loss']}")
print(f"Student L1 Loss: {loss_dict['student_l1_loss']}")
```

## 🔄 **Migration Guide: src/ → src_new/**

### **Migration Benefits**
- ✅ **25% Code Reduction**: From ~23k to ~17k lines with better functionality
- ✅ **Simplified Architecture**: Clear separation of concerns and single responsibility
- ✅ **Enhanced Reliability**: Fail-fast validation with comprehensive error handling
- ✅ **Better Testing**: 127 comprehensive tests vs scattered legacy tests
- ✅ **Production Ready**: Proven stability and performance optimization

### **Key Architecture Changes**

#### **Configuration System**
```python
# OLD (src/): Complex multi-class configuration
from src.config import BBUConfig, DomainConfigs
config = BBUConfig()
domain_configs = DomainConfigs()

# NEW (src_new/): Single unified configuration
from src_new.config.config import load_config
config = load_config('configs/bbu_v2.yaml')
```

#### **Model Loading**
```python
# OLD (src/): Complex model loader
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, processor = load_model_and_processor_unified(model_path)

# NEW (src_new/): Composition-based DetectionModel
from src_new.models.wrapper import DetectionModel
model = DetectionModel.from_pretrained(model_path, config, tokenizer)
```

#### **Training System**
```python
# OLD (src/): Complex inheritance chains
from src.training.trainer import BBUTrainer
from src.training.training_coordinator import TrainingCoordinator
coordinator = TrainingCoordinator(config)
trainer = coordinator.create_trainer()

# NEW (src_new/): Direct BBUTrainer usage
from src_new.training.bbu_trainer import BBUTrainer
trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

### **Migration Steps**

#### **Step 1: Update Imports**
```python
# Replace all src/ imports with src_new/ equivalents
# OLD
from src.config import BBUConfig
from src.models.wrapper import ModelWrapper
from src.training.trainer import BBUTrainer

# NEW
from src_new.config.config import load_config
from src_new.models.wrapper import DetectionModel
from src_new.training.bbu_trainer import BBUTrainer
```

#### **Step 2: Update Configuration**
```python
# OLD: Complex configuration initialization
config = BBUConfig()
config.load_from_yaml('configs/bbu_v2.yaml')
config.validate_all_domains()

# NEW: Simple configuration loading with automatic validation
config = load_config('configs/bbu_v2.yaml')
```

#### **Step 3: Update Training Code**
```python
# OLD: Complex training setup
coordinator = TrainingCoordinator(config)
trainer = coordinator.create_trainer()
trainer.setup_training()
trainer.train()

# NEW: Direct training with DetectionModel
model = DetectionModel.from_pretrained(config.model_path, config, tokenizer)
trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
```

#### **Step 4: Update Data Processing**
```python
# OLD: Complex data pipeline
from src.data import BBUDataset
from src.teacher_pool import TeacherPoolManager
dataset = BBUDataset(config, tokenizer, teacher_pool)

# NEW: Simplified data pipeline
from src_new.data.dataset import BBUDataset
from src_new.data.teacher_pool import TeacherPoolManager
from src_new.data.collator import create_data_collator

dataset = BBUDataset(config, tokenizer, split='train')
data_collator = create_data_collator(tokenizer, processor)
```

### **Compatibility Notes**

#### **Breaking Changes**
1. **Configuration**: Single `Config` class replaces multiple config classes
2. **Model Loading**: `ModelWrapper.from_config()` replaces complex loader functions
3. **Training**: Direct `BBUTrainer` usage replaces coordinator pattern
4. **Error Handling**: Fail-fast validation replaces silent failures

#### **Preserved Functionality**
1. **Coordinate Token System**: Full compatibility with enhanced performance
2. **Teacher-Student Training**: Same API with improved reliability
3. **Multi-Geometry Support**: All geometry types supported
4. **Data Formats**: Same JSONL format and data structure

### **Testing Migration**
```bash
# Test new implementation
python -m pytest src_new/tests/ -v

# Compare outputs (optional)
python scripts/compare_implementations.py --old_config configs/bbu_v2_old.yaml --new_config configs/bbu_v2.yaml

# Validate migration
python scripts/validate_migration.py --config configs/bbu_v2.yaml
```

---

**Migration Support**: For troubleshooting specific issues, see **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)** or run the migration validation scripts provided in the `scripts/` directory.
