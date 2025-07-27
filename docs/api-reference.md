# BBU Training Pipeline API Reference

**Complete API documentation for the BBU training pipeline components**

## 🏗️ **Core Components**

### **Configuration System**
```python
from src.config import load_config, init_config

# Load configuration from file
config = load_config('configs/bbu_v2.yaml')

# Initialize global configuration
init_config('configs/bbu_v2.yaml')

# Access configuration parameters
print(f"Model path: {config.model_path}")
print(f"Coordinate tokens: {config.coordinate_tokens_enabled}")
```

### **Model Loading**
```python
from src.models.model_loader import load_model_and_processor_unified

# Load model, tokenizer, and processor
model, tokenizer, processor = load_model_and_processor_unified(
    model_path="model_cache/Qwen/Qwen2.5-VL-3B-Instruct",
    for_inference=False,  # Enables embedding resize for new tokens
    attn_implementation="flash_attention_2"  # Optional
)

# Check vocabulary size
vocab_size = len(tokenizer.get_vocab())
print(f"Vocabulary size: {vocab_size}")
```

### **Coordinate Token Management**
```python
from src.utils.tokens.special_tokens import UnifiedTokenManager

# Create tokenizer with coordinate tokens
tokenizer = UnifiedTokenManager.create_tokenizer_with_coordinate_tokens()

# Check coordinate token availability
has_coord_tokens = UnifiedTokenManager.has_coordinate_tokens(tokenizer)

# Get coordinate token range
coord_range = UnifiedTokenManager.get_coordinate_token_range(tokenizer)
print(f"Coordinate token range: {coord_range}")
```

### **Coordinate Processing**
```python
from src.core.coordinate_manager import SimpleCoordinateManager

# Standard mode (integer coordinates)
manager_std = SimpleCoordinateManager(coordinate_tokens_enabled=False)
result_std = manager_std.format_object("device", "box", [150,10,211,35])
print(f"Standard: {result_std}")

# Coordinate mode (token coordinates)
manager_coord = SimpleCoordinateManager(
    coordinate_tokens_enabled=True, 
    tokenizer=tokenizer
)
result_coord = manager_coord.format_object("device", "box", [150,10,211,35])
print(f"Coordinate: {result_coord}")

# Wrap coordinates in tokens
wrapped = manager_coord.wrap_coordinates([150,10,211,35])
print(f"Wrapped: {wrapped}")
```

## 📊 **Data Processing**

### **Data Processor**
```python
from src.core.data_processor import DataProcessor

# Create data processor
processor = DataProcessor(tokenizer, processor, model, config=config)

# Create datasets
train_dataset, eval_dataset = processor.create_datasets()
print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# Create data collator
data_collator = processor.create_data_collator()
print(f"Collator type: {type(data_collator)}")
```

### **Dataset Classes**
```python
from src.data import BBUDataset

# Create dataset directly
dataset = BBUDataset(
    data_path="data/train.jsonl",
    chat_processor=chat_processor,
    teacher_ratio=0.5,
    is_training=True,
    config=config
)

# Access samples
sample = dataset[0]
print(f"Sample keys: {list(sample.keys())}")
```

### **Data Collators**
```python
from src.data import create_data_collator

# Create standard collator
collator = create_data_collator("standard", tokenizer)

# Create packed collator
packed_collator = create_data_collator("packed", tokenizer)

# Test collation
samples = [dataset[i] for i in range(2)]
batch = collator(samples)
print(f"Batch keys: {list(batch.keys())}")
```

## 🏋️ **Training System**

### **BBU Trainer**
```python
from src.training.trainer import BBUTrainer
from transformers import TrainingArguments

# Create training arguments
training_args = TrainingArguments(
    output_dir="checkpoints/run_001",
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    remove_unused_columns=False,  # Required for coordinate mode
)

# Create trainer
trainer = BBUTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=processor,
)

# Start training
trainer.train()
```

### **Training Factory**
```python
from src.training.trainer_factory import create_trainer_with_coordinator

# Create trainer with coordinator
trainer = create_trainer_with_coordinator(
    model=model,
    tokenizer=tokenizer,
    processor=processor,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    config=config
)
```

## 🔧 **Utility Functions**

### **Token Validation**
```python
from src.utils.tokens.special_tokens import UnifiedTokenManager

# Validate token format
def validate_coordinate_tokens(tokenizer):
    vocab = tokenizer.get_vocab()
    coord_tokens = [token for token in vocab.keys() 
                   if token.startswith('<|coord_') and token.endswith('|>')]
    
    print(f"Found {len(coord_tokens)} coordinate tokens")
    
    # Check specific tokens
    for i in range(min(10, len(coord_tokens))):
        expected = f"<|coord_{i}|>"
        if expected in vocab:
            print(f"✅ {expected} found")
        else:
            print(f"❌ {expected} missing")

validate_coordinate_tokens(tokenizer)
```

### **Configuration Validation**
```python
def validate_config(config):
    """Validate configuration for common issues."""
    issues = []
    
    # Check coordinate mode requirements
    if config.coordinate_tokens_enabled and config.remove_unused_columns:
        issues.append("Coordinate mode requires remove_unused_columns: false")
    
    # Check file paths
    import os
    if not os.path.exists(config.model_path):
        issues.append(f"Model path not found: {config.model_path}")
    
    if not os.path.exists(config.train_data_path):
        issues.append(f"Training data not found: {config.train_data_path}")
    
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Configuration validation passed")
        return True

validate_config(config)
```

## 🧪 **Testing Utilities**

### **System Health Check**
```python
def system_health_check():
    """Comprehensive system health check."""
    import torch
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
    
    # Check imports
    try:
        from src.training.trainer import BBUTrainer
        print("✅ BBUTrainer import OK")
    except ImportError as e:
        print(f"❌ BBUTrainer import failed: {e}")
    
    try:
        from src.core.coordinate_manager import SimpleCoordinateManager
        print("✅ SimpleCoordinateManager import OK")
    except ImportError as e:
        print(f"❌ SimpleCoordinateManager import failed: {e}")
    
    # Check model loading
    try:
        from src.models.model_loader import load_model_and_processor_unified
        model, tokenizer, processor = load_model_and_processor_unified(
            "model_cache/Qwen/Qwen2.5-VL-3B-Instruct", for_inference=True
        )
        print(f"✅ Model loaded, vocab size: {len(tokenizer.get_vocab())}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

system_health_check()
```

### **Data Validation**
```python
def validate_training_data(data_path):
    """Validate training data format."""
    import json
    from pathlib import Path
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Check first 3 samples
                    break
                
                sample = json.loads(line)
                
                # Check required fields
                required_fields = ['conversations', 'images']
                for field in required_fields:
                    if field not in sample:
                        print(f"❌ Missing field '{field}' in sample {i}")
                        return False
                
                print(f"✅ Sample {i} validation passed")
        
        print(f"✅ Data validation passed for {data_path}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

validate_training_data("data/train.jsonl")
```

## 📚 **Command Line Interface**

### **Training Script**
```bash
# Basic training
python scripts/train.py --config configs/bbu_v2.yaml

# Training with custom parameters
python scripts/train.py \
    --config configs/bbu_v2.yaml \
    --output_dir checkpoints/experiment_001 \
    --max_steps 1000 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6

# Resume from checkpoint
python scripts/train.py \
    --config configs/bbu_v2.yaml \
    --resume_from_checkpoint checkpoints/run_001/checkpoint-500

# Dry run (validation only)
python scripts/train.py --config configs/bbu_v2.yaml --dry-run
```

### **Testing Commands**
```bash
# Run all tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_coordinate_tokens.py -v
python -m pytest tests/test_data_pipeline.py -v
python -m pytest tests/test_training_components.py -v

# Test with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 🔍 **Debugging Tools**

### **Debug Mode**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Debug coordinate token processing
from src.core.coordinate_manager import SimpleCoordinateManager
manager = SimpleCoordinateManager(coordinate_tokens_enabled=True, tokenizer=tokenizer)
manager.debug = True  # Enable debug output
result = manager.format_object("device", "box", [150,10,211,35])
```

### **Memory Profiling**
```python
# Monitor GPU memory
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# Use during training
print_gpu_memory()
```

## 📚 **Related Documentation**

- [**Getting Started**](getting-started.md) - Quick setup and first run
- [**Configuration**](configuration.md) - Configuration system details
- [**Coordinate Tokens**](coordinate-tokens.md) - Coordinate token system
- [**Training**](training.md) - Training workflows and monitoring
- [**Troubleshooting**](troubleshooting.md) - Common issues and solutions

---

**Need more technical details?** Check the source code in `src/` directory for complete implementation details.
