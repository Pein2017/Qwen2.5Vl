# Qwen2.5-VL New Architecture - Quick Start Guide

> **📋 Implementation Guide**: New to the project? See [**docs/IMPLEMENTATION_GUIDE.md**](../docs/IMPLEMENTATION_GUIDE.md) to choose between `src/` vs `src_new/`.
>
> **📚 Full Documentation**: For comprehensive documentation, see [**docs/**](../docs/) directory.

## 🚀 **Getting Started in 5 Minutes**

### **1. Validate Your Setup**
```bash
cd /data3/Qwen2.5-VL-main

# Test configuration loading
python scripts/train_new.py --config bbu_v2 --validate-only

# Test trainer creation  
python scripts/train_new.py --config bbu_v2 --test-trainer

# Run test suite
python -m pytest src_new/tests/ -v --tb=short
```

### **2. Start Training**
```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=0 python scripts/train_new.py --config bbu_v2 --log_level INFO

# Multi-GPU training with DeepSpeed
BBU_DEEPSPEED_ENABLED=true ./scripts/run_train_new.sh
```

### **3. Monitor Training**
```bash
# View tensorboard logs
tensorboard --logdir output-*/tb

# Check training logs
tail -f output-*/logs/training.log
```

## 📋 **Common Usage Patterns**

### **Configuration Management**
```python
from src_new import Config

# Load configuration
config = Config.from_yaml("configs/bbu_v2.yaml")

# Access configuration values
print(f"Model: {config.model_path}")
print(f"Learning Rate: {config.learning_rate}")
print(f"Coordinate Tokens: {config.coordinate_tokens_enabled}")

# Validate configuration
config.validate()  # Raises detailed errors if invalid
```

### **Dataset Creation**
```python
from src_new import Dataset, TeacherPoolManager

# Create teacher pool manager
teacher_pool = TeacherPoolManager(
    teacher_pool_file=config.teacher_pool_file,
    data_root=config.data_root
)

# Create dataset
train_dataset = Dataset(
    data_path=config.train_data_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    teacher_pool_manager=teacher_pool,
    config=config
)

print(f"Dataset size: {len(train_dataset)}")
```

### **Model Creation**
```python
from src_new.models import DetectionModel

# Load model
model = DetectionModel.from_pretrained(
    config.model_path,
    config=config,
    torch_dtype=getattr(torch, config.torch_dtype),
    attn_implementation=config.attn_implementation
)

# Enable coordinate token mode (if not already enabled in config)
model.enable_coordinate_mode()

# Check model status
print(f"Coordinate mode: {model.coordinate_mode_enabled}")
print(f"Coordinate tokens: 2049 tokens (coord_0 to coord_2048)")
print(f"Model device: {model.device}")
```

### **Training Setup**
```python
from src_new.training import Trainer
from src_new.data import create_data_collator

# Create data collator
collator = create_data_collator(
    collator_type="standard",  # or "packed"
    tokenizer=tokenizer
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    tokenizer=tokenizer
)

# Start training
trainer.train()
```

## 🔧 **Development Workflow**

### **Adding New Features**

#### **1. Configuration Changes**
```python
# Add new field to src_new/config/config.py
@dataclass
class Config:
    # ... existing fields ...
    new_feature_enabled: bool = False
    new_feature_param: float = 1.0
    
    def _validate_new_feature(self):
        if self.new_feature_enabled and self.new_feature_param <= 0:
            raise ValueError("new_feature_param must be positive")
```

#### **2. Model Changes**
```python
# Add new component to src_new/models/wrapper.py
class DetectionModel(nn.Module):
    def __init__(self, base_model, config):
        # ... existing components ...
        if config.new_feature_enabled:
            self.new_component = NewComponent(config)
    
    def forward(self, **inputs):
        # ... existing forward pass ...
        if hasattr(self, 'new_component'):
            outputs = self.new_component(outputs, **inputs)
        return outputs
```

#### **3. Add Tests**
```python
# Create test in src_new/tests/test_models/test_new_feature.py
class TestNewFeature:
    def test_new_feature_enabled(self):
        config = create_test_config(new_feature_enabled=True)
        model = DetectionModel(mock_base_model, config)
        assert hasattr(model, 'new_component')
    
    def test_new_feature_disabled(self):
        config = create_test_config(new_feature_enabled=False)
        model = DetectionModel(mock_base_model, config)
        assert not hasattr(model, 'new_component')
```

### **Debugging Common Issues**

#### **Configuration Errors**
```python
# Error: FileNotFoundError: Model path does not exist
# Solution: Check model_path in config
config = Config.from_yaml("configs/bbu_v2.yaml")
print(f"Model path: {config.model_path}")
print(f"Exists: {Path(config.model_path).exists()}")
```

#### **Data Loading Errors**
```python
# Error: FileNotFoundError: Data file not found
# Solution: Check data paths
print(f"Train data: {config.train_data_path}")
print(f"Val data: {config.val_data_path}")
print(f"Teacher pool: {config.teacher_pool_file}")
```

#### **Memory Errors**
```python
# Error: CUDA out of memory
# Solution: Reduce batch size or use packed collator
config.per_device_train_batch_size = 1  # Reduce batch size
collator = create_data_collator("packed", tokenizer)  # Use packed collator
```

#### **Vision Token Errors**
```python
# Error: Image features and image tokens do not match
# Solution: Check image processing pipeline
from src_new.tests.test_data.test_image_processing import TestImageProcessing
test = TestImageProcessing()
test.test_vision_token_calculation()  # Should pass
```

### **Performance Optimization**

#### **Memory Optimization**
```python
# Use packed data collator
collator = create_data_collator("packed", tokenizer)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use DeepSpeed ZeRO
training_args.deepspeed = "configs/deepspeed_zero2.json"
```

#### **Speed Optimization**
```python
# Use Flash Attention 2
config.attn_implementation = "flash_attention_2"

# Optimize dataloader
config.dataloader_num_workers = 8
config.prefetch_factor = 4
config.pin_memory = True

# Use mixed precision
training_args.fp16 = True  # or bf16 = True
```

## 🧪 **Testing Guidelines**

### **Running Tests**
```bash
# Run all tests
python -m pytest src_new/tests/ -v

# Run specific test category
python -m pytest src_new/tests/test_config/ -v
python -m pytest src_new/tests/test_data/ -v
python -m pytest src_new/tests/test_models/ -v

# Run with coverage
python -m pytest src_new/tests/ --cov=src_new --cov-report=html
```

### **Writing Tests**
```python
# Test template
class TestNewComponent:
    def test_basic_functionality(self):
        # Arrange
        config = create_test_config()
        component = NewComponent(config)
        
        # Act
        result = component.process(test_input)
        
        # Assert
        assert result is not None
        assert isinstance(result, expected_type)
    
    def test_error_handling(self):
        # Test that component handles errors gracefully
        with pytest.raises(ValueError, match="Expected error message"):
            component.process(invalid_input)
```

## 📚 **Reference**

### **Key Files**
- `src_new/ARCHITECTURE.md`: Comprehensive architecture documentation
- `configs/bbu_v2.yaml`: Reference configuration
- `scripts/train_new.py`: Production training script
- `scripts/run_train_new.sh`: Multi-GPU training launcher

### **Important Classes**
- `Config`: Unified configuration management
- `Dataset`: Data processing with teacher-student support
- `DetectionModel`: Composition-based model wrapper
- `Trainer`: Minimal HF Trainer extension
- `LossManager`: Multi-component loss computation

### **Useful Commands**
```bash
# Configuration validation
python scripts/train_new.py --config bbu_v2 --validate-only

# Print configuration
python scripts/train_new.py --config bbu_v2 --print-config

# Test trainer creation
python scripts/train_new.py --config bbu_v2 --test-trainer

# Debug mode training
python scripts/train_new.py --config bbu_v2 --log_level DEBUG
```

---

## 📚 **Additional Resources**

### **Complete Documentation**
- **[Main Documentation](../docs/README.md)** - Complete documentation hub
- **[Implementation Guide](../docs/IMPLEMENTATION_GUIDE.md)** - Choose src/ vs src_new/
- **[Troubleshooting](../docs/troubleshooting/common-issues.md)** - Common issues and solutions
- **[Migration Guide](../docs/guides/migration-src-to-src-new.md)** - Upgrade from src/ to src_new/

### **Technical References**
- **[Architecture Overview](../docs/reference/architecture.md)** - System architecture
- **[API Reference](../docs/api-reference/src-new-api.md)** - Complete src_new/ API
- **[Configuration Guide](../docs/implementation/configuration.md)** - Configuration options

### **Need Help?**
- Check [**troubleshooting guide**](../docs/troubleshooting/common-issues.md) for common issues
- Review [**implementation documentation**](../docs/implementation/) for technical details
- Run tests: `python -m pytest src_new/tests/ -v`
```

This quick start guide provides everything you need to begin working with the new `src_new/` architecture effectively!
