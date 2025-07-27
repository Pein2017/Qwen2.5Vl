# Getting Started with BBU Training Pipeline

**Get up and running with the BBU (Bounding Box Understanding) training system in 15 minutes**

## 🎯 **What You'll Accomplish**

By the end of this guide, you'll have:
- ✅ Environment set up and validated
- ✅ Data processed and ready for training
- ✅ First training run completed
- ✅ Understanding of both Standard and Coordinate modes

## 📋 **Prerequisites**

- Linux environment with CUDA GPUs
- Python 3.10+ with PyTorch
- Access to the project directory
- Basic familiarity with command line

## 🚀 **Quick Setup (5 minutes)**

### **Step 1: Environment Validation**
```bash
# Navigate to project
cd /data3/Qwen2.5-VL-main

# Verify Python environment
python --version  # Should be Python 3.10+

# Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Quick system health check
python -c "from src.training.trainer import BBUTrainer; print('✅ System OK')"
```

### **Step 2: Install Dependencies**
```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import transformers, torch; print('✅ Dependencies OK')"
```

### **Step 3: Data Preparation**
```bash
# Process your raw data (assumes data in ds_v2/ directory)
bash data_conversion/convert_dataset.sh

# Verify output
ls -la data/  # Should see train.jsonl, val.jsonl, teacher.jsonl
head -1 data/train.jsonl | python -m json.tool  # Check format
```

**Expected output**: 3 JSONL files with properly formatted conversations.

## 🎮 **First Training Run (10 minutes)**

### **Option A: Standard Mode (Recommended)**
```bash
# Standard mode with integer coordinates
python scripts/train.py --config configs/bbu_v2.yaml --max_steps 100

# Monitor training
tail -f checkpoints/run_*/training.log
```

**What happens**: Training with integer coordinates `[150,10,211,35]` format.

### **Option B: Coordinate Mode (Advanced)**
```bash
# Coordinate mode with token coordinates
python scripts/train.py --config configs/bbu_coordinate.yaml --max_steps 100

# Monitor training
tail -f checkpoints/run_*/training.log
```

**What happens**: Training with coordinate tokens `[<|coord_150|>,<|coord_10|>,...]` format.

## 📊 **Understanding the Output**

### **Training Logs**
```
Step 10: loss=2.345, lm_loss=1.234, coordinate_loss=0.567
Step 20: loss=2.123, lm_loss=1.123, coordinate_loss=0.456
```

### **Key Metrics**
- **`loss`**: Combined training loss
- **`lm_loss`**: Language modeling loss
- **`coordinate_loss`**: Coordinate prediction loss (if enabled)

### **Checkpoints**
```
checkpoints/run_001/
├── checkpoint-100/          # Model checkpoint
├── training.log            # Training logs
└── config.yaml            # Training configuration
```

## 🔧 **Configuration Basics**

### **Standard Mode Configuration**
```yaml
# configs/bbu_v2.yaml
coordinate_tokens_enabled: false    # Use integer coordinates
remove_unused_columns: false       # Required for data compatibility
max_steps: 1000                    # Training steps
per_device_train_batch_size: 4     # Batch size
```

### **Coordinate Mode Configuration**
```yaml
# configs/bbu_coordinate.yaml
coordinate_tokens_enabled: true     # Use coordinate tokens
remove_unused_columns: false       # REQUIRED for coordinate mode
max_coord_value: 2048              # Maximum coordinate value
coordinate_loss_weight: 0.05       # Coordinate loss weight
```

## 🧪 **Testing Your Setup**

### **Run System Tests**
```bash
# Test data pipeline
python -m pytest tests/test_data_pipeline.py -v

# Test training components
python -m pytest tests/test_training_components.py -v

# Test coordinate tokens
python -m pytest tests/test_coordinate_tokens.py -v
```

### **Quick Validation**
```bash
# Test model loading
python -c "
from src.models.model_loader import load_model_and_processor_unified
model, tokenizer, processor = load_model_and_processor_unified('model_cache/Qwen/Qwen2.5-VL-3B-Instruct')
print(f'✅ Model loaded, vocab size: {len(tokenizer.get_vocab())}')
"

# Test data processing
python -c "
from src.core.data_processor import DataProcessor
from src.config import load_config
config = load_config('configs/bbu_v2.yaml')
print(f'✅ Configuration loaded: {config.coordinate_tokens_enabled=}')
"
```

## 🎯 **Next Steps**

### **For New Developers**
1. [**Configuration Guide**](configuration.md) - Understand all configuration options
2. [**Training Guide**](training.md) - Advanced training workflows
3. [**Troubleshooting**](troubleshooting.md) - Common issues and solutions

### **For Researchers**
1. [**Coordinate Tokens**](coordinate-tokens.md) - Deep dive into coordinate token system
2. [**Architecture**](architecture.md) - System design and components
3. [**API Reference**](api-reference.md) - Customization and extension

### **For Production Use**
1. [**Training Guide**](training.md) - Production training workflows
2. [**Configuration**](configuration.md) - Production configuration best practices
3. [**Migration**](migration.md) - Upgrading and deployment

## ❓ **Common Questions**

### **Which mode should I use?**
- **Standard Mode**: Recommended for most use cases, production-ready
- **Coordinate Mode**: Advanced features, requires `remove_unused_columns: false`

### **How long does training take?**
- **Quick test**: 100 steps (~5 minutes)
- **Full training**: 1000+ steps (varies by data size and hardware)

### **What if something goes wrong?**
Check [Troubleshooting](troubleshooting.md) for solutions to common issues.

---

**Ready for more?** Continue to [Configuration](configuration.md) to understand all available options.
