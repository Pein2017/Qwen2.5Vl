# Training Modes and Loss Computation

**Complete guide for BBU training modes with coordinate token system and advanced loss computation**

## 🎯 **Overview**

The BBU training system supports two distinct training modes with sophisticated loss computation strategies:

1. **Standard Mode**: Traditional language modeling with integer coordinates
2. **Coordinate Mode**: Advanced coordinate token system with specialized loss computation

Both modes use teacher-student learning with span-based loss splitting and enhanced loss management.

## 🏗️ **Training Mode Architecture**

### **Mode Comparison**

| Feature | Standard Mode | Coordinate Mode |
|---------|---------------|-----------------|
| **Coordinates** | Integer values in text | Specialized coordinate tokens |
| **Loss Computation** | Standard LLM loss only | LLM loss + coordinate loss |
| **Token Expansion** | 4 geometry tokens | Geometry + coordinate tokens |
| **Training Strategy** | Teacher-student learning | Teacher-student + coordinate regression |
| **Configuration** | `coordinate_tokens_enabled: false` | `coordinate_tokens_enabled: true` |

### **Loss Components**

```python
# Standard Mode
total_loss = teacher_llm_loss + student_llm_loss

# Coordinate Mode  
total_loss = teacher_llm_loss + student_llm_loss + coordinate_loss
```

## 🚀 **Training Workflows**

### **1. Standard Mode Training (Recommended)**

```bash
# Quick start with integer coordinates
python scripts/train.py --config configs/bbu_v2.yaml

# Monitor standard training progress
tail -f checkpoints/run_*/training.log
grep "loss=" checkpoints/run_*/training.log | tail -20
```

**Expected Log Output:**
```
Step 100: loss=2.345, lm_loss=2.345, eval_loss=2.123
Step 200: loss=2.234, teacher_llm_loss=1.125, student_llm_loss=1.109, eval_loss=2.089
```

### **2. Coordinate Mode Training (Advanced)**

```bash
# Start coordinate token training
python scripts/train.py --config configs/bbu_coordinate.yaml

# Monitor coordinate-specific metrics
grep "coordinate_loss" checkpoints/run_*/training.log | tail -10
```

**Expected Log Output:**
```
Step 100: loss=2.345, lm_loss=1.890, coordinate_loss=0.455, eval_loss=2.123
Step 200: loss=2.234, teacher_llm_loss=0.956, student_llm_loss=0.934, coordinate_loss=0.344, eval_loss=2.089
```

## 🔧 **Loss Computation System**

### **Enhanced Loss Manager**

The training system uses a sophisticated loss computation strategy with multiple components:

#### **1. Teacher-Student Loss Splitting**

```python
def compute_span_based_loss(self, labels, logits, teacher_spans, student_spans):
    """Compute loss split between teacher and student spans."""
    
    # Calculate total tokens for each type
    total_teacher_tokens = sum(end - start for start, end in teacher_spans)
    total_student_tokens = sum(end - start for start, end in student_spans)
    total_assistant_tokens = total_teacher_tokens + total_student_tokens
    
    # Handle edge cases
    if total_assistant_tokens == 0:
        return 0.0, total_llm_loss, coord_loss_total
    
    if total_teacher_tokens == 0:
        self.logger.warning("⚠️  No teacher tokens found in batch - teacher_llm_loss will be 0.0")
        return 0.0, total_llm_loss, coord_loss_total
    
    # Compute proportional losses
    teacher_llm_loss = (total_teacher_tokens / total_assistant_tokens) * total_llm_loss
    student_llm_loss = (total_student_tokens / total_assistant_tokens) * total_llm_loss
    
    return teacher_llm_loss, student_llm_loss, coord_loss_total
```

#### **2. Coordinate Loss Computation**

```python
def compute_coordinate_loss(self, coordinate_spans, labels, logits):
    """Compute specialized loss for coordinate tokens."""
    coord_loss_total = 0.0
    
    for start_idx, end_idx in coordinate_spans:
        # Extract coordinate token predictions
        coord_logits = logits[start_idx:end_idx]
        coord_labels = labels[start_idx:end_idx]
        
        # Apply coordinate-specific loss function
        coord_loss = F.cross_entropy(coord_logits.view(-1, coord_logits.size(-1)), 
                                   coord_labels.view(-1), 
                                   ignore_index=-100)
        coord_loss_total += coord_loss
    
    return coord_loss_total
```

### **Fixed Teacher Assignment Strategy**

The system now uses an improved teacher assignment algorithm that maintains consistent ratios:

```python
def assign_teacher_with_consistency(self, teacher_ratio=0.7):
    """Assign teachers with improved consistency to maintain target ratio."""
    
    total_samples = self._teacher_assignment_stats["total_samples"]
    expected_with_teacher = int(total_samples * teacher_ratio)
    current_with_teacher = self._teacher_assignment_stats["samples_with_teacher"]
    
    # Force teacher assignment if we're behind the expected ratio
    if current_with_teacher < expected_with_teacher:
        use_teachers = True
    else:
        # Calculate remaining ratio and assign probabilistically
        remaining_ratio = max(0.0, teacher_ratio - (current_with_teacher / total_samples))
        use_teachers = random.random() < remaining_ratio
    
    return use_teachers
```

**Before Fix**: Random 74.0% teacher assignment (highly variable)
**After Fix**: Consistent 70.0% teacher assignment (±5% variance)

## 📊 **Token Management and Embedding Expansion**

### **Token Expansion Scenarios**

#### **Standard Mode Expansion**
```python
# Adds 4 geometry tokens
new_tokens = ["<|line_start|>", "<|line_end|>", "<|square_start|>", "<|square_end|>"]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
```

#### **Coordinate Mode Expansion**  
```python
# Adds geometry tokens + coordinate tokens
geometry_tokens = ["<|line_start|>", "<|line_end|>", "<|square_start|>", "<|square_end|>"]
coordinate_tokens = [f"<|coord_{i}|>" for i in range(max_coord_value)]

# Add in batches to avoid memory issues
for batch in chunks(coordinate_tokens, 1000):
    tokenizer.add_tokens(batch)
```

### **Model Embedding Resizing**

```python
def resize_model_embeddings(self, num_new_tokens: int):
    """Resize model embeddings to accommodate new tokens."""
    old_size = self.model.get_input_embeddings().num_embeddings
    
    # Resize embedding layers
    self.model.resize_token_embeddings(len(self.tokenizer))
    
    new_size = self.model.get_input_embeddings().weight.shape[0]
    logger.info(f"🔧 Resized embeddings: {old_size} → {new_size} tokens")
    
    # Initialize new token embeddings with reasonable values
    with torch.no_grad():
        # Use mean and covariance of existing embeddings
        existing_embeddings = self.model.get_input_embeddings().weight.data[:old_size]
        new_embeddings = torch.normal(
            mean=existing_embeddings.mean(dim=0),
            std=existing_embeddings.std(dim=0),
            size=(num_new_tokens, existing_embeddings.size(1))
        )
        self.model.get_input_embeddings().weight.data[old_size:] = new_embeddings
```

## 📈 **Configuration Guide**

### **Standard Mode Configuration** (`configs/bbu_v2.yaml`)

```yaml
# Model and data paths
model_id: "/data3/model_cache/Qwen2.5-VL-7B-Instruct"
data_root: "data/ds_v2_full"
output_dir: "output-722/standard-mode"

# Training parameters
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 1e-5

# Standard mode settings
coordinate_tokens_enabled: false  # Standard integer coordinates
teacher_ratio: 0.7                 # 70% teacher samples
loss_type: "standard"              # Standard LLM loss only

# Performance optimization
fp16: true
gradient_checkpointing: true
dataloader_num_workers: 8
```

### **Coordinate Mode Configuration** (`configs/bbu_coordinate.yaml`)

```yaml
# Model and data paths  
model_id: "/data3/model_cache/Qwen2.5-VL-7B-Instruct"
data_root: "data/ds_v2_full"
output_dir: "output-722/coordinate-mode"

# Training parameters
num_train_epochs: 3
per_device_train_batch_size: 2     # Smaller batch for coordinate mode
gradient_accumulation_steps: 4
learning_rate: 1e-5

# Coordinate mode settings
coordinate_tokens_enabled: true    # Enable coordinate tokens
max_coord_value: 4096             # Maximum coordinate value
coordinate_loss_weight: 0.1       # Weight for coordinate loss
teacher_ratio: 0.7                # 70% teacher samples
loss_type: "coordinate"           # LLM + coordinate loss

# Token management
token_expansion_strategy: "coordinate"
embedding_initialization: "statistical"

# Performance optimization
fp16: true
gradient_checkpointing: true
dataloader_num_workers: 4         # Reduced for coordinate mode
```

## 🧪 **Training Validation and Testing**

### **Quick Validation Tests**

```bash
# Test both training modes
python scripts/train.py --config configs/bbu_v2.yaml --max_steps 5
python scripts/train.py --config configs/bbu_coordinate.yaml --max_steps 5

# Validate data loading
python -c "
from src.core.data_processor import DataProcessor
from src.config import load_config
config = load_config('configs/bbu_v2.yaml')
processor = DataProcessor(None, None, None, config=config)
train_dataset, eval_dataset = processor.create_datasets()
print(f'Train samples: {len(train_dataset)}')
print(f'Eval samples: {len(eval_dataset)}')
"

# Test model loading with token expansion
python -c "
from src.models.model_loader import load_model_and_processor_unified
from src.config import load_config
config = load_config('configs/bbu_coordinate.yaml')
model, tokenizer, processor = load_model_and_processor_unified(
    config.model_path, for_inference=False
)
print(f'Model loaded: {model.__class__.__name__}')
print(f'Vocabulary size: {len(tokenizer.get_vocab())}')
print(f'Coordinate tokens enabled: {config.coordinate_tokens_enabled}')
"
```

### **Loss Computation Validation**

```bash
# Monitor teacher assignment consistency
grep "samples_with_teacher\|teacher_ratio" checkpoints/run_*/training.log

# Verify loss components
grep "teacher_llm_loss\|student_llm_loss\|coordinate_loss" checkpoints/run_*/training.log | tail -20

# Check for loss computation warnings
grep "⚠️\|WARNING" checkpoints/run_*/training.log
```

## 🎯 **Training Strategies**

### **Progressive Training Approach**

```bash
# Stage 1: Quick validation (100 steps)
python scripts/train.py --config configs/bbu_v2.yaml --max_steps 100

# Stage 2: Loss component validation (500 steps)
python scripts/train.py --config configs/bbu_v2.yaml --max_steps 500

# Stage 3: Full training run (2000+ steps)
python scripts/train.py --config configs/bbu_v2.yaml --max_steps 2000
```

### **Hyperparameter Tuning**

#### **Learning Rate Optimization**
```bash
# Learning rate sweep for standard mode
for lr in 1e-5 5e-6 2e-5; do
    python scripts/train.py \
        --config configs/bbu_v2.yaml \
        --learning_rate $lr \
        --output_dir checkpoints/standard_lr_${lr} \
        --max_steps 500
done

# Learning rate sweep for coordinate mode
for lr in 5e-6 1e-5 2e-5; do
    python scripts/train.py \
        --config configs/bbu_coordinate.yaml \
        --learning_rate $lr \
        --output_dir checkpoints/coordinate_lr_${lr} \
        --max_steps 500
done
```

#### **Coordinate Loss Weight Tuning**
```bash
# Coordinate loss weight optimization
for weight in 0.01 0.05 0.1 0.2; do
    python scripts/train.py \
        --config configs/bbu_coordinate.yaml \
        --coordinate_loss_weight $weight \
        --output_dir checkpoints/coord_weight_${weight} \
        --max_steps 500
done
```

#### **Teacher Ratio Optimization**
```bash
# Teacher ratio sweep
for ratio in 0.5 0.7 0.8; do
    python scripts/train.py \
        --config configs/bbu_v2.yaml \
        --teacher_ratio $ratio \
        --output_dir checkpoints/teacher_ratio_${ratio} \
        --max_steps 500
done
```

## 📊 **Performance Monitoring**

### **Key Metrics to Track**

#### **Standard Mode Metrics**
- **`loss`**: Combined training loss
- **`lm_loss`**: Language modeling loss  
- **`teacher_llm_loss`**: Teacher span loss
- **`student_llm_loss`**: Student span loss
- **`eval_loss`**: Validation loss

#### **Coordinate Mode Metrics**
- **`loss`**: Combined training loss
- **`lm_loss`**: Language modeling component
- **`coordinate_loss`**: Coordinate prediction loss
- **`teacher_llm_loss`**: Teacher span LLM loss
- **`student_llm_loss`**: Student span LLM loss
- **`eval_loss`**: Validation loss

### **Monitoring Commands**

```bash
# Real-time training monitoring
tail -f checkpoints/run_*/training.log

# Extract loss trends
grep "loss=" checkpoints/run_*/training.log | tail -50

# Teacher assignment analysis
grep "teacher_llm_loss\|student_llm_loss" checkpoints/run_*/training.log | \
    awk '{print $2, $3, $4}' | tail -20

# Performance metrics
grep "train_samples_per_second" checkpoints/run_*/training.log | tail -10
watch -n 1 nvidia-smi
```

## 🔍 **Checkpoint Management**

### **Model Saving with Expanded Vocabulary**

```python
def save_checkpoint_with_expanded_vocab(self, output_dir: str):
    """Save model and tokenizer with expanded vocabulary."""
    
    # Save model (handles wrapper models)
    if hasattr(self.model, "base_model"):
        self.model.base_model.save_pretrained(output_dir)
    else:
        self.model.save_pretrained(output_dir)
    
    # Save tokenizer with new tokens
    self.tokenizer.save_pretrained(output_dir)
    
    # Save training state
    trainer_state = {
        "vocabulary_size": len(self.tokenizer),
        "coordinate_tokens_enabled": self.config.coordinate_tokens_enabled,
        "token_expansion_applied": True,
        "training_mode": "coordinate" if self.config.coordinate_tokens_enabled else "standard"
    }
    
    with open(os.path.join(output_dir, "training_state.json"), "w") as f:
        json.dump(trainer_state, f, indent=2)
```

### **Checkpoint Verification**

```bash
# Verify checkpoint integrity
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load and verify checkpoint
model_path = 'checkpoints/run_001/checkpoint-1000'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

print(f'Tokenizer vocab size: {len(tokenizer)}')
print(f'Model embedding size: {model.get_input_embeddings().num_embeddings}')
print(f'Sizes match: {len(tokenizer) == model.get_input_embeddings().num_embeddings}')
"

# Check coordinate tokens in vocabulary
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('checkpoints/run_001/checkpoint-1000')
coord_tokens = [token for token in tokenizer.get_vocab().keys() if 'coord_' in token]
print(f'Coordinate tokens found: {len(coord_tokens)}')
print(f'Sample tokens: {coord_tokens[:10]}')
"
```

## 🚨 **Troubleshooting Training Issues**

### **Common Problems and Solutions**

#### **1. Teacher Loss Always Zero**
```bash
# Check teacher assignment statistics
grep "teacher_assignment_stats\|samples_with_teacher" checkpoints/run_*/training.log

# Verify teacher ratio configuration
grep "teacher_ratio" configs/bbu_v2.yaml

# Monitor batch-level teacher presence
grep "⚠️.*No teacher tokens" checkpoints/run_*/training.log
```

**Solution**: The fixed teacher assignment algorithm ensures consistent teacher ratios.

#### **2. Loss Components Don't Sum Correctly**
```bash
# Verify loss computation
grep "teacher_llm_loss.*student_llm_loss.*coordinate_loss" checkpoints/run_*/training.log | \
    python -c "
import sys
for line in sys.stdin:
    # Parse and verify loss components
    parts = line.strip().split()
    # Add verification logic
"
```

**Solution**: Use the enhanced loss manager with proper span-based computation.

#### **3. Coordinate Loss Computation Errors**
```bash
# Check coordinate token detection
grep "coordinate.*tokens.*detected" checkpoints/run_*/training.log

# Verify coordinate spans
grep "coordinate_spans\|coord_loss" checkpoints/run_*/training.log
```

**Solution**: Ensure coordinate tokens are properly added to the tokenizer and detected in the loss computation.

#### **4. Memory Issues with Token Expansion**
```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Check embedding layer size
python -c "
from src.models.model_loader import load_model_and_processor_unified
model, _, _ = load_model_and_processor_unified('model_path')
embedding_params = model.get_input_embeddings().weight.numel()
print(f'Embedding parameters: {embedding_params:,}')
"
```

**Solution**: Use smaller batch sizes and gradient accumulation for coordinate mode.

## 📚 **Summary**

The BBU training system provides two sophisticated training modes:

### **Standard Mode Benefits**
- **✅ Faster training**: No coordinate token expansion overhead
- **✅ Smaller memory footprint**: Fewer parameters
- **✅ Proven reliability**: Well-tested integer coordinate approach
- **✅ Easier deployment**: Standard model format

### **Coordinate Mode Benefits**  
- **✅ Precise coordinate prediction**: Specialized coordinate tokens
- **✅ Advanced loss computation**: Multi-component loss optimization
- **✅ Better spatial accuracy**: Dedicated coordinate regression
- **✅ Enhanced multi-geometry support**: Native token-based geometry handling

Both modes include:
- **Enhanced teacher assignment strategy** for consistent training ratios
- **Robust loss computation** with proper error handling
- **Token embedding expansion** with statistical initialization
- **Comprehensive monitoring** and debugging capabilities
- **Flexible checkpoint management** for production deployment

Choose **Standard Mode** for fast, reliable training with proven integer coordinates, or **Coordinate Mode** for advanced spatial precision with specialized coordinate token prediction.