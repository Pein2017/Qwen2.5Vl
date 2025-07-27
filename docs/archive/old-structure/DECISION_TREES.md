# Decision Trees - Choose the Right Path

**Quick decision guides to help developers choose the right components and approaches**

## 🎯 I Want to Train a Model

### What Type of Training?

```
Do you have processed training data (train.jsonl, val.jsonl)?
├─ YES → Go to "Training Setup"
└─ NO → Go to "Data Processing" first

Training Setup:
├─ First time training?
│  ├─ YES → Use: docs/quick-start/first-training.md
│  └─ NO → Continue below
├─ Standard training?
│  ├─ YES → Use: create_trainer_with_coordinator(training_args)
│  └─ NO → Continue below
├─ Custom loss logic?
│  ├─ YES → Extend: src/training/loss_manager.py
│  └─ NO → Continue below
├─ Custom training loop?
│  ├─ YES → Extend: src/training/trainer.py
│  └─ NO → Use factory pattern
└─ Memory constraints?
   ├─ YES → Use: configs/low_memory.yaml template
   └─ NO → Use: configs/base_flat_v2.yaml template
```

### Training Configuration Decision Tree

```
What's your GPU setup?
├─ 4x A100 (40GB each)
│  └─ Use: per_device_train_batch_size: 2, gradient_accumulation_steps: 4
├─ 2x A100 (80GB each)  
│  └─ Use: per_device_train_batch_size: 4, gradient_accumulation_steps: 2
├─ 4x RTX 3090 (24GB each)
│  └─ Use: per_device_train_batch_size: 1, gradient_accumulation_steps: 8
└─ Less than 24GB per GPU
   └─ Use: per_device_train_batch_size: 1, gradient_checkpointing: true

What's your training goal?
├─ Quick experiment (< 1 hour)
│  └─ Use: num_train_epochs: 1, max_steps: 500
├─ Standard training (2-4 hours)
│  └─ Use: num_train_epochs: 3, learning_rate: 1e-5
├─ Production model (8+ hours)
│  └─ Use: num_train_epochs: 5, learning_rate: 1e-5, save_steps: 1000
└─ Research/experimentation
   └─ Use: Custom config with detailed logging
```

## 🔄 I Want to Process Data

### Data Processing Decision Tree

```
What's your input data format?
├─ Raw vendor JSON files in ds_v2/
│  └─ Use: bash data_conversion/convert_dataset.sh
├─ Already have clean JSON files
│  └─ Use: python data_conversion/pipeline_manager.py --skip-cleaning
├─ Custom data format
│  └─ Extend: data_conversion/unified_processor.py
└─ Need specific object types only
   └─ Set: export OBJECT_TYPES="bbu label" (see below)

What object types do you need?
├─ Equipment detection only
│  └─ Use: OBJECT_TYPES="bbu bbu_shield"
├─ Text recognition only
│  └─ Use: OBJECT_TYPES="label"
├─ Cable system only
│  └─ Use: OBJECT_TYPES="fiber wire"
├─ Hardware components only
│  └─ Use: OBJECT_TYPES="connect_point"
├─ Everything
│  └─ Use: OBJECT_TYPES="full"
└─ Custom combination
   └─ Use: OBJECT_TYPES="bbu label fiber" (space-separated)

What's your data size?
├─ Small dataset (< 1000 images)
│  └─ Use: Standard processing
├─ Medium dataset (1000-10000 images)
│  └─ Use: PARALLEL_WORKERS=4
├─ Large dataset (> 10000 images)
│  └─ Use: STREAMING_MODE=true, BATCH_SIZE=100
└─ Very large dataset (> 100000 images)
   └─ Use: Distributed processing across multiple machines
```

## 🤖 I Want to Load a Model

### Model Loading Decision Tree

```
What's your use case?
├─ Training
│  └─ Use: load_model_and_processor_unified(model_path, for_inference=False)
├─ Inference
│  └─ Use: load_model_and_processor_unified(model_path, for_inference=True)
├─ Custom model modifications
│  └─ Use: Manual loading + patches
└─ Research/experimentation
   └─ Use: Custom wrapper

What's your model source?
├─ Base Qwen2.5-VL model
│  └─ Use: model_path="/path/to/qwen2.5-vl-7b-instruct"
├─ Trained checkpoint
│  └─ Use: model_path="output/checkpoint-1000"
├─ Final trained model
│  └─ Use: model_path="output/final_model"
└─ HuggingFace model
   └─ Use: model_path="Qwen/Qwen2.5-VL-7B-Instruct"

What optimizations do you need?
├─ Maximum speed
│  └─ Use: attn_implementation="flash_attention_2", torch_dtype="bfloat16"
├─ Maximum memory efficiency
│  └─ Use: torch_dtype="float16", gradient_checkpointing=True
├─ Maximum compatibility
│  └─ Use: Default settings
└─ Custom optimizations
   └─ Modify: src/models/patches.py
```

## 🔧 I Want to Configure the System

### Configuration Decision Tree

```
What configuration approach?
├─ Use existing template
│  ├─ Standard training → configs/base_flat_v2.yaml
│  ├─ Memory constrained → configs/low_memory.yaml  
│  ├─ Fast experiment → configs/fast_experiment.yaml
│  └─ Production → configs/production.yaml
├─ Modify existing template
│  └─ Copy template → Edit parameters → Use custom config
├─ Create from scratch
│  └─ Use: DirectConfig parameter reference
└─ Environment-specific
   └─ Use: Environment variable overrides

What parameters do you need to change?
├─ Model settings
│  └─ Edit: model_path, model_max_length, torch_dtype
├─ Training settings  
│  └─ Edit: learning_rate, num_train_epochs, batch_size
├─ Data settings
│  └─ Edit: train_data_path, val_data_path, teacher_ratio
├─ Detection settings
│  └─ Edit: detection_enabled, coordinate_tokens_enabled
├─ Memory settings
│  └─ Edit: gradient_checkpointing, dataloader_num_workers
└─ Logging settings
   └─ Edit: logging_steps, save_steps, output_dir

How do you want to apply configuration?
├─ YAML file
│  └─ Use: --config path/to/config.yaml
├─ Environment variables
│  └─ Use: export PARAMETER_NAME=value
├─ Command line
│  └─ Use: --parameter_name value
└─ Python code
   └─ Use: config = get_config(); config.parameter_name = value
```

## 🐛 I Have a Problem

### Troubleshooting Decision Tree

```
What type of problem?
├─ Import errors
│  ├─ ModuleNotFoundError: src → Check: pwd, PYTHONPATH
│  ├─ AttributeError: DirectConfig → Check: parameter names in global_config.py
│  └─ ImportError: component → Check: component exists, spelling
├─ Memory errors
│  ├─ CUDA out of memory → Reduce: batch_size, enable gradient_checkpointing
│  ├─ CPU out of memory → Enable: streaming_mode, reduce workers
│  └─ Disk space → Clean: intermediate files, old checkpoints
├─ Training errors
│  ├─ Loss not decreasing → Check: learning_rate, data quality
│  ├─ Training too slow → Enable: flash_attention_2, increase batch_size
│  ├─ Coordinator setup failed → Check: model/tokenizer initialization
│  └─ Loss components wrong → Check: coordinate token setup
├─ Data processing errors
│  ├─ Pipeline stage failed → Check: logs, resume from stage
│  ├─ Invalid coordinates → Check: coordinate transformation
│  ├─ No samples found → Check: object_types, data format
│  └─ File not found → Check: input directory, file paths
└─ Inference errors
   ├─ Model loading failed → Check: model path, file integrity
   ├─ Generation failed → Check: coordinate tokens, memory
   ├─ Parsing failed → Check: response format, parser logic
   └─ Results invalid → Check: model training, coordinate scaling

Where to get help?
├─ Quick fixes → docs/quick-start/common-issues.md
├─ Comprehensive troubleshooting → docs/reference/troubleshooting.md
├─ Component-specific → docs/components/
└─ Architecture understanding → docs/MENTAL_MODEL.md
```

## 🔍 I Want to Understand the System

### Learning Path Decision Tree

```
What's your background?
├─ New to the project
│  ├─ 5 minutes → docs/MENTAL_MODEL.md
│  ├─ 15 minutes → docs/quick-start/README.md
│  └─ 30 minutes → docs/quick-start/first-training.md
├─ Familiar with ML/VLM
│  ├─ Architecture → docs/ARCHITECTURE.md
│  ├─ Training system → docs/components/training-system.md
│  └─ Coordinate tokens → docs/MENTAL_MODEL.md#coordinate-token-innovation
├─ Experienced developer
│  ├─ Code structure → docs/PROJECT_MAP.md
│  ├─ API reference → docs/reference/api.md
│  └─ Extension points → docs/components/
└─ Researcher/experimenter
   ├─ Innovation details → docs/ARCHITECTURE.md#coordinate-token-innovation
   ├─ Training methodology → docs/workflows/training.md
   └─ Data pipeline → docs/workflows/data-processing.md

What do you want to do?
├─ Understand the innovation
│  └─ Read: Coordinate token system, teacher-student learning
├─ Extend the system
│  └─ Read: Component contracts, extension points
├─ Debug issues
│  └─ Read: Component interaction, error handling
├─ Optimize performance
│  └─ Read: Performance characteristics, optimization guides
└─ Research applications
   └─ Read: Multi-geometry support, hierarchical descriptions
```

## 🚀 I Want to Deploy/Use the System

### Deployment Decision Tree

```
What's your deployment target?
├─ Development/testing
│  └─ Use: Local training and inference
├─ Production inference
│  ├─ Single image → Use: src/inference.py
│  ├─ Batch processing → Use: Batch inference script
│  ├─ REST API → Use: Flask/FastAPI wrapper
│  └─ Real-time → Use: Optimized inference pipeline
├─ Distributed training
│  └─ Use: torchrun with multiple GPUs/nodes
└─ Cloud deployment
   └─ Use: Docker containers with GPU support

What performance requirements?
├─ Maximum accuracy
│  └─ Use: Full model, no quantization
├─ Balanced accuracy/speed
│  └─ Use: FP16, Flash Attention 2
├─ Maximum speed
│  └─ Use: Quantization, model pruning
└─ Minimum memory
   └─ Use: Model sharding, gradient checkpointing

What integration needs?
├─ Standalone application
│  └─ Use: Command-line interface
├─ Web service
│  └─ Use: REST API with JSON responses
├─ Python library
│  └─ Use: Import components directly
├─ Batch processing system
│  └─ Use: Batch inference scripts
└─ Real-time system
   └─ Use: Optimized inference pipeline
```

## 🔄 Quick Reference Paths

### Common Workflows
```bash
# New user getting started
docs/MENTAL_MODEL.md → docs/quick-start/README.md → docs/quick-start/first-training.md

# Developer extending system
docs/PROJECT_MAP.md → docs/components/ → docs/reference/api.md

# Researcher understanding innovation
docs/ARCHITECTURE.md → docs/workflows/ → docs/advanced/

# Troubleshooter fixing issues
docs/quick-start/common-issues.md → docs/reference/troubleshooting.md → docs/components/

# Production deployment
docs/workflows/inference.md → docs/reference/api.md → Custom integration
```

### Component Selection Guide
```python
# Training: Always use factory pattern unless customizing
trainer = create_trainer_with_coordinator(training_args)

# Model loading: Always use unified loader
model, tokenizer, image_processor = load_model_and_processor_unified(...)

# Data processing: Use shell script for standard, Python API for custom
# Standard: bash data_conversion/convert_dataset.sh
# Custom: PipelineManager(custom_config).run_pipeline()

# Configuration: Use DirectConfig for all parameter access
config = get_config()
parameter_value = config.parameter_name

# Inference: Use inference.py for standard, custom wrapper for integration
# Standard: python src/inference.py --model_path ... --image_path ...
# Custom: load_model_and_processor_unified() + custom logic
```

---

**Quick Navigation**:
- **Mental Model**: [MENTAL_MODEL.md](MENTAL_MODEL.md) - Understand the system
- **Project Map**: [PROJECT_MAP.md](PROJECT_MAP.md) - Navigate the codebase  
- **Quick Start**: [quick-start/README.md](quick-start/README.md) - Get started fast
- **Components**: [components/](components/) - Deep dive into components
- **Workflows**: [workflows/](workflows/) - End-to-end processes
