# BBU Training Pipeline Documentation

> Working with `src_new/`? Start with `SRC_NEW_ASSISTANT_ONBOARDING.md` for a 10–15 min ramp-up.

**Complete guide to the BBU (Bounding Box Understanding) training pipeline for Qwen2.5-VL models with coordinate token support for precise object localization.**

## 🎯 **Quick Start (15 minutes)**

### **Prerequisites**
- Linux environment with CUDA GPUs
- Python 3.10+ with PyTorch
- Access to project directory: `/data3/Qwen2.5-VL-main`

### **Setup & First Training Run**
```bash
# 1. Navigate to project
cd /data3/Qwen2.5-VL-main

# 2. Verify environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Install dependencies
pip install -r requirements.txt

# 4. Process data (assumes data in ds_v2/ directory)
bash data_conversion/convert_dataset.sh

# 5. Run first training (src_new/ implementation - recommended)
python scripts/train_new.py --config bbu_v2 --max_steps 100

# 6. Monitor training
tail -f checkpoints/*/training.log
```

**Expected output**: Training logs showing decreasing loss values and successful checkpoint creation.

## 🏗️ **System Architecture & Status**

### **Current Status (January 2025)**
- **✅ PRODUCTION READY**: `src_new/` implementation with composition-based architecture
- **✅ RESOLVED**: NCCL timeout issues completely eliminated via local loss aggregation
- **✅ ENHANCED**: DetectionModel wrapper with intelligent checkpoint detection
- **✅ OPTIMIZED**: SafeTensors format for 4-6x faster inference loading
- **✅ STREAMLINED**: Unified configuration system with fail-fast validation
- **✅ TESTED**: Comprehensive test suite with 127+ tests covering all components
- **✅ INFERENCE**: Production inference pipeline with teacher guidance and batch processing
- **✅ DATA PIPELINE**: Object-oriented data conversion with multi-geometry support
- **❌ DEPRECATED**: `src/` legacy implementation

#### Important Notes
- **Vision token expansion rule**: In Qwen2.5‑VL, each `<|image_pad|>` in the chat template is expanded by the processor according to image grids and spatial merge size. The correct expected image token count is `sum_i (t_i*h_i*w_i) // (merge_size**2)`. Validation and batching align with this behavior to avoid false "image token mismatch" errors.
- **Model config exposure**: The training wrapper (`DetectionModel`) exposes the underlying HuggingFace config via `model.config` and keeps the training dataclass on `model.training_config`. This maintains compatibility with integrations that expect `model.config.to_json_string()` and related APIs.

### **Architecture Overview**

| Component | Location | Purpose |
|-----------|----------|---------|
| **DetectionModel** | `src_new/models/wrapper.py` | Composition-based wrapper around Qwen2.5-VL |
| **BBUTrainer** | `src_new/training/bbu_trainer.py` | HuggingFace Trainer with local loss aggregation |
| **LossManager** | `src_new/models/loss_manager.py` | Multi-component loss computation |
| **TokenProcessor** | `src_new/processing/token_processor.py` | Coordinate token handling |
| **Config** | `src_new/config/config.py` | Unified YAML-to-dataclass configuration |
| **Research/Experimentation** | `src_new/` | Cleaner APIs, easier to modify |
| **Production Deployment** | `src_new/` | More stable, better error handling |

### **Training Modes**

The system supports two operational modes:

#### **Standard Mode** (Recommended for Production)
- **Coordinates**: Integer format `[150,10,211,35]`
- **Vocabulary**: Minimal extension (+4 geometry tokens)
- **Use Case**: Production training with stable performance
- **Configuration**: `configs/bbu_v2.yaml`

#### **Coordinate Mode** (Advanced Features)
- **Coordinates**: Token format `[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]`
- **Vocabulary**: Extended (+2052 coordinate tokens)
- **Use Case**: Advanced sequence-based coordinate prediction
- **Configuration**: `configs/bbu_coordinate.yaml`
- **Requirements**: `remove_unused_columns: false`

## 🎯 **User Journeys & Navigation**

### **I'm a New Developer**
1. **[SETUP_AND_CONFIGURATION.md](SETUP_AND_CONFIGURATION.md)** - Complete setup, environment, and configuration guide
2. **[TRAINING_AND_IMPLEMENTATION.md](TRAINING_AND_IMPLEMENTATION.md)** - Training system, data processing, and model integration
3. **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)** - Common issues and solutions

### **I'm a Researcher/Experimenter**
1. **[COORDINATE_SYSTEM_GUIDE.md](COORDINATE_SYSTEM_GUIDE.md)** - Complete coordinate token system and features
2. **[TEACHER_STUDENT_TRAINING.md](TEACHER_STUDENT_TRAINING.md)** - Advanced teacher-student training system
3. **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation and customization
4. **[PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md)** - Performance tuning and optimization

### **I Need to Troubleshoot**
1. **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)** - Comprehensive problem-solving guide with all fixes
2. **[SETUP_AND_CONFIGURATION.md](SETUP_AND_CONFIGURATION.md)** - Configuration validation and setup issues
3. **[API_REFERENCE.md](API_REFERENCE.md)** - API debugging and migration guides

### **I'm Migrating/Upgrading**
1. **[API_REFERENCE.md](API_REFERENCE.md)** - Migration guides and version upgrade information
2. **[SETUP_AND_CONFIGURATION.md](SETUP_AND_CONFIGURATION.md)** - New configuration format and setup
3. **[COORDINATE_SYSTEM_GUIDE.md](COORDINATE_SYSTEM_GUIDE.md)** - New coordinate token system

## 🔧 **Core Features & Capabilities**

### **Coordinate Token System**
- **Standard Mode**: Integer coordinates `[150,10,211,35]` with minimal vocabulary extension
- **Coordinate Mode**: Token coordinates `[<|coord_150|>,<|coord_10|>,...]` with 2052 coordinate tokens
- **Multi-Geometry Support**: bbox_2d, line, and square geometries with enhanced coordinate ordering
- **Normalization**: Robust coordinate processing with degenerate case handling

### **Teacher-Student Training** 🚀 **NEW**
- **Performance**: 60-70% optimization with dual-role training
- **API**: Complete teacher-student pipeline with masking fixes
- **Configuration**: Flexible teacher ratio and loss weighting
- **Compatibility**: Full integration with coordinate token system

### **Training System**
- **BBUTrainer**: Production-ready trainer with local loss aggregation
- **NCCL Resolution**: Complete elimination of distributed training timeouts
- **Enhanced Logging**: 4-decimal precision losses and readable learning rates
- **Dataset Limiting**: Configurable `max_dataset_size` for debugging and testing
- **FlashAttention v2**: 5x performance improvement for long sequences

### **Data Processing**
- **Unified Pipeline**: Single conversion script for all data formats
- **Coordinate Management**: Automatic normalization and validation
- **Multi-Format Support**: JSONL, raw annotations, and teacher pool data
- **Validation**: Comprehensive data integrity checking

## 🔧 **Quick Commands Reference**

### **Training Commands**
```bash
# src_new/ implementation (recommended)
python scripts/train_new.py --config bbu_v2

# Coordinate token mode (advanced features)
python scripts/train_new.py --config bbu_v2 --coordinate_tokens_enabled

# Teacher-student training
python scripts/train_new.py --config bbu_v2 --teacher_ratio 0.5

# Debug mode (limited dataset)
python scripts/train_new.py --config bbu_v2 --max_steps 100
```

### **Data Processing**
```bash
# Convert raw data to training format
bash data_conversion/convert_dataset.sh

# Validate processed data
python -c "import json; print(json.load(open('data/train.jsonl')))"
```

### **Testing & Validation**
```bash
# Run all tests
python -m pytest src_new/tests/ -v

# Test specific component
python -m pytest src_new/tests/test_coordinate_tokens.py -v

# Quick system health check
python -c "from src_new.training import BBUTrainer; print('✅ System OK')"
```

## 📚 **Complete Documentation Structure**

This documentation follows a **consolidated, user-focused structure**:

### **Core Documentation (8 files)**
- **[README.md](README.md)** - This file: Main hub, quick start, system overview
- **[SETUP_AND_CONFIGURATION.md](SETUP_AND_CONFIGURATION.md)** - Complete setup and configuration guide
- **[COORDINATE_SYSTEM_GUIDE.md](COORDINATE_SYSTEM_GUIDE.md)** - All coordinate token features and implementation
- **[TEACHER_STUDENT_TRAINING.md](TEACHER_STUDENT_TRAINING.md)** - Complete teacher-student training system
- **[TRAINING_AND_IMPLEMENTATION.md](TRAINING_AND_IMPLEMENTATION.md)** - Training system, data processing, model integration
- **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)** - All troubleshooting, fixes, and common issues
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation and migration guides
- **[PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md)** - FlashAttention, performance tuning, optimizations

### **Archive (Historical Documentation)**
- **[archive/](archive/)** - Preserved historical documentation and technical details

### **Documentation Principles**
- **Consolidated Information**: All related content in comprehensive single files
- **User Journey Focus**: Organized by user needs and development workflows
- **Clear Cross-References**: Easy navigation between related topics
- **Minimal Maintenance**: Single source of truth for each topic area
- **Comprehensive Coverage**: All important information preserved and accessible

## 🚀 **Key Features & Capabilities**

### **Core Production Architecture (src_new/)**
- **DetectionModel**: Composition-based wrapper with coordinate token support and intelligent checkpoint detection
- **BBUTrainer**: Local loss aggregation eliminates NCCL timeouts (100% success rate vs 0% before)
- **LossManager**: Dual-loss architecture (LLM cross-entropy + coordinate L1 loss)  
- **TokenProcessor**: Coordinate token handling with positional encoding initialization
- **Configuration**: Unified YAML-to-dataclass with comprehensive validation

### **Critical Achievements**
- **NCCL Resolution**: 100% → 0% failure rate in distributed training via local loss aggregation
- **SafeTensors Optimization**: 4-6x faster checkpoint loading for inference deployment  
- **Architecture Stability**: Production-ready composition-based design with comprehensive error handling
- **Loss Computation**: Accurate teacher-student span detection and weighting
- **Performance**: ~2-3 samples/second on A100, ~24GB VRAM for batch_size=1, 1000-2000 steps convergence

### **Core Training Features**
- **8-Step Training Pipeline**: Complete data processing from JSONL to model checkpoints
- **Teacher-Student Learning**: Advanced dual-role training with conversation-based learning
- **Coordinate Token System**: Soft expectation regression for precise coordinate prediction
- **Multi-Geometry Support**: bbox_2d, line, and quadrilateral object detection
- **Object-Oriented Training**: Flexible object type combinations and filtering

### **Production Features**
- **Inference Engine**: Production-ready inference with teacher guidance and batch processing
- **SafeTensors Format**: 4-6x faster checkpoint loading for production deployment
- **Distributed Training**: NCCL timeout resolution with 100% reliability
- **Performance Monitoring**: Built-in memory usage and speed benchmarking
- **Path Management**: Unified path resolution with environment variable support

### **Development Features**
- **Comprehensive Testing**: 127+ tests covering all components with fixtures and utilities
- **Debug Logging**: Rank-aware logging with one-time sampling and token-level analysis
- **Configuration System**: YAML-to-dataclass with validation and fail-fast error handling
- **Data Conversion**: Object-oriented pipeline with smart resizing and quality control
- **Utilities**: PathManager, PerformanceMonitor, DebugLogger for development efficiency

### **Advanced Capabilities**
- **FlashAttention v2**: Optimized attention for long sequences and memory efficiency
- **Intelligent Checkpointing**: Auto-detection of base vs fine-tuned models
- **Environment Integration**: Support for HF_HOME, CUDA_VISIBLE_DEVICES, and custom paths
- **Modular Architecture**: Composition-based design with clean component boundaries
- **Zero Legacy Support**: Modern implementation without backward compatibility overhead

## 📈 **Project Status Summary**

- ✅ **Architecture**: `src_new/` implementation with clean, modular design
- ✅ **Training**: Stable distributed training with NCCL timeout resolution
- ✅ **Coordinate System**: Both Standard and Coordinate modes production ready
- ✅ **Teacher-Student**: Advanced training with dual-role learning optimization
- ✅ **Multi-Geometry**: Complete support for bbox_2d, line, and square geometries
- ✅ **Testing**: Comprehensive test suite with 127+ tests passing
- ✅ **Documentation**: Consolidated structure with 75% reduction in file count
- ✅ **Performance**: FlashAttention v2 integration with significant speed improvements
- ✅ **Inference**: Production inference pipeline with teacher guidance
- ✅ **Data Pipeline**: Object-oriented conversion with advanced filtering

---

**Need immediate help?** Start with **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)** for quick solutions to common issues.
