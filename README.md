# Qwen2.5-VL Multi-modal AI for Engineering Quality Inspection

This project implements an end-to-end AI quality inspection system using the Qwen2.5-VL multi-modal vision-language model. The system specializes in BBU (Base-Band Unit) equipment detection and captioning with sophisticated object detection and natural language descriptions.

## 🚀 **Quick Start**

### **New to the Project?**
1. **[Choose Implementation](docs/IMPLEMENTATION_GUIDE.md)** - Decide between `src/` (legacy) vs `src_new/` (recommended)
2. **[Getting Started](docs/getting-started.md)** - Quick setup and first training run
3. **[Complete Documentation](docs/README.md)** - Full documentation index and navigation

### **Key Documentation**
- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Choose between src/ vs src_new/ implementations
- **[Teacher-Student Training](docs/TEACHER_STUDENT_TRAINING_GUIDE.md)** - 🚀 **NEW**: Production-ready dual-role training
- **[Teacher-Student API](docs/TEACHER_STUDENT_API.md)** - Complete API documentation for teacher-student pipeline
- **[Configuration Guide](docs/implementation/configuration.md)** - Detailed configuration options
- **[Troubleshooting](docs/troubleshooting/common-issues.md)** - Common issues and solutions (both implementations)
- **[Migration Guide](docs/guides/migration-src-to-src-new.md)** - Upgrade from src/ to src_new/

## 🏗️ **Architecture Overview**

This project provides **two implementations**:

### **src_new/ (Recommended)**
- ✅ **25% less code** with better functionality
- ✅ **127 comprehensive tests** - well tested and reliable
- ✅ **Composition-based architecture** - easier to understand and maintain
- ✅ **Fail-fast validation** - immediate error detection
- ✅ **Production ready** - optimized for performance
- 🚀 **Teacher-Student Training** - Dual-role learning with 60-70% performance optimization

### **src/ (Legacy)**
- 🔄 **Mature codebase** with extensive history
- ⚠️ **Complex inheritance** - harder to maintain
- ⚠️ **Limited test coverage** - less reliable
- 📋 **Maintained for compatibility** - consider migrating to src_new/

## ⚡ **Quick Training Commands**

### **For src_new/ (Recommended)**
```bash
# 1. Activate environment
conda activate ms

# 2. Process your data
bash data_conversion/convert_dataset.sh

# 3. Start training (src_new)
python scripts/train_new.py --config configs/bbu_v2/base.yaml

# 4. Run tests
python -m pytest src_new/tests/ -v
```

### **For src/ (Legacy)**
```bash
# 1. Activate environment
conda activate ms

# 2. Process your data
bash data_conversion/convert_dataset.sh

# 3. Start training (legacy)
python -m src.training.trainer --config configs/bbu_v2/base.yaml

# 4. Run inference
python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg
```

> **💡 Tip**: New projects should use `src_new/` for better performance and maintainability. See [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md) for details.

## 📋 **Key Features**

### **Core Capabilities**
- **Multi-modal Vision-Language**: End-to-end object detection with English/Chinese descriptions
- **Coordinate Token System**: Soft expectation regression with automatic bbox→token conversion
- **Multi-Geometry Support**: bbox_2d, line, and quad geometries
- **Teacher-Student Learning**: Multi-task training with span-based loss splitting
- **Production Ready**: Memory optimization, gradient scaling, and comprehensive testing

### **Technical Highlights**
- **Modular Architecture**: Component-based design with clear separation of concerns
- **Enhanced Loss Management**: Mode-aware coordinate vs. standard LLM loss computation
- **HuggingFace Integration**: Direct compatibility with HF Trainer and ecosystem
- **Comprehensive Testing**: 127+ tests ensuring reliability and performance

## 📚 **Documentation Structure**

For complete documentation, see the [**docs/**](docs/) directory:

- **[docs/README.md](docs/README.md)** - Main documentation hub
- **[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)** - Choose src/ vs src_new/
- **[docs/getting-started.md](docs/getting-started.md)** - Quick setup guide
- **[docs/troubleshooting/](docs/troubleshooting/)** - Common issues and solutions
- **[docs/implementation/](docs/implementation/)** - Technical implementation details
- **[docs/features/](docs/features/)** - Feature-specific documentation

## 🆘 **Getting Help**

### **Common Issues**
- Check [**Troubleshooting Guide**](docs/troubleshooting/common-issues.md) for solutions
- Review [**Implementation Guide**](docs/IMPLEMENTATION_GUIDE.md) to choose the right approach
- Run tests: `python -m pytest src_new/tests/ -v` (for src_new) or `python -m pytest tests/ -v` (for src)

### **Migration Support**
- See [**Migration Guide**](docs/guides/migration-src-to-src-new.md) to upgrade from src/ to src_new/
- Compare implementations in [**Implementation Guide**](docs/IMPLEMENTATION_GUIDE.md)

---

**Recommendation**: Use `src_new/` for all new projects. Consider migrating existing `src/` projects for better maintainability and performance.

## 🔧 **Development**

### **Testing**
```bash
# Test src_new/ (recommended)
python -m pytest src_new/tests/ -v

# Test src/ (legacy)
python -m pytest tests/ -v
```

### **Documentation**
```bash
# Browse documentation
ls docs/  # See all available documentation

# Key documentation files
cat docs/IMPLEMENTATION_GUIDE.md  # Choose implementation
cat docs/getting-started.md       # Quick start
cat docs/troubleshooting/common-issues.md  # Common issues
```

## 📄 **License**

MIT License - see LICENSE file for details.