# BBU Training Pipeline Documentation

Welcome to the BBU (Bounding Box Understanding) training pipeline documentation. This system provides fine-tuning capabilities for Qwen2.5-VL models with coordinate token support for precise object localization.

## 🚀 **Quick Navigation**

### **New to BBU?** Start Here
- [**Getting Started**](getting-started.md) - 15-minute setup and first training run
- [**Configuration**](implementation/configuration.md) - Complete configuration guide
- [**Architecture**](reference/architecture.md) - System overview and components

### **Core Features**
- [**Coordinate Tokens**](features/coordinate-tokens.md) - Advanced coordinate token system (Standard + Coordinate modes)
- [**Multi-Geometry Support**](features/multi-geometry.md) - Support for bbox_2d, line, and square geometries
- [**Coordinate Normalization**](features/coordinate-normalization.md) - Robust coordinate processing and degenerate case handling
- [**Training System**](implementation/training-system.md) - Training workflows, monitoring, and optimization
- [**API Reference**](guides/api-reference.md) - Complete API documentation

### **Implementation Documentation**
- [**Data Conversion**](implementation/data-conversion.md) - Data processing pipeline and coordinate management
- [**Model System**](implementation/model-system.md) - Qwen2.5-VL integration and coordinate token support
- [**Training System**](implementation/training-system.md) - Loss management and training coordination

### **Need Help?**
- [**Troubleshooting**](troubleshooting/common-issues.md) - Common issues and solutions
- [**Migration**](guides/migration.md) - Upgrading from older versions

## 📋 **Complete Documentation Index**

For a comprehensive list of all documentation files, see [**INDEX.md**](INDEX.md) - Complete documentation index with 30+ files organized by topic and user type.

## 📊 **System Overview**

The BBU training pipeline supports two operational modes:

### **Standard Mode** (Recommended for Production)
- **Coordinates**: Integer format `[150,10,211,35]`
- **Vocabulary**: Minimal extension (+4 geometry tokens)
- **Use Case**: Production training with stable performance
- **Status**: ✅ Production ready

### **Coordinate Mode** (Advanced Features)
- **Coordinates**: Token format `[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]`
- **Vocabulary**: Extended (+2052 coordinate tokens)
- **Use Case**: Advanced sequence-based coordinate prediction
- **Status**: ✅ Production ready (requires `remove_unused_columns: false`)

## 🎯 **User Journeys**

### **I'm a New Developer**
1. [Getting Started](getting-started.md) - Setup and first run
2. [Configuration](implementation/configuration.md) - Understanding configuration options
3. [Training System](implementation/training-system.md) - Running your first training job
4. [Troubleshooting](troubleshooting/common-issues.md) - Common issues

### **I'm a Researcher/Experimenter**
1. [Coordinate Tokens](features/coordinate-tokens.md) - Advanced coordinate token features
2. [Architecture](reference/architecture.md) - System design and extensibility
3. [API Reference](guides/api-reference.md) - Customization and extension points
4. [Training Modes](features/training-modes.md) - Advanced training configurations

### **I Need to Troubleshoot**
1. [Troubleshooting](troubleshooting/common-issues.md) - Comprehensive problem-solving guide
2. [Configuration](implementation/configuration.md) - Configuration validation
3. [API Reference](guides/api-reference.md) - API debugging

### **I'm Migrating/Upgrading**
1. [Migration](guides/migration.md) - Version upgrade guides
2. [Configuration](implementation/configuration.md) - New configuration format
3. [Coordinate Tokens](features/coordinate-tokens.md) - New coordinate token system

## 🔧 **Quick Commands**

### **Setup**
```bash
# Clone and setup
git clone <repository>
cd Qwen2.5-VL-main
pip install -r requirements.txt

# Quick test
python scripts/train.py --config configs/bbu_v2.yaml --dry-run
```

### **Training**
```bash
# Standard mode (recommended)
python scripts/train.py --config configs/bbu_v2.yaml

# Coordinate mode (advanced)
python scripts/train.py --config configs/bbu_coordinate.yaml
```

### **Testing**
```bash
# Run all tests
python -m pytest tests/ -v

# Test specific component
python -m pytest tests/test_coordinate_tokens.py -v
```

## 📚 **Documentation Structure**

This documentation follows a simplified structure for easy navigation:

- **Single files per topic** - No nested directories for main content
- **Consolidated information** - All related content in one place
- **Clear cross-references** - Easy navigation between related topics
- **User-focused organization** - Organized by user needs, not system components

## 🏗️ **System Architecture**

```
BBU Training Pipeline
├── Data Conversion     → implementation/data-conversion.md
├── Model System        → implementation/model-system.md
├── Training System     → implementation/training-system.md
├── Configuration       → implementation/configuration.md
├── Coordinate Tokens   → features/coordinate-tokens.md
├── Multi-Geometry      → features/multi-geometry.md
└── Training Modes      → features/training-modes.md
```

## 📈 **Current Status**

- ✅ **Coordinate Token System**: Both modes production ready
- ✅ **Multi-Geometry Support**: Complete support for bbox_2d, line, and square geometries
- ✅ **Loss Computation**: Fixed teacher assignment and duplicate loss variables
- ✅ **Training Pipeline**: Fully functional with comprehensive testing
- ✅ **Documentation**: Simplified and consolidated structure
- ✅ **Testing**: 30/30 tests passing
- ✅ **Configuration**: Simplified and validated

## 🤝 **Contributing**

When updating documentation:
1. Keep information in the appropriate single file
2. Update cross-references when adding new content
3. Follow the user journey approach
4. Test all examples and commands

---

**Need immediate help?** Check [Troubleshooting](troubleshooting/common-issues.md) for quick solutions to common issues.
