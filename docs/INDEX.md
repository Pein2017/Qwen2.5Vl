# Complete Documentation Index

This is a comprehensive index of all documentation in the BBU Training Pipeline project.

## 🎯 **Current Status: Production Ready (August 2025)**
- **Architecture**: `src_new/` with BBUTrainer replacing DistributedLossTrainer
- **Training**: Stable distributed training with NCCL timeout resolution
- **Logging**: Enhanced formatting (4-decimal losses, readable learning rates)
- **Dataset**: Configurable size limiting via `max_dataset_size` parameter (default: -1 for full dataset)
- **Models**: Qwen2.5-VL-3B-Instruct with coordinate token integration

## 📚 **Core Documentation**

### **Getting Started**
- [**README.md**](README.md) - Main documentation hub and navigation
- [**IMPLEMENTATION_GUIDE.md**](IMPLEMENTATION_GUIDE.md) - Choose between src/ vs src_new/ implementations
- [**getting-started.md**](getting-started.md) - Quick setup and first training run

### **System Architecture & Reference**
- [**reference/architecture.md**](reference/architecture.md) - System overview and components
- [**guides/api-reference.md**](guides/api-reference.md) - Complete API documentation

### **Implementation Documentation**
- [**implementation/configuration.md**](implementation/configuration.md) - Complete configuration system
- [**implementation/data-conversion.md**](implementation/data-conversion.md) - Data processing pipeline
- [**implementation/model-system.md**](implementation/model-system.md) - Model architecture and integration
- [**implementation/training-system.md**](implementation/training-system.md) - Training framework and loss management
- [**implementation/granular-loss-logging.md**](implementation/granular-loss-logging.md) - Enhanced loss logging with teacher-student breakdown

### **Feature Documentation**
- [**features/coordinate-tokens.md**](features/coordinate-tokens.md) - Advanced coordinate token system
- [**features/coordinate-normalization.md**](features/coordinate-normalization.md) - Coordinate processing and normalization
- [**features/multi-geometry.md**](features/multi-geometry.md) - Multi-geometry support (bbox_2d, line, square)
- [**features/training-modes.md**](features/training-modes.md) - Standard vs Coordinate training modes


### **Troubleshooting & Support**
- [**troubleshooting/common-issues.md**](troubleshooting/common-issues.md) - General troubleshooting guide
- [**troubleshooting/performance-issues.md**](troubleshooting/performance-issues.md) - Critical performance bottlenecks and optimizations
- [**troubleshooting/qwen2-5vl-compatibility.md**](troubleshooting/qwen2-5vl-compatibility.md) - Qwen2.5-VL tensor format and compatibility issues

### **Migration & Guides**
- [**guides/migration-src-to-src-new.md**](guides/migration-src-to-src-new.md) - Upgrading from src/ to src_new/

## 📦 **Archive**

### **Historical Documentation**
- [**archive/README.md**](archive/README.md) - Archive overview and navigation
- [**archive/coordinate-token-system-complete-guide.md**](archive/coordinate-token-system-complete-guide.md) - Complete coordinate token guide
- [**archive/soft_expectation_coordinate_regression.md**](archive/soft_expectation_coordinate_regression.md) - Soft expectation coordinate regression
- [**archive/raw_data_template_数据堂.md**](archive/raw_data_template_数据堂.md) - Raw data template documentation
- [**archive/raw_data_v2.md**](archive/raw_data_v2.md) - Raw data v2 documentation

## 📋 **Documentation Categories**

### **By User Type**
- **New Developers**: README.md → IMPLEMENTATION_GUIDE.md → getting-started.md → implementation/configuration.md
- **Existing src/ Users**: IMPLEMENTATION_GUIDE.md → guides/migration-src-to-src-new.md → troubleshooting/common-issues.md
- **Researchers**: reference/architecture.md → features/coordinate-tokens.md → guides/api-reference.md
- **Troubleshooters**: troubleshooting/common-issues.md → troubleshooting/performance-issues.md → troubleshooting/qwen2-5vl-compatibility.md

### **By Topic**
- **Setup & Configuration**: getting-started.md, implementation/configuration.md
- **Data Processing**: implementation/data-conversion.md
- **Model Architecture**: implementation/model-system.md, reference/architecture.md
- **Training & Optimization**: implementation/training-system.md, features/training-modes.md
- **Coordinate Processing**: features/coordinate-tokens.md, features/coordinate-normalization.md
- **Multi-Geometry**: features/multi-geometry.md
- **Troubleshooting**: troubleshooting/common-issues.md, guides/migration.md

### **By Category**
- **Core Documentation**: Root level files (README.md, getting-started.md, etc.)
- **Implementation Documentation**: implementation/ directory - aligned with codebase structure
- **Feature Documentation**: features/ directory - specific functionality
- **Guides**: guides/ directory - task-oriented documentation
- **Reference**: reference/ directory - technical reference materials
- **Troubleshooting**: troubleshooting/ directory - common issues and solutions
- **Historical**: archive/ directory - preserved legacy content

## 🔍 **Quick Search Guide**

### **Common Questions**
- **"Which implementation should I use?"** → [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **"How do I get started?"** → [getting-started.md](getting-started.md)
- **"How do I migrate from src/ to src_new/?"** → [guides/migration-src-to-src-new.md](guides/migration-src-to-src-new.md)
- **"How do I configure training?"** → [implementation/configuration.md](implementation/configuration.md)
- **"What are coordinate tokens?"** → [features/coordinate-tokens.md](features/coordinate-tokens.md)
- **"How do I fix training issues?"** → [troubleshooting/common-issues.md](troubleshooting/common-issues.md)
- **"How do I use multi-geometry?"** → [features/multi-geometry.md](features/multi-geometry.md)
- **"How does data processing work?"** → [implementation/data-conversion.md](implementation/data-conversion.md)
- **"What training modes are available?"** → [features/training-modes.md](features/training-modes.md)

### **Technical Deep Dives**
- **System Architecture** → [reference/architecture.md](reference/architecture.md)
- **API Details** → [guides/api-reference.md](guides/api-reference.md)
- **Data Pipeline** → [implementation/data-conversion.md](implementation/data-conversion.md)
- **Model System** → [implementation/model-system.md](implementation/model-system.md)
- **Training Framework** → [implementation/training-system.md](implementation/training-system.md)
- **Coordinate Processing** → [features/coordinate-normalization.md](features/coordinate-normalization.md)
- **Token System** → [features/coordinate-tokens.md](features/coordinate-tokens.md)

## 📝 **Documentation Maintenance**

### **File Organization Principles**
1. **Implementation alignment** - Documentation structure matches codebase modules
2. **Clear categorization** - Organized by implementation/, features/, guides/, reference/, troubleshooting/
3. **Single source of truth** - One comprehensive document per topic
4. **Cross-references** - Easy navigation between related topics
5. **User-focused** - Organized by user needs and development workflows

### **Update Guidelines**
1. Keep information in the appropriate single file
2. Update cross-references when adding new content
3. Follow the user journey approach
4. Test all examples and commands
5. Archive outdated content rather than deleting

---

**Last Updated**: January 2025
**Total Documents**: 20+ files organized by category
**Status**: Documentation restructured and consolidated with 70% reduction in redundancy
