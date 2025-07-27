# Documentation Archive

**Historical documentation preserved from the complex nested structure (pre-reorganization)**

## 📋 **What's Archived Here**

This directory contains the previous documentation structure that was reorganized on 2025-01-27 to create a simplified, maintainable documentation system.

### **Archived Structure**
```
archive/
├── old-structure/           # Complete old documentation structure
│   ├── core/               # Core system documentation
│   ├── components/         # Component-specific guides
│   ├── guides/             # User guides and tutorials
│   ├── quick-start/        # Quick start materials
│   ├── status/             # Implementation status docs
│   ├── troubleshooting/    # Troubleshooting guides
│   ├── reference/          # API and command references
│   ├── workflows/          # Workflow documentation
│   ├── refactoring/        # Refactoring notes and plans
│   ├── implementation/     # Implementation details
│   ├── legacy/             # Legacy documentation
│   ├── advanced/           # Advanced topics
│   ├── 专利/               # Chinese documentation
│   └── archive-old/        # Previously archived content
└── README.md               # This file
```

## 🎯 **Why Was This Reorganized?**

### **Problems with Old Structure**
- **12+ nested subdirectories** with overlapping content
- **50+ scattered files** with duplicate information
- **Unclear navigation** and hierarchy
- **Mixed languages** and inconsistent naming
- **Maintenance burden** due to scattered information
- **User confusion** about where to find information

### **New Simplified Structure**
```
docs/
├── README.md               # Main entry point with navigation
├── getting-started.md      # Quick start guide (consolidated)
├── configuration.md        # Complete configuration guide (consolidated)
├── coordinate-tokens.md    # Coordinate token system (consolidated)
├── training.md            # Training guide and workflows (consolidated)
├── troubleshooting.md     # All troubleshooting in one place (consolidated)
├── api-reference.md       # API documentation (consolidated)
├── architecture.md        # System architecture overview (consolidated)
├── migration.md           # Migration guides (consolidated)
└── archive/               # Historical content (this directory)
```

### Legacy Data Formats
- **`raw_data_template_数据堂.md`** - Legacy data template documentation
- **`raw_data_v2.md`** - Legacy V2 data format documentation

## Current Documentation (2025 Modular Architecture)

The current system uses a modular architecture with simplified components. For up-to-date documentation, please refer to:

### Core Documentation
- **[Architecture](../core/architecture.md)** - Current modular system architecture
- **[Training Architecture](../core/training-architecture-2025.md)** - Current training system
- **[Data Pipeline](../core/data-pipeline.md)** - Current 5-stage processing pipeline

### Reference Documentation
- **[API Reference](../reference/api-core-components.md)** - Current modular APIs
- **[Source Code Reference](../reference/src-code-reference.md)** - Current codebase structure

## Migration Summary

The system has migrated from the legacy coordinate token approach to a simpler token system:

### Legacy Coordinate Token System (Archived)
- Complex coordinate token scheme with special encoding
- Separate token for each coordinate value
- Higher memory and computation requirements
- Large tokenizer vocabulary size

### Current Simple Token System
- Lightweight token addition using standard HuggingFace infrastructure
- Comma-separated coordinate values within geometry tokens
- Support for multiple geometry types (bbox_2d, square, line)
- Reduced memory usage and improved performance

For any questions about the archived documentation or the migration process, please refer to the [V2 Migration Complete](../V2_MIGRATION_COMPLETE.md) document. 