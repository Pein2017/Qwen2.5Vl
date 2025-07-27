# Advanced Topics Index (2025 Modular Architecture)

This directory contains detailed technical documentation for advanced features and implementation details of the Qwen2.5-VL BBU fine-tuning project with the current modular architecture.

## Available Topics

### [Collator Notes](collator_notes.md)
- **Focus**: Data collation for multi-geometry training
- **Key Topics**: BBU-specific collation, coordinate token handling, batch processing
- **Audience**: Developers working on data loading optimization
- **Status**: Updated for current architecture

### [PEFT Adapter](peft_adapter.md)
- **Focus**: Parameter-Efficient Fine-Tuning integration
- **Key Topics**: LoRA adapter configuration with coordinate tokens, training efficiency
- **Audience**: Researchers exploring efficient training methods
- **Status**: Compatible with current model wrapper

### [Teacher-Student Learning](teacher_student.md)
- **Focus**: Teacher-student training methodology in modular system
- **Key Topics**: LossManager implementation, span-based learning, teacher pool selection
- **Audience**: ML engineers implementing multi-task learning
- **Status**: Updated for TrainingCoordinator and LossManager

## Related Documentation

For comprehensive coverage of advanced topics, also see:

### Core Documentation
- [Architecture](../core/architecture.md) - Complete modular system architecture
- [Training Architecture](../core/training-architecture-2025.md) - Current training system
- [Data Pipeline](../core/data-pipeline.md) - 5-stage processing pipeline

### Reference Documentation
- [API Reference](../reference/api-core-components.md) - Current modular APIs
- [Source Code Reference](../reference/src-code-reference.md) - Current codebase structure
- [Troubleshooting](../guides/troubleshooting.md) - Common issues with modular architecture

## Navigation Guide

```
For Implementation Details → Use individual topic files in this directory
For System Overview → Start with ../architecture.md  
For Troubleshooting → Check ../critical_fixes.md and ../troubleshooting.md
For Operations → Reference ../runbook.md
For Getting Started → Begin with ../getting_started.md
```

## Contributing to Advanced Documentation

When adding new advanced topics:

1. **Create focused documents** covering specific technical areas
2. **Include code examples** and implementation details
3. **Reference related components** in the main codebase
4. **Update this index** with the new topic
5. **Cross-reference** from main documentation when relevant

---

This index helps organize advanced technical knowledge while keeping the main documentation accessible to all users.