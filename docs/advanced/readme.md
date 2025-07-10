# Advanced Topics Index

This directory contains detailed technical documentation for advanced features and implementation details of the Qwen2.5-VL BBU fine-tuning project.

## Available Topics

### [Collator Notes](collator_notes.md)
- **Focus**: Packed sequence collation internals
- **Key Topics**: Memory efficiency, boundary masking, position ID handling
- **Audience**: Developers working on data loading optimization

### [PEFT Adapter](peft_adapter.md)  
- **Focus**: Parameter-Efficient Fine-Tuning integration
- **Key Topics**: LoRA adapter configuration, training efficiency
- **Audience**: Researchers exploring efficient training methods

### [Teacher-Student Learning](teacher_student.md)
- **Focus**: Teacher-student training methodology
- **Key Topics**: Loss splitting, span-based learning, teacher pool selection
- **Audience**: ML engineers implementing multi-task learning

## Related Documentation

For comprehensive coverage of advanced topics, also see:

### Main Documentation
- [Architecture](../architecture.md) - Complete system architecture with DETR detection
- [Critical Fixes](../critical_fixes.md) - Advanced troubleshooting and patches
- [Lessons Learned](../lessons_learned.md) - Historical knowledge and pitfalls

### Specialized Topics
- **Detection System**: Covered in [Architecture](../architecture.md#67-detr-style-detection-system-current-implementation)
- **Model Patches**: Detailed in [Critical Fixes](../critical_fixes.md#model-implementation-fixes)
- **Coordinate Transformations**: Explained in [Data Schema](../data_schema.md#24-3-stage-coordinate-transformation)

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