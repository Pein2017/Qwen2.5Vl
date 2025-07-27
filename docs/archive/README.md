# Documentation Archive (2025 Update)

This directory contains legacy documentation that has been archived due to the 2025 modular architecture migration. These files are preserved for historical reference but are no longer actively maintained.

## Archived Documentation

### Legacy Architecture (Pre-2025)
The following files related to the monolithic architecture have been archived after the migration to the modular system:

- **`coordinate-token-system-complete-guide.md`** - Legacy coordinate token system guide
- **`soft_expectation_coordinate_regression.md`** - Legacy soft expectation approach
- **`REORGANIZATION_PLAN.md`** - Historical reorganization planning
- **`cleanup-candidates.md`** - Legacy cleanup planning

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