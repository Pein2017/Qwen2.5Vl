# Data Conversion Cleanup Summary

## Files Removed/Moved

The following outdated files have been moved to `legacy_backup/` folder:

### Replaced by New Modular Architecture
- **`convert_pure_json.py`** → Replaced by `sample_processor.py`
  - Old: Single monolithic converter
  - New: Modular sample processing with clear separation of concerns

- **`create_teacher_pool.py`** → Replaced by `teacher_selector.py` 
  - Old: Basic teacher selection
  - New: Advanced multi-objective selection with diversity metrics

- **`split_train_val.py`** → Replaced by `data_splitter.py`
  - Old: Simple splitting script
  - New: Proper class-based splitter with validation

- **`extract_candidates.py`** → No longer needed
  - Candidate phrase extraction is no longer part of the pipeline

- **`qwen_converter_unified.py`** → Replaced by `processor.py`
  - Old: Legacy unified converter
  - New: Modern orchestrator with component coordination

### Temporary Files Removed
- `convert.log` - Generated log file
- `candidate_phrases.json` - Generated temporary file

## Current Clean Architecture

### Core Pipeline Files
- `processor.py` - Main orchestrator
- `data_loader.py` - JSON/image loading and validation
- `sample_processor.py` - Individual sample processing
- `teacher_selector.py` - Teacher pool selection
- `data_splitter.py` - Train/validation splitting

### Utility Files
- `core_modules.py` - Shared utilities (TokenMapper, ObjectProcessor)
- `vision_process.py` - Image processing utilities
- `simple_validate.py` - Data validation
- `strip_exif_orientation.py` - Image preprocessing utility

### Configuration Files
- `convert_dataset.sh` - Main entry point
- `label_hierarchy.json` - Object type hierarchy
- `token_map.json` / `token_map_zh.json` - Token mappings

### Testing & Documentation
- `test_pipeline.py` - Component testing (keep for validation)
- `README_refactored.md` - Updated documentation
- `guidance.md` - Implementation guidance

### Legacy Backup
- `legacy_backup/` - Contains all replaced files for reference

## Benefits of Cleanup

✅ **Reduced Complexity**: 20 files → 15 active files  
✅ **Clear Purpose**: Each file has a single responsibility  
✅ **No Duplication**: Eliminated redundant functionality  
✅ **Maintainable**: Easier to understand and modify  
✅ **Testable**: Individual components can be tested separately  
✅ **Safe Migration**: Old files preserved in backup folder

## Migration Notes

If you need to reference old implementations:
- Check `legacy_backup/` folder
- Old functionality is now distributed across the new modular components
- The new architecture provides the same functionality with better organization

## Test File Decision

**`test_pipeline.py` - KEPT**
- Successfully tested during development
- Useful for validating component functionality
- Small and focused testing utility
- No dependencies on removed files