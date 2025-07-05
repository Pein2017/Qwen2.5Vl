# Data Conversion Pipeline - Codebase Structure

## Core Pipeline Files
- `convert_dataset.sh` - Main entry point script
- `processor.py` - Main orchestrator class
- `sample_processor.py` - Individual sample processing
- `core_modules.py` - Shared utilities (TokenMapper, ObjectProcessor)

## Data Handling
- `data_loader.py` - Sample loading and validation
- `data_splitter.py` - Train/val splitting
- `teacher_selector.py` - Teacher sample selection

## Preprocessing
- `clean_raw_json.py` - JSON cleaning and standardization
- `apply_token_mapping.py` - Token mapping application
- `vision_process.py` - Image processing utilities

## Configuration
- `label_hierarchy.json` - Object type hierarchy for filtering
- `token_map.json` / `token_map_zh.json` - Token mapping definitions

## Validation
- `simple_validate.py` - Output validation

## Documentation
- `README.md` - Pipeline documentation

## Legacy/Backup
- `legacy_backup/` - Old implementation files for reference

## Key Features
✅ **Automatic coordinate scaling** - Sample processor handles image resize with proper bbox scaling
✅ **Multi-format support** - Handles both dataList and markResult JSON formats  
✅ **Token mapping** - Standardizes labels across different annotation sources
✅ **Hierarchy filtering** - Filters objects based on valid type/property combinations
✅ **Smart image resizing** - Maintains aspect ratio while meeting size constraints
✅ **EXIF orientation handling** - Properly handles image orientation metadata

## Pipeline Flow
1. Clean raw JSON files (`clean_raw_json.py`)
2. Apply token mapping (`apply_token_mapping.py`)
3. Load and process samples (`processor.py` + `sample_processor.py`)
4. Split into train/val (`data_splitter.py`)
5. Select teacher samples (`teacher_selector.py`)
6. Generate JSONL files
7. Validate output (`simple_validate.py`)

## Integration Notes
- The coordinate scaling issue discovered during validation was already properly handled in `sample_processor.py`
- No additional coordinate fixing step is needed in the pipeline
- All image processing includes EXIF orientation correction
- The pipeline is production-ready and handles edge cases gracefully
