# Temporal Directory Cleanup Plan

## Files to Keep (Evidence/Reference)

### 1. Core Testing Files
- `test_coordinate_system.py` - **KEEP** - Final working coordinate system test
- `investigate_token_ids.py` - **KEEP** - Token ID investigation evidence

### 2. Documentation Files
- `coordinate_migration_summary.md` - **KEEP** - Migration documentation
- `default_values_fixes_summary.md` - **KEEP** - Error handling fixes documentation
- `test_summary_results.md` - **KEEP** - Test results summary

### 3. Production Example
- `production_training_example.py` - **KEEP** - Production training example

## Files to Delete (Obsolete/Redundant)

### Debug Files (Superseded by final implementation)
- `debug_config.py` - **DELETE** - Config debugging (obsolete)
- `debug_pipeline_comprehensive.py` - **DELETE** - Pipeline debugging (obsolete)
- `debug_pipeline_issue.py` - **DELETE** - Pipeline debugging (obsolete)
- `debug_sample_extraction.py` - **DELETE** - Sample debugging (obsolete)

### Redundant Test Files (Superseded by test_coordinate_system.py)
- `test_config_loading.py` - **DELETE** - Config testing (obsolete)
- `test_coordinate_fixes.py` - **DELETE** - Coordinate fixes testing (obsolete)
- `test_coordinate_integration.py` - **DELETE** - Integration testing (obsolete)
- `test_coordinate_loss_extraction.py` - **DELETE** - Loss extraction testing (obsolete)
- `test_coordinate_loss_simple.py` - **DELETE** - Simple loss testing (obsolete)
- `test_coordinate_setup.py` - **DELETE** - Setup testing (obsolete)
- `test_coordinate_tokens.py` - **DELETE** - Token testing (obsolete)
- `test_dimension_fix.py` - **DELETE** - Dimension testing (obsolete)
- `test_enhanced_detection_loss.py` - **DELETE** - Enhanced loss testing (obsolete)
- `test_logging_consistency.py` - **DELETE** - Logging testing (obsolete)
- `test_no_defaults.py` - **DELETE** - No defaults testing (obsolete)
- `test_refactored_pipeline.py` - **DELETE** - Refactored pipeline testing (obsolete)
- `test_training_integration.py` - **DELETE** - Training integration testing (obsolete)
- `test_validation_errors.py` - **DELETE** - Validation testing (obsolete)

### Validation Files (Superseded by final implementation)
- `validate_fixes.py` - **DELETE** - Validation script (obsolete)

## Summary
- **Keep**: 6 files (core tests, documentation, production example)
- **Delete**: 19 files (debug scripts, redundant tests, validation scripts)
- **Net cleanup**: Remove 76% of temporal files while preserving evidence and documentation