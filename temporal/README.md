# Temporal Directory

This directory contains test files, documentation, and examples from the coordinate token system development process.

## Contents

### 📋 Documentation
- `coordinate_migration_summary.md` - Summary of coordinate token migration process
- `default_values_fixes_summary.md` - Documentation of error handling improvements
- `test_summary_results.md` - Test results and validation summary
- `cleanup_plan.md` - Cleanup plan for temporal directory

### 🧪 Core Tests
- `test_coordinate_system.py` - **Final working coordinate system test**
- `investigate_token_ids.py` - Token ID investigation and resolution

### 🚀 Production Example
- `production_training_example.py` - Production training example

## Purpose

These files serve as:
1. **Evidence** of successful coordinate token system implementation
2. **Reference** for future development and debugging
3. **Documentation** of the development process and key decisions

## Usage

To run the final coordinate system test:
```bash
cd /data3/Qwen2.5-VL-main
/root/miniconda3/envs/ms/bin/python temporal/test_coordinate_system.py
```

## Cleanup Status

✅ **Completed**: Removed 19 obsolete files (debug scripts, redundant tests)
✅ **Preserved**: 6 essential files (tests, documentation, examples)
✅ **Organized**: Files properly documented and structured