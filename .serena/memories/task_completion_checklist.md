# Task Completion Checklist

## Code Quality Checks
1. **Linting**: Run `ruff check .` to identify code issues
2. **Formatting**: Run `ruff format .` to ensure consistent formatting
3. **Type Safety**: Ensure all functions have type annotations
4. **Error Handling**: Verify no silent failures or bare except blocks

## Data Processing Tasks
1. **Validation**: Run `python data_conversion/simple_validate.py`
2. **Consistency**: Run `python scripts/validate_consistency.py` 
3. **Config Validation**: Run `python scripts/validate_config.py`
4. **Teacher-Student**: Run `python scripts/validate_teacher_student_loss.py`

## Training Tasks
1. **Model Consistency**: Run `python scripts/validate_teacher_ratio.py`
2. **Config Check**: Ensure using `bbu_v2.yaml` (not outdated `base_flat_det.yaml`)
3. **Environment**: Verify `ms` conda environment is active
4. **GPU Setup**: Check `CUDA_VISIBLE_DEVICES` is properly set

## Testing Approach
- **No pytest framework**: Use validation scripts in `scripts/` directory
- **Temporal Testing**: Create test files in `./temporal/` for development
- **Evidence Results**: Leave verification artifacts to prove task completion

## Environment Verification
1. **Python Path**: Use `/root/miniconda3/envs/ms/bin/python` directly
2. **Environment Variables**: Verify `HF_HOME`, `CUDA_VISIBLE_DEVICES`
3. **Model Cache**: Ensure model cache is accessible at configured location

## Documentation
- **No automatic documentation creation**: Only create docs if explicitly requested
- **Update CLAUDE.md**: Add important discoveries to project memory
- **Code Comments**: Keep minimal - prefer clear code over verbose comments

## Final Steps
1. Test the specific functionality that was modified
2. Verify integration with existing systems
3. Check that no regression was introduced
4. Leave evidence of successful completion (output files, logs, etc.)