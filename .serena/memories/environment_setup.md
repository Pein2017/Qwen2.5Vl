# Environment Setup

## Python Environment
- **Required Conda Environment**: `ms`
- **Python Path**: Always use `/root/miniconda3/envs/ms/bin/python` directly instead of generic `python` command
- **Activation**: Avoid conda activation inconsistencies by using full path

## GPU Configuration
- Use `CUDA_VISIBLE_DEVICES` for GPU selection
- HF_HOME for model cache: `/data3/Qwen2.5-VL-main/model_cache`

## Network Constraints
- **Location**: China - cannot access foreign websites like `github` `google` `huggingface`
- **Solution**: Use local mirrors and cached resources when possible

## Data Migration Status
- Currently migrating to v2 data annotation structure
- Reference documentation:
  * `@docs/raw_data_v2.md`
  * `@docs/raw_data_template_数据堂.md`
- Updated data structure follows new V2 format