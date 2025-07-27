# Essential Development Commands

## Environment Setup
- **Python Path**: Always use `/root/miniconda3/envs/ms/bin/python` directly instead of generic `python` command
- **Conda Environment**: `conda activate ms` (but use full Python path to avoid activation inconsistencies)
- **Environment Variables**:
  - `export CUDA_VISIBLE_DEVICES=0,1,2,3` (adjust based on GPU setup)
  - `export HF_HOME=/data3/Qwen2.5-VL-main/model_cache`

## Training Commands
- **Main Training**: `bash scripts/run_train.sh`
- **Direct Training**: `python scripts/train.py --config bbu_v2 --log_level INFO`
- **Training Config**: Use `bbu_v2.yaml` (current/updated) instead of `base_flat_det.yaml` (outdated)

## Data Processing
- **Process Dataset**: `bash data_conversion/convert_dataset.sh`
- **Custom Processing**: `INPUT_DIR="custom_ds" OUTPUT_DIR="custom_data" bash data_conversion/convert_dataset.sh`
- **Validation**: `python data_conversion/simple_validate.py`

## Testing & Validation
- **Run Tests**: `python -m pytest eval/test_all_evaluations.py`
- **Config Validation**: `python scripts/validate_config.py`
- **Model Consistency**: `python scripts/validate_consistency.py`

## Code Quality
- **Linting**: `ruff check .` (configured in pyproject.toml)
- **Formatting**: `ruff format .`
- **Import Organization**: Handled automatically by ruff

## Inference
- **Run Inference**: `python src/inference.py --model_path path/to/checkpoint --image_path path/to/image.jpg`

## When Task is Completed
1. Run validation scripts if data processing was involved
2. Run `ruff check .` and `ruff format .` for code quality
3. Test relevant functionality with validation scripts
4. No explicit test framework - use validation scripts in `scripts/` directory