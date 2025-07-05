# Runbook (Commands Only)

> This file intentionally concentrates **all** shell & Python one-liners.  The functional docs stay clean; update commands here only.

---

## Data Conversion

### Enhanced Pipeline (Recommended)
```bash
conda activate ms
bash data_conversion/convert_dataset.sh
```

**What it does:**
1. Cleans raw JSON files (strips metadata, preserves structure)
2. Copies images to unified `ds_output/` directory  
3. Processes data with smart resize and bbox scaling
4. Generates `train.jsonl`, `val.jsonl`, `teacher.jsonl` with correct paths

### Custom Parameters
```bash
# Use different directories
INPUT_DIR="custom_ds" OUTPUT_DIR="custom_data" OUTPUT_IMAGE_DIR="custom_output" bash data_conversion/convert_dataset.sh

# English mode with different settings
LANGUAGE="english" RESIZE="true" MAX_TEACHERS="20" bash data_conversion/convert_dataset.sh
```

### Individual Steps (Debug)
```bash
# Step 1: Clean JSON only
python data_conversion/clean_raw_json.py ds ds_clean --lang zh

# Step 2: Full processing
python data_conversion/processor.py --input_dir ds_clean --output_dir data --language chinese --resize --output_image_dir ds_output
```

---

## Training
```bash
conda activate ms
python scripts/train.py --config base_flat --log_level INFO --log_verbose true
```

### With DeepSpeed
```bash
export BBU_DEEPSPEED_ENABLED=true
export BBU_DEEPSPEED_CONFIG=scripts/zero2.json
python scripts/train.py --config base_flat --log_level INFO --log_verbose true
```

## Validation-only
```bash
python scripts/train.py --config base_flat --validate-only
```

## Inference
```python
from src.inference import Qwen25VLInference
predictor = Qwen25VLInference('checkpoint_dir')
boxes, captions = predictor.predict_detection(images, prompt)
```

*(More examples were moved here from legacy docs; trim/update as workflows change.)* 