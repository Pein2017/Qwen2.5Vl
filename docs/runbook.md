# Runbook (Commands Only)

> This file intentionally concentrates **all** shell & Python one-liners for the project.

---

## Data Conversion
```bash
conda activate ms
bash data_conversion/convert_dataset.sh
```
This script runs the complete data conversion pipeline, preparing the raw data for training.

---

## Training (New System)
```bash
conda activate ms
python scripts/train.py --config base_flat_v2 --use-new-config --log_level INFO
```

### With DeepSpeed
```bash
export BBU_DEEPSPEED_ENABLED=true
export BBU_DEEPSPEED_CONFIG=scripts/zero2.json
python scripts/train.py --config base_flat_v2 --use-new-config --log_level INFO
```

## Training (Legacy System)
```bash
conda activate ms
python scripts/train.py --config base_flat --log_level INFO
```

## Validation-only
```bash
# New System
python scripts/train.py --config base_flat_v2 --use-new-config --validate-only

# Legacy System
python scripts/train.py --config base_flat --validate-only
```

## Inference
```python
from src.inference import Qwen25VLInference
predictor = Qwen25VLInference('checkpoint_dir')
boxes, captions = predictor.predict_detection(images, prompt)
```

*(More examples were moved here from legacy docs; trim/update as workflows change.)* 