# Data Conversion Reference Hub

Thin pointer for AI assistants.

## Canonical
- AI KB: `AI_ASSISTANT_KB.md`
- Deep dive: `../data_conversion/README.md`

## Quick
```bash
./data_conversion/convert_dataset.sh
```

Edit `data_conversion/convert_dataset.sh` essentials:
```bash
INPUT_DIR="ds_v2"
OUTPUT_DIR="data"
DATASET_NAME="ds_v2_bbu"
OBJECT_TYPES="bbu bbu_shield"    # or "full"
VAL_RATIO="0.1"
MAX_TEACHERS="10"
RESIZE="true"
SEED="17"
```

Object types: `bbu`, `bbu_shield`, `connect_point`, `label`, `fiber`, `wire`

Outputs: `data/<dataset>/{train.jsonl,val.jsonl,teacher.jsonl,all_samples.jsonl,label_vocabulary.json,images/}`

Python API:
```python
from data_conversion.unified_processor import UnifiedProcessor
from data_conversion.config import DataConversionConfig
config = DataConversionConfig(input_dir="ds_v2", output_dir="data", object_types=["bbu","label"], resize=True, val_ratio=0.1, max_teachers=10, seed=17)
processor = UnifiedProcessor(config)
results = processor.process()
```

Gotchas: use exact object names; `fiber`/`wire` require line geometry; try `OBJECT_TYPES="full"` if filtering to zero.
 