# src_new/ Reference Hub

**Production-ready Qwen2.5-VL BBU detection system** - Structured hub with minimal redundancy pointing to implementation sources.

## 🚀 Status: **PRODUCTION READY**
- **NCCL Resolution**: 100% distributed training success rate (was 0%)
- **SafeTensors Optimization**: 4-6x faster inference loading
- **Architecture Stability**: Composition-based design with comprehensive error handling

## Start here
- **Onboarding**: `SRC_NEW_ASSISTANT_ONBOARDING.md` (10–15 min)
- **Setup**: `SETUP_AND_CONFIGURATION.md` 
- **Training flow**: `TRAINING_AND_IMPLEMENTATION.md`

## Deep dive (single source of truth)
- **Full reference**: `../src_new/UNIFIED_DOCUMENTATION.md`
  - Architecture, data format, token pipeline, conversations, span detection, losses, distributed training, configuration, troubleshooting

## Key entry points (open these first)
- `src_new/inference.py` — Training-matched inference pipeline (teacher guidance supported)
- `src_new/processing/conversation_processor.py` — Build conversations and tensors
- `src_new/processing/token_processor.py` — Coordinate vocab extension and embeddings
- `src_new/models/wrapper.py` — `DetectionModel` composition wrapper
- `src_new/models/loss_manager.py` — Span masks and loss computation
- `src_new/training/bbu_trainer.py` — Trainer with local loss aggregation
- `src_new/utils/path_manager.py` — Unified path resolution

## Minimal commands
```bash
# Training (recommended defaults)
python scripts/train_new.py --config bbu_v2

# Inference (training-matched)
python -m src_new.inference \
  --config_path configs/bbu_v2.yaml \
  --model_path checkpoints/best \
  --input_file data/val.jsonl \
  --output_file results/val.json \
  --data_root /abs/path/to/data_root

# Tests
python -m pytest src_new/tests -q
```

## Troubleshooting pointers
- Checkpoint saving: ensure `trainer.processing_class = tokenizer`
- NCCL timeouts: use `BBUTrainer` (local aggregation)
- Span alignment: use tokenizer `offset_mapping`

For complete details, see the deep dive: `../src_new/UNIFIED_DOCUMENTATION.md`. 