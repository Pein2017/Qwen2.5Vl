# src_new/ Reference Hub

Minimal hub for AI assistants. For details, use the canonical deep dive and AI KB.

## Canonical
- AI KB: `AI_ASSISTANT_KB.md`
- Deep dive: `../src_new/UNIFIED_DOCUMENTATION.md`

## Open These First
- `src_new/inference.py`
- `src_new/processing/conversation_processor.py`
- `src_new/processing/token_processor.py`
- `src_new/models/wrapper.py`
- `src_new/models/loss_manager.py`
- `src_new/training/bbu_trainer.py`
- `src_new/utils/path_manager.py`

## Commands
```bash
python scripts/train_new.py --config bbu_v2
python -m src_new.inference --config_path configs/bbu_v2.yaml --model_path checkpoints/best --input_file data/val.jsonl --output_file results/val.json --data_root /abs/path
python -m pytest src_new/tests -q
```

## Pointers
- Checkpoint saving: set `trainer.processing_class = tokenizer`
- NCCL timeouts: use `BBUTrainer`
- Span alignment: tokenizer `offset_mapping`

Full details: `../src_new/UNIFIED_DOCUMENTATION.md`.  