# src_new Essentials — Assistant Onboarding (10–15 min)

Read this, then jump to `AI_ASSISTANT_KB.md` and the deep dives.

- What: Vision-language fine-tuning for BBU detection, Chinese descriptions + coordinates (bbox/quad/line), optional coordinate tokens
- Entrypoints: train `scripts/train_new.py`; infer `src_new/inference.py`
- Flow: JSONL → Conversation+Images → Tokenization → Model → Loss → Checkpoints
- Modules: conversations, token processor, wrapper, loss manager, trainer, path manager, inference

Commands:
```bash
python scripts/train_new.py --config bbu_v2
python -m src_new.inference --config_path configs/bbu_v2.yaml --model_path checkpoints/best --input_file data/val.jsonl --output_file results/val.json --data_root /abs/path
python -m pytest src_new/tests -q
```

Pointers:
- Data schema and spans, coord tokens, masking, and loss weights → `AI_ASSISTANT_KB.md`
- Full algorithms and code citations → `../src_new/UNIFIED_DOCUMENTATION.md`
 