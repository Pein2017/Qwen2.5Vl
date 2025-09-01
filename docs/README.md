# BBU Training Pipeline Docs (AI-Oriented)

Canonical sources for assistants:
- AI KB: `AI_ASSISTANT_KB.md` (open first)
- src_new deep dive: `../src_new/UNIFIED_DOCUMENTATION.md`
- data conversion deep dive: `../data_conversion/README.md`

Quick commands:
```bash
python scripts/train_new.py --config bbu_v2
python -m src_new.inference --config_path configs/bbu_v2/base.yaml --model_path checkpoints/best --input_file data/val.jsonl --output_file results/val.json --data_root /abs/path
python -m pytest src_new/tests -q
```

Pointers:
- Troubleshooting: `TROUBLESHOOTING_GUIDE.md`
- Inference fixes: `INFERENCE_ROOT_CAUSE_AND_FIXES.md`
- Performance: `PERFORMANCE_OPTIMIZATION.md`

For comprehensive narratives and diagrams, prefer the deep dives above. This README intentionally stays minimal to avoid duplication.
