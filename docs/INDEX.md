# Documentation Index

Minimal, AI-assistant–oriented navigation. Canonical facts live in two deep dives; all other docs are pointers.

## Canonical Sources
- AI KB: `AI_ASSISTANT_KB.md` (open first)
- src_new deep dive: `../src_new/UNIFIED_DOCUMENTATION.md`
- data_conversion deep dive: `../data_conversion/README.md`

## Quick Links
- Onboarding summary: `SRC_NEW_ASSISTANT_ONBOARDING.md` (condensed; redirects to AI KB + deep dives)
- src_new reference hub: `SRC_NEW_REFERENCE.md` (thin pointer to deep dive + code map)
- data_conversion reference hub: `DATA_CONVERSION_REFERENCE.md` (thin pointer + scripts)
- Troubleshooting: `TROUBLESHOOTING_GUIDE.md` (kept; refer to deep dives for details)
- Inference fixes: `INFERENCE_ROOT_CAUSE_AND_FIXES.md` (kept; summarized in AI KB)

## Commands
- Train: `python scripts/train_new.py --config bbu_v2`
- Infer: `python -m src_new.inference --config_path configs/bbu_v2.yaml --model_path checkpoints/best --input_file data/val.jsonl --output_file results/val.json --data_root /abs/path`
- Tests: `python -m pytest src_new/tests -q`

## Files to Read First
1) `AI_ASSISTANT_KB.md`
2) `../src_new/UNIFIED_DOCUMENTATION.md`
3) `../data_conversion/README.md`

## Archive
- Historical content is under `archive/`; prefer canonical sources above.

Last Updated: August 2025