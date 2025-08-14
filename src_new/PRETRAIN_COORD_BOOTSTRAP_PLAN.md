## Qwen2.5‑VL Coord Bootstrap Pre‑Training Plan (Knowledge Migration)

Purpose: add a short, fail‑fast pre‑training stage to teach the model a new “coordinate language” where integers map to `<|coord_*|>` tokens and basic arithmetic answers are emitted in coordinate space. This migrates numeric competence from standard text tokens to the new vocabulary before full grounding/detection training.

---

### Scope & Outcomes
- **Teach identity mapping**: `N → <|coord_N|>` for `0 ≤ N ≤ max_coord` (currently 1024)
- **Teach small arithmetic**: `a + b`, `a − b`, `a × b`, `a ÷ b` where the result is in `[0, max_coord]`
- **Strict output**: assistant replies MUST contain a single coordinate token only; no raw numbers
- **No geometry yet**: this stage is token‑usage alignment, not detection; keep sequences short
- **Deterministic**: reproducible generation and model init (`coordinate_init_mode: fourier_ramp`)

---

### Design Principles (Repository‑Aligned)
- **Absolute paths** only in scripts/CLI (see Global Dev Guide)
- **Conda env**: run in `ms` (`source ~/.bashrc && conda activate ms`)
- **Fail‑fast**: explicit validation with actionable errors (no silent coercion)
- **TDD first**: add unit tests under `src_new/tests/` before implementing logic
- **No hidden defaults**: all hyperparameters/configs passed via YAML or entry script

---

### Deliverables
- Dataset generation script (synthetic, deterministic)
  - `/data3/Qwen2.5-VL-main/scripts/generate_coord_bootstrap.py`
- Bootstrap dataset (JSONL)
  - Example default output: `/data3/Qwen2.5-VL-main/data/coord_bootstrap/coord_bootstrap.jsonl`
  - Shared blank image: `/data3/Qwen2.5-VL-main/assets/blank_1x1.png` (generated if missing)
- New dataset class (text+single‑image bootstrap)
  - `/data3/Qwen2.5-VL-main/src_new/data/bootstrap_coord_dataset.py`
- Config for pre‑training
  - `/data3/Qwen2.5-VL-main/configs/bbu_v2_coord_bootstrap.yaml`
- Training entry script
  - `/data3/Qwen2.5-VL-main/scripts/run_coord_bootstrap.sh`
- Tests
  - `/data3/Qwen2.5-VL-main/src_new/tests/coord_bootstrap/test_dataset_and_spans.py`
  - `/data3/Qwen2.5-VL-main/src_new/tests/coord_bootstrap/test_tokenizer_and_vocab.py`
  - `/data3/Qwen2.5-VL-main/src_new/tests/coord_bootstrap/test_generation_strictness.py`
- Minimal docs
  - This plan and a short README in `/data3/Qwen2.5-VL-main/src_new/`

---

### Dataset Specification (Bootstrap)
- **Conversation shape**: 1 user + 1 assistant (simple conversation)
- **Images**: exactly 1 image (use shared 1×1 blank to satisfy current validator)
- **Assistant content**: single coord token (e.g., `<|coord_123|>`) with `<|im_end|>` in labels
- **Examples**
  - Identity: “What is `123` in coordinate space?” → `<|coord_123|>`
  - Arithmetic: “what is 1+5 in coordinate space?” → `<|coord_6|>`
- **Sampling**
  - Identity: uniform 0..max_coord, with extra weight on edges {0,1,max‑1,max}
  - Arithmetic: sample operands so result in range; include +-×÷
- **JSONL line schema** (per sample)
  - `{"images": ["/data3/Qwen2.5-VL-main/assets/blank_1x1.png"], "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<|coord_N|>"}], "meta": {"task": "identity|arithmetic", "result": N}}`
- **Reason for single image**: current `ConversationValidator` enforces 1 image for simple conversations; this keeps the pipeline unchanged and avoids intrusive branching in core code.

---

### New Dataset Class
File: `/data3/Qwen2.5-VL-main/src_new/data/bootstrap_coord_dataset.py`
- Responsibilities
  - Read JSONL bootstrap samples
  - Enforce exactly one image per sample
  - Use the existing `ConversationProcessor.create_simple_conversation(...)`
  - Reuse span/label creation identical to detection dataset
- Validation (fail‑fast)
  - `messages` a list with exactly 2 turns (user, assistant)
  - Assistant content is exactly one `<|coord_*|>` token; reject otherwise
  - `images` length is exactly 1; path must exist
  - `0 ≤ N ≤ max_coord`; reject lines that violate range
- Outputs
  - Same dictionary with `input_ids`, `pixel_values`, `labels`, and spans as detection dataset

Note: We intentionally avoid touching `src_new/data/dataset.py` strict object/geometry validation; the bootstrap dataset is isolated to minimize impact.

---

### Generation Script
File: `/data3/Qwen2.5-VL-main/scripts/generate_coord_bootstrap.py`
- Inputs (CLI, required; absolute paths)
  - `--output /abs/path/to/coord_bootstrap.jsonl`
  - `--num_identity 50000` (no defaults in code; required)
  - `--num_arithmetic 20000`
  - `--max_coord 1024`
  - `--seed 1337`
- Behavior
  - Create `/data3/Qwen2.5-VL-main/assets/blank_1x1.png` if missing (RGB)
  - Generate identity and arithmetic samples under constraints
  - Write JSONL; validate each line before writing
- Fail‑fast rules
  - Range, schema, uniqueness checks where appropriate

---

### Config (YAML)
File: `/data3/Qwen2.5-VL-main/configs/bbu_v2_coord_bootstrap.yaml`
- Required keys (aligned with existing configs)
  - `data_path`: `/data3/Qwen2.5-VL-main/data/coord_bootstrap/coord_bootstrap.jsonl`
  - `data_root`: `/` (or a real base for path manager)
  - `dataset_kind`: `coord_bootstrap`
  - `max_dataset_size`: explicit integer or `-1`
  - `teacher_ratio`: `0` and `num_teacher_samples`: `0`
  - `coordinate_tokens_enabled`: `true`
  - `max_coord_value`: `1024`
  - `coordinate_init_mode`: `fourier_ramp`
  - Loss weights and aux settings explicitly provided (no defaults inside libraries)
- Model path points to a tokenizer with coord tokens (see validation below)

---

### Trainer Integration
- Entry script: `/data3/Qwen2.5-VL-main/scripts/run_coord_bootstrap.sh`
  - Activates env and runs the standard trainer with the bootstrap config
- Dataset selection
  - In training setup, condition on `dataset_kind == "coord_bootstrap"` to instantiate `BootstrapCoordDataset`
  - All other components (collator, conversation processor, loss, logging) remain unchanged

---

### Initialization & Freezing Schedule
- Stage 0 (optional, very short)
  - Freeze all modules except token embeddings and LM head projection
  - LR: `5e-5`, steps: `200–500`
- Stage 1
  - Unfreeze full model
  - LR: `1e-5` to `2e-5`, steps: `1k–3k`
- Keep auxiliary coord losses enabled if your training uses them; harmless here and tends to sharpen distributions

---

### Validation & Gates (Fail‑Fast)
At startup (trainer side):
- Tokenizer must contain `<|coord_0|>` … `<|coord_{max_coord}|>`; raise with clear message if missing
- `coordinate_tokens_enabled` must be `True`; raise if not
- ConversationProcessor and validator confirm exactly 1 image placeholder per sample
- Enforce assistant span contains exactly one `<|coord_` token and includes `<|im_end|>` in labels (test via span extraction)

During training:
- Running metric: identity accuracy and arithmetic accuracy on held‑out template set (log every `eval_steps`)
- Reject/skip any batch where decoded assistant content contains raw digits or multiple coord tokens

---

### Tests (TDD)
Files under `/data3/Qwen2.5-VL-main/src_new/tests/coord_bootstrap/`
- `test_tokenizer_and_vocab.py`
  - Asserts coord token presence and contiguous ID range consistency with `max_coord_value`
- `test_dataset_and_spans.py`
  - Loads a tiny JSONL with 4 samples and the blank image
  - Verifies labels mask only assistant span, which contains `<|coord_*|>` and covers `<|im_end|>`
  - Verifies image token counts match expectations
- `test_generation_strictness.py`
  - Runs a tiny forward+decode and asserts the response contains exactly one `<|coord_*|>` and no raw digits

All tests must run via:
```
cd /data3/Qwen2.5-VL-main/src_new/tests
python run_comprehensive_tests.py
```

---

### Example Commands
- Generate data
```
source ~/.bashrc && conda activate ms
python /data3/Qwen2.5-VL-main/scripts/generate_coord_bootstrap.py \
  --output /data3/Qwen2.5-VL-main/data/coord_bootstrap/coord_bootstrap.jsonl \
  --num_identity 50000 --num_arithmetic 20000 --max_coord 1024 --seed 1337
```
- Train (bootstrap)
```
cd /data3/Qwen2.5-VL-main
source ~/.bashrc && conda activate ms
bash /data3/Qwen2.5-VL-main/scripts/run_coord_bootstrap.sh
```

---

### Migration to Full Detection Training
- Use the bootstrap checkpoint as `--checkpoint` for the main training config (e.g., `bbu_v2_use_coord.yaml`)
- Expect faster convergence and cleaner coord usage in geometry spans
- Keep evaluation hooks to ensure the model never falls back to raw digits

---

### Risks & Mitigations
- Risk: text‑only bias. Mitigation: always keep 1 image (blank) to preserve image token accounting
- Risk: overfitting wording. Mitigation: diversify prompts; hold out template families for eval
- Risk: coord range leakage. Mitigation: strict generator validation; skip any out‑of‑range arithmetic

---

### Implementation Order (Recommended)
1) Add tests (vocab, dataset/spans, strictness)
2) Implement `generate_coord_bootstrap.py` (deterministic, validated)
3) Add `BootstrapCoordDataset` + config flag wiring
4) Add run script and minimal docs
5) Run tests; then run short bootstrap; verify metrics; proceed to main training
