# Tasks Document

- [x] 1. Scaffold module structure `src_rl`
  - Files/Dirs:
    - `src_rl/__init__.py`
    - `src_rl/trainer.py` (VisionGRPOTrainer subclass of TRL GRPO)
    - `src_rl/rewards/format_rewards.py` (reward components + combiner)
    - `src_rl/rewards/registry.py` (map string keys -> reward callables)
    - `src_rl/prompting/conversation.py` (wrap ConversationBuilder)
    - `src_rl/data/dataset.py` (JSONL → tensors using unified processor)
    - `src_rl/tools/parity_check.py` (SFT↔GRPO input parity utility)
    - `src_rl/eval.py` (evaluation harness)
    - `src_rl/__main__.py` (allow `python -m src_rl`)
    - `src_rl/runner.py` (CLI, config load, trainer wiring)
    - `configs/rl/dense_grpo.yaml` (all knobs; no in-code defaults)
  - Purpose: Establish clean separation and HF-first parity with SFT.
  - _Requirements: 1,2,3,4,5,6_
  - _Prompt: Role: Python ML Engineer | Task: Create `src_rl` skeleton and placeholders with correct imports from `src_new` (no logic yet), add config file stub under `configs/rl/dense_grpo.yaml` | Restrictions: Do not add in-code defaults; ensure files import without circular deps | Success: Module imports succeed; CLI entry `python -m src_rl --help` works (stub)

- [x] 2. Implement tokenizer/processor/model loader (HF-first parity)
  - Files: `src_rl/runner.py`
  - Use `DetectionModel.from_pretrained_fast` and build a unified `Qwen2_5_VLProcessor` with inherited chat_template; set eos `<|im_end|>`, pad=eos, left padding; attention eager; bf16.
  - _Requirements: 1,6_
  - _Prompt: Role: HF Integration Engineer | Task: Implement loader matching `src_new` inference/training parity, fail fast on token/image invariants | Restrictions: No silent fallbacks; no coord-token mode | Success: Loads SFT ckpt and prints device/dtype/vocab parity

- [x] 3. Implement `VisionGRPOTrainer` adapter
  - File: `src_rl/trainer.py`
  - Forward `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` to `model.generate`; preserve vision tensors after concat; EOS `<|im_end|>`; sample_K from config.
  - _Requirements: 4,6_
  - _Prompt: Role: RL Engineer | Task: Subclass TRL’s GRPOTrainer to support multimodal generation and group sampling | Restrictions: Do not alter model forward; keep batching=1; robustness to missing images | Success: Trainer can sample K completions and step without runtime errors

- [x] 4. Port robust parsing utilities for rewards
  - File: `src_rl/rewards/format_rewards.py`
  - Port tolerant geometry parsing logic (no imports across modules); expose pure functions: `parse_geometry_blocks`, `check_wrappers`, `check_ascii_separators`, `check_coords_counts`, `check_banned_vocab`, `length_score`, `compute_rewards`.
  - _Requirements: 2,3,5_
  - _Prompt: Role: NLP/Text Processing Engineer | Task: Implement pure-text reward checks mirroring `src_new/inference.py` behavior (no torch), with unit-testable functions | Restrictions: No file IO; deterministic | Success: Reward unit tests pass on edge cases

- [x] 5. Conversation builder wrapper for RL
  - File: `src_rl/prompting/conversation.py`
  - Use `ConversationBuilder.create_simple_conversation_for_generation(...)` to produce tensors for single-turn student prompts; adopt SFT augmentation gates when enabled in RL YAML.
  - _Requirements: 1,6_
  - _Prompt: Role: Data Pipeline Engineer | Task: Build inputs identical to SFT (shapes/validations), return dict for trainer | Restrictions: Must pass `<|image_pad|>` alignment checks | Success: A sample resolves to valid tensors and decodes 1+ image tokens

- [x] 6. Dataset for GRPO
  - File: `src_rl/data/dataset.py`
  - Read JSONL, resolve images with `create_path_manager`, call the wrapper to materialize tensors; attach meta (width/height, object_count) for rewards.
  - _Requirements: 1,5,6_
  - _Prompt: Role: Data Engineer | Task: Implement minimal dataset yielding ready-to-generate batches; batch size 1 | Restrictions: No teacher pairing; no augmentation unless enabled in YAML | Success: DataLoader iterates and yields valid dicts

- [x] 7. Wire rewards into TRL loop with registry
  - Files: `src_rl/trainer.py`, `src_rl/rewards/format_rewards.py`, `src_rl/rewards/registry.py`, `src_rl/runner.py`
  - Compose component scores with YAML weights; support selecting a subset via keys; log per-component metrics.
  - _Requirements: 2,3,4,5_
  - _Prompt: Role: RL Engineer | Task: Integrate reward registry into GRPO trainer hooks; add metrics logging and best-checkpoint promotion by avg_reward and valid_parse_rate | Restrictions: No in-loop model edits; deterministic seed | Success: Training runs for 100+ steps and logs rising reward

- [x] 8. Config (`configs/rl/dense_grpo.yaml`)
  - Keys: paths (model_path, data_root, train/val), sample_k, max_new_tokens, temperature/top_k/p, repetition_penalty, pixel budgets, freeze policy, reward weights, augmentation gates, logging/save cadence.
  - _Requirements: 4,5,6_
  - _Prompt: Role: MLOps Engineer | Task: Author YAML with explicit keys only; no hidden defaults | Restrictions: Absolute/normalized paths; match repo style | Success: Loader validates and runs

- [x] 9. Unit tests for rewards and parity
  - Files: `src_rl/rewards/test_format_rewards.py`, `src_rl/tools/test_parity_check.py`
  - Edge cases for rewards + a parity test asserting RL vs SFT pre-generation equivalence.
  - _Requirements: 2,3,5,6_
  - _Prompt: Role: QA Engineer | Task: Add tests to guard reward regressions and parity drift | Restrictions: Pure functions only for reward; minimal fixture for parity | Success: pytest passes locally

- [x] 10. Sanity GRPO run script
  - File: `scripts/run_dense_grpo.sh`
  - Launch with accelerate/deepspeed; log dirs and save paths; sample_K=4.
  - _Requirements: 4,5,6_
  - _Prompt: Role: DevOps | Task: Provide a one-liner to start RL | Restrictions: Non-interactive flags; no TTY prompts | Success: Script completes a short run

- [x] 11. Validation harness
  - File: `src_rl/eval.py`
  - Compute valid_parse_rate, wrapper_ok_rate, coords_ok_rate, ascii_sep_rate, banned_vocab_hit_rate, avg_reward on val split.
  - _Requirements: 5,6_
  - _Prompt: Role: Evaluation Engineer | Task: Implement evaluation loop using the same parser functions | Restrictions: No teacher mode | Success: Writes JSON report for best checkpoint