# TRL-Based GRPO Migration Plan (Minimal, Fresh Build)

## Goals
- Replace the manual GRPO stack (`src_new/rl/grpo_trainer.py`, `generation_buffer.py`, `metrics_aggregator.py`, etc.) with a lean implementation that leverages Hugging Face `trl.GRPOTrainer`.
- Preserve HF-first multimodal processing defined in `src_new/processing` and documented in `src_new/UNIFIED_DOCUMENTATION.md` without introducing compatibility layers.
- Reduce infrastructure surface area to the essentials required for dense captioning GRPO (single-turn, no teacher/student pairing).

## Reference Baseline
- Manual pipeline entry: `src_new/rl/runner.py:252` orchestrates dataset loading, reward wiring, and `BBUGRPOTrainer`.
- Dataset builder: `src_new/rl/data/dataset.py:24` ensures prompts, `pixel_values`, `image_grid_thw`, and metadata align with the SFT stack.
- Trust-region mechanics: `src_new/rl/generation.py`, `logprobs.py`, and reward registry `src_new/rl/rewards/registry.py`.
- Logging/diagnostics: spread across `metrics_aggregator.py`, `tensorboard_logger.py`, and console formatting utilities.

These inform success criteria for the new path (same contracts, fewer bespoke components).

## Target Minimal Architecture

1. **Configuration**
   - Define a new YAML schema section (e.g., `trl_grpo`) mapping 1:1 onto `trl.GRPOConfig`. Strict validation still lives in `RLConfig`, but the legacy manual-only fields (buffer reuse, manual schedulers) are dropped.
   - Keep only one entrypoint (`scripts/train_new.py` continues to call `src_new/rl/runner.py`, which now builds TRL args directly). No manual/TRL branching.

2. **Dataset & Processing**
   - Re-implement the RL dataset class to emit a TRL-friendly record:
     ```python
     {
         "prompt": ctx.builder.build_prompt(sample),  # list[dict{role, content}]
         "images": loaded PIL.Image list,
         "meta": {...},  # optional reward payload
     }
     ```
   - `ConversationBuilder` handles chat template + vision tensorization inside a new minimal collator invoked inside the trainer; no persisted `input_ids`/`pixel_values` on disk or adapter layer.

3. **Trainer**
   - Create `src_new/rl/trl_trainer.py` by forking the required pieces of `trl/trainer/grpo_trainer.py` to:
     - Accept multimodal batches (`prompt`, `images`, `meta`) and call the existing Qwen processor to produce tensors before generation.
     - Ensure stored generation log-probs use the generation policy (trust-region requirement).
     - Stream rewards via lightweight hooks, reusing only the reward registry functions.
   - Strip unused features (vLLM integration, PEFT toggles, dataset repetition heuristics) unless explicitly required by configs.

4. **Rewards**
   - Keep the existing reward registry functions but expose them directly through a thin list of callables passed to the new trainer. No observe-only plumbing unless needed—each active reward maps to its weight in YAML.
   - Maintain optional normalization/standardization using current `RewardStandardizer` if empirically necessary; otherwise remove.

5. **Logging**
   - Replace the manual aggregators with simple per-step logging inside the trainer (reward mean/std, advantage stats, clip ratios). Use Accelerator’s `gather` utilities directly where cross-rank sync is needed.
   - Retain TensorBoard logging only if required; otherwise emit console + JSONL summaries via standard `transformers.Trainer` callbacks.

6. **Runner**
   - Rewrite `src_new/rl/runner.py` to: load YAML → instantiate tokenizer/processor/model (still via `build_hf_components`) → build the new dataset → initialize the TRL-based trainer → call `train()` or `evaluate()`. Remove manual buffer scheduling and Accelerate wiring now handled by TRL.

## Implementation Steps
1. **Scaffold**
   - Delete manual trainer modules once the new trainer compiles (commit in feature branch).
   - Add new dataset + trainer files with the minimal feature set above.
2. **Wire Config**
   - Update `RLConfig` dataclasses (+ schema tests) to reflect the trimmed parameter set.
   - Adjust YAML configs under `configs/dense_rl/` to align with TRL fields (e.g., `num_generations`, `max_completion_tokens`, `temperature`).
3. **Integrate Rewards**
   - Port registry functions unchanged; pass them directly to the trainer in `runner.py`.
   - Validate meta-dependent rewards still receive object data from the dataset.
4. **Testing**
   - Run `python -m src_new.rl.runner --config ... --mode load` to ensure processors and datasets build.
   - Execute a short training dry run (small dataset, `max_steps=5`) to confirm generation, reward computation, and logging.
   - Compare reward statistics against the manual baseline to confirm parity.
5. **Cleanup**
   - Remove now-unused modules (`generation_buffer.py`, `metrics_aggregator.py`, etc.) and update documentation (`GRPO_README.md`) to describe the new flow.

## Risks & Mitigations
- **Multimodal breakage**: Direct TRL use expects text-only prompts. Forked trainer must be tested with image-heavy samples and validated against `src_new/processing` invariants (token counts, `<|image_pad|>` alignment). Add assertions mirroring SFT fail-fast checks.
- **Performance regressions**: TRL’s default batching may diverge from Swift-style buffer reuse. Start with `num_iterations=1`, `steps_per_generation=gradient_accumulation_steps` to match old behavior; profile generation cadence before tuning.
- **Reward drift**: Simplifying logging/standardization can shift reward scaling. Monitor reward histograms on the first migration runs and reintroduce normalization only if instability appears.
- **Code removal**: Deleting manual modules reduces traceability for past experiments. Archive a tag or branch before wiping to preserve historical reference.

## Next Actions
1. Approve the trimmed config schema and new trainer surface.
2. Implement dataset + trainer skeleton, focusing on end-to-end multimodal generation.
3. Port reward wiring, validate on a debug config, then iterate on logging.
4. Finalize documentation updates, remove legacy files, and cut a migration summary for the team.

