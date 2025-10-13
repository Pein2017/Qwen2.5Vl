# TRL-Based GRPO Migration Plan (VLM‑ready, Consolidated)

## Decision & Scope
- Migrate from the manual GRPO stack to a TRL-based trainer for dense captioning RL, while preserving HF-first multimodal contracts (chat template → processor tensors) and trust-region correctness.
- Target: A minimal fork/wrapper of TRL’s `GRPOTrainer` that supports image+text batches and true generation-policy logprobs (with and without vLLM).
- Non-goals: Feature parity for all manual diagnostics; only essential metrics/logging will be kept initially.

## Hard Constraints (must remain true)
- HF-first processing: prompts built with `Qwen2_5_VLProcessor.apply_chat_template()`; images passed as typed data; never hand-craft `<|image_pad|>`.
- Multimodal invariants: `<|image_pad|>` count ⇔ `image_grid_thw`; packed patch rows equal THW sum; fail fast on mismatch.
- EOS discipline: Train on completion tokens up to the first `<|im_end|>`.
- Trust region: Use generation-policy per-token logprobs as denominator (avoid ratio ≈ 1 degeneracy).

## Target Minimal Architecture
1) Configuration (1:1 mapping)
- Add `trl_grpo` (or reuse `grpo`) fields that map directly to TRL’s `GRPOConfig`:
  - `num_generations`, `max_prompt_length`, `max_completion_length`, `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `loss_type`, `epsilon`, `epsilon_high`, `beta`, `scale_rewards`, `mask_truncated_completions`, `steps_per_generation`, `num_iterations`, `use_vllm`, `vllm_mode`, `vllm_tensor_parallel_size`.
- Keep a single entrypoint (`scripts/train_new.py` → `src_new/rl/runner.py`) that builds TRL args directly; no manual/TRL branching.

2) Dataset & Collator (VLM)
- Reuse the SFT data helpers to avoid duplication:
  - Import `read_jsonl`, `create_path_manager`, and the `ConversationBuilder` wrapper already used in SFT (`src_new/data/dataset.py`, `src_new/processing/conversation/builder.py`).
  - Factor a prompt-only view of `StandardDataCollator` that reuses `_pad_sequence` and the multimodal checks; wrap it with `TrainerCompatibleDataCollator` so TRL still receives 2‑D tensors even without labels.
- Dataset emits TRL-friendly records:
```python
{
  "prompt": list[dict{role, content}],   # strictly via ConversationBuilder
  "images": list[PIL.Image],            # raw RGB images, length = num images in prompt
  "meta": dict | None                   # reward payload (objects, geometry, etc.)
}
```
- Prompt-only collator (called inside the trainer) applies the processor to each batch:
  - Uses `ConversationBuilder` + `Qwen2_5_VLProcessor.apply_chat_template()` to build canonical prompt text (same callsites as SFT).
  - Runs the official processor to produce tensors: `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, optional `images_per_sample`, plus `conversation_text` for reward logging.
  - Reuses the SFT collator’s invariant checks (token counts vs `image_grid_thw`, packed patch sums, `<|im_end|>` present). Any mismatch → `ValueError` (fail fast).
  - Keeps dtype/device expectations identical to `build_hf_components()` so the tensors feed directly into `model.generate()` and reward code.

3) Trainer (minimal fork/wrapper)
- Add `src_new/rl/trl_vlm_trainer.py` that subclasses or lightly forks TRL’s `GRPOTrainer` with the following overrides:
  - Data preparation: accept batches containing `prompt`+`images`; apply chat template and processor to build tensors.
  - Generation paths:
    - Transformers path: call `model.generate()` with vision tensors; compute per-token logprobs for the *chosen* tokens (the actual sequence produced) using `outputs.scores`, mirroring the manual implementation so the denominator reflects the generation policy.
    - vLLM path (server/colocate): request chosen-token logprobs via vLLM’s `logprobs` API. If the deployed vLLM build does not expose them, document that the trust-region guarantee is weakened (ratios collapse toward 1) and gate this behavior behind a config toggle.
  - Trust region: during loss, use stored generation logprobs as denominator and keep TRL’s clipping logic. KL regularization stays disabled by default (`beta=0`); no reference model will be instantiated unless explicitly configured later.
  - Masking & length: build completion masks that stop after first `<|im_end|>`; honor `mask_truncated_completions`.
  - Steps-per-generation: rely on TRL’s existing buffering (`steps_per_generation`) without rebuilding the CPU buffer; ensure stored generation logprobs persist across reuse cycles so ratios remain correct.
  - Metrics: log core scalars (reward mean/std, clip ratios, completion lengths, termination ratio). Rich diagnostics can be reintroduced later as needed.
  - Reuse existing helpers where possible: `src_new/rl/logprobs.get_per_token_logps` (vision slicing + trust-region math) and dynamic-cap logic from `src_new/rl/buffer.py`.

> **Chosen-token logprobs** refer to the log probabilities that the generation policy assigned to each token it actually produced. They are required for GRPO’s trust-region ratio; we only store these values (no need for full distribution snapshots).

4) Rewards
- Reuse the existing registry; wire reward callables directly into TRL via `reward_funcs=[...]` and `reward_weights=[...]`.
- Pass `meta` transparently from dataset → reward functions. Support sanitize/tail logic if required by metrics.

5) Runner
- Update `src_new/rl/runner.py` to:
  - Load YAML → build tokenizer/processor/model.
  - Build RL dataset and new collator.
  - Initialize `trl_vlm_trainer.GRPOVLMTrainer` (subclass/fork of TRL) with config, reward functions, datasets.
  - Call `train()`; keep `evaluate()` path using the existing lightweight eval harness.

## Config Mapping (old → TRL)
- `sample_k` → `num_generations`
- `max_new_tokens` → `max_completion_length`
- `min_new_tokens` → include in `generation_kwargs`
- `temperature/top_p/repetition_penalty` → same
- `epsilon_low/epsilon_high` → `epsilon` / `epsilon_high`
- `beta` → same (KL proxy)
- `scale_rewards` → same
- `mask_truncated_completions` → same
- `steps_per_generation` → same (reuse) — ensure stored generation logprobs persist across steps
- `gradient_accumulation_steps` → TRL’s Trainer arg (no change)
- `generation.dynamic_length.*` → map to per-batch caps inside the wrapper while TRL still receives a fixed `max_completion_length`
- (removed) `rewards.standardizer` → rely on TRL `scale_rewards`; no custom standardizer
- vLLM: `use_vllm`, `vllm_mode`, `vllm_tensor_parallel_size`, plus a new flag `vllm_logprobs: true` (local extension) to request chosen-token logprobs

## Milestones & Deliverables
1) M0 — Baseline TRL text-only dry run (same repo, tiny dataset)
- Verify end-to-end training loop and logging without images.
- Deliver: small overfit run and logs.

2) M1 — Multimodal collation + transformers generation
- Extract the reusable prompt-only collator from SFT code; validate invariants; feed tensors to `generate()`; store generation logprobs from `outputs.scores`.
- Deliver: 1–2 steps with images; show ratios not ≈ 1 and sane clip stats.

3) M2 — vLLM integration with logprobs
- Enable vLLM in colocate mode; request token logprobs; store generation logprobs; confirm trust-region ratios deviate from 1.
- Deliver: short training confirming speedup and correct ratios.

4) M3 — Steps-per-generation reuse parity
- Ensure `steps_per_generation > 1` reuses completions while preserving stored generation logprobs across reuse; verify stability.
- Deliver: compare reward/clip trends vs manual trainer at `S=1` and `S>1`.

5) M4 — Validation & parity checks
- Compare against manual trainer on a fixed debug config: reward mean/std, termination ratio, length stats, and clip ratios.
- Deliver: brief report; accept within agreed tolerance.

6) M5 — Cleanup & documentation
- Update README/GRPO docs; mark manual modules deprecated; keep eval harness.
- Deliver: documentation PR and deprecation note.

## Success Criteria (must meet to cutover)
- Multimodal invariants enforced with fail-fast errors.
- Trust-region ratios not degenerate (distribution spans <1 and >1); clip ratios in expected ranges.
- Reward/length/termination stats comparable to manual baseline on debug run.
- vLLM path yields a measurable generation-time speedup without breaking memory.
- Config validation remains strict (missing/extra fields surface as errors).
- With `beta=0` (default path), training stays stable without a reference model.

## Risks & Mitigations
- VLM data path mismatch in TRL: Mitigation → subclass/fork trainer to accept processor outputs and vision tensors; add validators.
- Missing vLLM logprobs: Mitigation → short-term recompute via current policy (warn), long-term require vLLM with logprob API and gate via `vllm_logprobs` flag.
- Performance regression from buffer differences: Mitigation → use TRL `steps_per_generation`; profile; optionally add a minimal CPU-side cache if needed later.
- Reward drift: Mitigation → keep weights/standardization identical; compare histograms on debug runs; iterate.

## Rollback / Hybrid Plan
- Keep the manual trainer runnable behind a feature flag while validating TRL runs.
- If vLLM logprobs are unavailable on the target infra, run transformers path until the dependency is upgraded.

## Immediate Next Actions
1) Extract shared dataset/collator utilities from SFT (`read_jsonl`, `_pad_sequence`, invariant checks) and expose prompt-only variants for RL.
2) Add `trl_vlm_trainer.py` (subclass/fork scope as above) and hook the shared collator in.
3) Extend config loader to build TRL `GRPOConfig` from YAML; add `vllm_logprobs` toggle.
4) Wire rewards and dataset in `runner.py`; compile and run M0/M1.
5) Add vLLM logprobs path (M2); validate ratios; iterate.
