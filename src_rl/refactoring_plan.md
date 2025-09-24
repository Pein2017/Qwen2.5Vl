## Qwen2.5‑VL RL (GRPO) Refactor Plan – Natural Extension of src_new SFT

### 1) Goals
- Make `src_rl/` a first‑class extension of `src_new/` SFT for the same dense captioning task.
- Preserve HF‑first invariants and prompt/tensor parity with he vision path.SFT; never lose t
- Keep CE in SFT; use online sampling with reward shaping in RL; no schema drift.
- Fail fast on missing config/invalid shapes; avoid hidden defaults.

### 2) Parity invariants with src_new (authoritative)
- Tokenizer/processor/model loading identical to training/inference:
  - Fast tokenizer with `offset_mapping`, chat template required, `pad=eos`, left padding, `attn=eager` for stability, bf16 when supported.
  - Image path resolution via `PathManager`, enforce `<|image_pad|>` alignment with `pixel_values` and `image_grid_thw`.
  - Coordinate tokens are deprecated by default. Geometry is emitted as wrappers + raw integers.
- Conversation building is a single source of truth: use `src_new.processing.conversation.ConversationBuilder` for RL prompts too.
- Eval/inference behavior mirrors SFT: single‑turn prompts only; teacher pairing disabled in RL.

### 3) Current gaps (what to fix)
- Vision tensors are dropped before TRL, so generation runs text‑only.
```401:410:/data3/Qwen2.5-VL-main/src_rl/runner.py
    # Use the underlying HF model for TRL compatibility
    hf_model = getattr(bundles["model"], "base_model", bundles["model"])  # type: ignore[assignment]
    trainer = GRPOTrainer(
        model=hf_model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_ds,  # type: ignore[arg-type]
        eval_dataset=val_ds,     # type: ignore[arg-type]
        processing_class=bundles["tokenizer"],
    )
```
- The dataset is vision‑aware, but is wrapped into a prompt‑only dataset, losing `pixel_values`/`image_grid_thw`.
```19:55:/data3/Qwen2.5-VL-main/src_rl/data/dataset.py
class RLDenseJSONLDataset:
  ...
  def __getitem__(self, idx: int) -> Dict[str, Any]:
      ...
      tensors = build_simple_generation_inputs(sample, self.ctx)
      meta = {"width": sample.get("width"), "height": sample.get("height"),
              "object_count": len(sample.get("objects", []) if isinstance(sample.get("objects"), list) else [])}
      out = {**tensors, "meta": meta}
      return out
```
- Rewards are formatting‑centric; no structural/object‑level coverage checks, and no light geometry fidelity checks beyond counts.

### 4) Design: minimal, safe refactor
- Replace the prompt‑only text flow with a multimodal flow end‑to‑end while keeping TRL GRPO semantics.
- Use a thin, vision‑aware trainer adapter (`VisionGRPOTrainer`) that:
  - Accepts batches that contain `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`.
  - Overrides generation plumbing to pass vision tensors to `model.generate`.
  - Concatenates queries+responses while preserving vision tensors.
  - Leaves GRPO algorithmic logic intact (rewards/updates unchanged).

### 5) Concrete changes (modules and edits)
- runner.py
  - Instantiate `VisionGRPOTrainer` instead of `GRPOTrainer`.
  - Remove `_PromptOnlyDataset` and feed `RLDenseJSONLDataset` directly.
  - Keep loader parity: patches, tokenizer/processor, eager attention, bf16 probe, chat template inheritance.
- trainer.py
  - Keep/extend `VisionGRPOTrainer` to actually be used by the runner.
  - Ensure helpers are used inside GRPO flow: `prepare_generate_inputs`, `generate_completions`, `concatenate_queries_and_responses`.
  - Add robust diagnostics on image token alignment before generation.
- data/dataset.py
  - Keep as‑is (already emits vision tensors); add optional `prompt_text` only for logging (not consumed by model).
- prompting/conversation.py
  - Keep single‑turn builder via `ConversationBuilder.create_simple_conversation_for_generation` (training parity).
  - Add optional `build_summary_generation_inputs` for Stage‑A SFT alignment reuse if needed.
- rewards/
  - Extend registry with object‑ and geometry‑aware components (see §7) using the same parsing logic as `src_new.inference`.
  - Keep all rewards deterministic and pure; surface weights via YAML.

### 6) Config surface (RL YAML)
- Required (align names with `src_new` where applicable):
  - model_path, train_data_path, val_data_path, data_root
  - attn_implementation (RL forces eager at runtime), bf16 (auto‑probe), image.max_pixels
  - max_coord_value, coordinate_tokens_enabled=false (default)
  - sampling: sample_k, max_new_tokens, temperature, top_p, repetition_penalty
  - GRPO: per_device_train_batch_size, update_steps, learning_rate, warmup_steps, max_steps
  - rewards: mapping of names→weights (non‑negative, not all zeros)
- Validation: fail fast on missing required keys and invalid ranges; no silent defaults.
- Optional: allow using `src_new.config.load_config` to hydrate model/data fields, but keep RL‑specific keys separate.

### 7) Reward extensions (beyond formatting)
- parse: success if at least one geometry block is present (existing)
- wrappers: object/geometry wrappers balanced (existing)
- coords: count correctness (4/8/even≥4) (existing)
- separators: ASCII commas+pairs (existing)
- vocab: banned terms (existing)
- NEW coverage: predicted object count within a tolerance of GT count
  - Uses dataset `objects` length when available; reward = 1.0 if |pred−gt| ≤ 1 else decays.
- NEW geometry_sanity: clamp and bounds check against `max_coord_value`; penalize huge outliers
- NEW bbox_giou: for any predicted bbox_2d, compute Generalized IoU (GIoU) against matched GT bbox (Hungarian when available, greedy fallback), reward = mean mapped GIoU ([0..1]).
  - For quads, compare GIoU on their axis‑aligned bounding boxes as an approximation.
  - For lines, compare endpoints using normalized L1 distance with Hungarian matching.
- NEW taxonomy_valid: ensure desc strings conform to Level‑0/Level‑1 rules from `src_new/data_details.md` (structural regex checks)
- Implementation detail: reuse a shared parser to turn generated text into `{desc, bbox_2d|quad|line}`
  - Extract from `src_new/inference.py::_normalize_prediction_to_vis_objects` and factor into a small shared utility (no code duplication).

### 8) Shared parsing – single source of truth
- Extract a lightweight parser into `src_new/processing/parse_generated.py`:
  - Functions: `parse_geometry_wrapped_text_to_objects(text, coordinate_tokens_enabled=False) -> List[Dict]`
  - Move tolerant/strict regexes from `src_new/inference.py` here.
  - Import this in RL rewards to reduce drift.

### 9) Diagnostics & alignment checks
- Add `src_rl/tools/parity_check.py` (already present) enhancements:
  - Validate image token vs tensor alignment per batch (counts, density, THW grid)
  - Dump short previews of prompts and shapes
- Add optional offline scorer (`src_rl/eval.py`) that logs each reward component and aggregate.

### 10) Testing strategy
- Unit
  - Rewards: synthetic strings covering wrappers/coords/coverage/IoU/taxonomy
  - Parser: golden inputs for bbox/quad/line; Chinese punctuation normalization
  - Loader: tokenizer offset_mapping presence; chat template propagation
- Integration
  - Build components on a tiny JSONL; run one GRPO step; assert no image/text mismatch
  - Evaluate on 10 samples and produce a JSON report with all reward components
- Regression
  - Ensure RL prompts decode to include `<|image_pad|>`; ensure shapes are stable under pad/eos changes

### 11) Migration plan
1. Wire trainer
  - Switch `runner.py` to `VisionGRPOTrainer` and feed `RLDenseJSONLDataset` directly (remove prompt‑only adapter)
  - Add a minimal collator (if needed) that returns the multimodal dict untouched
2. Parser extraction
  - Factor parser from `src_new/inference.py` into `src_new/processing/parse_generated.py`
  - Update `src_rl/rewards/*` to import it
3. Rewards
  - Implement `coverage`, `geometry_sanity`, `bbox_iou_approx`, `taxonomy_valid`; register in `registry.py`
  - Expose weights in RL YAML; default new weights to small values (e.g., 0.1–0.3)
4. Validation & logging
  - Add pre‑generation validation (image token vs tensor alignment) in trainer before every `generate`
  - Extend `eval.py` to print per‑component means and valid parse rate (already present; keep keys stable)
5. Docs & examples
  - Update `src_rl/README.md` with the new vision‑aware flow and YAML snippet
  - Provide a `scripts/run_dense_grpo.sh` example matching SFT loader flags

### 12) Backward compatibility
- Keep current YAML keys; add new reward names as optional.
- If users still want text‑only experiments, keep a `text_only` dataset flag that bypasses image loading (default off).

### 13) Risks & mitigations
- TRL integration assumptions: ensure `VisionGRPOTrainer` fully overrides the generation call path.
  - Mitigation: add a smoke test that asserts `pixel_values` participates in forward/generate on GPU.
- Parser drift: sharing parser with inference avoids duplication.
- Reward brittleness: keep weights configurable; start small; monitor stability.

### 14) Acceptance criteria
- GRPO training runs with images on; no "Image features and image tokens do not match" errors.
- Prompts/tensors identical to SFT for the same sample.
- Reward report includes both formatting and geometry metrics; valid parse rate > 0.9 on val subset of SFT data.
- Checkpoints generated by RL can be loaded by `src_new/inference.py` without changes.

### Appendix: Evidence of current single‑turn, image‑aware builder (keep)
```38:55:/data3/Qwen2.5-VL-main/src_rl/prompting/conversation.py
def build_simple_generation_inputs(sample: Dict[str, Any], ctx: RLConversationContext) -> Dict[str, Any]:
  ...
  images = _load_images_abs(images_field, data_root=ctx.data_root)
  return ctx.builder.create_simple_conversation_for_generation(sample=sample, images=images)
```
