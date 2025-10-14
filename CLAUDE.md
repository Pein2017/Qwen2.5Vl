<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Qwen2.5‑VL — Architecture & Training (Concise Guide)

This file is a lightweight index that stays attached to every chat. It keeps only the shared background/global context and points you to the detailed docs for each main workflow. Use the links below based on the task you’re working on.

### Quick map (choose your task)
- **SFT (supervised fine‑tuning; detection stack)**: see `src_new/UNIFIED_DOCUMENTATION.md`
- **GRPO (dense captioning, post‑SFT)**: see `src_new/rl/GRPO_README.md`
- **Group‑level quality control RL**: see `src_post/README.md`
- **Data conversion pipeline**: see `data_conversion/README.md`

### Shared background (global and always‑on)
- **HF‑first reasoning**: Build typed messages → `Qwen2_5_VLProcessor.apply_chat_template()` → official processor tensors. Never hand‑craft `<|image_pad|>`.
- **Fail‑fast validation**: Validate shapes, token counts, multimodal alignment, special tokens, and paths before forward/generate.
- **Single‑pass loss**: Cross‑entropy computed once; teacher/student and grouped masks (caption/grounding/formatting) applied afterward.
- **Configuration‑driven**: Strict YAML schemas with inheritance; no in‑code defaults. Missing keys surface as errors.
- **Separation of concerns**: Processing, conversation building, model wrapper, losses, training, inference, RL, and conversion are independent with shared contracts.

### Pairing policy (important)
- **SFT**: may include a single teacher turn before the student; only the final assistant (student) contributes to gradients.
- **Dense GRPO (post‑SFT)**: single‑turn prompts; **no teacher‑student pairing**.
- **Group QC GRPO**: Stage‑A (per‑image summary) and Stage‑B (group decision) are single‑turn; **no teacher‑student pairing**.

### Business goals & targets (what the model learns)
- **Dense captioning capacity**: object descriptions + geometry (bbox/quad/line) with canonical wrapper tokens.
- **Single‑image summary ability**: one‑line Chinese summary per image (no geometry/special tokens).
- **Group decision making**: aggregate per‑image summaries to output a final Pass/Fail (with reason) for a multi‑image quality control scene.

### One shared conversation “diagonal” (SFT and dense GRPO)
- The conversation/template → processor tensors path is identical for SFT and dense GRPO; group GRPO reuses this only in Stage‑A (summary variant). Do not hand‑craft `<|image_pad|>`; it is inserted by the official chat template.

```text
<|im_start|>system
你是一个BBU检测助手，请严格遵循格式。
<|im_end|>
<|im_start|>user
请识别图中设备与几何位置，并用规范标记输出。
<|image_pad|><|image_pad|> ... <|image_pad|>   # inserted by HF chat template for the attached image
<|im_end|>
<|im_start|>assistant
<|object_ref_start|>BBU设备/型号未知,安装牢固<|object_ref_end|><|box_start|>[264, 144, 326, 201]<|box_end|>
<|im_end|>
```
- **Geometry wrappers**: `<|box_start|>…<|box_end|>`, `<|quad_start|>…<|quad_end|>`, `<|line_start|>…<|line_end|>`
- **Object reference wrappers**: `<|object_ref_start|>…<|object_ref_end|>` (caption tokens live inside)
- **Turn markers**: `<|im_start|>`, `<|im_end|>`; image placeholders are `<|image_pad|>`

### Minimal entrypoints (for orientation)
- **SFT training**: `/root/miniconda3/envs/ms/bin/python scripts/train_new.py --config /abs/config.yaml`
- **Dense captioning RL (post‑SFT)**: `/root/miniconda3/envs/ms/bin/python -m src_new.rl.runner --config /abs/config.yaml --mode {load|train}`
- **Group QC RL**: `python -m src_post.runner --config /abs/config.yaml` (or `bash scripts/run_group_qc_rl.sh`)

### Where details live
- SFT pipeline, processing stack, losses, inference: `src_new/UNIFIED_DOCUMENTATION.md`
- Dense GRPO after SFT: config schema, runner, trainer, dynamic length, rewards, logging: `src_new/rl/GRPO_README.md`
- Group QC post‑training: two‑stage prompts (Stage‑A/Stage‑B), rewards, configs, DDP notes: `src_post/README.md`
- Data conversion: canonical geometry, hierarchical CN descriptions, smart resize, teacher pool, outputs: `data_conversion/README.md`

### Configurations
- All YAML configs are under `@/configs/`, grouped by task directories.
- Each task provides `debug`, `base`, and `standard` variants (e.g., `base/sft_base.yaml`, `phase_1/standard.yaml`, `phase_1/debug.yaml`). See `@/configs/README.md` for the full structure and usage.

### Critical contracts & health checks (always enforced)
- **Chat template & images**: Typed images in user content; placeholder count equals image count; mismatches raise an error.
- **Multimodal alignment**: `<|image_pad|>` count equals expected tokens from `image_grid_thw`; `pixel_values` rows equal packed patch sums; `image_grid_thw` has shape `[num_images, 3]`.
- **Spans & labels (SFT)**: Assistant spans include `<|im_end|>`; labels outside spans set to −100; `<|image_pad|>` always masked.
- **Grouping invariants (SFT)**: Caption/grounding/formatting masks are disjoint; union equals the shifted assistant mask per role.
- **Geometry tokens**: Use geometry/object‑ref wrappers with raw integers only (no coordinate tokens).
- **RL trust region**: Use generation‑policy log‑probs for denominators to avoid ratio≈1 degeneracy.
- **Checkpoints**: Save tokenizer/processor; wrapper exposes `model.config` for HF compatibility; best‑checkpoint rotation is atomic.

### Troubleshooting pointers (short)
- **Image token mismatch**: Confirm decoded prompt `<|image_pad|>` count vs `image_grid_thw` and packed `pixel_values`; see SFT/Inference sections in `src_new/UNIFIED_DOCUMENTATION.md`.
- **RL ratio degeneracy**: Ensure `generation_logps` are computed/stored; see trust‑region notes in `src_new/rl/GRPO_README.md`.
- **Formatting/geometry parse issues**: Use training‑matched templates and wrappers; see parsing details in `src_new/UNIFIED_DOCUMENTATION.md` and RL rewards notes in `src_new/rl/GRPO_README.md`.

This concise guide intentionally avoids task‑specific deep dives. For any workflow, jump to the corresponding markdown listed above.