# Design Document

## Overview
We introduce a new module `src_rl` that performs GRPO post-training (via TRL) on top of SFT checkpoints from `src_new`. The objective is to stabilize dense-caption outputs: strict wrapper structure, integer coordinate lists with correct counts, ASCII separators, and domain-constrained vocabulary. The module keeps HF-first invariants (chat template parity, image token alignment), reusing the `src_new` conversation builder and unified processor. No changes to `src_new` code paths.

## Steering Document Alignment

### Technical Standards (tech.md)
- HF-first data flow (typed messages → chat template → processor tensors) retained.
- No in-code defaults: all RL hyperparameters and reward weights live in `configs/rl/dense_grpo.yaml`.
- Strict validations on tokens and image/text counts; fail fast on mismatch.

### Project Structure (structure.md)
- New top-level `src_rl/` holding only RL artifacts: trainer adapter, rewards, prompting wrapper, dataset, and runner.
- Configs under `configs/rl/` (no reuse of `src_post`).

## Code Reuse Analysis
- Reuse `src_new.processing.conversation.ConversationBuilder` to build prompts and tensors identical to SFT.
- Reuse tokenizer/processor/model loading discipline from `src_new.models.wrapper.DetectionModel.from_pretrained_fast` and unified processor creation.
- Reuse parsing logic/ideas from `src_new/inference.py` for reward computation (ported as pure utilities to avoid cross-dependencies).

### Existing Components to Leverage
- `ConversationBuilder`: to produce `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` with validations.
- `Qwen2_5_VLProcessor` and tokenizer with inherited chat_template; left-padding and eos `<|im_end|>`.
- Model wrapper `DetectionModel` for stable loading and vocabulary consistency.

### Integration Points
- Checkpoints saved under a new RL output directory and include tokenizer/processor.
- Optional Deepspeed Zero2/Zero3 via env/switch; keep attention eager.

## Architecture
A thin adapter around TRL’s `GRPOTrainer` to support multimodal inputs, with a plug-in reward layer operating on generated text. Dataset yields preprocessed vision+text tensors built exactly like SFT.

```mermaid
graph TD
    A[JSONL samples] --> B[ConversationBuilder]
    B --> C[Unified Processor -> tensors]
    C --> D[VisionGRPOTrainer (TRL)]
    D --> E[Model.generate K samples]
    E --> F[Rewards: parse/wrappers/coords/separators/vocab]
    F --> D
    D --> G[Checkpoint+metrics]
```

## Components and Interfaces

### src_rl/trainer.py — VisionGRPOTrainer
- **Purpose:** Minimal subclass of TRL GRPO to pass vision tensors during generation and keep them when concatenating query+response.
- **Interfaces:**
  - `generate_completions(batch_dict) -> generated_ids`
  - `concatenate_queries_and_responses(queries, responses)` ensures `pixel_values`/`image_grid_thw` persist.
- **Dependencies:** TRL `GRPOTrainer`, Accelerate, Transformers; our `DetectionModel` and processor.
- **Reuses:** None beyond TRL base; shares tokenizer/model from loader.

### src_rl/rewards/format_rewards.py
- **Purpose:** Compute scalar reward ∈ [0,1] per completion from components:
  - R_parse, R_wrappers, R_coords, R_separators, R_vocab, R_len.
- **Interfaces:**
  - `compute_rewards(texts: List[str], metas: List[dict], weights: dict) -> List[float]`
  - Individual pure functions per component for unit tests.
- **Dependencies:** Ported string parsers inspired by `inference.py` (no torch dependency).
- **Reuses:** Token and geometry wrappers defined in `special_tokens.py` (copied literals locally to avoid imports).

### src_rl/prompting/conversation.py
- **Purpose:** Wrap `ConversationBuilder` to create student-only prompts for dense-caption GRPO.
- **Interfaces:** `build_inputs(sample: dict) -> Dict[str, Tensor]`.
- **Dependencies:** `ConversationBuilder`, unified processor.

### src_rl/data/dataset.py
- **Purpose:** JSONL loader that emits already-processed tensors for GRPO.
- **Interfaces:** HF Dataset-style `__getitem__` returning dict with `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, plus `width/height` for reward checks and `raw_text_prompt` for debugging.
- **Dependencies:** `create_path_manager` for image resolution; `ConversationBuilder`.

### src_rl/runner.py
- **Purpose:** CLI entry to run GRPO with config; initializes loader, trainer, reward fn, and training loop; logs metrics and saves best checkpoint.
- **Interfaces:** `main()` with args or YAML path.
- **Dependencies:** TRL, Accelerate, Transformers; our loader utilities.

## Pipeline Parity Details (SFT ↔ GRPO)
1. Prompts: Use `get_system_prompt(format_mode="special_tokens")` and `BASE_USER_PROMPT`; `add_generation_prompt=True` for a single assistant target.
2. Chat template: Inherit template from tokenizer; enforce `pad_token_id=eos_token_id=id(<|im_end|>)`; left padding.
3. Conversation builder: Call `create_simple_conversation_for_generation(...)`; teacher pairing disabled.
4. Image pipeline: Use same processor with `max_pixels` override from RL YAML; smart-resize budgets (factor=28) configurable in RL YAML to match SFT.
5. Invariants: Validate `<|image_pad|>` counts vs THW and tensor-row sums; fatal on mismatch.
6. Augmentation: Gate reuse of SFT augmentations via RL YAML: `features.augmentation.enabled`; default false.
7. Parity test: Add a dev utility to compare RL vs SFT pre-generation `input_ids` up to the generation prompt index and image tensors.

## Reward Registry
- Multiple reward candidates are supported via a simple registry pattern:
  - `src_rl/rewards/registry.py` maps keys to callables; YAML selects active ones and weights.
  - Example keys: `parse`, `wrappers`, `coords`, `separators`, `vocab`, `length`.

## Data Models
- Sample schema: same as SFT JSONL (`images`, `objects`, `width`, `height`, optional `summary`). For RL we only need `images` and `width/height`; `objects` optionally used by a soft line-count reward.
- Rewards metadata: `{width:int,height:int,object_count:int|null}`.

## Error Handling
- Missing tokens/wrappers → fail fast with explicit list.
- Image/text alignment mismatch → raise with decoded preview and shapes.

## Testing Strategy
- Unit tests for reward functions and registry selection.
- Parity utility test on 1–2 samples asserting equal pre-generation `input_ids` and image tensor shapes.
- Small GRPO run (K=4) to observe improvements in `valid_parse_rate`, `avg_reward`.