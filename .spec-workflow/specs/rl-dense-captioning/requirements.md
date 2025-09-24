# Requirements Document

## Introduction
We will add a new post-training module `src_rl` to perform GRPO reinforcement learning (using TRL) on top of an SFT checkpoint produced by `src_new`. The goal is to stabilize dense-caption formatting (wrappers, separators, geometry counts) and reduce domain drift (ban irrelevant vocabulary), improving valid-parse rates and final prediction utility without changing SFT data contracts.

## Alignment with Product Vision
This extends the Qwen2.5-VL pipeline beyond SFT with a HF-first, explicit, and well-validated RL phase. It keeps strict invariants from `src_new` (chat template parity, image-token alignment) and focuses on real deployment pain points (malformed geometry, off-domain text). It is separate from existing `src_post` (group QC) to avoid scope collision.

## Requirements

### Requirement 1: Training Compatibility
**User Story:** As an ML engineer, I want RL to reuse the SFT tokenizer/processor/model and conversation builder, so that RL training strictly matches SFT inputs and avoids token/image mismatches.

#### Acceptance Criteria
1. WHEN RL trainer loads the checkpoint THEN it SHALL load tokenizer, processor, and model via the same HF-first flow and set eos to `<|im_end|>` and pad to eos.
2. WHEN building prompts THEN it SHALL use the same conversation builder to produce `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` with identical shape validations.
3. WHEN running generate THEN it SHALL pass both text and vision tensors and preserve left-padding.

### Requirement 2: Rewards for Format Validity
**User Story:** As a researcher, I want explicit rewards for correct wrappers, integer coordinates, ASCII separators and bracket pairing, so that the model internalizes formatting.

#### Acceptance Criteria
1. WHEN a completion is parsed THEN the system SHALL compute R_parse, R_wrappers, R_coords, R_separators in [0,1].
2. IF any geometry is malformed (counts/order) THEN R_coords SHALL reduce accordingly; missing wrappers reduce R_wrappers.
3. WHEN separators include Chinese punctuation or missing spaces THEN R_separators SHALL reduce.

### Requirement 3: Domain Cleanliness
**User Story:** As a product owner, I want to discourage irrelevant professional vocabulary outside our BBU taxonomy, so that outputs remain in-domain.

#### Acceptance Criteria
1. WHEN banned terms are present (e.g., PPDU, DCDU, CPRI) THEN reward SHALL be penalized.
2. WHEN taxonomy-aligned words are used and no banned terms exist THEN reward SHALL not be penalized.

### Requirement 4: RL Loop via TRL GRPO
**User Story:** As an engineer, I want to use TRL’s GRPO to minimize custom RL code, so that we can iterate quickly and scale with Accelerate.

#### Acceptance Criteria
1. WHEN training starts THEN a GRPO trainer SHALL sample K responses per prompt and compute group advantages from a single reward function, no critic.
2. WHEN saving checkpoints THEN tokenizer/processor SHALL be saved; best-by-reward checkpoint SHALL be promoted.
3. WHEN resuming THEN training SHALL continue seamlessly from the last checkpoint.

### Requirement 5: Evaluation & Metrics
**User Story:** As a QA engineer, I want val metrics to track valid-parse rate and component rates, so that we can compare improvements.

#### Acceptance Criteria
1. WHEN running validation THEN the system SHALL report: valid_parse_rate, wrapper_ok_rate, coords_ok_rate, ascii_sep_rate, banned_vocab_hit_rate, avg_reward.
2. WHEN training completes THEN it SHALL write a JSON report with these metrics for the best checkpoint.

### Requirement 6: SFT↔GRPO Pipeline Parity (Prompts, Templates, Invariants, Augmentation)
**User Story:** As a maintainer, I want GRPO to mirror SFT data preparation so that generation quality gains come from rewards rather than pipeline drift.

#### Acceptance Criteria
1. Prompts: RL SHALL use the same system and user prompts as SFT (system via `get_system_prompt(format_mode="special_tokens")`; user via `BASE_USER_PROMPT`), encoded by the tokenizer’s chat_template with `add_generation_prompt=True` to target a single assistant round.
2. Chat template: RL processor SHALL inherit `chat_template` from tokenizer; padding side left; `eos_token_id` equals `<|im_end|>` and `pad_token_id` equals eos.
3. Conversation builder: RL SHALL reuse `ConversationBuilder.create_simple_conversation_for_generation(...)` and disable teacher pairing; assistant is the only generated turn.
4. Image pipeline: RL SHALL use the same `Qwen2VLImageProcessor` with `max_pixels` override from YAML; smart-resize budgets (factor=28) SHALL match SFT settings configurable in RL YAML.
5. Invariants: RL SHALL validate `<|image_pad|>` count vs THW-derived expectation and `pixel_values` rows vs THW sum exactly like SFT; failures are fatal.
6. Augmentation: RL SHALL optionally reuse SFT augmentation stages (at minimum smart-resize; photometric/line ops when explicitly enabled in RL YAML); default off. When enabled, it SHALL call the same implementations to ensure parity.
7. Parity tests: A parity check SHALL confirm for a sample that RL and SFT produce identical `input_ids` up to the generation prompt boundary and identical image tensors (`image_grid_thw`, `pixel_values`) modulo non-deterministic low-level pixel differences.
8. Config: All parity toggles (augmentation reuse, budgets) SHALL live in RL YAML; no in-code defaults.

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: `src_rl` contains trainers, rewards, prompting, and a thin dataset; it does not modify `src_new`.
- **Modular Design**: Rewards isolated under `src_rl/rewards/`; VLM adapter under `src_rl/trainer.py`.
- **Dependency Management**: Use TRL/Accelerate/Transformers already in environment. No hidden defaults; all knobs in a YAML under `configs/rl/`.
- **Clear Interfaces**: Reward function signature `List[str] -> List[float]` with access to sample meta (width/height).

### Performance
- Attn eager, bf16, per-device batch 1, sample_K 4–8; max_new_tokens ≤ 512; training 2–5k steps.

### Security
- No network calls; local data only. Deterministic seeds; safe file IO.

### Reliability
- Fail fast on missing tokens, processor mismatch, or vision/text count mismatch. Unit tests for reward edge cases and parity checks.

### Usability
- Single CLI entry: `/root/miniconda3/envs/ms/bin/python -m src_rl.runner --config /abs/configs/rl/dense_grpo.yaml`. Clear logs and metrics.