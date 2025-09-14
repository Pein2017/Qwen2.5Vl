## RL Post-Training for Group-Level QC (Qwen2.5‑VL, GRPO)

### What this module does
- **Goal**: Fine‑tune an SFT‑trained Qwen2.5‑VL to make a single “pass/fail + reason” decision for a group of images (a work order), with one short Chinese summary per image.
- **Two stages**:
  - **Stage A (per image)**: generate one concise, informative Chinese summary per image。
  - **Stage B (group aggregation)**: aggregate per‑image summaries and output: 总评: 通过/不通过 (+ 原因 when 不通过)。
- **Post‑training method**: **GRPO (Group Relative Policy Optimization)**
  - Sample multiple responses to the same prompt; compute rewards; convert to relative advantages via z‑score; push up higher‑reward samples and push down lower‑reward samples.
  - Optimizes Stage‑B (final decision) and, optionally, Stage‑A (image summaries: conditional per‑image or joint‑set).

### Start here: GRPO in 5 minutes
- **One‑liner**: Try several answers, score them, and make the model more likely to produce the better ones next time. No critic/value head.
- **You need**:
  - An SFT checkpoint + its processor (from `src_new/`).
  - A dataset organized by group labels (`审核通过|审核不通过/{group_id}/*.jpg`) or a JSONL file.
- **How it updates your SFT model**: Fine‑tunes the LM head and a few top text layers (vision usually mostly frozen). GRPO gradients flow through Stage‑B replies and optionally Stage‑A summaries.
- **Quick run**:
  ```bash
  conda activate ms
  bash scripts/run_group_qc_rl.sh /abs/path/to/config.yaml
  ```
  Minimal keys in `configs/rl/group_qc_grpo.yaml`:
  ```yaml
  checkpoint: /abs/path/sft_ckpt
  processor:  /abs/path/processor
  train_data_dir: /abs/path/groups
  output_dir: /abs/path/output_post/grpo
  mission: bbu安装方式检查
  # Sampling
  K_B: 3                 # Stage-B replies per group
  K_A: 3                 # Stage-A candidates per image (if enabled)
  train_stage_a_mode: conditional   # conditional | joint | off
  stage_a_weight: 1.0
  ```

---

## Rewards (mission‑aware overview)
- Registry: `src_post/rewards/` with name→fn in `rewards/__init__.py`.
- Inputs to reward functions: `{ gt_label, pred_label, summary_lines, stage_b_reason, checklist_lines, tf_p_pass, tf_p_fail, mission }`（按需使用）。
- Mission‑aware behavior:
  - `coverage`: 解析 `MISSION_CHECKS_COVERAGE`（见 `prompting/conversation.py`）并按规范短语计算覆盖度；当摘要含“无需安装”时自动满足挡风板符合性；缺省回退为规范槽位覆盖。
  - `taxonomy`: 词汇表来自 `data_conversion/hierarchical_attribute_mapping.json`，并与 `MISSION_CHECKS_COVERAGE` 的 token 取交集（若提供 mission）。
  - `violations`: 负面词表对齐 SFT 摘要规范，集中于 `rewards/lexicon.py`。
  - `decision_strict`: 仅校验“总评/原因”的严格格式（含禁用词，见 `lexicon.py`）。
- 推荐组合（示例）：
  ```yaml
  reward_fns: group_margin,decision_strict,label_match,coverage,consistency,violations,cleanliness,taxonomy,rep_penalty,quote_penalty,special_penalty,pass_prior
  reward_weights: 1.0,3.0,0.5,0.8,0.3,0.3,0.2,0.2,-0.2,-0.1,-0.2,0.05
  ```

### Shared lexicon & helpers
- `rewards/lexicon.py`: 统一维护 `NEGATIVE_TOKENS`, `FORBIDDEN_DECISION_WORDS`, `CANONICAL_SLOTS`。
- `prompting/schema.py`: 解析 `MISSION_CHECKS_COVERAGE` 为 mission→check→token 集（带缓存）。
- `generation/generation.py::build_decision_prefix_constraint(...)`: 统一的 Stage‑B 决策前缀约束构建。

---

## How to run
```bash
# From repo root
conda activate ms
bash scripts/run_group_qc_rl.sh /abs/path/to/config.yaml
```
- The script auto‑detects single vs multi‑GPU and launches with PyTorch DDP when `GPU_DEVICES` has multiple GPUs.
- Entry: `python -m src_post.runner --config <yaml>`

Data layout (recommended):
```text
/abs/path/to/train_groups/
  审核通过/
    <group_id>/image_*.jpg
  审核不通过/
    <group_id>/image_*.jpg
```
Or JSONL: one group per line
```json
{"images": ["/abs/.../1.jpg", "/abs/.../2.jpg"], "label": "pass", "meta": {"group_id": "g1"}}
```
Label normalization: 审核通过|通过|pass → pass；审核不通过|不通过|fail → fail。

---

## Process workflow (components)

1) Data loading (group level)
- File: `src_post/data/dataset_group_qc.py` (`RLGroupQCDataset`)
- Yields `{ image_paths, images, label, meta, num_images }` from the directory layout above or JSONL.

2) Stage A summarization (per image, typed chat)
- File: `src_post/prompting/conversation.py` → `GroupQCConversationBuilder.build_stage_a_messages(mission)` builds:
  - system: 限制一行中文摘要；仅 BBU 场景对象；禁止坐标/几何/特殊标记 `<|...|>`；避免引号与重复清单；偏好简洁短语（品牌/挡风板/安装/是否需要/合规/标签可读性/光纤弯曲/电线整齐度等）。
  - user: typed list `[{"type": "image"}, {"type": "text", "text": "请只输出一行摘要"}]`
- Tokenization: `Qwen2VLProcessor.apply_chat_template(...)` → `processor(text=[...], images=[img], return_tensors='pt')`
- Generation:
  - Greedy one line per image becomes Stage‑B context.
  - When training Stage‑A GRPO, sample `K_A` candidates per image (conditional) or `K_set` candidate‑sets (joint).
  - Stopping: Stage‑A decoding stops early on newline or `。` if tokenized as a single token.

3) Stage B GRPO (group decision)
- Entry: `src_post/runner.py` → builds prompt via `build_stage_b_messages(summary_lines, checklist_lines)`.
- Sampling: sample `K_B` short replies; parse with `src_post/prompting/span_parser.py` to `{label, reason}`.
- Reward: composed via `src_post/rewards/` (see “Rewards” below).
- GRPO update: teacher‑forcing on each sampled reply to sum log‑probs over the reply; z‑score rewards → advantages → optimize `-A_k * logp_k` (+ optional KL to reference).

4) Stage A GRPO (optional)
- Modes: `conditional` (per‑image variants against fixed others) or `joint` (sample full sets across all images).
- Rebuild Stage‑B prompt with the candidate(s), recompute group reward, z‑score to advantages, teacher‑force summary tokens, and apply GRPO loss (+ optional KL).

5) Outputs
- Per‑rank JSONL: `results.rank{RANK}.jsonl` with `{group_index, mission, gt_label, pred_label, reward, images[captions], stage_b_raw}`.
- Checkpoints: `output_dir/checkpoints/grpo_final/` with the policy and saved processor.

---

## Tensor flow (forward and backward)

### Stage A (context generation unless training Stage‑A)
- Encode typed image message → tensors: `input_ids [1, L]`, `pixel_values`, `image_grid_thw`.
- Generate one‑line summary (greedy) for each image; no gradients.
- If training Stage‑A: teacher‑force over the sampled summary tokens to compute per‑candidate log‑prob sums.

### Stage B (GRPO – training focus)
- Prompt‑only (text): `input_ids [1, L_prompt]`.
- For each sampled reply `y_k` with token IDs:
  1) Teacher‑force once over `[prompt + y_k]` → `logits [1, L-1, V]` and sum `logp_k` over reply positions (length‑norm optional).
  2) Compute rewards `{r_k}` → `A_k = zscore(r_k)` (clipped).
  3) Loss: `L_k = - stop_grad(A_k) * logp_k` (+ optional KL to reference policy on reply positions).
  4) Backprop; gradient clip; optimizer step.

### Stage A (GRPO – conditional or joint)
- Conditional: vary one image’s line at a time; recompute Stage‑B reward; z‑score; optimize on that image’s summary tokens.
- Joint: sample `K_set` sets of summaries across all images; score each set via Stage‑B; z‑score across sets; optimize per‑set sum of summary log‑probs; optional set‑averaged KL.

---

## Multi‑GPU with PyTorch DDP (supported)

- When launched with multiple GPUs via `torch.distributed.run` (script handles this), the runner initializes PyTorch DDP and wraps the trainable policy.
- Dataset is manually sharded by index position: each rank iterates items where `idx % world_size == rank`. Do not use `DistributedSampler`.
- Routing:
  - Sampling/generation (Stage‑A lines and Stage‑B replies) are executed per‑rank with the plain policy (no gradients).
  - Teacher‑forcing log‑prob and optional KL forwards use the DDP‑wrapped policy; gradients are all‑reduced.
- Effective batch size: behaves like increasing TF/KL batch size by `world_size`; `K_B/K_A` still control sample diversity per rank.
- Launch via the script: set `GPU_DEVICES="0,1"` then `bash scripts/run_group_qc_rl.sh`.

---

## Selective unfreeze and per‑group LRs (src_new‑style)

- We mirror `src_new` behavior via `PhaseFreezeManager`:
  - Always unfreeze the aligner/merger bridge.
  - Optionally unfreeze only the last‑K LLM decoder layers and/or last‑K vision blocks.
  - Keep `visual.patch_embed` frozen by default for stability.
- YAML keys:
  ```yaml
  # Selective unfreeze
  llm_top_k_block: 2           # unfreeze last-K LLM layers
  vision_top_k_block: 1        # unfreeze last-K vision blocks
  freeze_patch_embed: true

  # Differential learning rates (three groups)
  aligner_lr: 1.5e-5
  llm_lr: 8.0e-6
  vision_lr: 4.0e-6
  ```
- Implementation: `src_post/grpo_runner.py` uses `PhaseFreezeManager` (phase_3) and builds three optimizer groups: `aligner`, `llm_topk`, `vision_topk`.

---

## Learning‑rate scheduler (cosine + warmup)

- Optional scheduler via HuggingFace `get_scheduler` with warmup ratio:
  ```yaml
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  ```
- Total steps per rank derive from `num_updates * batch_size` (no grad accumulation in this runner). If `num_updates` isn’t set, a minimal default is used.

---

## Rank‑synchronized logging, TensorBoard, and checkpointing

- At the end of each update, metrics are aggregated across ranks and only rank‑0 logs:
  - Printed fields: `loss`, `reward_best (mean±std)`, `acc_best`, `any_hit`, `resp_len`, `grad_norm`, `eta_hours`, and optional `kl_b`.
- JSONL:
  - Results: `results.rank{RANK}.jsonl` written by every rank.
  - Metrics: `metrics.rank0.jsonl` written only by rank‑0 (others stay empty).
- TensorBoard (optional): set `tb_log_dir` and `run_name` to enable rank‑0 TB scalars under `<tb_log_dir>/<run_name>/`.
- Saving: rank‑0 saves final checkpoint to `output_dir/checkpoints/grpo_final/` (set `SKIP_SAVE=1` env or `skip_save_checkpoints: true` to skip).

---

## Minimal config (excerpt)
```yaml
# Required paths
checkpoint: /abs/path/to/sft_checkpoint
processor:  /abs/path/to/processor
output_dir: /abs/path/to/output_post/grpo
train_data_dir: /abs/path/to/train_groups
mission: bbu安装方式检查

# Generation (short outputs; repetition control)
temperature: 0.5
top_p: 0.9
max_new_tokens_stage_a: 32
max_new_tokens_stage_b: 128
mask_geometry_tokens: false
mask_coordinate_tokens: false

# GRPO sampling
K_B: 3
K_A: 3
adv_clip: 1.5
length_norm: true

# Training loop
train: true
epochs: 1
batch_size: 1
learning_rate: 1.0e-5
weight_decay: 1.0e-3
max_grad_norm: 0.5

# Stage‑A training mode
train_stage_a_mode: conditional   # conditional | joint | off
stage_a_weight: 1.0
K_set: 1                          # used when train_stage_a_mode=joint
max_images_tf: 1                  # cap images per update for conditional Stage-A

# KL to reference policy (optional)
use_ref_kl: true
ref_checkpoint: /abs/path/to/sft_checkpoint
lambda_kl_stage_b: 0.02
lambda_kl_stage_a: 0.02

# Selective unfreeze + per‑group LRs
llm_top_k_block: 1
vision_top_k_block: 0
freeze_patch_embed: true
aligner_lr: 1.5e-5
llm_lr: 8.0e-6
vision_lr: 1.0e-6

# LR scheduler (optional)
lr_scheduler_type: cosine
warmup_ratio: 0.1

# Rewards (include penalties)
reward_fns: label_match,decision_prob,violations,coverage,formatting,cleanliness,rep_penalty,quote_penalty,special_penalty
reward_weights: 1.0,0.7,0.5,0.2,0.1,0.2,-0.7,-0.4,-1.0

# Logging & outputs
results_jsonl: /abs/path/output_post/results.jsonl
metrics_jsonl: /abs/path/output_post/metrics.jsonl
tb_log_dir: tb_post
run_name: debug

# Misc
device: cuda
seed: 17
limit_groups: -1
```
Notes:
- `reward_fns`/`reward_weights` must have the same length; unknown names will raise with a list of valid options.
- Optional `sanitize_stage_a` exists in config but is off by default and not applied during training; prefer reward shaping and logits masking.

---

## Key files
- `src_post/runner.py`: entry logic, dataset iteration, Stage‑A/B generation, reward composition, GRPO update loop (no value head).
- `src_post/prompting/conversation.py`: Stage‑A/B prompt builders and mission hints.
- `src_post/prompting/schema.py`: mission check/token parsing from `MISSION_CHECKS_COVERAGE` (cached).
- `src_post/rewards/lexicon.py`: shared canonical tokens (negatives/slots/forbidden words).
- `src_post/generation/generation.py`: generators and `build_decision_prefix_constraint` for strict Stage‑B headers.
- `src_post/data/dataset_group_qc.py`: group‑level dataset (directory or JSONL).
- `src_post/generation/logits_processors.py`: decode‑time masking for geometry/coordinate tokens.
- `src_post/rewards/`: modular rewards and penalties.
- `src_post/prompting/span_parser.py`: robust parsing of Stage‑B decision text.

---

## Differences from SFT in `src_new/`
- **Objective**: SFT minimizes CE to match reference text; RL here maximizes a composed task reward using GRPO (relative advantages across samples). No value head.
- **Where gradients flow**: Stage‑B reply tokens (always), and optionally Stage‑A summary tokens (conditional or joint). Vision stack largely frozen; LM head + last‑K text layers and last‑K vision blocks are trained.
- **Images**: Reuse the official `Qwen2VLProcessor` and the SFT‑style EXIF‑aware resize so vision token alignment stays correct.
- **Decoding constraints**: Prefer reward‑based shaping; optional runtime masking removes geometry/coord tokens during generation without affecting gradients.

---

## Troubleshooting
- **Repetition in Stage‑A summaries**: raise generation `repetition_penalty`/`no_repeat_ngram_size` or increase negative weight of `rep_penalty`.
- **Special tokens leak (`<|...|>`)**: increase negative weight of `special_penalty`; optionally enable decode‑time masking.
- **Quotes or long rambles**: penalize with `quote_penalty`; you may also shorten Stage‑A `max_new_tokens_stage_a`.
- **Advantage variance issues (std≈0)**: skip update (A=0) or increase `K_B`/`K_A` moderately.
- **Instability/drift**: increase KL weight(s) slightly; reduce learning rate; keep most layers frozen initially.

## New knobs (refactor additions)

- Stage‑B training control
  - `train_stage_b: bool` (freeze Stage‑B updates when false)
  - `stage_b_weight: float` (weight of Stage‑B loss)
  - `freeze_stage_b_steps: int` (freeze Stage‑B for the first N updates)
- Prompt bias toggle
  - `use_mission_checklist: bool` (include checklist when true; use minimal Stage‑B prompt when false)
- Group reward mode
  - `group_reward_mode: {margin_only|label_match|combined}`
    - `margin_only` uses TF log‑margin only via `group_margin`
    - `label_match` uses binary only
    - `combined` uses configured reward names/weights
- Stage‑A credit assignment
  - `train_stage_a_mode: {off|conditional|joint}`
  - `pairwise_credit_enabled: bool` (enable pairwise fallback)
  - `pairwise_pairs_per_group: int` (budget ≤ 1 by default)
  - `pairwise_delta_threshold: float` (trigger only if best single‑image Δ < threshold)
- Uncertainty gate (optional)
  - `use_uncertainty_gate: bool`
  - `uncertainty_gate_min_entropy: float` (per‑line entropy threshold)

### Examples

Warm‑up (Stage‑A only; minimal prompt; margin‑only):
```yaml
checkpoint: /abs/sft_ckpt
processor:  /abs/sft_processor
output_dir: /abs/out/grpo_warmup
train_data_dir: /abs/data/groups
mission: bbu安装方式检查

train_stage_b: false
stage_b_weight: 1.0
freeze_stage_b_steps: 1000
use_mission_checklist: false

group_reward_mode: margin_only
reward_fns: group_margin,formatting,cleanliness,rep_penalty,quote_penalty,special_penalty
reward_weights: 1.0,0.3,0.3,-0.7,-0.4,-1.0

train_stage_a_mode: conditional
pairwise_credit_enabled: true
pairwise_pairs_per_group: 1
pairwise_delta_threshold: 0.02

use_ref_kl: true
ref_checkpoint: /abs/sft_ckpt
lambda_kl_stage_b: 0.02
lambda_kl_stage_a: 0.02

K_B: 3
K_A: 3
max_new_tokens_stage_a: 48
max_new_tokens_stage_b: 128

device: cuda
epochs: 1
batch_size: 1
```

End‑to‑end (both stages; checklist; combined reward):
```yaml
checkpoint: /abs/sft_ckpt
processor:  /abs/sft_processor
output_dir: /abs/out/grpo_e2e
train_data_dir: /abs/data/groups
mission: bbu安装方式检查

train_stage_b: true
stage_b_weight: 1.0
freeze_stage_b_steps: 0
use_mission_checklist: true

group_reward_mode: combined
reward_fns: group_margin,label_match,formatting,cleanliness,rep_penalty
reward_weights: 1.0,0.3,0.3,0.2,-0.7

train_stage_a_mode: conditional
pairwise_credit_enabled: false
pairwise_pairs_per_group: 0
pairwise_delta_threshold: 0.00

use_ref_kl: true
ref_checkpoint: /abs/sft_ckpt
lambda_kl_stage_b: 0.02
lambda_kl_stage_a: 0.02

K_B: 3
K_A: 3
max_new_tokens_stage_a: 48
max_new_tokens_stage_b: 128

device: cuda
epochs: 1
batch_size: 1
```
