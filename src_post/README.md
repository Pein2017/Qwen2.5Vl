## RL Post-Training for Group-Level QC (Qwen2.5‑VL, GRPO)

### What this module does
- **Goal**: Fine‑tune an SFT‑trained Qwen2.5‑VL to make a single “pass/fail + reason” decision for a group of images (a work order), with one short Chinese summary per image.
- **Two stages**:
  - **Stage A (per image)**: generate one concise, informative Chinese summary per image.
  - **Stage B (group aggregation)**: aggregate per‑image summaries and output: 总评: 通过/不通过 (+ 原因 when 不通过).
- **Post‑training method**: **GRPO (Group Relative Policy Optimization)**.
  - We sample multiple responses for the same prompt, compute group‑level rewards, convert them to relative advantages via z‑score, and push up the log‑probability of higher‑reward samples while pushing down lower‑reward ones.
  - Optimizes Stage‑B (final decision) and, optionally, Stage‑A (image summaries) using conditional GRPO so per‑image summaries learn subjective biases directly from group labels.

---

## Previous background: SFT stage in `src_new/`

- Purpose and result:
  - Fine‑tune Qwen2.5‑VL for BBU detection/captioning with optional coordinate‑token support.
  - Produces a base checkpoint and a matching `Qwen2VLProcessor` used as the starting point for RL post‑training.
- Core mechanics (high level):
  - Conversations built with the official processor’s `chat_template`; images are passed as typed content so `<|image_pad|>` tokens align with vision features.
  - Dynamic coordinate tokens discovered via `src_new.processing.special_tokens.get_coord_token_range`; model/tokenizer expansion handled in `src_new.models.wrapper.DetectionModel` before distributed setup.
  - Loss path is single‑pass CE over assistant spans; optional auxiliary coordinate losses (Kernelized‑KL, Unlikelihood) can be enabled in `src_new` but are not used during RL.
  - Qwen2.5‑VL compatibility patches are applied (attention/prepare_inputs fixes). EOS `<|im_end|>` is included during SFT to teach proper termination.
  - Alignment validated: expected `<|image_pad|>` count equals `sum_i(t_i*h_i*w_i)/(merge_size^2)` as computed from `image_grid_thw`.
- From SFT to RL (bridge):
  - SFT teaches the model to understand the domain and produce clean captions with proper EOS.
  - RL reuses the SFT checkpoint + processor to keep tokenizer vocab, vision merge size, and chat template consistent.

---

## How to run
```bash
# From repo root
conda activate ms
bash scripts/run_group_qc_rl.sh
```
- The script handles single/multi‑GPU and shards the dataset by rank.
- Default config path: `configs/rl/group_qc_grpo.yaml`.

Directory data layout (recommended):
```text
/abs/path/to/test_data/
  审核通过/
    <group_id>/image_*.jpg
  审核不通过/
    <group_id>/image_*.jpg
```
Labels are normalized: 审核通过|通过|pass → pass；审核不通过|不通过|fail → fail。

---

## Process workflow (component interactions)

1) Data loading (group level)
- File: `src_post/dataset_group_qc.py` (`RLGroupQCDataset`)
- Yields a group: `{ image_paths, images, label, meta, num_images }` from the directory layout above (or JSONL).

2) Stage A summarization (per image, typed chat)
- File: `src_post/conversation.py` → `GroupQCConversationBuilder.build_stage_a_messages(mission)` builds:
  - system: “只输出一行摘要；纯中文；不要坐标/特殊符号；不要使用引号；不要罗列重复清单；尽量单句短语” (+ optional mission hints)
  - user: typed list `[{'type': 'image'}, {'type': 'text', 'text': '请只输出一行摘要'}]`
- Tokenization: `Qwen2VLProcessor.apply_chat_template(...)` → `processor(text=[...], images=[img], return_tensors='pt')`
- Generation: keep one greedy line per image for Stage‑B context; optionally sample K_A variants during Stage‑A GRPO.

3) Stage B GRPO (group decision)
- File: `src_post/grpo_runner.py` → builds prompt via `build_stage_b_messages(summary_lines, checklist_lines)`.
- Sampling: sample K_B short answers; parse with `src_post/span_parser.py` to `{label, reason}`.
- Reward: composed via `src_post/rewards/` (e.g., label_match, formatting, cleanliness, plus negative penalties for repetition/quotes/special tokens leaking into summaries).
- GRPO update: teacher‑forcing on each sampled answer to compute log‑prob sums; use z‑scored rewards as advantages.

4) Stage A GRPO (optional, conditional)
- Conditional credit assignment:
  - For image i, fix other images’ summaries to their current best; sample K_A variants for image i.
  - Rebuild Stage‑B prompt with each candidate inserted; recompute group reward.
  - Convert per‑image rewards to per‑image advantages via z‑score; teacher‑force the summary tokens and apply GRPO loss.

5) Outputs
- JSONL per rank: `results.rank{RANK}.jsonl` with `{group_index, mission, gt_label, pred_label, reward, images[captions], stage_b_raw}`
- Checkpoints: `output_post/checkpoints/grpo/<tag>/` saving the policy (no value head needed).

---

## Tensor flow (forward and backward)

### Stage A (generation only unless training Stage‑A GRPO)
- Encode with typed image message:
  - `processor(..., images=[img])` → tensors: `input_ids [1,Lp]`, `pixel_values [B,T,H,W,C]` (internal), `image_grid_thw`.
- Generate one‑line summary; no gradients for context building.
- When training Stage‑A (conditional GRPO), do teacher‑forcing over the sampled summary tokens to compute log‑prob sums.

### Stage B (GRPO – training focus)
- Prompt only (text): `input_ids [1, L_prompt]`.
- Sample K_B responses; for each response with tokens `response_ids [L_resp]`:
  1) Build `full_ids = [prompt_ids + response_ids]` → shape `[1, L_total]`.
  2) Forward once (teacher‑forcing):
     - `logits [1, L_total-1, V]` (shifted next‑token prediction)
     - `targets = full_ids[:, 1:]`
     - Build response mask: `resp_mask[:, prompt_len-1 :] = True`
     - Compute per‑token `logprobs = log_softmax(logits)`; sum over response positions → `logp_k`.
  3) Compute rewards {`r_k`} and advantages:
     - `A_k = (r_k - mean(r)) / (std(r)+eps)`; clip to a small range (e.g., ±1.5).
  4) Loss (per response):
     - `L_k = - stop_grad(A_k) * logp_k` (optionally divide by response length for length‑norm)
  5) KL to reference (optional):
     - `KL(π || π_ref)` over response positions; add `λ_KL * KL` to the loss to prevent drift.
  6) Backprop; gradient clip; optimizer step (LM head + last N layers unfrozen).

### Stage A (conditional GRPO – training)
- For each image i, sample K_A summary candidates; for each candidate:
  1) Replace the i‑th line in summaries and evaluate Stage‑B reward → `r_i^k`.
  2) Compute `A_i^k = zscore(r_i^k)`; clip.
  3) Teacher‑force over the i‑th summary tokens to get `logp_i^k`; apply `- stop_grad(A_i^k) * logp_i^k` (length‑norm optional).
  4) Optional KL to reference on Stage‑A tokens.

---

## GRPO vs PPO (theory quick primer)

- **PPO** adds a value head to estimate returns per token and uses the ratio‑clipped objective to keep policy updates stable; great sample efficiency, higher engineering overhead (critic, masks, advantage estimation), and more sensitive to absolute reward scale/noise.
- **GRPO** discards the critic and uses relative advantages from multiple samples of the same prompt:
  - Sample `K` responses → rewards `{r_k}` → `A_k = zscore(r_k)`.
  - Optimize `-A_k * log_prob(y_k)` so higher‑reward samples get increased probability.
  - Works particularly well for short generations and noisy/subjective rewards (only rankings/relative differences matter).
- In our two‑stage setup:
  - Stage‑B: short decision text, ideal for GRPO.
  - Stage‑A: conditional GRPO gives image‑level credit without per‑image ground truth.

---

## Intuitive demo examples

### Example 1: Stage‑B GRPO with K_B=3
- Prompt built from 3 summaries → sample 3 decisions:
  - y1: “总评: 通过” → r1 = 0.2 (format looks OK but wrong label)
  - y2: “总评: 不通过\n原因: 挡风板缺失” → r2 = 1.0 (label matches, clean format)
  - y3: “总评: 不通过” → r3 = 0.8 (label matches, reason missing)
- Advantages (z‑score): A ≈ [-1.1, +0.8, +0.3]
- Loss: `L = -A1*logp(y1) - A2*logp(y2) - A3*logp(y3)` → pushes up y2,y3 and down y1.

### Example 2: Stage‑A conditional GRPO (per‑image, K_A=3)
- Fix other images to current best; for image i, sample 3 candidates:
  - s1: “BBU/华为；挡风板/已安装” → r_i^1 = 0.9
  - s2: “螺丝、螺丝、螺丝、螺丝...” → r_i^2 = 0.1 (repetition penalty)
  - s3: “BBU/华为；挡风板/无需” → r_i^3 = 0.7
- Advantages (z‑score): A_i ≈ [+0.8, -1.0, +0.2]
- Loss on summary tokens: `L_i = -A_i^1*logp(s1) - A_i^2*logp(s2) - A_i^3*logp(s3)`.

---

## Rewards (modular registry)

- File: `src_post/rewards/`, registry at `rewards/__init__.py`.
- Inputs to rewards: `{ gt_label, pred_label, summary_lines, stage_b_reason, checklist_lines }`.
- Typical composition (positive terms):
  - **label_match** (binary)
  - **formatting** (on‑domain vocabulary, basic structure)
  - **cleanliness** (no illegal markers)
  - **consistency**/**taxonomy** (optional domain rules)
- Negative penalties (use negative weights in config):
  - **rep_penalty**: penalize token repetition (e.g., ‘螺丝、螺丝、…’)
  - **quote_penalty**: penalize quotes around summaries
  - **special_penalty**: penalize `<|...|>` leakage into Stage‑A summaries

Config example (see `configs/rl/group_qc_grpo.yaml`):
```yaml
reward_fns: label_match,formatting,cleanliness,rep_penalty,quote_penalty,special_penalty
reward_weights: 1.0,0.5,0.7,-0.7,-0.4,-1.0
```

Notes:
- Prefer shaping via rewards over hard masking/sanitization during training so the model learns to produce clean outputs by itself.
- You can still enable decode‑time masking via `src_post/logits_processors.py` for evaluation.

---

## Multi‑GPU behavior and sharding
- The launcher selects single vs multi‑GPU automatically and uses `torchrun` when needed.
- Each rank processes `idx % world_size == rank` groups. Results are written to `results.rank{RANK}.jsonl`.

---

## Key files (where to look)
- `src_post/grpo_runner.py`: entry logic, dataset iteration, Stage‑A/B generation, reward composition, GRPO update loop (no value head).
- `src_post/conversation.py`: Stage‑A and Stage‑B prompt builders, mission rules and hints.
- `src_post/dataset_group_qc.py`: group‑level dataset (directory or JSONL).
- `src_post/logits_processors.py`: optional runtime masking for geometry/coordinate tokens (decode‑time only).
- `src_post/rewards/`: modular rewards and penalties.
- `src_post/span_parser.py`: robust parsing of Stage‑B decision text.

(Deprecated in this GRPO version: `ppo_runner.py`, `policy_vhead.py`.)

---

## Differences from SFT in `src_new/`
- **Objective**: SFT minimizes CE to match reference text; RL here maximizes a composed task reward using GRPO (relative advantages across samples). No value head.
- **Where gradients flow**: Stage‑B response tokens (always), and optionally Stage‑A summary tokens (conditional GRPO). Vision stack typically frozen; LM head + last N transformer layers trained.
- **Images**: We reuse the official `Qwen2VLProcessor` and SFT‑style resizing so the vision token alignment stays correct.
- **Decoding constraints**: Prefer reward‑based shaping; optional masking at decode time does not affect gradients.

---

## Troubleshooting
- Repetition in Stage‑A summaries:
  - Increase generation `repetition_penalty` / `no_repeat_ngram_size` (already higher by default), or raise the negative weight of `rep_penalty`.
- Special tokens leak (`<|...|>`):
  - Increase negative weight of `special_penalty`; optionally enable `logits_processors` masking during evaluation.
- Quotes or long rambles:
  - Penalize with `quote_penalty`; optionally cap Stage‑A summary length in preprocessing for logging only.
- Advantage variance issues (std≈0):
  - Skip update (A=0) or increase K (K_B/K_A) moderately.
- Instability / drift:
  - Increase KL weight(s) slightly; reduce learning rate; keep most layers frozen initially.

---

## Minimal config (excerpt)
```yaml
checkpoint: /abs/path/to/sft_checkpoint
processor:  /abs/path/to/processor
output_dir: /abs/path/to/output_post/grpo
train_data_dir: /abs/path/to/train_groups

# Generation (short outputs; repetition control)
temperature: 0.5
top_p: 0.9
max_new_tokens_stage_a: 64
max_new_tokens_stage_b: 160

# GRPO sampling
K_B: 4
K_A: 3
adv_clip: 1.5
length_norm: true

# Trainable parts
train_lm_head: true
train_last_n_layers: 1

# KL to reference
use_ref_kl: true
lambda_kl_stage_b: 0.02
lambda_kl_stage_a: 0.02

# Rewards (include penalties)
reward_fns: label_match,formatting,cleanliness,rep_penalty,quote_penalty,special_penalty
reward_weights: 1.0,0.5,0.7,-0.7,-0.4,-1.0
```

This README reflects the GRPO‑based implementation and should be used as the source of truth for training and evaluation of group‑level QC with Stage‑A shaping.
