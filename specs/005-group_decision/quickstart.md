# Quickstart — Group QC GRPO

## Prereqs
- 激活环境：`conda activate ms`
- 硬件：8×A100‑80G；bf16；CUDA 可用
- 检查点与处理器：SFT 模型目录（含 tokenizer/processor）

## Minimal YAML (关键键)
- `checkpoint, processor, output_dir, train_data_dir, mission`
- `K_B=2, temperature=0.7, top_p=0.95`
- `use_mission_checklist=true`
- `train_stage_a_mode=conditional, stage_a_weight=1.0`
- `use_ref_kl=true, ref_checkpoint=<sft_dir>, lambda_kl_stage_b=0.02, lambda_kl_stage_a=0.02`
- `use_uncertainty_gate=true, uncertainty_gate_min_entropy=1.2`
- 并行：8卡，Accelerate/或 DDP（WORLD_SIZE=8）

## Defaults（建议，不改变现有代码）
- Stage‑B 裁剪/熵掩码：默认关闭
  - `enable_clipped_grpo: false`
  - `enable_entropy_mask_stage_b: false`
- 解码屏蔽（仅影响生成）
  - Stage‑A：`mask_geometry_tokens: true`, `mask_coordinate_tokens: true`
  - Stage‑B：`mask_geometry_tokens: false`, `mask_coordinate_tokens: false`

## Tuning Tips — Diagnostics‑Driven

- 若 `reward_best_std≈0` 且候选文本几乎一致：
  - 提升 `temperature`/`top_p` 或增大 `K_B`（已达上限则考虑奖励塑形对差异更敏感）。
  - 在奖励侧对“标签/无法识别/不清楚”结合上下文施加轻惩，避免“一票通过”。
- 若 Stage‑A 候选重复、`phase_a_entropy_mean` 低：
  - 设置 `no_repeat_ngram_size_stage_a: 12`，必要时将 `K_A: 4`。
- 类别不均衡：
  - 开启 `balance_pass_fail: true`，保证每步 pass/fail 均衡采样。

### Diversity & Credit — Quick Config
- 完整 YAML 请见 `spec.md` 的 Focus 章节（单一事实源）。
  - Stage‑A：提高温度/去重、K_A=4、放宽熵门控；
  - Stage‑B：轻度温度+去重、允许少量重采样；
  - 奖励日程：前 200–500 步 `margin_only`，后切回 `combined`；KL≈0.02；冻结 B≈300 步。

观察指标（建议阈，详见 spec.md）：
- `stage_b` 采样 `std>0` 占比 ≥ 40%
- `phase_a_entropy_mean ≥ 0.10`
- `best_single_delta>0` 样本占比 ≥ 30%

## Run (Accelerate)
```
accelerate launch --num_processes 8 --mixed_precision bf16 \
  /root/miniconda3/envs/ms/bin/python -m src_post.runner --config /abs/path/config.yaml
```

## Outputs
- `results.rank{R}.jsonl`、`metrics.rank0.jsonl`、TB：`<tb_dir>/<run_name>/`
- 最终检查点：`output_dir/checkpoints/grpo_final/`

## Notes
- 顺序化采样与前向；`no_sync` + 累积；
- 解析失败/路径缺失/方差为0均会记录或 fail‑fast；
- 若 FN 偏高，提升 KL 或启用 pairwise 回退。

## Next Steps（实践建议）
- Stage‑B 提示与多样化
  - 提示中新增：
    - “鼓励给出不同但合理的简洁理由，避免重复用语/模板化句式。”
    - “若关键项不可确认（标签缺失/只显示部分/安装方向不明），请输出不通过，并给出简要原因。”
  - YAML 侧增量：
    - `temperature_stage_b: 1.0~1.2`, `top_p_stage_b: 0.97~0.98`, `K_B: 3~4`
    - `no_repeat_ngram_size_stage_b: 12~16`, `repetition_penalty_stage_b: 1.05~1.10`, `max_resample_times: 2~3`
- Stage‑A 摘要的负向线索强化
  - 在系统/用户提示中强调：明确描述“无法确认/只显示部分/标签缺失/安装方向可疑”等；
  - YAML：`no_repeat_ngram_size_stage_a: 12`，必要时 `K_A: 4`。
- 奖励组合（降低 FN）
  - 建议 `group_reward_mode: combined`，并引入：
    - `label_match: +0.3~+0.5`，`neg_alignment: +0.2~+0.4`
    - 保持 `coverage≈0.7`，`soft_lexicon/special_penalty/rep_penalty/quote_penalty` 维持轻惩方向；
  - 不确定门：`uncertainty_gate_min_entropy: 1.5`；配对回退：`pairwise_pairs_per_group: 2`, `pairwise_delta_threshold: 0.15`。
