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
