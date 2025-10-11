# Tasks — 005-group_decision

Feature: AI质检（BBU）多图组级决策 — Stage‑A 摘要 + Stage‑B 汇总（GRPO）
Root: /data3/Qwen2.5-VL-main

Note: 所有测试脚本需放置在 `./tests/group_qc` 目录下。

## Dependency‑Ordered Task List (numbered)

- T003 代码审查: 关键路径核对（不改架构，仅核对）
  - 文件: `src_post/runner.py, src_post/config.py, src_post/data/dataset_group_qc.py, src_post/prompting/conversation.py, src_post/prompting/span_parser.py`
  - 核对点: HF‑first、占位/THW 校验、metrics 聚合是否包含 accuracy/FN 相关字段、KL/采样/门控键是否可用。

- T010 集成冒烟: 现有实现最小闭环（单机单卡）
  - YAML: 使用你的 SFT ckpt 和小样本（≤10组）；`K_B=2, temperature=0.7, top_p=0.95, use_mission_checklist=true`
  - 命令: `python -m src_post.runner --config /abs/config.yaml`
  - 验证: 生成 `results.rank*.jsonl` 与 `metrics.rank0.jsonl`；日志无异常；`reward_std>0`。

- T020 G2 实现: 组目录读取与标签归一（增量实现，保持现有结构） [X]
  - 文件: `src_post/data/dataset_group_qc.py::RLGroupQCDataset`
  - 功能: 支持目录结构 `<root>/<mission>/{审核通过|审核不通过}/<group_id>/*.{jpg,jpeg,png}` 遍历；
    - mission=第二级目录；label 归一: {"审核通过","通过","pass"}→"pass"；{"审核不通过","不通过","fail"}→"fail"；
    - group_id=第三级目录名；组内多图自然序读取。
  - 不修改 API；保留 JSONL 输入兼容。

- T021 G2 单测: 目录/标签读取与异常场景 [X]
  - 路径: `tests/group_qc/test_dataset_group_qc.py`
  - 覆盖: 最小目录树（各1组）；空目录/未知标签名/不可读图片→fail‑fast；mission 取自二级目录。

- T030 A1/G4 文档与默认策略更新（最小侵入） [X]
  - 文档: `/specs/005-group_decision/quickstart.md` 与 `/specs/005-group_decision/contracts/rl_runner_config.md`
  - 内容: 默认 `enable_clipped_grpo=false, enable_entropy_mask_stage_b=false`；
    - 若启用，示例: `epsilon_low=0.2, epsilon_high=0.0`；熵掩码二选一：`entropy_top_quantile_stage_b=0.2` 或 `entropy_min_threshold_stage_b=1.5`；
    - 解码屏蔽: Stage‑A `mask_geometry_tokens=true, mask_coordinate_tokens=true`；Stage‑B 两项均 false。

- T031 A1/G4 验证: 开关影响对比（小样本） [P]
  - 运行两组 YAML：默认关闭 vs 开启裁剪/熵掩码；
  - 采集: `loss`/`reward_std`/`sb_clip_*`/`sb_entropy_mask_ratio`；确认默认关闭下方差稳定、无退化。

- T040 C1 单测: fail‑fast 校验 [X]
  - 路径: `tests/group_qc/test_failfast.py`
  - 覆盖: 路径不存在/图片不可读；未知 mission；未知标签目录名；
    - JSONL 行缺少 images/label → fail‑fast。

- T050 A3 指标输出: accuracy + FN_rate 显式化 [X]
  - 文件: `src_post/runner.py`（已存在导出；本任务提供写出校验用例）
  - 要求: rank‑0 `metrics.rank0.jsonl` 中包含 `accuracy` 与 `fn_rate` 字段；TB 同步记录。
  - 验证: `tests/group_qc/test_metrics_logging.py` 断言字段存在且为数值。

- T060 Debug & 收敛性核查（48–200步短跑）
  - 运行: 8 张组样本，200 步；记录 `decision_ce_ema` 下降趋势、`accuracy` 上扬或不跌、`reward_std>0`。
  - 若发散: 小幅提高 KL；若 FN 偏高: 保持 KL 并尝试 pairwise 回退 1 对。

- T070 文档微调与交付核对 [X]
  - 更新 `/specs/005-group_decision/spec.md` 的 Success Criteria：保留主KPI=accuracy、辅KPI=group_margin，并加入 `FN_rate` 作为次级监控（不高于基线或下降）。
  - 快速核对 `/specs/005-group_decision/contracts/logging_outputs.md` 是否包含新增指标。

## Parallel Guidance
- 可并行 [P]: T021/T030/T031/T040/T070（不同文件/文档/测试；互不冲突）。
- 顺序依赖: T003→T010→T020→T021；T030→T031；T050 在 T010 后执行；T060 最后。

## Optional Test Pruning

- 为缩短调参迭代周期，可临时跳过以下较重或重复覆盖的测试（提交前再恢复）：
  - 复杂组合奖励路径的端到端对比（若日志+TB 指标已覆盖）。
  - 与 `src_new` 无关的边缘用例重复校验（已由 fail‑fast 捕获的场景）。

## Example Commands
- 单卡冒烟: `python -m src_post.runner --config /abs/config.yaml`
- 8卡 Accelerate: `accelerate launch --num_processes 8 --mixed_precision bf16 /root/miniconda3/envs/ms/bin/python -m src_post.runner --config /abs/config.yaml`
- 指标校验脚本（示意）:
  - `python - <<'PY'
import json,sys; p=sys.argv[1]; d=json.load(open(p)); assert 'accuracy' in d and 'fn_rate' in d
PY metrics.rank0.jsonl`
