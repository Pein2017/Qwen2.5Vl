# Tasks — 005-group_decision

## Story P1 — 端到端组级判定
- [T] 实现最小可运行 YAML（含默认 K_B/KL/checklist/门控）
- [T] 结果/指标 JSONL 与 TB 写出；ETA/显存/方差日志
- [T] 加入 fail‑fast：路径/mission/奖励名/图文占位/THW 检查
- [Test] 集成：小样本端到端运行（CPU mock 或 1 组 GPU）

## Story P2 — 条件式信用分配（Stage‑A）
- [T] 逐图候选替换与 z‑score 优势回传
- [T] 不确定性门控（熵阈=1.2）与 `stage_a_top_m`
- [T] 配对回退触发与日志（Δ 阈）
- [Test] 单元：优势裁剪/门控/回退路径覆盖

## Story P3 — 决策一致性与最小偏置
- [T] checklist 与最小提示双路支持，默认 checklist
- [T] 决策前缀约束与解析健壮性（容错空行/空格）
- [Test] 解析器单元测试（正负样例 + 中文标点）

## Infrastructure / Constitution
- [T] Accelerate/或 DDP 启动脚本：8×A100 bf16；顺序化 generate/TF
- [T] `no_sync` + 累积 + grad clip；学习率调度（余弦+warmup）
- [T] RLRunnerConfig 校验与默认项落地；错误消息可读化
- [Test] ruff+pyright 通过；pytest 覆盖关键路径

## Milestones
- M1：单机单卡跑通（P1最小闭环）
- M2：8卡稳定训练（加速与指标齐备）
- M3：验证集 KPI 达成（acc↑3% 或 margin↑0.2）
