# Implementation Plan: AI质检（BBU）多图组级决策 — Stage‑A 摘要 + Stage‑B 汇总（GRPO）

**Branch**: `005-group_decision` | **Date**: 2025-10-11 | **Spec**: `/data3/Qwen2.5-VL-main/specs/005-group_decision/spec.md`
**Input**: Feature specification from `/specs/005-group_decision/spec.md`

## Summary

在仅有工单级（多图）GT 的前提下，复用已完成 SFT 的 Qwen2.5‑VL 模型：
- Stage‑A：逐图生成一行中文摘要（不含坐标/特殊标记）。
- Stage‑B：基于全组摘要输出两行决策（通过/不通过 + 原因），并使用 GRPO 将组级奖励（以对数边际为主）回传到 Stage‑B 回答与各图摘要（条件式信用分配）。
- 默认：包含 mission checklist 提示；Stage‑B 采样 K_B=2, temperature=0.7, top_p=0.95；KL 参考为当前 SFT；Stage‑A 门控启用，熵阈=1.2；8×A100‑80G，Accelerate 多卡。

## Technical Context

**Language/Version**: Python 3.12（conda `ms` 环境）  
**Primary Dependencies**: PyTorch, Transformers (Qwen2.5‑VL), Accelerate, PIL, TensorBoard  
**Storage**: N/A（文件型 JSONL/ckpt/specs）  
**Testing**: pytest + ruff + pyright（按宪章启用）  
**Target Platform**: Linux（bf16，flash_attention_2 或 eager）  
**Project Type**: 单仓库（训练/推理/RL 共存）  
**Performance Goals**: 显存稳定、无 OOM；奖励方差>0；ETA/吞吐稳定；顺序化采样与前向保证可复现  
**Constraints**: bf16-only；严格 HF‑first 与 fail‑fast 校验；顺序化 generate/forward + grad accumulation  
**Scale/Scope**: 8×A100‑80G，DDP/Accelerate；K_B=2；K_A∈{2,3}；组规模~数万样本可扩展

## Constitution Check

| Principle / Rule | Status | Notes |
|------------------|--------|-------|
| Single Source of Truth & Explicit Contracts | PASS | 复用 `src_new/processing` 与 HF processor；禁止手工 `<|image_pad|>` |
| Fail-Fast Validation & Observability | PASS | 强制图文占位/THW 校验；日志+TB 指标齐备 |
| Configuration & Reproducibility | PASS | 严格 `RLRunnerConfig`；无隐式默认；固定种子 |
| Separation of Concerns | PASS | 数据/提示/奖励/训练分层；`src_post` 不侵入 `src_new` |
| Testing & TDD | PASS WITH ACTION | 增补最小单元/集成用例（见 tasks） |
| Simplicity & Minimal Surface Area | PASS | 默认开关有限；强约束留在解码期 |
| Decoupling & Reuse | PASS | 奖励/信用分配/提示构造为可复用组件 |
| Versioning (No Backward Compatibility Guarantees) | PASS | 变更仅影响 `src_post`；无破坏性接口 |
| Security & Compliance | PASS | 无外部秘密；对象/几何白名单遵循 |
| Documentation & Traceability | PASS | `spec/plan/tasks` 成套落地，提交即追踪 |
| Performance & Resource Stewardship | PASS | 8×A100；顺序化采样；显存监控与裁剪 |
| Sequential Processing & GPU Resource Constraints | PASS | 严格顺序化 generate/TF + `no_sync` 累积 |
| Code Review & CI Gates | PASS | PR 启用 lint/type/test 阶段性门禁 |
| Execution Environment & Tooling Rules | PASS | 仅 `ms` 环境；绝对路径；bf16 |
| Operational Workflow & Artifact Expectations | PASS | 本 plan 及产物按模板生成 |

## Project Structure

### Documentation (this feature)
```
specs/005-group_decision/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── rl_runner_config.md
│   ├── reward_registry.md
│   └── logging_outputs.md
└── tasks.md
```

### Source Code (repository root)
```
src_post/
├── runner.py                # 组级GRPO训练主循环
├── prompting/
│   ├── conversation.py      # Stage‑A/B 提示构造 + checklist
│   └── span_parser.py       # 决策两行解析
├── rewards/                 # 奖励注册与组合
├── tf/                      # Teacher-forcing与KL工具
├── generation/              # 采样与前缀/掩码
├── data/                    # 组级数据集与加载
├── models.py                # 模型加载/冻结/分组学习率
└── config.py                # RLRunnerConfig 校验

scripts/
├── run_group_qc_rl.sh       # 多卡/Accelerate 启动脚本
└── ...
```

**Structure Decision**: 采用现有 `src_post` 分层；新增/更新严格限制在提示/奖励/训练与配置校验层，不改动 `src_new`。

## Phase 0 — Research
- 对比最小提示 vs checklist 提示对 `group_margin` 与解析稳定性的影响；
- 评估 K_B=2 的方差与收敛（对比 K_B=3 的算力性价比）；
- 记录失败模式：std≈0、超长、特殊标记泄露、FN率。

## Phase 1 — Design / Data Model & Contracts
- 数据模型：Group/Image/SummaryLine/StageBDecision/Reward 定义与字段约束；
- 合同：RLRunnerConfig 必填项与默认值；奖励注册默认组合；日志/指标产物契约；
- Quickstart：8×A100 + Accelerate 启动指引；YAML 最小示例键集合。

## Phase 2 — Tasks Generation
- 依据用户故事与宪章门禁，生成可并行的任务清单（tests 优先）。

## Execution Notes
- 顺序化 generate/TF 与 `no_sync` 累积；
- DDP/Accelerate：rank 切分、独立采样、全归约 TF/KL；
- 默认门控与 KL/采样参数写入 YAML；
- 所有失败回退（std≈0、路径/清单缺失、解析失败）均 fail‑fast 或计数并显式日志。

## Progress Tracking
- Phase 0: DONE（research.md 已生成）
- Phase 1: DONE（data-model.md / contracts/* / quickstart.md 已生成）
- Phase 2: DONE（tasks.md 已生成）
