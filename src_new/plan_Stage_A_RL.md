## Stage‑A One‑Line Summary: SFT + RL Plan (free‑length)

### Goal
- Produce a single Chinese sentence per image that summarizes the key QC attributes (no coordinates, no special tokens). Length is not hard‑constrained (free‑length), but it must remain one line.
- Base the content on the hierarchical mapping (ignore occlusion attributes). Use the one‑line as Stage‑A context to stabilize Stage‑B GRPO.

---

## Checklist (from hierarchical_attribute_mapping.json; ignore occlusion)
The sentence should, when relevant, mention:
- BBU设备（bbu）
  - 品牌（brand）: {华为, 中兴, 爱立信}
  - 挡风板需求（windshield_requirement）: {无需安装, 机柜空间充足需要安装}
  - 挡风板配置符合性（windshield_conformity，当且仅当需要安装时）: {按要求配备, 未按要求配备}
- 挡风板（bbu_shield）
  - 品牌（brand）: {华为, 中兴}
  - 安装方向（direction）: {安装方向正确, 安装方向错误}
- 螺丝、光纤插头（connect_point）
  - 合规性（compliance）: {符合要求, 不符合要求}
  - 具体问题（specific_issues，当“不符合要求”时）: {未拧紧, 露铜, 复接, 生锈}
- 光纤（fiber）
  - 保护措施（protection）: {无保护措施, 有保护措施 (蛇形管/铠装/同时)}
  - 弯曲半径（bend_radius）: {弯曲半径合理, 弯曲半径不合理（<4cm或成环）}
- 电线（wire）
  - 整齐度（organization）: {捆扎整齐, 分布散乱}
- 标签（label）
  - 可读性/内容（text_content）: 概述“标签文字清晰/可读/不可读”（无需坐标）

Guidance for the one‑line:
- 一行中文自然语言短句（不限长度，但保持一行）。
- 不输出任何坐标或 `<|...|>` 特殊标记；不使用列表/编号；避免冗长解释和“总评/原因”字样。
- 关键词聚焦：品牌/挡风板（是否需要、是否按需配备、方向）/螺丝与插头合规与具体问题/光纤保护与弯曲/电线整齐/标签可读性。
- 优先覆盖影响质量的关键信息（例如：挡风板缺失或方向错误、螺丝不合规、弯曲不合理、线缆散乱、标签不可读）。

---

## Conversation #4: One‑Line Summary (image → one sentence)

### Minimal code additions in `src_new`
1) `src_new/processing/templates.py` (add a new constant; do not change existing ones)
- Add a dedicated system prompt constant (no coordinates; ignore occlusion):
```python
SUMMARY_ONE_LINE_SYSTEM_PROMPT = (
    "你是通信机房质检助手。请根据图像只输出一行中文自然语言摘要，"
    "不得包含坐标或任何 <|...|> 特殊标记，不要编号/列表/引号/英文，"
    "仅限 BBU 场景相关要点：品牌、挡风板需求与方向（如需）、螺丝/插头合规与具体问题、"
    "光纤是否有保护及弯曲半径、电线整齐度、标签可读性。"
)
```
- (可选) 在 `CONSTANTS` 中加入别名：`"SUMMARY_ONE_LINE_SYSTEM_PROMPT": SUMMARY_ONE_LINE_SYSTEM_PROMPT`

2) `src_new/processing/conversation_processor.py`（新增一个 HF‑first builder）
- Add a new public method that mirrors existing builders:
```python
from typing import Any, Dict, List, Optional
from .templates import SUMMARY_ONE_LINE_SYSTEM_PROMPT

# inside ConversationProcessor
def build_summary_one_line_messages(self, mission: Optional[str]) -> List[Dict[str, Any]]:
    # 这里可按需拼接 mission hints（不含遮挡），也可保持通用
    sys_text = SUMMARY_ONE_LINE_SYSTEM_PROMPT
    user_text = (
        "现在给你一张图像，请只输出一行摘要（不得包含 < 或 > 或任意 <|...|> 标记）：<image>"
    )
    return [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_text},
    ]
```
- Tokenization remains via `apply_chat_template(...)`; no change to validators or variants is required.

3) No changes required in `src_new/processing/variants.py`/`templates.get_system_prompt()`.
- The summary builder is a direct system+user conversation; no object variant handler is needed.

---

## Training flows

### A) SFT (optional but recommended)
- Data: image → one‑line targets（若无标注，可先伪标签）
  - 伪标签策略：
    - 用当前模型按新 builder 生成 K 个候选，清洗（去 `<|...|>`、坐标、英文数字、重复），
      选取包含任一 checklist 关键词的最短干净句；或从已有密集描述中按关键词提炼压缩为一行。
- Setup（src_new）
  - Builder：`build_summary_one_line_messages(mission)`
  - 冻结视觉；仅微调 LLM last‑K（K=1–2）；`llm_lr ~ 5e-6`；epochs 1–2；batch_size 1–2
  - 保存 checkpoint + processor，用作 RL 初始 policy/reference（ref_kl 可相同）

### B) RL (GRPO) as Stage‑A context
- Stage‑A 生成：
  - 使用新 builder 输出一行摘要；不做强长度限制（自由长度），但设置合理 `max_new_tokens_stage_a: 64` 以防失控。
  - `mask_geometry_tokens: true, mask_coordinate_tokens: true`；`sanitize_stage_a: true`（去特殊标记与坐标、合并重复、清理英文数字）。
  - 去重复：`repetition_penalty` 适中（比如 1.5–1.8），`no_repeat_ngram_size: 24–40`。
- Stage‑A GRPO：
  - `train_stage_a_mode: conditional`；`K_A: 2–3`；`max_images_tf: 1`；`stage_a_weight: 1.0`
  - KL to reference：`lambda_kl_stage_a: 0.01–0.02`
- Stage‑B GRPO（不变）：
  - `K_B: 3–4`；`lambda_kl_stage_b: 0.01–0.02`

### Rewards（先稳风格后提升判定）
- 初期建议（示例）：
  - `formatting:0.3, cleanliness:0.3, coverage:0.2, decision_prob:0.1, rep_penalty:-0.8, quote_penalty:-0.5, special_penalty:-1.0`
- 风格稳定（1–2k steps）后提高 `decision_prob` 权重。

---

## Decoding & Diagnostics
- Decoding：自由长度的一行摘要；屏蔽坐标/几何；适度去重复；`max_new_tokens_stage_a: 64`（仅上限，不是硬性剪短）。
- Diagnostics：开启 `enable_phase_a_diagnostics: true`，追踪 `formatting/coverage/cleanliness/taxonomy/consistency ∈ [0,1]`。

---

## Configuration keys（excerpt）
```yaml
# Stage‑A one‑line
train_stage_a_mode: conditional
K_A: 3
stage_a_weight: 1.0
max_images_tf: 1
max_new_tokens_stage_a: 64
sanitize_stage_a: true
mask_geometry_tokens: true
mask_coordinate_tokens: true

# Diagnostics
enable_phase_a_diagnostics: true

# Rewards (example)
reward_fns: formatting,cleanliness,coverage,decision_prob,rep_penalty,quote_penalty,special_penalty
reward_weights: 0.3,0.3,0.2,0.1,-0.8,-0.5,-1.0

# KL
use_ref_kl: true
lambda_kl_stage_a: 0.02
lambda_kl_stage_b: 0.02
```

---

## DDP & Performance Notes
- Reductions：metrics 向量 `all_reduce(SUM)`；wall‑time `all_reduce(MAX)`；全局进度 `all_reduce(SUM)`。
- Mixed precision：BF16 优先。
- Param groups：`aligner + llm_topk (+ vision_topk 可选)`；大多情况下视觉保持冻结；LLM top‑K=1–2。
- 保证 `logits_processor` 在 Stage‑A/B 生成时启用（屏蔽几何/坐标）。

---

## Implementation tasks（minimal）
1) Add `SUMMARY_ONE_LINE_SYSTEM_PROMPT` to `src_new/processing/templates.py`（不涉及坐标与遮挡）。
2) Add `build_summary_one_line_messages(mission)` to `src_new/processing/conversation_processor.py`（system+user，`<image>` + “请只输出一行摘要”）。
3) （可选）SFT：用该 builder 做小规模风格微调；保存 checkpoint 作为 RL 初始 policy。
4) RL：将 Stage‑A context 切换为该 builder；开启 sanitize/掩码；设置自由长度上限（如 64）。
5) 打开 diagnostics 并观察 `formatting/coverage` 走高；再逐步提高 `decision_prob` 权重。
