## SFT Fourth Variant Plan — Image → One‑Line Summary (Integrated in SFT)

### 目标
- 在 SFT 阶段新增“第4种对话变体”：输入图片 → 输出一行中文摘要（不含坐标/特殊标记/英文长串）。
- 训练素材直接来源于现有 SFT 样本的 `objects[].desc`，用固定规则抽取摘要作为 Teacher Forcing 的 GT。
- 与现有三变体（dense_caption/coords_to_desc/desc_to_coords）混合训练，降低 RL 前的分布落差，稳定 Stage‑A。

— 已删除旧的“小规模指令微调（200–500条）”等与本方案冲突的内容 —

### 一、摘要对话变体（第4种）
- 名称（建议）：`summary`（内部使用）
- 约束：
  - 一行中文短语；不得包含 `<|...|>`、`<`、`>`、`[`、`]`、坐标或长串英数；避免引号与清单重复。
  - 只在 BBU 场景范围内（BBU设备、挡风板、螺丝/光纤插头、标签、光纤、电线）。
  - 标签仅判断“清晰/不清晰”，不抄写OCR文字。
  - 可在末尾简短加入 `extra_info`（如“无法判断/已整改/空间有限/螺丝不全/部分未套蛇形管”等），但必须简洁。
- 内容优先级（从重到轻）：
  1) 螺丝/插头合规异常：未拧紧/露铜/复接/生锈/螺丝不全（任一出现即强违规）
  2) 光纤弯曲异常：弯曲半径不合理（<4cm 或成环）
  3) 电线整齐度异常：分布散乱
  4) 挡风板要求未满足或方向错误：需安装但未配；或安装方向错误
  5) BBU 完整性：只显示部分
  6) 标签可读性：清晰/不清晰（不抄文字）
  7) extra_info（可选）
- 正常/无异常兜底：输出“关键项正常，光纤弯曲合理，电线捆扎整齐，标签清晰”。

### 二、固定的摘要提取器（从 `objects[].desc` 生成一行摘要）
- 输入：单张图片的 `objects: List[Object]`（每对象仅一种几何 + `desc`）。
- 输出：一行中文摘要字符串。
- 规则（伪代码/参考实现）：
```python
from typing import Any, Dict, List

def extract_summary_from_objects(objects: List[Dict[str, Any]]) -> str:
    bad_screw_issues: List[str] = []
    bad_fiber: bool = False
    bad_wire: bool = False
    shield_needed_bad: bool = False
    shield_dir_bad: bool = False
    bbu_partial: bool = False
    label_clear: bool | None = None
    extra: List[str] = []

    def tail_remark(desc: str) -> str:
        # 取末尾“/备注文本”（如存在且非层级关键词）
        if "/" in desc:
            last = desc.rsplit("/", 1)[-1].strip()
            if last and last not in (
                "这个BBU设备按要求配备了挡风板",
                "这个BBU设备未按要求配备挡风板",
                "蛇形管", "铠装", "同时有蛇形管和铠装",
            ):
                return last
        return ""

    for o in objects:
        desc = str(o.get("desc", ""))
        if desc.startswith("螺丝、光纤插头/"):
            if "不符合要求" in desc:
                if "未拧紧" in desc: bad_screw_issues.append("未拧紧")
                if "露铜" in desc:   bad_screw_issues.append("露铜")
                if "复接" in desc:   bad_screw_issues.append("复接")
                if "生锈" in desc:   bad_screw_issues.append("生锈")
            if "螺丝不全" in desc or "固定螺丝缺少" in desc:
                bad_screw_issues.append("螺丝不全")
            tr = tail_remark(desc)
            if any(k in tr for k in ("无法判断", "已整改", "空间", "未套", "螺丝")):
                extra.append(tr)

        elif desc.startswith("光纤/"):
            if "弯曲半径不合理" in desc:
                bad_fiber = True
            tr = tail_remark(desc)
            if any(k in tr for k in ("无法判断", "已整改", "部分未套")):
                extra.append(tr)

        elif desc.startswith("电线/"):
            if "分布散乱" in desc:
                bad_wire = True

        elif desc.startswith("BBU设备/"):
            if "只显示部分" in desc:
                bbu_partial = True
            if "机柜空间充足需要安装" in desc and "按要求配备" not in desc:
                shield_needed_bad = True

        elif desc.startswith("挡风板/"):
            if "安装方向错误" in desc:
                shield_dir_bad = True
            if "固定螺丝缺少" in desc:
                bad_screw_issues.append("螺丝不全")

        elif desc.startswith("标签/"):
            text = desc.replace("标签/", "").strip()
            label_clear = bool(text)

    parts: List[str] = []
    if bad_screw_issues:
        parts.append("、".join(sorted(set(bad_screw_issues))))
    if bad_fiber:
        parts.append("光纤弯曲半径不合理")
    if bad_wire:
        parts.append("电线分布散乱")
    if shield_needed_bad:
        parts.append("需安装挡风板未按要求配备")
    if shield_dir_bad:
        parts.append("挡风板安装方向错误")
    if bbu_partial:
        parts.append("BBU只显示部分")

    if label_clear is not None:
        parts.append("标签清晰" if label_clear else "标签不清晰")

    if not parts:
        parts = ["关键项正常", "光纤弯曲合理", "电线捆扎整齐", "标签清晰"]

    # 附加最短的一个备注（控长）
    if extra:
        parts.append(min(extra, key=len))

    # 连接与清理（控长、去特殊符号）
    summary = "，".join(parts)
    summary = summary.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
    return summary[:40]
```

### 三、摘要变体的固定 Prompts（新增常量）
- 在与 `DENSE_USER_PROMPT / COORD_TO_DESC_USER_PROMPT / DESC_TO_COORD_USER_PROMPT` 同级的位置，新增（命名建议）：
```python
SUMMARY_SYSTEM_PROMPT = (
    "你是通信机房质检助手。请根据图片仅用中文输出简洁的一行摘要，"
    "禁止任何坐标或几何标记；禁止出现任意 <|...|> 特殊标记、< >、[ ] 或坐标数字。"
    "不要使用引号，不要逐字重复，不要清单式罗列。只在 BBU 场景范围内描述。"
    "标签仅判断‘清晰/不清晰’，不要抄写文字。若有影响判定的特殊情况（extra_info），可在末尾简要说明。"
)

SUMMARY_USER_PROMPT = "现在给你一张图像，请只输出一行摘要（仅中文自然语言短语，不要出现 < 或 > 或任意 <|...|> 标记）：<image>"
```
- 训练时该变体的 assistant 目标即为上文提取器生成的 `summary`。

### 四、数据构造与文件格式
- 产物：`data/summary_sft/train.jsonl`、`data/summary_sft/val.jsonl`
- 每行（单图）示例：
```json
{"images": ["/abs/path/to/img.jpg"], "summary": "露铜，光纤弯曲半径不合理，电线分布散乱", "meta": {"source": "ds_v2_full", "id": "..."}}
```
- 多图样本：建议按图逐条抽取各自摘要，分别训练（避免组内依赖进入 SFT）。

### 五、训练融合（不改RL；SFT中混合四变体）
- 变体占比（起点建议，可在训练后期退火）：
```yaml
conversation_variant_ratios:
  dense_caption: 0.45
  coords_to_desc: 0.20
  desc_to_coords: 0.20
  summary: 0.15
```
- 其余 SFT 配置（学习率/解冻/损失等）保持原样。若启用坐标令牌，请确保 `coordinate_tokens_enabled: true` 且 `max_coord_value` 正确。

### 六、质量门槛（SFT 验证集）
- 符号泄漏率（`<|...|>`, `<`, `>`, `[`, `]`）：0。
- 中文占比 > 95%；长度 10–40 字；无清单式重复；不含 OCR 文本内容。
- 命中关键槽位（未拧紧/露铜/复接/生锈/弯曲不合理/分布散乱/挡风板未配或方向错/标签清晰度）按分布抽查。

### 七、与 RL 的衔接（参考）
- 在 SFT 收敛后再进入 RL（先 Stage‑B，后开启 Stage‑A conditional）。
- RL 仍沿用现有 mission‑only 判定与解码遮罩；因 SFT 已学到“摘要样式”，RL 只需做偏好对齐与尺度学习，收敛更快。

### 八、里程碑
- D1：跑摘要提取脚本，生成 `summary_sft/*.jsonl`；抽检清洗。
- D1–D2：四变体混合 SFT 训练；过门槛后保存 checkpoint。
- D3：切入 RL（Stage‑B→Stage‑A conditional），观察奖励方差与格式指标，微调权重与采样参数。
