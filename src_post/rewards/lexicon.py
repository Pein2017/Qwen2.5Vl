#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Set

# Canonical negative tokens (align with SFT summary templates)
NEGATIVE_TOKENS: Set[str] = set([
    # Connection issues
    "未拧紧",
    "露铜",
    "复接",
    "生锈",
    "不符合要求",
    # Fiber / wire
    "无保护措施",
    "弯曲半径不合理",
    "弯曲半径<4cm或者成环",
    "分布散乱",
    # Shield install direction
    "安装方向错误",
    # Label readability (negative case only)
    "标签/无法识别",
])

# Forbidden decision words for Stage‑B strict format
FORBIDDEN_DECISION_WORDS: Set[str] = set([
    "需要进一步确认",
    "可以",
    "可通过",
    "待确认",
    "继续审查",
    "合格",
    "不合格",
])

# Canonical slot tokens (fallback coverage slots)
CANONICAL_SLOTS: Dict[str, Set[str]] = {
    # Shield requirement & conformity
    "shield_requirement": {"机柜空间充足需要安装", "无需安装"},
    "shield_conformity": {"这个BBU设备按要求配备了挡风板", "这个BBU设备未按要求配备挡风板"},
    # Installation direction for shields
    "install_direction": {"安装方向正确", "安装方向错误"},
    # Connection points
    "connect_issues": {"未拧紧", "露铜", "复接", "生锈", "不符合要求"},
    # Fibers
    "fiber_protection": {"无保护措施", "有保护措施", "蛇形管", "铠装", "同时有蛇形管和铠装"},
    "fiber_bend": {"弯曲半径合理", "弯曲半径不合理", "弯曲半径<4cm或者成环"},
    # Wires
    "wire_org": {"捆扎整齐", "分布散乱"},
    # Labels (only negative is emitted in summaries)
    "label_unreadable": {"标签/无法识别"},
}
