# Contract — Reward Registry (src_post)

## Registered Names (subset)
- group_margin
- decision_strict
- label_match
- coverage
- taxonomy
- consistency
- cleanliness
- rep_penalty
- quote_penalty
- special_penalty

## Combination (spec default)
- names: `group_margin, decision_strict, cleanliness, special_penalty, rep_penalty, quote_penalty, coverage, taxonomy, consistency`
- weights (suggested): `1.0, 3.0, 0.2, -0.2, -0.7, -0.4, 0.8, 0.2, 0.3`

## Notes
- 名称必须存在于注册表；未知名直接 fail‑fast；
- 权重长度必须与名称长度一致；
- 允许软超长惩罚（命中 max_new_tokens 时）。
