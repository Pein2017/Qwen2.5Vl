# Contract — Reward Registry (src_post)

## Registered Names (subset)
- group_margin
- label_match
- coverage
- taxonomy
- consistency
- cleanliness
- rep_penalty
- quote_penalty
- special_penalty
- soft_lexicon
- table_lexicon
- neg_alignment

## Combination (spec default)
- names: `group_margin, label_match, neg_alignment, cleanliness, special_penalty, rep_penalty, quote_penalty, coverage, taxonomy, consistency`
- weights (suggested): `1.0, 0.5, 0.3, 0.2, -0.2, -0.7, -0.4, 0.7, 0.2, 0.3`

## Notes
- 名称必须存在于注册表；未知名直接 fail‑fast；
- 权重长度必须与名称长度一致；
- 允许软超长惩罚（命中 max_new_tokens 时）。
