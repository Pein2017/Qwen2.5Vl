# Config Fix Applied: Dynamic Length Alpha

## Issue
`term_ratio=0.000` - No completions ending with EOS token.

## Root Cause
Dynamic length cap too restrictive:
```yaml
alpha: 1.1  # Only 10% headroom over GT length
```

For GT=400 tokens, cap=440. Dense captioning often needs >500 tokens.

## Fix Applied
**File**: `configs/dense_rl/standard.yaml` (line 46)
```yaml
alpha: 1.5  # Increased from 1.1, allows 50% headroom
```

## Expected Outcome
- **term_ratio**: 0.000 → 0.60-0.80 (60-80% ending with EOS)
- **Reward variance**: More stable (proper endings)
- **Learning**: Improved signal quality

## Verification
Run diagnostic and check logs:
```bash
bash run_8gpu_diagnostic.sh
grep "term_ratio" quick_diagnostic_8gpu_clean.log
```

Should see `term_ratio > 0.5` within first few steps.

## Fallback
If still `term_ratio=0.000`, disable dynamic length entirely:
```yaml
dynamic_length:
  enabled: false
```
