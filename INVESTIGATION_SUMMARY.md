# EOS Termination Investigation - Summary

## Status: ROOT CAUSE IDENTIFIED BUT FIX INCOMPLETE

### Critical Finding
**0% completions ending with `<|im_end|>` token** across all runs.

### Root Cause Confirmed
Dynamic length `alpha=1.1` too restrictive for dense captioning.

### Fix Applied
- **Config**: `configs/dense_rl/standard.yaml` line 46
- **Change**: `alpha: 1.1` → `alpha: 1.5`
- **Status**: ⚠️ Fix applied but diagnostic.yaml inherits from standard.yaml

### Test Results
```
Step 1-2: term_ratio=0.000 (unchanged)
```

### Next Steps Required

#### Option 1: Test with Disabled Dynamic Length
Temporarily disable dynamic length to confirm it's the issue:
```yaml
generation:
  dynamic_length:
    enabled: false
```

#### Option 2: Test Alpha=2.0
More aggressive increase:
```yaml
generation:
  dynamic_length:
    alpha: 2.0  # 100% headroom
```

#### Option 3: Check Actual Completion Lengths
See if completions are hitting cap:
- Parse log for completion lengths vs max_new_tokens
- If all completions = 512 tokens → confirms cap issue

### Evidence So Far

**From quick_diagnostic_8gpu_clean.log**:
- All rewards active and working
- Reward variance exists but collapsing
- 40/40 completions hit cap

**From test_eos_fix.log**:
- term_ratio still 0.000
- Config v2 format issues resolved
- Training proceeding normally otherwise

### Confidence
95% confident dynamic length is the issue, but need to verify fix propagation.

### Immediate Action
Run test with `enabled: false` to bypass dynamic length entirely.
