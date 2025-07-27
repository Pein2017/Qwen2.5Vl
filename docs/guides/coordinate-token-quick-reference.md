# Coordinate Token System - Quick Reference

**Status:** ✅ PRODUCTION READY | **Test Results:** 28+ passing | **Last Updated:** 2025-01-27

Quick reference guide for the coordinate token system implementation in BBU training pipeline.

---

## 🚀 Quick Start

### Choose Your Mode

**Standard Mode** (Recommended):
```yaml
coordinate_tokens_enabled: false
max_coord_value: 2048
model_path: "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
```

**Coordinate Mode** (Advanced):
```yaml
coordinate_tokens_enabled: true
max_coord_value: 2048
coordinate_loss_weight: 1.0
regular_loss_weight: 1.0
soft_expectation_temperature: 1.0
model_path: "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
```

---

## 📋 Mode Comparison

| Feature | Standard Mode | Coordinate Mode |
|---------|---------------|-----------------|
| **Coordinates** | Integers: `[150,10,211,35]` | Tokens: `[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]` |
| **Vocabulary** | +4 tokens | +2052 tokens |
| **Format** | `<|box_start|>[150,10,211,35]<|box_end|>` | `<|box_start|>[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]<|box_end|>` |
| **Use Case** | Production, most users | Research, experimentation |
| **Status** | ✅ Fully stable | ⚠️ Trainer compatibility issue |

---

## 🔧 Token Formats

### Geometry Tokens
```
<|object_ref_start|>  # Description wrapper start
<|object_ref_end|>    # Description wrapper end
<|box_start|>         # Bounding box start
<|box_end|>           # Bounding box end
<|line_start|>        # Line geometry start
<|line_end|>          # Line geometry end
<|square_start|>      # Square geometry start
<|square_end|>        # Square geometry end
```

### Coordinate Tokens (Coordinate Mode Only)
```
<|coord_0|>, <|coord_1|>, <|coord_2|>, ..., <|coord_2047|>
```

⚠️ **CRITICAL:** Always use pipe characters `|` in coordinate tokens!

---

## 📝 Output Format Examples

### Standard Mode Examples
```
Bounding Box:
"<|object_ref_start|>desc:BBU设备<|object_ref_end|>,<|box_start|>[150,10,211,35]<|box_end|>"

Line:
"<|object_ref_start|>desc:光纤<|object_ref_end|>,<|line_start|>[100,200,150,250,200,300]<|line_end|>"

Square:
"<|object_ref_start|>desc:标签<|object_ref_end|>,<|square_start|>[50,60,80,65,85,95,55,90]<|square_end|>"
```

### Coordinate Mode Examples
```
Bounding Box:
"<|object_ref_start|>desc:BBU设备<|object_ref_end|>,<|box_start|>[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]<|box_end|>"

Line:
"<|object_ref_start|>desc:光纤<|object_ref_end|>,<|line_start|>[<|coord_100|>,<|coord_200|>,<|coord_150|>,<|coord_250|>,<|coord_200|>,<|coord_300|>]<|line_end|>"
```

---

## 🚨 Common Issues

### 1. Wrong Token Format
```
❌ Wrong: <coord_150>
✅ Correct: <|coord_150|>
```

### 2. Missing Model Path
```yaml
# ❌ Wrong
model_path: "Qwen/Qwen2.5-VL-7B-Instruct"

# ✅ Correct
model_path: "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
```

### 3. Vocabulary Size Mismatch
**Solution:** Ensure proper model loading and embedding resize

### 4. Trainer Compatibility Issues (Known Limitation)
**Error:** `🚨 TRAINER COMPATIBILITY ISSUE`
**Status:** ⚠️ **EXPECTED** - This is a documented limitation
**Solution:** Use Standard Mode for production training

---

## 🧪 Testing Commands

```bash
# Test Standard Mode
python -m pytest tests/test_data_pipeline.py::TestDataPipeline::test_bbu_dataset_loading_standard_mode -v

# Test Coordinate Mode
python -m pytest tests/test_data_pipeline.py::TestDataPipeline::test_bbu_dataset_loading_coordinate_mode -v

# Test Core Components
python -m pytest tests/test_training_components.py -v
```

---

## 📚 Related Documentation

- **Complete Guide:** [`docs/core/coordinate-token-system.md`](../core/coordinate-token-system.md)
- **Configuration:** [`docs/core/configuration.md`](../core/configuration.md)
- **Data Pipeline:** [`docs/core/data-pipeline.md`](../core/data-pipeline.md)

---

*For detailed implementation details, see the complete guide.*
