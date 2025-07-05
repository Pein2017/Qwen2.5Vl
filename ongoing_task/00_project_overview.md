# 📚 Qwen-BBU-VL — Project Overview (Updated)

*Last updated: 2025-01-18 – comprehensive refresh against latest source tree*

---
## 0. 30-Second TL;DR
We fine-tune **Qwen-2.5-VL-3B** end-to-end for dense object detection *and* captioning in BBU (Base-Band Unit) rooms.

**REFACTORED ARCHITECTURE**: The project now features a sophisticated modular training system with domain-specific configuration:
- **Legacy System**: `configs/base_flat.yaml` + `python scripts/train.py --config base_flat --log_level INFO --log_verbose true`
- **New System**: `configs/base_flat_v2.yaml` + `python scripts/train.py --config base_flat_v2 --use-new-config --log_level INFO --log_verbose true`

**Key Improvements**:
- **Domain-Specific Configs**: ModelConfig, TrainingConfig, DataConfig, DetectionConfig, InfrastructureConfig
- **Component Managers**: LossManager, ParameterManager, TrainingCoordinator
- **Clean Separation**: Training, loss computation, parameter management, configuration validation
- **Backward Compatibility**: Both systems work seamlessly with same codebase

---
## 1. Canonical Data Schema (Current Implementation)
Each JSONL line contains training data with **teachers** and **student** structure:
```jsonc
{
  "teachers": [
    {
      "images": ["ds_rescaled/<img>.jpeg", ...],
      "objects": [
        {"bbox_2d": [x1, y1, x2, y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"},
        ...
      ]
    }
  ],
  "student": {
    "images": ["ds_rescaled/<img>.jpeg", ...],
    "objects": [
      {"bbox_2d": [x1, y1, x2, y2], "desc": "螺丝连接点/BBU安装螺丝/连接正确"},
      ...
    ]
  }
}
```
**Key Facts:**
1. **Boxes**: Absolute pixel coords `[x1, y1, x2, y2]` (left, top, right, bottom)
2. **Descriptions**: Natural language format (simplified from old slash-separated)
3. **Images**: Pre-scaled JPEGs matching vision processor constraints
4. **Teacher-Student**: 70% use teachers (configurable via `teacher_ratio`)

---
## 2. End-to-End Execution Flow (Current)
```
bash → python scripts/train.py --config base_flat --log_level INFO --log_verbose true
         ↳ init_config("configs/base_flat.yaml")      # DirectConfig system
         ↳ configure_global_logging()                 # Multi-level logging
         ↳ create_trainer() → BBUTrainer              # Custom trainer
              ↳ setup_model_and_tokenizer()           # Model loading + patches
              ↳ setup_data_module()                   # BBUDataset + collators
              ↳ init_param_groups()                   # 5 LR groups
              ↳ trainer.train()                       # Multi-task training
```
**Output**: `output-{run_name}/` contains checkpoints, logs, TensorBoard

---
## 3. Source Tree Reference (Updated)
| Path                               | Current Responsibility                                      |
| ---------------------------------- | ----------------------------------------------------------- |
| `src/config/global_config.py`     | DirectConfig with 149 parameters, auto-validation          |
| `src/schema.py`                    | Complete tensor type system with torchtyping validation    |
| `src/chat_processor.py`            | Conversation builder, vision token expansion, tokenization |
| `src/data.py`                      | BBUDataset, StandardDataCollator, PackedDataCollator      |
| `src/models/wrapper.py`            | Qwen2.5-VL loading + DetectionHead integration            |
| `src/models/detection_head.py`     | DETR decoder + bbox/objectness/caption heads              |
| `src/detection_loss.py`            | Hungarian matching + multi-task loss                      |
| `src/training/trainer.py`          | BBUTrainer with teacher-student loss splitting            |
| `src/inference.py`                 | Stand-alone inference pipeline                             |
| `scripts/train.py`                 | Main training entry point                                  |
| `data_conversion/`                 | Raw JSON → clean JSONL pipeline                           |

---
## 4. Configuration System (DirectConfig)
**Single Source**: `configs/base_flat.yaml` (149 parameters)
**Key Categories**:
- **Model**: `model_path`, `model_max_length`, `attn_implementation`
- **Training**: `per_device_train_batch_size`, `num_train_epochs`, `gradient_accumulation_steps`
- **Learning Rates**: `vision_lr`, `merger_lr`, `llm_lr`, `detection_lr`, `adapter_lr`
- **Detection**: `detection_enabled`, `detection_num_queries`, loss weights
- **Data**: `collator_type` ("standard"/"packed"), `teacher_ratio`, `max_examples`
- **Performance**: `bf16`, `gradient_checkpointing`, `dataloader_num_workers`

**No Defaults**: All parameters must be explicitly defined in YAML.

---
## 5. Launch Commands (Current)
```bash
# Basic training
conda activate ms
python scripts/train.py --config base_flat --log_level INFO --log_verbose true

# With DeepSpeed (via environment)
export BBU_DEEPSPEED_ENABLED=true
export BBU_DEEPSPEED_CONFIG=scripts/zero2.json
python scripts/train.py --config base_flat --log_level INFO --log_verbose true

# Config validation only
python scripts/train.py --config base_flat --log_level INFO --log_verbose true --validate-only

# Print config and exit
python scripts/train.py --config base_flat --log_level INFO --log_verbose true --print-config
```

**Inference**:
```python
from src.inference import Qwen25VLInference
predictor = Qwen25VLInference(checkpoint_dir)
boxes, captions = predictor.predict_detection(images, prompt)
```

---
## 6. Data Processing Pipeline (Current)
```
Raw vendor JSONs → data_conversion/convert_dataset.sh → Clean JSONL + ds_rescaled/
                   ↳ convert_pure_json.py (extraction + vision_process.smart_resize)
                   ↳ create_teacher_pool.py (curated demonstrations)
                   ↳ split_train_val.py (train/val split)
```

---
## 7. Multi-Task Training Features
- **Teacher-Student Learning**: 70% samples use teacher demonstrations
- **Packed Collator**: Zero-padding with Flash Attention 2 for efficiency
- **Detection Head**: DETR-style with bbox/objectness/caption prediction
- **Differential LR**: 5 parameter groups with separate learning rates
- **Dynamic Freezing**: `detection_freeze_epochs` for staged training
- **Loss Splitting**: Separate teacher/student loss components for analysis

---
## 8. Monitoring & Debugging
**TensorBoard Metrics**:
- Loss components: `lm_loss`, `teacher_lm_loss`, `student_lm_loss`
- Detection: `bbox_l1_loss`, `bbox_giou_loss`, `objectness_loss`, `caption_loss`
- Norms: `wn/*` (weight norms), `gn/*` (gradient norms)

**Fail-Fast Validation**:
- Tensor shape assertions via torchtyping
- Configuration parameter validation
- Image size compatibility checks
- Special token synchronization

---
## 9. FAQ / Current Gotchas
1. **Always use `predict_detection()`** for inference with bounding boxes
2. **Check `<|image_pad|>` count** matches `pixel_values` patches (auto-validated)
3. **No hardcoded defaults** - all config must be in YAML
4. **Teacher-student spans** are automatically extracted during processing
5. **Packed collator** requires `per_device_train_batch_size: 1` typically
6. **Detection head** can be disabled via `detection_enabled: false`

---
## 10. Recent Updates
- **DirectConfig System**: Eliminated parameter passing, flat access pattern
- **Teacher-Student Loss**: Implemented span extraction and separate loss tracking
- **Schema Validation**: Complete torchtyping integration for runtime checks
- **Enhanced Logging**: Multi-level logging with per-component loss tracking
- **Packed Collator**: Efficient token packing for variable-length sequences

Happy training! 🚀