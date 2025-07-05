# PEFT / LoRA Integration

> **Purpose:** Explain how lightweight adapters (LoRA + prompt tuning) are injected into Qwen-2.5-VL while keeping the original backbone frozen.

---

## 1. Injection points
* LoRA wraps `q_proj`, `k_proj`, `v_proj`, `o_proj`, and FFN projections in each transformer layer (rank **r=8** by default).
* Optional **soft prompt**: 16 virtual tokens prepended to every input.
* Existing vision & language adapters remain unchanged; LoRA receives gradients from *both* LM and detection losses.

## 2. Configuration (excerpt)
```yaml
lora_enabled: true
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

## 3. Implementation hooks
| Step | Code | Notes |
|------|------|-------|
| Apply LoRA | `src/models/wrapper.py::_apply_lora_config` | Uses `peft.get_peft_model` |
| Group params | `src/training/trainer.py::init_param_groups` | PEFT params share `llm_lr` |
| Save / load | `wrapper.save_pretrained` & `wrapper.from_pretrained` | Saves `adapter_config.json`, `adapter_model.safetensors` |

---

### Related source files
* `src/models/wrapper.py`
* `src/training/trainer.py` 