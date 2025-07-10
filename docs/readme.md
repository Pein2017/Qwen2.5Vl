# Qwen-BBU-VL Documentation ✨

Welcome! This project fine-tunes **Qwen-2.5-VL** for dense object detection and captioning of BBU (Base-Band Unit) equipment images.

## 📖 Quick links
| Topic | Document |
|-------|----------|
| Project scope & design | [overview.md](overview.md) |
| Model & training architecture | [architecture.md](architecture.md) |
| Data schema & conversion pipeline | [data_schema.md](data_schema.md) |
| **JSON cleaning & preprocessing** | **[json_cleaning.md](json_cleaning.md)** |
| YAML configuration reference | [configuration.md](configuration.md) |
| Prompt system & teacher-student logic | [prompts.md](prompts.md) |
| Diagnostics & fail-fast philosophy | [operations.md](operations.md) |
| Run commands & scripts (optional) | [runbook.md](runbook.md) |
| PEFT / LoRA adapters | [advanced/peft_adapter.md](advanced/peft_adapter.md) |
| Teacher-Student loss splitting | [advanced/teacher_student.md](advanced/teacher_student.md) |
| Packed-collator internals | [advanced/collator_notes.md](advanced/collator_notes.md) |

> If you only need **how to run** training or inference, jump straight to **runbook.md**.

---

Each document starts with a one-sentence purpose statement and ends with *Related source files* links so you can quickly jump from documentation to code.

Enjoy exploring the codebase! 🚀 