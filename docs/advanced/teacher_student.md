# Teacher-Student Loss Splitting

> **Purpose:** Detail how the language modeling loss is divided into teacher vs student components for analysis and research.

---

## 1. Conversation anatomy
```
System
└─ User (img T1)
   └─ Assistant (teacher resp.)  ← teacher_lm_loss
└─ User (img T2)
   └─ Assistant (teacher resp.)  ← teacher_lm_loss
└─ User (img S)
   └─ Assistant (student resp.) ← student_lm_loss + detection_loss
```

## 2. Span extraction algorithm
1. `ChatProcessor` tokenises every assistant response.
2. All but the last assistant span are tagged as *teacher*.
3. Start/end token indices are returned in `ChatProcessorOutput`.

## 3. Loss computation
`BBUTrainer._compute_teacher_student_losses()` iterates over spans and re-uses the official Qwen2.5-VL cross-entropy routine, ensuring:
* **Compatibility** – `teacher_lm_loss + student_lm_loss ≈ lm_loss` (tolerance 1e-6).
* **Gradient flow** – Both losses back-propagate.

## 4. Metrics logged
* `teacher_lm_loss`
* `student_lm_loss`
* `teacher_student_ratio`

---

### Related source files
* `src/chat_processor.py`
* `src/training/trainer.py` 