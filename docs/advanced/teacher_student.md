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
1. `ChatProcessor` tokenizes every assistant response.
2. All but the last assistant span are tagged as *teacher*.
3. Start/end token indices are returned in `ChatProcessorOutput` and passed to the `LossManager`.

## 3. Loss computation
The `LossManager` is responsible for computing the teacher and student losses. It iterates over the spans provided in the inputs and calculates a separate cross-entropy loss for each, ensuring:
* **Compatibility** – `teacher_lm_loss + student_lm_loss` is consistent with the total `lm_loss`.
* **Gradient flow** – Both losses are combined with configurable weights and contribute to the total loss for backpropagation.

## 4. Metrics logged
* `teacher_lm_loss`
* `student_lm_loss`
* `teacher_loss_weight`
* `student_loss_weight`

---

### Related source files
* `src/chat_processor.py`
* `src/training/loss_manager.py`
* `src/training/training_coordinator.py` 