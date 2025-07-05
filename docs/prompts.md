# Prompt System & Teacher-Student Logic

> **Purpose:** Document how conversation templates, system prompts, and teacher-student examples are constructed before tokenisation.

---

## 1. System Prompt (Chinese default)
```text
你是专业的BBU基站设备检测专家。请仔细分析图像中的设备状态，识别螺丝连接点、线缆、标签贴纸等关键部件，并描述它们的安装状态和连接情况。
```
An English fallback exists for non-Chinese datasets.

## 2. Conversation Templates
| Mode | Structure | Losses |
|------|-----------|--------|
| **Teacher-Student** | System → (User image / Assistant teacher)* → User image → Assistant student | Teacher & student LM loss + Detection loss |
| **Single-Shot** | System → User image → Assistant | Student LM + Detection loss |

`teacher_ratio` controls the sampling mix (default 0.7).

## 3. Token expansion
`<IMAGE>` placeholders become `<|vision_start|> <|image_pad|>×N <|vision_end|>` where `N` equals the number of ViT patches.

## 4. Span extraction
`ChatProcessor._extract_assistant_token_spans()` labels every assistant response so the trainer can split LM loss into *teacher* and *student* parts.

## 5. Prompt helper APIs
| Function | Purpose |
|----------|---------|
| `prompt.get_system_prompt(use_training_prompt, language)` | Returns the *system* prompt chosen between `CHINESE_TRAINING_PROMPT` / `CHINESE_EVALUATION_PROMPT` or English. |
| `prompt.get_user_prompt_prefix(use_training_prompt, language)` | Prefix for user turns (useful in few-shot mode). |
| `prompt.format_few_shot_prompt(examples, target_desc, use_training_prompt, language)` | Builds few-shot prompt by embedding example JSON snippets. |
| `prompt.get_optimized_prompt_for_context(context, language)` | One-stop helper that returns a dict with system prompt + flags for ChatProcessor.

Key constants exported from `prompt.py`:
* `CHINESE_TRAINING_PROMPT`
* `CHINESE_EVALUATION_PROMPT`
* `ENGLISH_BASE_PROMPT`
* `CHINESE_CANDIDATES_SECTION`

These are imported by `ChatProcessor` at runtime to build the final conversation skeleton.

---

### Related source files
* `src/prompt.py`
* `src/chat_processor.py` 