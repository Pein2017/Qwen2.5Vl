# Qwen2.5-VL Development Guide

## 1. Project Overview
Qwen2.5-VL fine-tunes large vision-language models for BBU (Base-Band Unit) equipment detection and captioning. Key features:

- **Multi-modal Vision-Language Integration** – end-to-end dense object detection with English/Chinese descriptions  
- **Coordinate Token System** – soft expectation regression with automatic bbox→token conversion  
- **Multi-task Training** – teacher-student learning with span-based loss splitting  
- **Modular Architecture** – component-based design, clear separation of concerns  
- **Enhanced Loss Management** – mode-aware coordinate vs. standard LLM loss computation  

## 2. Environment Setup
- **Conda environment**: `ms` (activate before any command)  
- **Python interpreter**: `/root/miniconda3/envs/ms/bin/python` (use explicitly)  
- **Environment variables**:  
  - `CUDA_VISIBLE_DEVICES` – GPU selection  
  - `HF_HOME` – model cache directory (e.g. `/data3/Qwen2.5-VL-main/model_cache`)  
- **Network constraints**: offline/China-only environment – rely on local mirrors & caches; GitHub/Google/HF are inaccessible.

## 3. Key Documentation
- Architecture – `docs/ARCHITECTURE.md`  
- Mental model – `docs/MENTAL_MODEL.md`  
- Project map – `docs/PROJECT_MAP.md`  
- Data migration – `docs/raw_data_v2.md`, `docs/raw_data_template_数据堂.md`  

## 4. Development Workflow & Standards
### Planning & Execution
- **Plan-First** – define goals & steps before coding  
- **Fail-Fast** – surface errors immediately; no silent exceptions  
- **Simple Solutions** – implement minimal-impact changes  
- **Type Safety** – add explicit typing & validate inputs  
- **Documentation** – update docs alongside code changes  

### Code Quality
- **Explicit Parameters** – define every hyper-parameter explicitly  
- **Type Annotations** – required for all functions/methods  
- **No Silent Failures** – avoid bare `except`/`except: pass`  
- **Validation** – use `@dataclass` for schema enforcement  
- **Logging** – reserve `logger.debug()` for information, never to hide errors  

### File Management
- **Refactoring** – override existing files; move superseded versions to `legacy/`  
- **No Duplicates** – reuse/merge rather than create near-identical files  
- **Clean-Up** – delete temporary/debug files after use  
- **Exploration** – traverse data & control flow depth-first before changing code  

### Testing & Validation
- **Temporal tests** – place in `./temporal` to keep main codebase clean  
- **Evidence** – leave verification results after completing tasks  
- **Root Cause** – fix underlying issues, avoid temporary patches  

## 5. Development Process Checklist
1. **Analysis** – thoroughly analyze the problem & codebase  
2. **Planning** – create a todo list in `tasks/todo.md`  
3. **Verification** – get plan approval  
4. **Implementation** – make small, targeted changes  
5. **Review** – summarize all changes with high-level explanation  
6. **Documentation** – update relevant documentation  

## 6. Common Entry Points
| Purpose        | Command |
|----------------|---------|
| Training       | `python scripts/train.py --config configs/bbu_v2.yaml` |
| Data Processing| `bash data_conversion/convert_dataset.sh` |
| Inference      | `python src/inference.py --model_path /path/to/model --image_path /path/to/image` |

---
