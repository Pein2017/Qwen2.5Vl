# 📚 Qwen2.5-VL Project Documentation

*Last updated: 2025-01-18 – Reorganized and consolidated documentation*

This directory contains the complete documentation for the Qwen2.5-VL BBU equipment detection project.

---

## 🎯 Quick Start - READ THIS FIRST

### 📖 Main Guide
- **[Complete Pipeline Guide](complete_pipeline_guide.md)** ⭐ - **START HERE** - Complete end-to-end workflow covering training, inference, and evaluation

### 📋 Historical Reference  
- **[Historical Inference Issues](historical_inference_issues.md)** - Resolved chat template issue (for reference only)

---

## 📁 Complete Documentation Library

### 🏗️ Core Architecture & Implementation
- **[00_project_overview.md](00_project_overview.md)** - Project scope, goals, and high-level architecture
- **[01_architecture_and_training.md](01_architecture_and_training.md)** - Detailed model architecture and training pipeline
- **[02_data_pipeline_and_formats.md](02_data_pipeline_and_formats.md)** - Data formats, conversion, and validation
- **[03_prompt_and_training.md](03_prompt_and_training.md)** - Prompt engineering and training strategies

### ⚙️ Configuration & Operations
- **[04_operations_and_diagnostics.md](04_operations_and_diagnostics.md)** - Operational procedures and diagnostics
- **[05_debug_and_hf_integration.md](05_debug_and_hf_integration.md)** - Debugging and HuggingFace integration
- **[06_configuration_reference.md](06_configuration_reference.md)** - Complete configuration reference

### 🧪 Advanced Topics & Research
- **[adding_peft_adapter.md](adding_peft_adapter.md)** - PEFT adapter implementation details
- **[split_teacher_student_loss.md](split_teacher_student_loss.md)** - Advanced teacher-student loss splitting
- **[packed_collator_issue.md](packed_collator_issue.md)** - Packed collator implementation notes

---

## 🚀 Quick Navigation by Task

### Training a Model
1. Start with **[Complete Pipeline Guide](complete_pipeline_guide.md)** sections 1-2
2. Reference **[06_configuration_reference.md](06_configuration_reference.md)** for advanced settings
3. Use **[04_operations_and_diagnostics.md](04_operations_and_diagnostics.md)** for troubleshooting

### Running Inference
1. **[Complete Pipeline Guide](complete_pipeline_guide.md)** section 3 (includes resolved issues)
2. **[Historical Inference Issues](historical_inference_issues.md)** - background on resolved chat template problem

### Evaluation & Metrics
1. **[Complete Pipeline Guide](complete_pipeline_guide.md)** section 4
2. Use the pipeline: Data → Training → Inference → Evaluation

### Understanding the Architecture
1. **[00_project_overview.md](00_project_overview.md)** for high-level understanding
2. **[01_architecture_and_training.md](01_architecture_and_training.md)** for technical details
3. **[02_data_pipeline_and_formats.md](02_data_pipeline_and_formats.md)** for data flow

---

## ✅ What's New (January 2025)

### 🎯 Major Reorganization
- **Consolidated Documentation**: Merged 4 separate guides into comprehensive pipeline guide
- **Resolved Issues**: All inference issues have been fixed and documented
- **Simplified Structure**: Cleaner navigation and reduced duplication

### 🔧 Key Achievements
- **Unified Inference Pipeline**: Both standard and teacher-guided modes now use identical ChatProcessor
- **Template Alignment**: Fixed critical chat template mismatch between training and inference
- **Batch Processing**: Resolved Qwen2.5-VL processor batch handling issues
- **Complete Evaluation**: COCO-style metrics with semantic matching and per-category analysis

### 🗂️ Removed/Consolidated
- ✅ Merged: `inferece_issue.md` → `complete_pipeline_guide.md` + `historical_inference_issues.md`
- ✅ Merged: `08_evaluation_pipeline.md` → `complete_pipeline_guide.md`
- ✅ Merged: `07_training_workflow_guide.md` → `complete_pipeline_guide.md`
- ✅ Merged: `07_improve_inference_with_teacher_guidance.md` → `complete_pipeline_guide.md`

---

## 📊 Document Status

| Document | Status | Last Updated | Purpose |
|----------|--------|--------------|---------|
| **complete_pipeline_guide.md** | ✅ **CURRENT** | 2025-01-18 | **Main guide - use this** |
| **historical_inference_issues.md** | ✅ RESOLVED | 2025-01-18 | Historical reference only |
| 00_project_overview.md | ✅ Current | 2024-12-XX | Project overview |
| 01_architecture_and_training.md | ✅ Current | 2024-12-XX | Architecture details |
| 02_data_pipeline_and_formats.md | ✅ Current | 2024-12-XX | Data pipeline |
| 03_prompt_and_training.md | ✅ Current | 2024-12-XX | Prompt engineering |
| 04_operations_and_diagnostics.md | ✅ Current | 2024-12-XX | Operations |
| 05_debug_and_hf_integration.md | ✅ Current | 2024-12-XX | Debugging |
| 06_configuration_reference.md | ✅ Current | 2024-12-XX | Configuration |
| adding_peft_adapter.md | ✅ Current | 2024-12-XX | PEFT implementation |
| split_teacher_student_loss.md | ✅ Current | 2024-12-XX | Advanced loss splitting |
| packed_collator_issue.md | ✅ Current | 2024-12-XX | Collator implementation |

---

## 💡 Tips for New Users

1. **Start Here**: Read **[Complete Pipeline Guide](complete_pipeline_guide.md)** first
2. **Follow the Flow**: Data Preparation → Training → Inference → Evaluation  
3. **Check Issues**: All major issues have been resolved and documented
4. **Use Examples**: Each section includes working command examples
5. **Reference Advanced**: Use numbered documents (00-06) for deep dives

---

## 🔗 External Resources

- **Model Checkpoints**: Store in `output-*/` directories
- **Data**: Use `data/` for JSONL files, `ds_rescaled/` for images
- **Results**: Save to `eval_results_*/` and `infer_result/`
- **Logs**: Check `logs/` for training and inference logs

---

**For any questions, start with the [Complete Pipeline Guide](complete_pipeline_guide.md) - it covers 95% of common use cases with step-by-step instructions.** 🚀 