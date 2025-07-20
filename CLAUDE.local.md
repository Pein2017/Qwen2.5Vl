## AI Assistant Codebase Guidelines

### Environment
- ~~`source activate ms`~~
- **Updated Environment Initialization**: Use `/root/miniconda3/envs/ms/bin/python` directly instead of generic `python` command
- Located in China, cannot access to foreign website like `github` `google` `huggingface`

### Workflow

* **Plan**: Define goals & end-to-end steps (a→b→c) before coding.
* **Execute**: Implement directly; commit iterations freely.
* **Iterate**: Continue until objectives are achieved or resource limits reached.

### Error Handling

* **Fail-fast**: No silent `try/except: pass` or bare `except`.
* **Surface Errors**: Let exceptions bubble; use explicit `raise` for illegal states.
* **Logging**: `logger.debug()` only for state/info—never to suppress errors.

### Hyperparameters & Attributes

* **Explicit**: All hyperparameters must be defined—**never use** `getattribute` or `dict.get(<key>, <default>)`.
* **Validate**: Enforce schemas via `@dataclass`.

### Code Exploration & Refactoring

* **Review**: Depth-first traversal of modules to map data/control flow.
* **Document**: For each file, note purpose, key classes/functions, inputs/outputs.
* **Plan & Act**: List refactors/fixes in order and implement immediately.
* **Refactoring Rule**: Directly override existing files. For important files, copy/move to `legacy` for reference. No fallback or backward compatibility design.
* **File Management**: Try to reuse/override/merge the files. Don't create new but similar files.

### Code Quality

* **Concise**: Keep code, comments, and commit messages focused.
* **Types**: Annotate every function/method.
* **Defaults**: Only universal defaults (e.g., `in_channels: int = 3`); otherwise require explicit args.
* **Reuse**: Extend existing files; delete temp/debug files when done.

> **Reminder:** Favor refactoring over file duplication. Keep the codebase DRY, transparent, and consistent.

### Data Annotations

* Recorded information about data annotations outline per user request.

### Testing Guidelines

* Create and run temporal tests under `./temporal` directory to keep main codebase clean
* After completing a task, leave one or few evidence results to verify task completion

### Package References

* HuggingFace Transformers Package Location: `/root/miniconda3/envs/ms/lib/python3.10/site-packages/transformers/models/qwen2_5_vl`
* Official HuggingFace transformers package is now available for Qwen2.5VL source code reference
* `/src/reference` is a copy from the transformers package, can be used for reference as well

### Architecture Deep Dive

#### Core Model Architecture
```
Qwen25VLWithDetection (src/models/wrapper.py)
├── Base Qwen2.5-VL Model (vision + language)
├── Coordinate Token Manager (src/utils/coordinate_token_manager.py)
│   ├── Extended Vocabulary: +2048 coordinate tokens [151665-153713)
│   ├── Box Tokens: <|box_start|> (151648), <|box_end|> (151649)
│   └── Soft Expectation Loss: focal + L1 + GIoU
├── Loss Manager (src/training/loss_manager.py)
│   ├── Mode Detection: coordinate vs standard LLM
│   ├── Multi-component Loss: coordinate + focal + regular + l1 + giou
│   └── Teacher-Student Splitting: span-based loss separation
└── Training Coordinator (src/training/training_coordinator.py)
    ├── Forward Pass Orchestration
    ├── Loss Computation Delegation  
    └── Component Integration
```

#### Data Flow Architecture
```
Raw JSONL → ChatProcessor → Coordinate Conversion → Model Training
├── Input: {"bbox_2d": [x1,y1,x2,y2], "desc": "object"}
├── Conversion: "<|box_start|><coord_x1><coord_y1><coord_x2><coord_y2><|box_end|>"
├── Tokenization: [151648, 151668, 151924, 151960, 152318, 151649]
├── Detection: bbox_spans = [(start_idx, end_idx)]
├── Loss: coordinate_loss + focal_loss + regular_loss + l1_loss + giou_loss
└── Training: mode-aware backpropagation
```

#### Key Component Interactions
- **ChatProcessor** (`src/chat_processor.py`): Automatic bbox→coordinate token conversion
- **CoordinateLossComputer** (`src/utils/coordinate_loss_computer.py`): Enhanced bbox span detection
- **LossManager** (`src/training/loss_manager.py`): Configuration-driven loss computation
- **ModelWrapper** (`src/models/wrapper.py`): Defensive loss component initialization

### Current Implementation Status

✅ **Coordinate Token System**: Fully operational with automatic conversion
✅ **Multi-component Loss**: Enhanced with validation and debugging  
✅ **Mode-aware Training**: Seamless coordinate vs standard LLM switching
✅ **Architecture Documentation**: Updated to reflect current implementation
✅ **Configuration Validation**: Comprehensive parameter checking

### Backward Compatibility

* Don't need any Backward compatibility, I want to make the codebase clean and consise.