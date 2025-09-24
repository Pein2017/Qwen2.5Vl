# Qwen2.5-VL RL Integration Operations Plan

## Executive Summary
Consolidate `src_rl/` into `src_new/rl/` to eliminate code duplication while preserving all existing functionality and maintaining strict backward compatibility. This unifies SFT and RL post-training under a single codebase with shared components and consistent HF-first invariants.

---

## Current State Analysis

### Code Duplication Identified
- **Data Processing**: Both pipelines parse JSONL, load images via PathManager, and apply identical multimodal validations
- **Chat Templates**: Both use HF-first conversation building with typed messages → template render → processor tensors
- **Model Wrapper**: Identical DetectionModel with coordinate token support and multimodal validations  
- **Inference Logic**: Same generation patterns needed for evaluation and RL sampling stages
- **Configuration**: Overlapping config structures with different field names for same concepts

### Existing Assets
- `src_new/merging_grpo_post_training_plan.md` - Comprehensive technical plan (231 lines)
- `src_rl/refactoring_plan.md` - Current RL improvements plan (161 lines)
- Working `VisionGRPOTrainer` in `src_rl/trainer.py` 
- Established config system in `src_rl/config.py` with dataclasses
- Full reward registry system in `src_rl/rewards/`

---

## Target Architecture

```
src_new/
├── rl/                                    # ← NEW: Consolidated RL module
│   ├── __init__.py
│   ├── runner.py                         # ← FROM src_rl/runner.py
│   ├── trainer.py                        # ← FROM src_rl/trainer.py  
│   ├── config.py                         # ← FROM src_rl/config.py
│   ├── eval.py                           # ← FROM src_rl/eval.py
│   ├── data/
│   │   └── dataset.py                    # ← FROM src_rl/data/dataset.py
│   ├── prompting/
│   │   └── conversation.py               # ← FROM src_rl/prompting/conversation.py
│   ├── rewards/                          # ← FROM src_rl/rewards/*
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── format_rewards.py
│   │   └── detection_rewards.py
│   ├── tools/                            # ← FROM src_rl/tools/*
│   └── README.md                         # ← FROM src_rl/README.md (updated paths)
├── processing/
│   ├── parse_generated.py                # ← NEW: Shared parser (extract from inference.py)
│   └── conversation/                     # ← EXISTING: Already shared by both
└── ...

src_rl/                                    # ← DEPRECATED: Backward compatibility shims
├── __init__.py                           # ← RE-EXPORT from src_new.rl
├── runner.py                             # ← SHIM with deprecation warning
└── ...
```

---

## Operations Sequence

### Phase 1: Foundation (Day 1)
1. **Create module skeleton**
   ```bash
   mkdir -p /data3/Qwen2.5-VL-main/src_new/rl/{data,prompting,rewards,tools}
   touch /data3/Qwen2.5-VL-main/src_new/rl/__init__.py
   ```

2. **Move core files with import updates**
   - `src_rl/runner.py` → `src_new/rl/runner.py`
   - `src_rl/trainer.py` → `src_new/rl/trainer.py` 
   - `src_rl/config.py` → `src_new/rl/config.py`
   - `src_rl/eval.py` → `src_new/rl/eval.py`
   - Update all imports: `from src_rl...` → `from src_new.rl...`

3. **Move subdirectories**
   - `src_rl/data/*` → `src_new/rl/data/*`
   - `src_rl/prompting/*` → `src_new/rl/prompting/*`
   - `src_rl/rewards/*` → `src_new/rl/rewards/*`  
   - `src_rl/tools/*` → `src_new/rl/tools/*`

### Phase 2: Shared Components (Day 1-2)
4. **Extract shared parser**
   - Create `src_new/processing/parse_generated.py`
   - Move parsing logic from `src_new/inference.py::_normalize_prediction_to_vis_objects`
   - Export: `parse_geometry_wrapped_text_to_objects(text: str, coordinate_tokens_enabled: bool = False) -> List[Dict[str, Any]]`
   - Update consumers:
     - `src_new/inference.py` to import shared parser
     - `src_new/rl/rewards/*` to use shared parser

5. **Update conversation building**
   - Ensure `src_new/rl/prompting/conversation.py` uses `src_new.processing.conversation.ConversationBuilder`
   - Validate single-turn prompt generation maintains HF-first invariants

### Phase 3: Configuration Consolidation (Day 2)
6. **Unify config system**
   - Port `EnhancedRLConfig` dataclasses into `src_new/rl/config.py`
   - Maintain field mapping compatibility with existing YAML configs
   - Add validation for required fields with fail-fast behavior
   - Ensure `configs/rl/*.yaml` continue working unchanged

### Phase 4: Backward Compatibility (Day 2)
7. **Create deprecation shims**
   - Modify `src_rl/__init__.py` to re-export from `src_new.rl`
   - Add `src_rl/runner.py` shim that imports and calls `src_new.rl.runner` with deprecation warning
   - Create `src_rl/__main__.py` shim for `-m src_rl.runner` compatibility

### Phase 5: Scripts and Documentation (Day 3)
8. **Update launcher scripts**
   - Modify `scripts/run_dense_grpo.sh` to call `src_new.rl.runner`
   - Preserve all existing arguments and environment variables
   - Add alias script for backward compatibility if needed

9. **Documentation migration**
   - Move `src_rl/README.md` → `src_new/rl/README.md`
   - Update all paths and examples to use new module structure  
   - Add migration notes and deprecation timeline

### Phase 6: Validation (Day 3-4)
10. **Comprehensive testing**
    - **Unit tests**: Parser, rewards, config loading
    - **Integration tests**: 
      ```bash
      # Loader validation
      python -m src_new.rl.runner --config configs/rl/dense_grpo.yaml --mode load
      
      # Short training run  
      python -m src_new.rl.runner --config configs/rl/dense_grpo.yaml --mode train --max_steps 2
      
      # Backward compatibility
      python -m src_rl.runner --config configs/rl/dense_grpo.yaml --mode load
      ```
    - **Validation checks**:
      - Image token ↔ tensor alignment (no "Image features and image tokens do not match" errors)
      - Vision tensors present in generation and log-prob paths
      - RL checkpoints loadable by `src_new/inference.py`
      - Prompt/tensor parity with SFT for identical samples

---

## Technical Validation Checklist

### HF-First Invariants (Must Hold)
- [ ] Placeholder count in rendered text equals number of images
- [ ] `pixel_values` rows == sum of `image_grid_thw` (t×h×w)  
- [ ] `<|image_pad|>` count in `input_ids` matches expected merge_size calculations
- [ ] Assistant spans properly detected with `<|im_end|>` inclusion
- [ ] Coordinate tokens remain deprecated (`coordinate_tokens_enabled: false`)

### Multimodal Tensor Flow
- [ ] `VisionGRPOTrainer.generate()` forwards `pixel_values` and `image_grid_thw`
- [ ] Per-token log-probs computed with same vision tensors
- [ ] Pre-generation validation catches alignment mismatches
- [ ] Vision tensors preserved through query+response concatenation

### Configuration Compatibility
- [ ] Existing `configs/rl/*.yaml` files load without modification
- [ ] Required fields validated with clear error messages
- [ ] No silent defaults for critical parameters
- [ ] bf16 auto-probe with graceful float32 fallback

### Backward Compatibility
- [ ] `python -m src_rl.runner` works with deprecation warning
- [ ] `scripts/run_dense_grpo.sh` unchanged from user perspective
- [ ] All existing configs, scripts, and workflows continue functioning

---

## Risk Mitigation

### Import/Module Issues
- **Risk**: Circular imports or missing dependencies during move
- **Mitigation**: Systematic import testing at each phase; use relative imports within RL module

### TRL/HF Version Compatibility  
- **Risk**: TRL integration assumptions break during refactor
- **Mitigation**: Pin tested versions; isolate TRL imports to `src_new/rl/` only

### Configuration Drift
- **Risk**: YAML configs become incompatible or develop silent failures
- **Mitigation**: Comprehensive config validation tests; maintain exact field mapping

### Performance/Stability
- **Risk**: Vision tensor handling affects training stability  
- **Mitigation**: Force eager attention in RL; validate tensor shapes pre-generation

---

## Success Criteria

### Functional Requirements
- [ ] GRPO training completes with images enabled and zero alignment errors
- [ ] Reward metrics include both formatting and geometry components  
- [ ] Valid parse rate ≥ 0.9 on validation subset
- [ ] Generated RL checkpoints consumable by existing inference pipeline

### Quality Requirements
- [ ] Zero code duplication between SFT and RL data/model/conversation handling
- [ ] Single source of truth for parsing, validations, and HF-first contracts
- [ ] All existing functionality preserved with identical behavior
- [ ] Clear deprecation path with 2-week grace period

### Operational Requirements  
- [ ] No changes required to user workflows, configs, or scripts during transition
- [ ] Comprehensive documentation with migration examples
- [ ] Rollback capability if issues discovered post-merge

---

## Timeline & Resources

**Total Duration**: 4 days
**Resource Requirements**: 1 engineer with familiarity of both codebases

### Daily Breakdown
- **Day 1**: Core file moves and import fixes (Phases 1-2)
- **Day 2**: Configuration consolidation and backward compatibility (Phases 3-4)  
- **Day 3**: Scripts, documentation, and initial testing (Phases 5-6)
- **Day 4**: Comprehensive validation and final acceptance testing

### Rollback Plan
- Git branch for all changes with atomic commits per phase
- Backward compatibility shims remain indefinitely if needed
- Option to revert by updating script targets back to `src_rl` if critical issues found

---

## Post-Merge Operations

### Immediate (Week 1)
- Monitor for any alignment errors or performance regressions
- Update any internal documentation or workflows referencing old paths
- Announce successful migration to users with new recommended paths

### Cleanup (Week 2+)  
- Remove `src_rl/` backward compatibility shims after grace period
- Archive old refactoring plans and update repository documentation
- Consider additional shared component extraction opportunities discovered during merge
