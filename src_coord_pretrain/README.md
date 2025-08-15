# Enhanced Coord Bootstrap Pre‑Training

**Advanced coordinate token training with Phase A, Unlikelihood, and Reverse Mapping**

Purpose: teach `<|coord_*|>` usage through sophisticated multi-phase training including identity mapping, arithmetic, reverse mapping, and unlikelihood training to achieve near-perfect coordinate token alignment before handing off to `src_new`.

## 🚀 Quick Start

### Basic Training
```bash
source ~/.bashrc && conda activate ms

# Generate enhanced training data
python src_coord_pretrain/scripts/generate_coord_bootstrap.py \
  --output src_coord_pretrain/data/coord_bootstrap.jsonl \
  --num_identity 50000 --num_arithmetic 20000 --num_reverse 10000 \
  --max_coord 1024 --seed 42

# Run enhanced training
python src_coord_pretrain/training/trainer.py \
  --config src_coord_pretrain/config/coord_bootstrap.yaml
```

### Debug Training (Fast Testing)
```bash
# Use debug configuration for quick testing
python src_coord_pretrain/training/trainer.py \
  --config src_coord_pretrain/config/coord_bootstrap_debug.yaml
```

## 🎯 Enhanced Features

### Phase A Training (Identity-Only Warm-Start)
- **Purpose**: Freeze backbone layers and train only embeddings + LM head for better coordinate token learning
- **Duration**: Configurable steps (default: 400 steps)
- **Benefits**: Faster convergence, better coordinate token alignment

### Unlikelihood Training
- **Digit Suppression**: Penalizes raw digit tokens at assistant positions
- **Coordinate Window Suppression**: Penalizes neighboring coordinate tokens
- **Configurable Weights**: Fine-tune suppression strength

### Reverse Mapping & Top-K Unlikelihood
- **Bidirectional Learning**: `"N" ↔ <|coord_N|>` conversion
- **Top-K Negative Sampling**: Intelligent selection vs simple suppression
- **Conflict Resolution**: Token-level masks for mixed batches
- **Numerical Stability**: Probability clamping and finite loss verification

### Advanced Evaluation
- **Identity Accuracy**: Validates `"N" → <|coord_N|>` conversion (target: ≥99.9%)
- **Reverse Accuracy**: Validates `<|coord_N|> → "N"` conversion (target: ≥99.5%)
- **Strictness Validation**: Ensures exactly one coordinate token, no raw digits

## 📋 Configuration Options

### Phase A Settings
```yaml
phase_a_enabled: true          # Enable Phase A training
phase_a_steps: 400            # Duration of Phase A in steps
phase_a_freeze_backbone: true # Freeze LLM backbone layers
phase_a_eos_off: true         # Disable EOS in labels during Phase A
identity_only_ratio: 0.8      # Portion of identity samples in Phase A
```

### Unlikelihood Settings
```yaml
unlikelihood_enabled: true           # Enable unlikelihood training
unlikelihood_lambda_digits: 1.0     # Weight for digit suppression
unlikelihood_lambda_coords: 1.0     # Weight for coordinate suppression
unlikelihood_coord_window: 8        # Neighbor suppression window

# Top-K Unlikelihood (Advanced Features)
ul_topk_noncoord: 100               # Top-K non-coordinate tokens for coordinate targets
ul_topk_coord: 100                  # Top-K coordinate tokens for text targets
ul_neighbor_window: 8               # Coordinate neighbor suppression window
```

### Reverse Mapping Settings
```yaml
reverse_mapping_ratio: 0.2          # Portion of reverse mapping samples
```

## 🔧 Data Generation Options

### Enhanced Data Generation
```bash
python src_coord_pretrain/scripts/generate_coord_bootstrap.py \
  --output data/coord_bootstrap.jsonl \
  --num_identity 50000 \
  --num_arithmetic 20000 \
  --num_reverse 10000 \
  --reverse_ratio 0.2 \
  --identity_only \
  --canonical_prompts \
  --max_coord 1024 \
  --seed 42
```

### Data Generation Parameters
- `--num_reverse`: Number of reverse mapping samples
- `--reverse_ratio`: Alternative to --num_reverse (ratio of total samples)
- `--identity_only`: Generate only identity samples (for Phase A)
- `--canonical_prompts`: Use canonical prompts (reduced variance)

### Tiny Dataset for Debug (NEW)
For rapid development and testing, generate a small dataset:

```bash
# Generate tiny dataset (100 samples total)
python src_coord_pretrain/scripts/generate_coord_bootstrap.py \
  --output src_coord_pretrain/data/coord_bootstrap_tiny.jsonl \
  --num_identity 50 \
  --num_arithmetic 30 \
  --num_reverse 20 \
  --max_coord 100 \
  --seed 42

# Use in config: data_path: "data/coord_bootstrap_tiny.jsonl"
```

**Benefits**: Fast loading, quick iteration, minimal memory usage

## 📊 Training Monitoring

### Key Metrics
- **Phase Information**: Current training phase (A or B)
- **Identity Accuracy**: Forward mapping accuracy
- **Reverse Accuracy**: Reverse mapping accuracy
- **Strictness Accuracy**: Coordinate token usage compliance
- **Unlikelihood Components**: Digit and coordinate suppression losses

### Enhanced Loss Component Logging
The training system now provides detailed loss component tracking:

```
📊 step: 100
📊 loss: 0.845                    # Total combined loss
📊 llm_loss: 0.723               # Standard cross-entropy loss
📊 llm_loss_ratio: 0.856         # Proportion of total loss
📊 unlikelihood_loss: 0.089      # Digit/coordinate suppression loss
📊 unlikelihood_loss_ratio: 0.105 # Proportion of total loss
📊 auxiliary_loss_total: 0.122   # Sum of all non-LLM losses
📊 auxiliary_loss_ratio: 0.144   # Proportion of auxiliary losses
📊 current_phase: A              # Training phase
📊 phase_a_progress: 0.25        # Phase A completion
```

### Example Training Output
```
🔄 Transitioning from Phase A to Phase B at step 400
✅ Phase transition completed - backbone unfrozen, optimizer recreated
📊 eval_identity_accuracy: 0.995
📊 eval_reverse_accuracy: 0.987
📊 eval_strictness_accuracy: 0.999
```

## 🔗 Integration with src_new

### Checkpoint Validation
```bash
# Validate final checkpoint compatibility with src_new
python src_coord_pretrain/scripts/validate_integration.py \
  src_coord_pretrain/output/checkpoint-3120

# Or validate any specific checkpoint
python src_coord_pretrain/scripts/validate_integration.py \
  src_coord_pretrain/output/checkpoint-1800
```

### Using Checkpoint in src_new
```yaml
# In src_new config, point to the final checkpoint subfolder
model_path: "/absolute/path/to/src_coord_pretrain/output/checkpoint-3120"
coordinate_tokens_enabled: true
max_coord_value: 1024
```

**Note**: Always use the specific `checkpoint-xxxx/` subfolder path, not the root output directory.

### Inference-Ready Checkpoints
The enhanced training pipeline saves inference-ready checkpoints with:
- ✅ SafeTensors format (4-6x faster loading)
- ✅ All configuration files (tokenizer, processor, coordinate mappings)
- ✅ Training metrics and validation results
- ✅ Usage instructions and compatibility info
- ✅ **Consistent folder structure**: Final checkpoint saved in `checkpoint-xxxx/` subfolder

#### Checkpoint Structure
```
src_coord_pretrain/output/
├── checkpoint-600/          # Regular training checkpoint
├── checkpoint-1200/         # Regular training checkpoint
├── checkpoint-1800/         # Regular training checkpoint
└── checkpoint-3120/         # Final inference-ready checkpoint
    ├── config.json
    ├── coordinate_config.json
    ├── coord_token_ids.json
    ├── metrics-final.json
    ├── model-*.safetensors
    ├── tokenizer files...
    └── README_INFERENCE.json
```

## 🧪 Testing

### Run Test Suite
```bash
# Run comprehensive test suite
python src_coord_pretrain/tests/test_enhanced_features.py

# Expected output:
# 🧪 Running enhanced features test suite...
# ✅ test_data_generation_reverse_mapping
# ✅ test_phase_a_config_validation
# ✅ test_unlikelihood_config_validation
# ✅ test_coordinate_token_detection
# ✅ test_strictness_validation
# 🎉 Test suite completed!
```

## 🔧 Troubleshooting

### Common Issues

#### Phase Transition Problems
```
Error: Phase transition failed
Solution: Check that phase_a_steps is reasonable (100-500 steps)
```

#### Low Accuracy Metrics
```
Issue: Identity accuracy < 99%
Solutions:
- Increase phase_a_steps for better warm-start
- Enable canonical_prompts for reduced variance
- Check unlikelihood_lambda_* weights (try 0.5-2.0)
```

#### Checkpoint Compatibility
```
Issue: src_new cannot load checkpoint
Solutions:
- Run validate_integration.py script
- Check coordinate_config.json exists
- Verify SafeTensors format is used
```

### Debug Configuration
Use `coord_bootstrap_debug.yaml` for faster testing:
- Shorter Phase A (50 steps)
- Smaller batch sizes
- More frequent logging
- Reduced dataset size

## 🔧 Recent Enhancements & Fixes

### Runtime Error Fixes (NEW)
- **Dataset Validation**: Enhanced to handle both coordinate tokens and raw numbers
- **Collator Validation**: Supports forward (`"N" → <|coord_N|>`) and reverse (`<|coord_N|> → "N"`) mapping
- **Trainer Tokenizer Access**: Robust tokenizer access with multiple fallbacks
- **Coordinate Token Ranges**: Uses known ranges from src_new analysis (151667-152691)
- **Digit Token Detection**: Tokenization-based approach for single digits

### Top-K Unlikelihood Implementation (NEW)
- **Advanced Negative Sampling**: Replaces simple suppression with intelligent Top-K selection
- **Token-Level Conflict Resolution**: Mutually exclusive masks for mixed batches
- **Configurable Parameters**: `ul_topk_noncoord: 100`, `ul_topk_coord: 100`
- **Numerical Stability**: Probability clamping and finite loss verification
- **Comprehensive Testing**: Unit tests and integration tests for all components

### Enhanced Checkpoint Structure (NEW)
- **Consistent Naming**: Final checkpoint saved in `checkpoint-xxxx/` subfolder
- **SafeTensors Format**: 4-6x faster loading compared to PyTorch format
- **Complete Configuration**: All tokenizer, processor, and coordinate mappings included
- **Validation Scripts**: Automated compatibility checking with src_new


## 🎯 Best Practices

1. **Start with Debug Config**: Test your setup with debug configuration first
2. **Monitor Phase Transition**: Watch for successful Phase A → B transition
3. **Check Metrics Early**: Identity accuracy should be >95% after Phase A
4. **Validate Integration**: Always run validation script before using in src_new
5. **Use Canonical Prompts**: Enable for Phase A to reduce variance

## Notes
- Paths in config YAML can be absolute or relative (relative paths resolved from current working directory)
- Tokenizer/model must already include `<|coord_0|>` … `<|coord_1024|>`
- Assistant‑only labels via offset mapping; `<|im_end|>` included in span
- Supports warmup_ratio and lr_scheduler_type following src_new patterns
