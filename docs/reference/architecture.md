# BBU Training Pipeline Architecture

**System architecture overview for the BBU (Bounding Box Understanding) training pipeline**

## 🏗️ **System Overview**

The BBU training pipeline is built on a modular architecture that extends Qwen2.5-VL for precise object localization with coordinate token support.

```
BBU Training Pipeline
├── Configuration System    → Unified configuration management
├── Model System           → Qwen2.5-VL with coordinate token extensions
├── Data Processing        → JSONL data handling and coordinate conversion
├── Training System        → Multi-component loss and optimization
├── Coordinate Tokens      → Standard and Coordinate mode support
└── Testing Framework      → Comprehensive validation and testing
```

## 📊 **Core Components**

### **1. Configuration System (`src/config/`)**
```
src/config/
├── __init__.py           # Configuration loading and initialization
├── global_config.py      # Global configuration management
└── explicit_config.py    # Explicit configuration schema
```

**Purpose**: Centralized configuration management with validation
**Key Features**:
- Unified configuration loading from YAML files
- Type validation and default value handling
- Global configuration access throughout the system

### **2. Model System (`src/models/`)**
```
src/models/
├── model_loader.py          # Unified model, tokenizer, and processor loading
├── wrapper.py              # Model wrapper with coordinate token support
├── coordinate_handler.py    # Coordinate token handling and management
├── detection_integration.py # Detection capabilities integration
├── model_adapter.py        # Model adaptation utilities
├── loss_manager.py         # Model-level loss computation
└── patches.py              # Qwen2.5-VL compatibility patches
```

**Purpose**: Model loading, coordinate token integration, and detection capabilities
**Key Features**:
- Unified model loading for training and inference consistency
- Automatic vocabulary extension for coordinate tokens
- Embedding resize for new tokens
- Qwen2.5-VL model integration with patches
- Coordinate token handling and validation
- Detection capabilities integration

### **3. Data Processing (`src/data.py`, `src/core/`)**
```
src/
├── data.py              # Dataset and collator implementations
└── core/
    ├── data_processor.py    # High-level data processing
    └── coordinate_manager.py # Coordinate token management
```

**Purpose**: Data loading, processing, and coordinate conversion
**Key Features**:
- JSONL data format support
- Coordinate token conversion (integer ↔ token)
- Multi-geometry support (box, line, square)
- Trainer-compatible data collation

### **4. Training System (`src/training/`)**
```
src/training/
├── loss_manager.py          # Loss computation and component extraction
├── training_coordinator.py  # Multi-task training coordination
├── training_state_manager.py # Training state and metrics management
├── base_manager.py          # Common manager functionality
├── trainer.py              # Enhanced HuggingFace Trainer
├── trainer_factory.py      # Trainer creation utilities
├── callbacks.py            # Training callbacks
└── stability.py            # Training stability utilities
```

**Purpose**: Training orchestration, loss management, and monitoring
**Key Features**:
- Multi-component loss calculation (LLM + coordinate losses)
- Teacher-student learning coordination
- Training state management and metrics tracking
- Enhanced logging and monitoring
- Coordinate-aware training with span-based loss distribution

### **5. Coordinate Token System (`src/utils/tokens/`)**
```
src/utils/tokens/
├── __init__.py          # Token system exports
└── special_tokens.py    # Unified token management
```

**Purpose**: Coordinate token management and vocabulary extension
**Key Features**:
- Automatic coordinate token generation
- Token format validation
- Vocabulary extension management

## 🔄 **Data Flow Architecture**

### **Training Data Flow**
```
Raw JSONL Data
    ↓
Data Conversion (data_conversion/)
    ↓
Processed JSONL (data/)
    ↓
BBUDataset (src/data.py)
    ↓
Coordinate Processing (src/core/coordinate_manager.py)
    ↓
Data Collation (src/data.py)
    ↓
BBUTrainer (src/training/trainer.py)
    ↓
Model Training
```

### **Coordinate Token Flow**
```
Integer Coordinates [150,10,211,35]
    ↓
SimpleCoordinateManager
    ↓
Mode Detection (Standard vs Coordinate)
    ↓
Format Conversion:
    Standard:   [150,10,211,35]
    Coordinate: [<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]
    ↓
Token Integration
    ↓
Model Processing
```

## 🎯 **Dual-Mode Architecture**

### **Standard Mode Architecture**
```
Configuration: coordinate_tokens_enabled: false
    ↓
Base Tokenizer (151,665 tokens)
    ↓
Add Geometry Tokens (+4 tokens)
    ↓
Integer Coordinate Processing
    ↓
Standard Training Pipeline
```

**Characteristics**:
- Minimal vocabulary extension
- Integer coordinate format
- Production-optimized
- Stable training

### **Coordinate Mode Architecture**
```
Configuration: coordinate_tokens_enabled: true
    ↓
Base Tokenizer (151,665 tokens)
    ↓
Add Geometry Tokens (+4 tokens)
    ↓
Add Coordinate Tokens (+2048 tokens)
    ↓
Token Coordinate Processing
    ↓
Enhanced Training Pipeline
```

**Characteristics**:
- Extended vocabulary
- Token coordinate format
- Advanced sequence prediction
- Requires specific configuration

## 🔧 **Component Interactions**

### **Configuration → Model Loading**
```python
# Configuration drives model setup
config = load_config('configs/bbu_v2.yaml')
    ↓
model, tokenizer, processor = load_model_and_processor_unified(
    config.model_path, 
    for_inference=False  # Triggers embedding resize
)
    ↓
# Vocabulary automatically extended based on config.coordinate_tokens_enabled
```

### **Data Processing → Training**
```python
# Data processor creates training-ready datasets
data_processor = DataProcessor(tokenizer, processor, model, config=config)
    ↓
train_dataset, eval_dataset = data_processor.create_datasets()
data_collator = data_processor.create_data_collator()
    ↓
# Trainer uses processed data
trainer = BBUTrainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator,
    ...
)
```

## 🧪 **Testing Architecture**

### **Test Structure**
```
tests/
├── test_coordinate_tokens.py    # Coordinate token system tests
├── test_data_pipeline.py        # Data processing tests
├── test_training_components.py  # Training system tests
├── test_integration.py          # End-to-end integration tests
└── fixtures/                   # Test utilities and fixtures
    ├── config_factory.py       # Test configuration generation
    └── synthetic_data.py       # Synthetic data generation
```

### **Test Coverage**
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Configuration Tests**: Configuration validation
- **Data Tests**: Data format and processing validation

## 🔍 **Design Principles**

### **1. Modularity**
- Clear separation of concerns
- Independent component testing
- Easy component replacement

### **2. Configuration-Driven**
- Single source of truth for configuration
- Runtime behavior controlled by configuration
- Easy mode switching (Standard ↔ Coordinate)

### **3. Backward Compatibility**
- Standard mode maintains compatibility with base Qwen2.5-VL
- Minimal changes to existing workflows
- Graceful degradation when coordinate tokens unavailable

### **4. Extensibility**
- Easy addition of new geometry types
- Pluggable coordinate processing
- Extensible training components

## 🚀 **Performance Characteristics**

### **Memory Usage**
```
Standard Mode:
- Base model: ~6GB
- Vocabulary extension: ~16MB (+4 tokens)
- Total: ~6.02GB

Coordinate Mode:
- Base model: ~6GB
- Vocabulary extension: ~8GB (+2048 tokens)
- Total: ~14GB
```

### **Training Speed**
```
Standard Mode:
- Faster data processing (integer coordinates)
- Smaller vocabulary (faster forward pass)
- Recommended for production

Coordinate Mode:
- Slower data processing (token conversion)
- Larger vocabulary (slower forward pass)
- Advanced features for research
```

## 🔧 **Deployment Architecture**

### **Development Environment**
```
Local Development
├── Source Code (src/)
├── Configuration (configs/)
├── Test Data (data/)
├── Model Cache (model_cache/)
└── Checkpoints (checkpoints/)
```

### **Production Environment**
```
Production Deployment
├── Trained Model
├── Production Configuration
├── Data Processing Pipeline
├── Monitoring and Logging
└── Backup and Recovery
```

## 📚 **Related Documentation**

- [**Getting Started**](getting-started.md) - Quick setup and architecture walkthrough
- [**Configuration**](configuration.md) - Configuration system details
- [**Coordinate Tokens**](coordinate-tokens.md) - Coordinate token architecture
- [**API Reference**](api-reference.md) - Component APIs and interfaces
- [**Training**](training.md) - Training system architecture

---

**Want to extend the system?** Check [API Reference](api-reference.md) for component interfaces and extension points.
