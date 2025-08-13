# System Architecture Diagrams

Note: Supplementary diagrams. Canonical facts → `AI_ASSISTANT_KB.md` and `../src_new/UNIFIED_DOCUMENTATION.md`.

**Visual representations of the Qwen2.5-VL BBU detection system architecture and data flow**

> **Note**: For high-level overview and component descriptions, see [README.md](README.md). This document focuses on detailed architectural diagrams and technical schematics.

## Production System Architecture

### Component Interaction Diagram
```mermaid
graph TD
    Config["Config\n(YAML→Dataclass)"] --> Dataset["Dataset\n(JSONL Loading)"]
    Config --> TokenProcessor["TokenProcessor\n(Coordinate Tokens)"]
    Config --> DetectionModel["DetectionModel\n(Composition Wrapper)"]
    
    Dataset --> Collator["Collator\n(Batch Processing)"]
    TokenProcessor --> ConversationProcessor["ConversationProcessor\n(Teacher-Student)"]
    
    Collator --> BBUTrainer["BBUTrainer\n(Local Loss Aggregation)"]
    ConversationProcessor --> BBUTrainer
    DetectionModel --> BBUTrainer
    
    BBUTrainer --> LossManager["LossManager\n(Multi-Component Loss)"]
    BBUTrainer --> TrainingStateManager["TrainingStateManager\n(Metrics Tracking)"]
    
    style Config fill:#e1f5fe
    style DetectionModel fill:#fff3e0
    style BBUTrainer fill:#f3e5f5
    style LossManager fill:#e8f5e8
```

### Data Flow Architecture
```mermaid
flowchart LR
    subgraph Input["Input Data"]
        JSONL["JSONL Files"]
        Images["Image Files"]
        Config["YAML Config"]
    end
    
    subgraph Processing["Data Processing"]
        Dataset["Dataset Loading"]
        TokenConv["Coordinate→Token\nConversion"]
        ChatTemplate["Chat Template\nApplication"]
        Collation["Batch Collation"]
    end
    
    subgraph Training["Training Loop"]
        Forward["Forward Pass"]
        LossComp["Dual Loss\nComputation"]
        Backprop["Backpropagation"]
        StateUpdate["State Updates"]
    end
    
    subgraph Output["Outputs"]
        Checkpoints["Model Checkpoints"]
        Metrics["Training Metrics"]
        Logs["Training Logs"]
    end
    
    Input --> Processing
    Processing --> Training
    Training --> Output
    
    Training --> Processing
```

## Training Pipeline Architecture

### 8-Step Training Flow
```mermaid
sequenceDiagram
    participant Config as Configuration
    participant Data as Dataset
    participant Proc as TokenProcessor  
    participant Model as DetectionModel
    participant Trainer as BBUTrainer
    participant Loss as LossManager
    
    Config->>Data: Load & Validate JSONL
    Data->>Proc: Raw Coordinates
    Proc->>Proc: Convert to Tokens
    Proc->>Model: Extended Vocabulary
    Model->>Model: Initialize Coordinate Tokens
    
    loop Training Loop
        Data->>Trainer: Batched Samples
        Trainer->>Model: Forward Pass
        Model->>Loss: Compute Dual Loss
        Loss->>Trainer: Loss Components
        Trainer->>Trainer: Local Aggregation
        Trainer->>Model: Backpropagation
    end
    
    Trainer->>Config: Save Checkpoint
```

### Coordinate Token System Architecture
```mermaid
graph TB
    subgraph Input["Input Coordinates"]
        BBox["bbox_2d: [x1,y1,x2,y2]"]
        Quad["quad: [x1,y1,...,x4,y4]"]
        Line["line: [x1,y1,...,xn,yn]"]
    end
    
    subgraph Processing["Token Conversion"]
        Clamp["Coordinate Clamping\n(0-1023 range)"]
        Wrap["Geometry Wrapping\n(<|box_start|>...)"]
        Convert["Token Mapping\n(coord_123 → token_id)"]
    end
    
    subgraph Output["Token Sequences"]
        BoxSeq["<|box_start|> <|coord_150|> <|coord_10|>\n<|coord_211|> <|coord_35|> <|box_end|>"]
        QuadSeq["<|quad_start|> ... <|quad_end|>"]
        LineSeq["<|line_start|> ... <|line_end|>"]
    end
    
    Input --> Processing
    Processing --> Output
```

## Loss Computation Architecture

### Dual-Loss System Diagram
```mermaid
graph TD
    subgraph Forward["Forward Pass"]
        Input["Input Tokens"] --> Model["DetectionModel"]
        Model --> Logits["Model Logits"]
    end
    
    subgraph LossComputation["Loss Computation"]
        Logits --> SpanMask["Span-based Masking"]
        SpanMask --> LLMLoss["LLM Cross-Entropy Loss"]
        
        Logits --> CoordSlice["Coordinate Token Slicing"]
        CoordSlice --> CoordLoss["Coordinate L1 Loss"]
        
        LLMLoss --> WeightedSum["Weighted Loss Combination"]
        CoordLoss --> WeightedSum
    end
    
    subgraph Output["Training Outputs"]
        WeightedSum --> TotalLoss["Total Loss"]
        TotalLoss --> Metrics["Training Metrics"]
        TotalLoss --> Gradients["Gradient Computation"]
    end
    
    style LLMLoss fill:#e1f5fe
    style CoordLoss fill:#fff3e0
    style TotalLoss fill:#f3e5f5
```

## Distributed Training Architecture

### NCCL Timeout Resolution
```mermaid
graph LR
    subgraph Rank0["Rank 0"]
        LocalLoss0["Local Loss\nComputation"]
        LocalMetrics0["Local Metrics\nAggregation"]
    end
    
    subgraph Rank1["Rank 1"]
        LocalLoss1["Local Loss\nComputation"]
        LocalMetrics1["Local Metrics\nAggregation"]
    end
    
    subgraph Rank2["Rank N"]
        LocalLossN["Local Loss\nComputation"]
        LocalMetricsN["Local Metrics\nAggregation"]
    end
    
    LocalLoss0 --> Barrier["Gradient\nSynchronization\nBarrier"]
    LocalLoss1 --> Barrier
    LocalLossN --> Barrier
    
    Barrier --> Checkpoint["Rank 0 Only\nCheckpoint Saving"]
    
    style LocalLoss0 fill:#e8f5e8
    style LocalLoss1 fill:#e8f5e8  
    style LocalLossN fill:#e8f5e8
    style Barrier fill:#fff3e0
    style Checkpoint fill:#f3e5f5
```

**Key Improvements**:
- **Local Loss Aggregation**: Eliminates NCCL timeout issues (100% success rate)
- **Rank-Aware Logging**: Prevents cross-rank log spam
- **Unified Checkpoint Management**: SafeTensors format with best-copy creation

## Data Conversion Pipeline Architecture

### Unified Processing Flow
```mermaid
graph TD
    subgraph Input["Input (ds_v2/)"]
        JSONFiles["V2 JSON Files"]
        ImageFiles["Image Files"]
        ConfigFile["Processing Config"]
    end
    
    subgraph UnifiedProcessor["UnifiedProcessor"]
        SampleProcessor["Sample Processing"]
        CoordManager["CoordinateManager\n(EXIF, Rescaling, Resize)"]
        TaxonomyProc["FlexibleTaxonomyProcessor\n(Hierarchical Descriptions)"]
        ValidationMgr["ValidationManager\n(Strict Validation)"]
    end
    
    subgraph Output["Output (data/{name}/)"]
        TrainJSONL["train.jsonl"]
        ValJSONL["val.jsonl"]
        TeacherJSONL["teacher.jsonl"]
        ProcessedImages["images/"]
        Statistics["label_vocabulary.json"]
    end
    
    Input --> UnifiedProcessor
    UnifiedProcessor --> Output
    
    CoordManager --> TaxonomyProc
    TaxonomyProc --> ValidationMgr
    ValidationMgr --> SampleProcessor
```

### Coordinate Transformation Pipeline
```mermaid
sequenceDiagram
    participant Input as Raw Coordinates
    participant EXIF as EXIF Compensation
    participant Rescale as Dimension Rescaling
    participant Resize as Smart Resize
    participant Output as Final Coordinates
    
    Input->>EXIF: Original coordinates
    EXIF->>EXIF: Apply orientation transform
    EXIF->>Rescale: EXIF-adjusted coordinates
    Rescale->>Rescale: Handle dimension mismatch
    Rescale->>Resize: Rescaled coordinates
    Resize->>Resize: Apply smart resize (MAX_PIXELS)
    Resize->>Output: Training-ready coordinates
```

## Inference Pipeline Architecture

### Production Inference Flow
```mermaid
graph TD
    subgraph Initialization["Inference Engine Init"]
        LoadModel["Load DetectionModel"]
        LoadTeachers["Load Teacher Pool"]
        ConfigValidation["Validate Configuration"]
    end
    
    subgraph Processing["Inference Processing"]
        ImageLoad["Image Loading"]
        TeacherSelection["Teacher Selection"]
        ConversationBuild["Conversation Building"]
        TokenGeneration["Token Generation"]
    end
    
    subgraph PostProcess["Post-Processing"]
        TokenParsing["Coordinate Token Parsing"]
        CoordExtraction["Coordinate Extraction"]
        ValidationCheck["Output Validation"]
    end
    
    subgraph Output["Inference Output"]
        Predictions["Object Predictions"]
        Confidence["Confidence Scores"]
        Metadata["Processing Metadata"]
    end
    
    Initialization --> Processing
    Processing --> PostProcess
    PostProcess --> Output
    
    style LoadModel fill:#fff3e0
    style TeacherSelection fill:#e1f5fe
    style CoordExtraction fill:#e8f5e8
```

**Key Features**:
- **Teacher-Guided Inference**: Replicates training pipeline teacher assignment
- **Training-Compatible Processing**: Exact data preparation pipeline match
- **Robust Error Handling**: Comprehensive validation with detailed diagnostics
- **Batch Processing**: Configurable batch sizes with automatic adjustment for teacher mode