# BBU Detection System Documentation

**Optimized documentation structure for fast information retrieval and efficient development**

This documentation reflects the **2025 modular architecture** with role-based entry points and streamlined navigation.

---

## 🎯 Start Here (Choose Your Path)

### 🆕 New to the Project (5 minutes)
**Goal**: Understand what this system does and how it works
- **Start Here**: [Mental Model](MENTAL_MODEL.md) - Single source of truth for understanding
- **Then**: [Project Map](PROJECT_MAP.md) - Navigate the codebase
- **Quick Start**: [5-Minute Overview](quick-start/README.md)

### 👨‍💻 Ready to Code (15 minutes)
**Goal**: Get up and running with training
- **Environment Setup**: [Setup Guide](quick-start/setup.md)
- **First Training**: [Complete Training Setup](quick-start/first-training.md)
- **Coordinate Tokens**: [Quick Reference](guides/coordinate-token-quick-reference.md) - Choose Standard vs Coordinate mode
- **Common Issues**: [Top 5 Problems + Fixes](quick-start/common-issues.md)

### 🚨 Need to Fix Something (2 minutes)
**Goal**: Solve problems quickly
- **Decision Tree**: [Choose the Right Solution](DECISION_TREES.md#i-have-a-problem)
- **Quick Fixes**: [Common Issues](quick-start/common-issues.md)
- **Full Troubleshooting**: [Complete Problem Guide](reference/troubleshooting.md)

### 🔬 Want to Understand/Extend (30 minutes)
**Goal**: Deep understanding and system extension
- **Architecture**: [Complete System Architecture](ARCHITECTURE.md)
- **Components**: [Detailed Component Docs](components/)
- **Workflows**: [End-to-End Processes](workflows/)
- **API Reference**: [Complete API Guide](reference/api.md)

---

## 📚 Documentation Structure

### 🧠 Foundation Documents
**Start with these to understand the system**:

- **[Mental Model](MENTAL_MODEL.md)** - What this system does and how (5 min read)
- **[Architecture](ARCHITECTURE.md)** - Complete system architecture (15 min read)

### 📚 New Documentation Structure

#### 📖 Guides (`guides/`)
User-focused guides for common tasks and workflows:
- **[Differential Learning Rates](guides/DIFFERENTIAL_LEARNING_RATES.md)** - Complete guide to differential learning rates for coordinate tokens

#### 🔧 Implementation (`implementation/`)
Technical implementation details and improvements:
- **[Improvements Summary](implementation/IMPROVEMENTS_SUMMARY.md)** - Comprehensive analysis and improvements made to the training framework

#### 📋 Reference (`reference/`)
Technical reference materials and API documentation

#### 🔍 Troubleshooting (`troubleshooting/`)
Common issues, debugging guides, and solutions

### 🧪 Test Files (`../temporal/tests/`)
Test files are organized in `/temporal/tests/` for easy cleanup:
- **[Differential Learning Rate Tests](../temporal/tests/test_differential_lr.py)** - Comprehensive validation of differential learning rates implementation
- **[Project Map](PROJECT_MAP.md)** - File purposes and navigation (quick reference)
- **[Decision Trees](DECISION_TREES.md)** - Choose the right approach for your task

### 🚀 Quick Start Guides
**Get productive fast**:

- **[5-Minute Overview](quick-start/README.md)** - Fastest path to understanding
- **[Environment Setup](quick-start/setup.md)** - Detailed setup instructions
- **[First Training](quick-start/first-training.md)** - Complete training walkthrough
- **[Common Issues](quick-start/common-issues.md)** - Top 5 problems + quick fixes

### 🔧 Component Documentation
**Deep dive into system components**:

- **[Training System](components/training-system.md)** - BBUTrainer, Coordinator, LossManager
- **[Model System](components/model-system.md)** - ModelLoader, Wrapper, Patches
- **[Data Pipeline](components/data-pipeline.md)** - 5-stage processing system
- **[Configuration](components/configuration.md)** - DirectConfig system

### 🔄 Workflow Guides
**End-to-end processes**:

- **[Data Processing](workflows/data-processing.md)** - Raw data → training data
- **[Training](workflows/training.md)** - Training data → trained model
- **[Inference](workflows/inference.md)** - Trained model → detection results

### 📖 Reference Documentation
**Quick lookup and comprehensive guides**:

- **[API Reference](reference/api.md)** - Complete API documentation
- **[Commands](reference/commands.md)** - All command-line operations
- **[Troubleshooting](reference/troubleshooting.md)** - Comprehensive problem solving

### 🔬 Advanced Topics
**Specialized techniques and deep technical details**:

- **[Advanced Index](advanced/readme.md)** - Overview of advanced topics
- **[PEFT Adapter](advanced/peft_adapter.md)** - Parameter-efficient fine-tuning
- **[Teacher-Student Learning](advanced/teacher_student.md)** - Multi-task learning
- **[Collator Notes](advanced/collator_notes.md)** - Data collation internals

### 📦 Archive
**Historical documentation**:

- **[Archive](archive/README.md)** - Legacy documentation from pre-2025 architecture

## 🎯 System Highlights

### What Makes This System Special

1. **Dual-Mode Coordinate System**:
   - **Standard Mode**: Integer coordinates `[150,10,211,35]` with minimal vocabulary extension
   - **Coordinate Mode**: Token-based coordinates `[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]` for sequence prediction
2. **Multi-Geometry Support**: bbox, square (quadrilateral), and line geometries with proper token wrapping
3. **Object-Oriented Training**: Train on specific equipment types or combinations
4. **Teacher-Student Learning**: High-quality demonstrations + model predictions
5. **Modular Architecture**: Clean separation of concerns, easy to extend and debug

### 🎯 Coordinate Token System

**Two distinct modes for different use cases:**

**Standard Mode** (Recommended):
```yaml
coordinate_tokens_enabled: false  # Uses integer coordinates
```
- ✅ Minimal vocabulary extension (+4 geometry tokens)
- ✅ Compatible with pretrained weights
- ✅ **Production ready** - Fully stable and tested
- ✅ Format: `"<|object_ref_start|>desc:BBU设备<|object_ref_end|>,<|box_start|>[150,10,211,35]<|box_end|>"`

**Coordinate Mode** (Advanced):
```yaml
coordinate_tokens_enabled: true   # Uses coordinate tokens
```
- ✅ Extended vocabulary (+2052 tokens)
- ✅ Sequence-based coordinate prediction
- ⚠️ **Known limitation:** HuggingFace trainer compatibility issue
- ✅ Format: `"<|object_ref_start|>desc:BBU设备<|object_ref_end|>,<|box_start|>[<|coord_150|>,<|coord_10|>,<|coord_211|>,<|coord_35|>]<|box_end|>"`

**Quick Start:**
- **Complete Guide**: [Coordinate Token System](core/coordinate-token-system.md)
- **Quick Reference**: [Coordinate Token Quick Reference](guides/coordinate-token-quick-reference.md)
- **Configuration**: [Configuration Guide](core/configuration.md#coordinate-token-configuration)
6. **Chinese BBU Domain**: Specialized for Chinese BBU equipment detection

### Key Innovation
```
Traditional: "BBU设备" + regression head → [10, 20, 100, 200]
Our System: "BBU设备: <coord_10><coord_20><coord_100><coord_200>"
```

---

## 🧭 Navigation Guide

### 🆕 First Time Here?
1. **[Mental Model](MENTAL_MODEL.md)** (5 min) - Understand what this system does
2. **[Quick Start](quick-start/README.md)** (15 min) - Get up and running
3. **[First Training](quick-start/first-training.md)** (30 min) - Complete walkthrough

### 🎯 Need Something Specific?
- **Train a model** → [Decision Trees](DECISION_TREES.md#i-want-to-train-a-model)
- **Process data** → [Data Processing Workflow](workflows/data-processing.md)
- **Run inference** → [Inference Workflow](workflows/inference.md)
- **Fix a problem** → [Common Issues](quick-start/common-issues.md)
- **Understand architecture** → [Architecture Guide](ARCHITECTURE.md)
- **Extend the system** → [Component Documentation](components/)
- **Find a command** → [Commands Reference](reference/commands.md)
- **Use an API** → [API Reference](reference/api.md)

### ⏱️ Time-Based Paths
- **5 minutes**: [Mental Model](MENTAL_MODEL.md) - Core concepts
- **15 minutes**: [Quick Start](quick-start/README.md) - Basic usage
- **30 minutes**: [First Training](quick-start/first-training.md) - Complete setup
- **1 hour**: [Architecture](ARCHITECTURE.md) + [Components](components/) - Deep understanding
- **Half day**: [Workflows](workflows/) + [Advanced Topics](advanced/) - Expert level

---

**🚀 Ready to start?** Pick a path above based on your needs and time available.

**🤔 Not sure where to go?** Start with [Decision Trees](DECISION_TREES.md) to find the right approach.

### 🛠️ **Goal: Fix Broken Training**
1. [Troubleshooter Quick Start](user-journeys/troubleshooter-quickstart.md) (5 min)
2. [Problem-Solution Lookup](quick-reference/problem-solution-lookup.md) (lookup)
3. If not solved → [Critical Fixes Catalog](critical-fixes-problem-catalog.md)
4. [Lessons Learned Tips](quick-reference/lessons-learned-quick-tips.md) (prevention)

### 🏗️ **Goal: Extend the System**
1. [Advanced Developer Deep Dive](user-journeys/advanced-developer-deepdive.md) (45 min)
2. [Component Implementation Details](architecture-appendix-a-components.md) (30 min)
3. [Modular Architecture](modular-architecture.md) (30 min)
4. [API Reference](quick-reference/api-core-components.md) (implementation)

### 📊 **Goal: Production Deployment**
1. [V2 Migration Complete](V2_MIGRATION_COMPLETE.md#usage-examples) (20 min)
2. [Configuration Guide](configuration.md) (15 min)
3. [Runbook](runbook.md) (monitoring)
4. [Testing Framework](testing.md) (validation)

---

## 🧭 Navigation Conventions

### Document Cross-References
All documents use consistent cross-reference patterns:

- **← Back to [Parent Document]** - Navigate to parent/overview
- **→ Next: [Related Topic]** - Continue to related information  
- **See also: [Reference]** - Additional related information
- **Deep dive: [Detail Document]** - More detailed information

### Information Hierarchy
- **Overview** → High-level understanding
- **Quick Reference** → Immediate actionable information
- **Detailed Guide** → Comprehensive implementation details
- **Appendices** → Technical specifications and deep dives

### Priority Indicators
- **🔴 Critical** - Must read for basic functionality
- **🟠 Important** - Should read for effective development
- **🟡 Optional** - Nice to know for optimization
- **🔵 Advanced** - For system extension and research

---

## 📊 Documentation Health Dashboard

### Coverage Status  
- ✅ **User Onboarding**: Complete with 30-minute fast track
- ✅ **Troubleshooting**: Comprehensive problem → solution lookup  
- ✅ **API Reference**: All core components documented with examples
- ✅ **Configuration**: Templates and validation for all scenarios
- ✅ **Architecture**: Overview + detailed appendices structure
- ✅ **V2 Data Pipeline**: Complete migration to object-oriented training system **← UPDATED**
- ✅ **Token System**: Simple token approach documentation **← UPDATED**
- ✅ **Problem Solving**: Searchable database with categories
- ✅ **Modular Architecture**: Documentation of refactored component design **← NEW**

### Quick Access Validation
**Test your path**: Can you find the answer to these questions in <2 minutes?
- How to start training? → [New Developer Onboarding](user-journeys/new-developer-onboarding.md)
- Training crashed with memory error? → [Memory Quick Fix](quick-reference/problem-solution-lookup.md#memory--gpu-problems)
- How do simple tokens work? → [Token Validation System](token_validation_system.md#simple-token-approach)
- What config for production? → [Config Templates](quick-reference/config-templates.md#coordinate-token-training-recommended)
- How to extend the model? → [Advanced Developer Guide](user-journeys/advanced-developer-deepdive.md#system-extensions)

---

## 🔄 Maintenance & Updates

### Keeping Documentation Current
- **User Journey Updates**: Reflect any changes to onboarding process
- **API Reference Updates**: Keep in sync with `src/` implementations
- **Problem Database Updates**: Add new fixes and solutions as discovered
- **Cross-Reference Validation**: Ensure all links remain valid

### Feedback & Improvement
- **User Experience**: Test navigation paths with new developers
- **Information Retrieval**: Measure time-to-answer for common questions
- **Content Quality**: Validate technical accuracy with code changes
- **Coverage Gaps**: Identify and fill missing information areas

---

**💡 Key Design Principle**: This documentation system prioritizes **time-to-information** over comprehensive coverage. Every piece of information should be findable within 2 minutes for common queries and 5 minutes for complex problems.

**🎯 Success Metric**: A new developer should be able to train their first model within 30 minutes of accessing this documentation, and troubleshooters should resolve issues in under 5 minutes.

---

**Quick Links**: [🚀 Start Training](user-journeys/new-developer-onboarding.md) | [🚨 Fix Issues](user-journeys/troubleshooter-quickstart.md) | [🔬 V2 Migration](V2_MIGRATION_COMPLETE.md) | [🏗️ Extend](user-journeys/advanced-developer-deepdive.md)