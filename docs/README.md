# BBU Detection System Documentation

**Fast, user-centric information retrieval system for the BBU detection codebase**

This documentation is organized for **fast information retrieval** with role-based entry points and comprehensive cross-references.

---

## 🚀 Quick Start (Choose Your Path)

### 👨‍💻 New Developer (30 minutes to productive)
**Goal**: Get up and running with training in 30 minutes
- **Start Here**: [New Developer Onboarding](user-journeys/new-developer-onboarding.md)
- **Then Use**: [API Quick Reference](quick-reference/api-core-components.md)
- **Configure With**: [Config Templates](quick-reference/config-templates.md)

### 🚨 Troubleshooter (Problem → Solution <5 minutes)
**Goal**: Fix broken training quickly
- **Start Here**: [Troubleshooter Quick Start](user-journeys/troubleshooter-quickstart.md)
- **Problem Lookup**: [Problem-Solution Database](quick-reference/problem-solution-lookup.md)
- **Critical Fixes**: [Searchable Problem Catalog](critical-fixes-problem-catalog.md)

### 🔬 Researcher/Experimenter
**Goal**: Understand theory and conduct experiments
- **Start Here**: [Researcher & Experimenter Guide](user-journeys/researcher-experimenter-guide.md)
- **Theory**: [Coordinate Token Complete Guide](coordinate-token-system-complete-guide.md)
- **Architecture**: [System Architecture Overview](architecture-overview.md)

### 🏗️ Advanced Developer
**Goal**: Extend and optimize the system
- **Start Here**: [Advanced Developer Deep Dive](user-journeys/advanced-developer-deepdive.md)
- **Architecture**: [Component Implementation Details](architecture-appendix-a-components.md)
- **Best Practices**: [Lessons Learned Quick Tips](quick-reference/lessons-learned-quick-tips.md)

---

## 📚 Documentation Structure

### 🎯 User Journeys (Role-Based Entry Points)
Role-specific guides for different types of users:

- **[New Developer Onboarding](user-journeys/new-developer-onboarding.md)** - 30-minute path to productivity
- **[Troubleshooter Quick Start](user-journeys/troubleshooter-quickstart.md)** - Problem → solution in <5 minutes
- **[Advanced Developer Deep Dive](user-journeys/advanced-developer-deepdive.md)** - System extension and optimization
- **[Researcher & Experimenter Guide](user-journeys/researcher-experimenter-guide.md)** - Theory, experiments, and research

### ⚡ Quick Reference (Fast Lookup)
Bite-sized information for immediate use:

- **[API Core Components](quick-reference/api-core-components.md)** - Essential APIs with copy-paste examples
- **[Command Cheat Sheet](quick-reference/command-cheatsheet.md)** - Common operations and commands
- **[Configuration Templates](quick-reference/config-templates.md)** - Ready-to-use training configs
- **[Problem-Solution Lookup](quick-reference/problem-solution-lookup.md)** - Symptom → fix database
- **[Lessons Learned Quick Tips](quick-reference/lessons-learned-quick-tips.md)** - Actionable development insights

### 🏗️ Architecture & Design
System understanding and technical deep dives:

- **[Architecture Overview](architecture-overview.md)** - High-level system design with navigation
- **[Component Implementation Details](architecture-appendix-a-components.md)** - Detailed API specs
- **[Source Code Reference](src-code-reference.md)** - Current codebase structure and navigation
- **[Coordinate Token Complete Guide](coordinate-token-system-complete-guide.md)** - Authoritative coordinate token reference

### 🛠️ Problem Solving & Fixes
Comprehensive troubleshooting and historical knowledge:

- **[Critical Fixes Problem Catalog](critical-fixes-problem-catalog.md)** - Searchable fix database with categories
- **[Implementation Summary](implementation_summary.md)** - Current implementation status
- **[Migration Guide](migration_guide.md)** - System evolution and migration paths

### 📈 Operations & Advanced Topics
Production deployment and specialized techniques:

- **[Runbook](runbook.md)** - Training execution and monitoring
- **[Configuration Reference](configuration-reference-complete.md)** - Complete parameter documentation
- **[Configuration Guide](configuration.md)** - Configuration usage patterns
- **[Testing Framework](testing.md)** - Validation and quality assurance
- **[Advanced Topics](advanced/)** - Specialized techniques and research

---

## 🔍 Find Information By...

### By Problem Symptom
| **Symptom** | **Quick Fix** | **Detailed Guide** |
|-------------|---------------|-------------------|
| `CUDA out of memory` | [Memory Quick Fix](quick-reference/problem-solution-lookup.md#memory--gpu-problems) | [Memory Optimization](quick-reference/lessons-learned-quick-tips.md#-performance--memory) |
| Training not learning | [Loss Quick Fix](quick-reference/problem-solution-lookup.md#bad-lossmetrics) | [Training Troubleshooting](user-journeys/troubleshooter-quickstart.md#bad-lossmetrics) |
| Data processing errors | [Data Quick Fix](quick-reference/problem-solution-lookup.md#data-issues) | [Data Pipeline Guide](quick-reference/lessons-learned-quick-tips.md#-data-pipeline) |
| Model loading failures | [Loading Quick Fix](quick-reference/problem-solution-lookup.md#model-loading) | [Model Management](architecture-appendix-a-components.md#a2-model-management-components) |
| Configuration errors | [Config Quick Fix](quick-reference/problem-solution-lookup.md#config-problems) | [Config Validation](quick-reference/config-templates.md#configuration-validation-checklist) |

### By Development Task
| **Task** | **Quick Start** | **Deep Dive** |
|----------|----------------|---------------|
| First-time setup | [New Developer Onboarding](user-journeys/new-developer-onboarding.md) | [Getting Started](getting_started.md) |
| Configure training | [Config Templates](quick-reference/config-templates.md) | [Configuration Guide](configuration.md) |
| Debug training issues | [Troubleshooter Guide](user-journeys/troubleshooter-quickstart.md) | [Critical Fixes Catalog](critical-fixes-problem-catalog.md) |
| Understand coordinate tokens | [Coordinate Token Guide](coordinate-token-system-complete-guide.md) | [Researcher Guide](user-journeys/researcher-experimenter-guide.md) |
| Extend the system | [Advanced Developer Guide](user-journeys/advanced-developer-deepdive.md) | [Architecture Deep Dive](architecture-overview.md) |
| Production deployment | [Deployment Section](coordinate-token-system-complete-guide.md#migration--deployment) | [Runbook](runbook.md) |

### By Component/Technology
| **Component** | **API Reference** | **Implementation** | **Usage Examples** |
|---------------|------------------|-------------------|-------------------|
| Model Wrapper | [Model APIs](quick-reference/api-core-components.md#-model-components) | [Model Management](architecture-appendix-a-components.md#a2-model-management-components) | [Training Setup](quick-reference/api-core-components.md#complete-training-setup) |
| Data Processing | [Data APIs](quick-reference/api-core-components.md#-core-factories--processors) | [Data Components](architecture-appendix-a-components.md#a3-data-processing-components) | [Data Pipeline](user-journeys/new-developer-onboarding.md#-phase-3-hands-on-training-10-minutes) |
| Training System | [Training APIs](quick-reference/api-core-components.md#-training-orchestration) | [Training Components](architecture-appendix-a-components.md#a4-training-system-components) | [Training Example](coordinate-token-system-complete-guide.md#complete-training-script) |
| Coordinate Tokens | [Coordinate APIs](quick-reference/api-core-components.md#-coordinate-token-system) | [Coordinate System](architecture-appendix-a-components.md#a5-coordinate-token-system-components) | [Coordinate Usage](coordinate-token-system-complete-guide.md#usage-guide) |

---

## 📖 Reading Paths for Different Goals

### 🎯 **Goal: Get Training Working (Fast Track)**
1. [New Developer Onboarding](user-journeys/new-developer-onboarding.md) (30 min)
2. [Config Templates](quick-reference/config-templates.md) (5 min)
3. [Command Cheat Sheet](quick-reference/command-cheatsheet.md) (reference)
4. If issues → [Problem-Solution Lookup](quick-reference/problem-solution-lookup.md)

### 🔬 **Goal: Understand the Innovation**
1. [Architecture Overview](architecture-overview.md) (15 min)
2. [Coordinate Token Complete Guide](coordinate-token-system-complete-guide.md) (45 min)
3. [Researcher Guide](user-journeys/researcher-experimenter-guide.md) (30 min)
4. [Advanced Topics](advanced/) (varies)

### 🛠️ **Goal: Fix Broken Training**
1. [Troubleshooter Quick Start](user-journeys/troubleshooter-quickstart.md) (5 min)
2. [Problem-Solution Lookup](quick-reference/problem-solution-lookup.md) (lookup)
3. If not solved → [Critical Fixes Catalog](critical-fixes-problem-catalog.md)
4. [Lessons Learned Tips](quick-reference/lessons-learned-quick-tips.md) (prevention)

### 🏗️ **Goal: Extend the System**
1. [Advanced Developer Deep Dive](user-journeys/advanced-developer-deepdive.md) (45 min)
2. [Component Implementation Details](architecture-appendix-a-components.md) (30 min)
3. [Lessons Learned Tips](quick-reference/lessons-learned-quick-tips.md) (best practices)
4. [API Reference](quick-reference/api-core-components.md) (implementation)

### 📊 **Goal: Production Deployment**
1. [Deployment Guide](coordinate-token-system-complete-guide.md#migration--deployment) (20 min)
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
- ✅ **Coordinate Tokens**: Authoritative consolidated guide
- ✅ **Problem Solving**: Searchable database with categories

### Quick Access Validation
**Test your path**: Can you find the answer to these questions in <2 minutes?
- How to start training? → [New Developer Onboarding](user-journeys/new-developer-onboarding.md)
- Training crashed with memory error? → [Memory Quick Fix](quick-reference/problem-solution-lookup.md#memory--gpu-problems)
- How do coordinate tokens work? → [Coordinate Token Guide](coordinate-token-system-complete-guide.md#mathematical-foundations)
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

**Quick Links**: [🚀 Start Training](user-journeys/new-developer-onboarding.md) | [🚨 Fix Issues](user-journeys/troubleshooter-quickstart.md) | [🔬 Research](user-journeys/researcher-experimenter-guide.md) | [🏗️ Extend](user-journeys/advanced-developer-deepdive.md)