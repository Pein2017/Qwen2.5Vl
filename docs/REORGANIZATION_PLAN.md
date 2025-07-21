# Documentation Reorganization Plan

## Current State Analysis

### Redundancies Identified:
1. **Troubleshooting Content**: 
   - `troubleshooting.md` (comprehensive)
   - `critical-fixes-problem-catalog.md` (searchable database)
   - `legacy/critical_fixes.md` (outdated version)
   - `quick-reference/problem-solution-lookup.md` (quick reference)

2. **Architecture Documentation**:
   - `architecture-overview.md` (current)
   - `legacy/architecture.md` (outdated)
   - `architecture-appendix-a-components.md` (detailed)

3. **Configuration Guides**:
   - `configuration.md` (usage guide)
   - `configuration-reference-complete.md` (complete reference)
   - `quick-reference/config-templates.md` (templates)

4. **Getting Started Content**:
   - `getting_started.md` (comprehensive)
   - `user-journeys/new-developer-onboarding.md` (role-based)
   - `README.md` (overview)

## Proposed New Structure

```
docs/
├── README.md                           # Main entry point with navigation
├── QUICK_START.md                      # 15-minute getting started
├── 
├── core/                              # Core documentation
│   ├── architecture.md                # System architecture (consolidated)
│   ├── data-pipeline.md              # Data processing (from data_schema.md)
│   ├── coordinate-tokens.md           # Coordinate system (consolidated)
│   └── configuration.md               # Complete config guide (consolidated)
│
├── guides/                            # User guides by role
│   ├── developer-guide.md            # For developers (consolidated)
│   ├── researcher-guide.md           # For researchers
│   ├── troubleshooting.md            # Problem solving (consolidated)
│   └── deployment.md                 # Production deployment
│
├── reference/                         # Quick reference materials
│   ├── api-reference.md              # API documentation
│   ├── commands.md                   # Command cheatsheet
│   ├── config-templates.md           # Configuration templates
│   └── problem-solutions.md          # Quick problem lookup
│
├── advanced/                          # Advanced topics (keep existing)
│   ├── teacher-student.md
│   ├── performance-optimization.md
│   └── extending-system.md
│
└── archive/                          # Historical/legacy content
    ├── migration-notes.md
    ├── implementation-history.md
    └── deprecated/
```

## Consolidation Strategy

### 1. Merge Redundant Content
- Combine all troubleshooting content into single comprehensive guide
- Merge architecture documentation into unified view
- Consolidate configuration information

### 2. Create Clear Hierarchy
- **README.md**: Navigation hub with role-based entry points
- **QUICK_START.md**: Fast 15-minute onboarding
- **Core**: Essential system understanding
- **Guides**: Task-oriented documentation
- **Reference**: Quick lookup materials

### 3. Preserve Valuable Content
- Move legacy content to archive with clear migration notes
- Preserve historical implementation details
- Keep specialized advanced topics

### 4. Improve Navigation
- Clear cross-references between related sections
- Consistent formatting and structure
- Role-based entry points maintained

## Implementation Steps

1. Create new structure with consolidated content
2. Update cross-references and links
3. Move legacy content to archive
4. Update main README with new navigation
5. Validate all links and references