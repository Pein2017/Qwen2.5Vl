# Documentation Cleanup Candidates

## Outdated/Redundant Files (Safe to Archive or Remove)

### 1. Duplicate Architecture Documentation
- **`architecture.md`** - Legacy comprehensive architecture doc
- **Keep**: `architecture-overview.md` (current) + `architecture-appendix-a-components.md`
- **Action**: Archive to `legacy/` folder

### 2. Duplicate Critical Fixes Documentation  
- **`critical_fixes.md`** - Legacy comprehensive fixes doc
- **Keep**: `critical-fixes-problem-catalog.md` (current, searchable format)
- **Action**: Archive to `legacy/` folder

### 3. Fragmented Coordinate Token Documentation
- **`coordinate_loss_visibility_fix.md`** - Specific fix, now covered in problem catalog
- **`coordinate_regression_guide.md`** - Redundant with complete guide
- **`coordinate_token_system_update.md`** - Status update, now outdated
- **Keep**: `coordinate-token-system-complete-guide.md` (consolidated)
- **Action**: Archive to `legacy/` folder

### 4. Duplicate Lessons Learned
- **`lessons_learned.md`** - Legacy comprehensive doc
- **Keep**: `quick-reference/lessons-learned-quick-tips.md` (actionable format)
- **Action**: Archive to `legacy/` folder

### 5. Miscellaneous Files
- **`raw_data.txt`** - Appears to be scratch data
- **`readme.md`** - Lowercase, conflicts with main `README.md`
- **`patent_application_soft_expectation.md`** - Research doc, archive
- **Action**: Clean up or archive

## Recommended Cleanup Actions

1. **Create `legacy/` directory for archived docs**
2. **Move redundant files to avoid confusion**
3. **Update cross-references in remaining docs**
4. **Verify no important content is lost**

## Files to Keep (Current System)
- README.md (main navigation)
- quick-reference/* (fast access)
- user-journeys/* (role-based guides)  
- architecture-overview.md + architecture-appendix-a-components.md
- coordinate-token-system-complete-guide.md (consolidated)
- critical-fixes-problem-catalog.md (searchable)
- configuration.md, data_schema.md, getting_started.md, etc.