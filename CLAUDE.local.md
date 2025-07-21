# CLAUDE.local.md

Personal development guidelines and workflow preferences for this codebase.

## Development Workflow

### Planning & Execution
- **Plan**: Define goals & end-to-end steps (a→b→c) before coding
- **Execute**: Implement directly; commit iterations freely
- **Iterate**: Continue until objectives are achieved or resource limits reached

### Error Handling Standards
- **Fail-fast**: No silent `try/except: pass` or bare `except`
- **Surface Errors**: Let exceptions bubble; use explicit `raise` for illegal states
- **Logging**: `logger.debug()` only for state/info—never to suppress errors

### Code Quality Standards

#### Type Safety & Validation
- **Explicit**: All hyperparameters must be defined—**never use** `getattribute` or `dict.get(<key>, <default>)`
- **Validate**: Enforce schemas via `@dataclass`
- **Types**: Annotate every function/method

#### Code Organization
- **Concise**: Keep code, comments, and commit messages focused
- **Defaults**: Only universal defaults (e.g., `in_channels: int = 3`); otherwise require explicit args
- **Reuse**: Extend existing files; delete temp/debug files when done

### Refactoring Philosophy

#### File Management
- **Refactoring Rule**: Directly override existing files. For important files, copy/move to `legacy` for reference
- **No Backward Compatibility**: Clean refactoring without legacy support
- **File Reuse**: Try to reuse/override/merge files. Don't create new but similar files

#### Code Exploration Process
- **Review**: Depth-first traversal of modules to map data/control flow
- **Document**: For each file, note purpose, key classes/functions, inputs/outputs
- **Plan & Act**: List refactors/fixes in order and implement immediately

> **Reminder:** Favor refactoring over file duplication. Keep the codebase DRY, transparent, and consistent.

## Testing & Validation

### Testing Guidelines
- Create and run temporal tests under `./temporal` directory to keep main codebase clean
- After completing a task, leave one or few evidence results to verify task completion

### Data Annotations
- Recorded information about data annotations outline per user request

## Development Environment Notes

### Python Environment
- **Path**: Always use `/root/miniconda3/envs/ms/bin/python` directly instead of generic `python` command
- **Activation**: Avoid conda activation inconsistencies by using full path

### Network Constraints
- Located in China, cannot access foreign websites like `github` `google` `huggingface`
- Use local mirrors and cached resources when possible