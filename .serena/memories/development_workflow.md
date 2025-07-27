# Development Workflow

## Planning & Execution Philosophy
- **Plan**: Define goals & end-to-end steps (a→b→c) before coding
- **Execute**: Implement directly; commit iterations freely  
- **Iterate**: Continue until objectives are achieved or resource limits reached

## Error Handling Standards
- **Fail-fast**: No silent `try/except: pass` or bare `except`
- **Surface Errors**: Let exceptions bubble; use explicit `raise` for illegal states
- **Logging**: `logger.debug()` only for state/info—never to suppress errors

## Code Quality Standards

### Type Safety & Validation
- **Explicit**: All hyperparameters must be defined—**never use** `getattribute` or `dict.get(<key>, <default>)`
- **Validate**: Enforce schemas via `@dataclass`
- **Types**: Annotate every function/method

### File Management
- **Refactoring Rule**: Directly override existing files. For important files, copy/move to `legacy` for reference
- **No Backward Compatibility**: Clean refactoring without legacy support
- **File Reuse**: Try to reuse/override/merge files. Don't create new but similar files

## Testing Guidelines
- Create and run temporal tests under `./temporal` directory to keep main codebase clean
- After completing a task, leave one or few evidence results to verify task completion