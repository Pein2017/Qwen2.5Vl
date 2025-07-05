## AI Assistant Codebase Guidelines

### Environment
- `conda activate ms`
- Located in China, cannot access to foreign website like `github` `google` `huggingface`

### Workflow

* **Plan**: Define goals & end-to-end steps (a→b→c) before coding.
* **Execute**: Implement directly; commit iterations freely.
* **Iterate**: Continue until objectives are achieved or resource limits reached.

### Error Handling

* **Fail-fast**: No silent `try/except: pass` or bare `except`.
* **Surface Errors**: Let exceptions bubble; use explicit `raise` for illegal states.
* **Logging**: `logger.debug()` only for state/info—never to suppress errors.

### Hyperparameters & Attributes

* **Explicit**: All hyperparameters must be defined—**never use** `getattribute` or `dict.get(<key>, <default>)`.
* **Validate**: Enforce schemas via `@dataclass`.

### Code Exploration & Refactoring

* **Review**: Depth-first traversal of modules to map data/control flow.
* **Document**: For each file, note purpose, key classes/functions, inputs/outputs.
* **Plan & Act**: List refactors/fixes in order and implement immediately.
* **Refactoring Rule**: Directly override existing files. For important files, copy/move to `legacy` for reference. No fallback or backward compatibility design.

### Code Quality

* **Concise**: Keep code, comments, and commit messages focused.
* **Types**: Annotate every function/method.
* **Defaults**: Only universal defaults (e.g., `in_channels: int = 3`); otherwise require explicit args.
* **Reuse**: Extend existing files; delete temp/debug files when done.

> **Reminder:** Favor refactoring over file duplication. Keep the codebase DRY, transparent, and consistent.

### Data Annotations

* Recorded information about data annotations outline per user request.

### Testing Guidelines

* Create and run temporal tests under `./temporal` directory to keep main codebase clean
* After completing a task, leave one or few evidence results to verify task completion