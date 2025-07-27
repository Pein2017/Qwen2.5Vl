# Code Style and Conventions

## Code Organization
- **File Management**: Directly override existing files; copy important files to `legacy/` for reference
- **No Backward Compatibility**: Clean refactoring without legacy support
- **File Reuse**: Extend existing files rather than creating similar new ones
- **DRY Principle**: Keep codebase DRY, transparent, and consistent

## Type Safety & Validation
- **Explicit Configuration**: All hyperparameters must be defined - never use `getattribute` or `dict.get(<key>, <default>)`
- **Schema Validation**: Enforce schemas via `@dataclass`
- **Type Annotations**: Annotate every function/method
- **Fail-Fast**: No silent `try/except: pass` or bare `except`

## Coding Standards
- **Imports**: Organized by ruff with isort integration
- **Formatting**: Double quotes, 4-space indentation
- **Line Length**: Flexible (E501 ignored in ruff config)
- **Naming**: Descriptive variable and function names
- **Documentation**: Type hints preferred over verbose docstrings

## Error Handling Philosophy
- **Surface Errors**: Let exceptions bubble; use explicit `raise` for illegal states
- **No Silent Failures**: Every error must be exposed immediately
- **Logging**: `logger.debug()` only for state/info—never to suppress errors

## Configuration Style
- **Flat Configuration**: Direct access pattern via `get_config()`
- **YAML-based**: Type-safe configuration with dataclasses
- **Environment Integration**: Support for environment variables
- **Default Values**: Only universal defaults (e.g., `in_channels: int = 3`)

## Testing Approach
- **Temporal Testing**: Create tests under `./temporal` directory
- **Evidence Results**: Leave verification artifacts after task completion
- **Validation Scripts**: Use project-specific validation in `scripts/`