# CLAUDE.local.md

## Development Workflow & Standards

### Planning & Execution
- **Plan-First**: Define goals & end-to-end steps before coding
- **Fail-Fast**: Surface all errors immediately; no silent exceptions
- **Simple Solutions**: Implement changes with minimal code impact
- **Type Safety**: Use explicit typing and validate all inputs
- **Documentation**: Update docs alongside code changes

### Code Quality Rules
- **Explicit Parameters**: All hyperparameters must be defined explicitly
- **Type Annotations**: Required for all functions/methods
- **No Silent Failures**: Never use bare `except` or `except: pass`
- **Validation**: Use `@dataclass` for schema enforcement
- **Logging**: `logger.debug()` for info only, never to suppress errors

### File Management
- **Refactoring Philosophy**: Directly override existing files; move important files to `legacy/` for reference
- **No Duplicates**: Reuse/override/merge files; don't create similar variants
- **Temporary Files**: Delete debugging/temporary files when work is complete
- **Exploration Process**: Depth-first traversal to map data/control flow before changes

### Testing & Validation
- **Temporal Tests**: Create in `./temporal` directory to keep main codebase clean
- **Evidence**: Leave verification results after completing tasks
- **No Laziness**: Find and fix root causes; avoid temporary patches

### Environment Notes
- **Python Path**: Always use `/root/miniconda3/envs/ms/bin/python` directly
- **Network Constraints**: Use local mirrors and cached resources when needed
- **Tool Preferences**: Use serena MCP for development support

## Development Process

1. **Analysis**: Thoroughly analyze the problem and codebase
2. **Planning**: Create todo list in `tasks/todo.md`
3. **Verification**: Get plan approval before implementation
4. **Implementation**: Make small, targeted changes following todo list
5. **Review**: Summarize all changes with high-level explanation
6. **Documentation**: Update relevant documentation for completed work