# /git auto — minimal: infer message, commit, push
Your job:
- Run inside Cursor with terminal access. **Do the work, don’t just print commands.**
- Read the repo diff (prefer staged; else working tree).
- **Auto-generate** a clean Conventional Commit title and a structured body (schema below).
- Produce **only one Bash code block** that stages, commits, and pushes to the branch I mention.
- If I don't mention a branch, use the current branch.

## Input
- Natural language containing an optional branch (e.g., "push to feature/mm-proj"). If absent, use `git rev-parse --abbrev-ref HEAD`.

## Commit style (the AI decides; do not ask me)
- **Title**: `type(scope): subject` (Conventional Commits; ≤ 60 chars; imperative; no period).
  - type ∈ { feat, fix, refactor, perf, test, docs, chore, build, ci, style }
  - scope: concise area inferred from paths/diff (e.g., train, data, model, multimodal, eval, infra, pipeline, serving, tooling, docs, ci, build).
- **Body schema (stable, AI-friendly)** — keep headings, omit empty sections:
