# Qwen2.5-VL BBU Detection Test Suite

This directory contains fast, deterministic unit tests and higher-level integration tests for the Qwen2.5‑VL BBU detection project. Tests follow the fail‑fast principles defined in `.augment-guidelines`.

## Structure and Organization

- Unit tests (fast, isolated):
  - src_new/tests/unit/**
  - Focus on single modules or functions: tokenizer extension, coordinate converters, loss manager, masking, span identification
  - Use synthetic fixtures and minimal mocks; no I/O or network
- Integration tests (slower, real components allowed):
  - src_new/tests/integration/**
  - Validate end‑to‑end flows: conversation processing, inference preparation, packed collator, “real data” adapters
  - Prefer local cached models/tokenizers; still avoid network and heavy compute
- Shared fixtures/utilities:
  - src_new/tests/conftest.py (common fixtures)
  - src_new/tests/fixtures/** (mock/real component helpers)

## How to Run Tests (Correct Python Environment)

Always use the repository’s conda environment Python directly (do not `conda activate` in non‑interactive shells):

- Run entire suite
  - `/root/miniconda3/envs/ms/bin/python -m pytest -q src_new/tests`
- Run a subset or a single test
  - `/root/miniconda3/envs/ms/bin/python -m pytest -q src_new/tests/unit/token_processing/test_coordinate_system.py::TestCoordinateTokenSystem::test_token_processor_vocabulary_extension`

A quiet `-q` run with exit code 0 indicates success.

## Guidelines for Writing New Tests (Fail‑Fast)

- Validate inputs up front; raise on first violation with actionable messages
- Avoid hidden fallbacks: no implicit defaults, no dict.get/getattr defaults
- Keep tests fast and deterministic:
  - Use synthetic tensors and in‑memory JSON where possible
  - Avoid filesystem and network I/O unless the test explicitly targets those paths
- Small, focused assertions with clear failure messages
- Use explicit types and shapes; assert shapes/dtypes and key invariants

## Coordinate Tokenization: Tokenizer Extension Requirement

When testing any coordinate‑enabled path (coordinate_tokens_enabled=True):

1. Extend the tokenizer vocabulary before constructing models:
   ```python
   from src_new.processing.token_processor import TokenConfig, TokenProcessor
   tokenizer = TokenProcessor(
       TokenConfig(coordinate_tokens_enabled=True, max_coord_value=1024)
   ).extend_tokenizer_vocabulary(tokenizer)
   ```
2. Construct the DetectionModel (it will resize embeddings and validate alignment).
3. Alternatively, if the test does NOT target coordinate functionality, set `coordinate_tokens_enabled=False` in the test config to bypass extension.

Notes:
- The extension adds exactly 2 line tokens and (max_coord_value + 1) coordinate tokens.
- For the standard config below, max_coord_value is 1024 (total +1027 tokens).

## Standard Configuration

All tests that load a real configuration must use:

- `/data3/Qwen2.5-VL-main/configs/bbu_v2/coord_aux.yaml`

If specific paths (train/val JSONL, teacher pool) are required, create temporary files in fixtures and override the config accordingly (see `real_components.py`).

## Fixture Scoping and Quality Standards

- Define reusable fixtures at module or session scope when multiple classes need them
- Avoid class‑scoped fixtures that are used by other classes in the same module
- Prefer synthetic data in unit tests; reserve real components for integration tests
- Keep integration tests bounded: use small samples, local cache only, and strict assertions

## Helper Fixtures and Utilities (Recommendations)

To reduce duplication and keep tests robust:
- Provide a shared `extended_tokenizer` fixture pattern that extends a provided tokenizer with coordinate tokens
- Provide a `real_test_config` fixture that loads the standard config and overrides data paths to temporary directories
- Provide synthetic image generation utilities for image‑dependent tests

## Continuous Maintenance

- Validate the full test suite regularly with the standard config:
  - `/root/miniconda3/envs/ms/bin/python -m pytest -q src_new/tests`
- Document common pitfalls:
  - Forgetting to extend tokenizer before constructing DetectionModel in coordinate mode
  - Using class‑scoped fixtures across multiple classes (scope mismatch)
  - Hardcoding paths to deprecated debug configs
- When core APIs change, update tests concurrently:
  - Keep tests aligned to new interfaces and invariants
  - Prefer adding small helper fixtures to ease transitions
