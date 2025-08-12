XFail Allowlist (Explicit and Documented)

Rationale: These xfails are legitimate environment-dependent checks or reflect official upstream artifacts that may be unavailable or invalid in some environments. We do not want them to fail CI runs; they gate optional capabilities.

1) unit/token_processing/test_official_tokenizer_coordinate_range.py::test_official_tokenizer_coordinate_range[Qwen2.5-VL-7B-Instruct]
   - Reason: upstream tokenizer JSON may be missing or corrupt (e.g., "Token `box` out of vocabulary") when local cache is incomplete.
   - Action: Keep xfail until the 7B cache is guaranteed present and valid.

2) unit/token_processing/test_official_tokenizer_coordinate_range.py (parametrized)
   - Skip if /data3/Qwen2.5-VL-main/model_cache/Qwen is absent.
   - Skip when coordinate tokens are absent in the base tokenizer.

