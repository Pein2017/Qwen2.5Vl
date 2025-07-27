# Implementation Plan

- [x] 1. Fix model wrapper coordinate loss attachment
  - Ensure all coordinate loss components are properly attached to model outputs
  - Verify that loss attributes are accessible by the loss manager
  - Add defensive checks for coordinate token configuration
  - _Requirements: 1.1, 1.2, 3.3_

- [x] 2. Enhance loss manager coordinate loss extraction
  - Fix the `_extract_coordinate_losses()` method to reliably extract all coordinate loss components
  - Add robust handling for missing coordinate loss attributes
  - Ensure coordinate losses are properly tracked in current loss components
  - _Requirements: 1.2, 2.1, 3.1_

- [x] 3. Fix trainer coordinate loss accumulation and logging
  - Ensure coordinate losses are always accumulated, even when zero
  - Fix the logging logic to always include coordinate losses in training logs
  - Add proper initialization of coordinate loss accumulators
  - _Requirements: 1.3, 2.2, 2.3_

- [x] 4. Add coordinate loss validation and debugging
  - Add logging to verify coordinate token configuration is loaded correctly
  - Add debug logging to track coordinate loss computation and extraction
  - Implement validation to ensure coordinate losses are being computed when expected
  - _Requirements: 1.4, 3.2, 3.4_

- [x] 5. Test and validate the coordinate loss visibility fix
  - Run training with coordinate tokens enabled and verify losses appear in logs
  - Test with coordinate tokens disabled to ensure graceful degradation
  - Validate that all coordinate loss components are properly logged
  - _Requirements: 1.1, 1.3, 2.1_