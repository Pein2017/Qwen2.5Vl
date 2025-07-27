# Implementation Plan

- [x] 1. Fix model wrapper coordinate loss attachment
  - Ensure all coordinate loss components are always attached to model outputs in `_forward_with_coordinate_tokens()`
  - Verify that `_last_*` attributes are properly set before attaching to outputs
  - Add validation that loss attributes are accessible by the loss manager
  - Add debug logging to verify loss attachment process
  - _Requirements: 1.1, 1.2, 3.2_

- [x] 2. Enhance loss manager coordinate loss extraction
  - Fix the `_extract_coordinate_losses()` method to reliably extract all coordinate loss components
  - Ensure proper updating of `_current_*` loss tracking variables for coordinate losses
  - Add robust handling for missing coordinate loss attributes with proper defaults
  - Verify coordinate losses are properly included in loss components dictionary
  - _Requirements: 1.2, 2.1, 3.1_

- [x] 3. Fix trainer coordinate loss accumulation and logging
  - Ensure coordinate loss accumulators are properly initialized in trainer constructor
  - Fix the loss accumulation logic to include all coordinate loss components
  - Ensure coordinate losses are always logged in final output, even when zero
  - Add proper averaging of coordinate losses across gradient accumulation steps
  - _Requirements: 1.3, 1.4, 2.2, 3.4_

- [x] 4. Add coordinate loss validation and debugging
  - Add comprehensive debug logging to track coordinate loss flow through the pipeline
  - Implement validation to ensure coordinate losses match between debug output and final logs
  - Add checks to verify coordinate token configuration is loaded correctly
  - Create validation that coordinate losses are being computed when expected
  - _Requirements: 2.3, 3.3, 3.4_

- [ ] 5. Test and validate the coordinate loss logging fix
  - Run training with coordinate tokens enabled and verify all losses appear in logs
  - Validate that logged loss values match the debug output values (e.g., Detection loss: 76.333801 should appear as coordinate_loss: 76.333801)
  - Test with coordinate tokens disabled to ensure graceful degradation
  - Verify that focal_loss, l1_loss, and giou_loss are logged separately without wrapper summation
  - _Requirements: 1.4, 2.4, 3.2, 3.3_