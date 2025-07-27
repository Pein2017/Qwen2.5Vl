# Requirements Document

## Introduction

The user has configured coordinate token-based detection training in their BBU system using `configs/base_flat_det.yaml`, but the detection losses (coordinate_loss, focal_loss, etc.) are not appearing in the training logs. The system is computing coordinate token losses internally but they're not being properly extracted and logged during training. This prevents the user from monitoring the effectiveness of their coordinate token regression training.

## Requirements

### Requirement 1

**User Story:** As a developer training a coordinate token detection model, I want to see coordinate token losses in my training logs, so that I can monitor the detection training progress.

#### Acceptance Criteria

1. WHEN coordinate tokens are enabled in config THEN coordinate token losses SHALL appear in training logs
2. WHEN coordinate token losses are computed by the model wrapper THEN they SHALL be properly extracted by the loss manager
3. WHEN training logs are generated THEN they SHALL include coordinate_loss, focal_loss, regular_loss, l1_loss, and giou_loss values
4. WHEN coordinate token losses are zero THEN they SHALL still be logged as 0.0 for visibility

### Requirement 2

**User Story:** As a developer debugging coordinate token training, I want detailed loss component breakdown, so that I can understand which parts of the detection system are learning effectively.

#### Acceptance Criteria

1. WHEN coordinate token losses are computed THEN individual components SHALL be tracked separately
2. WHEN loss logging occurs THEN coordinate losses SHALL be included in component-wise logging
3. WHEN gradient accumulation is used THEN coordinate losses SHALL be properly accumulated and averaged
4. WHEN evaluation occurs THEN coordinate losses SHALL be preserved and restored correctly

### Requirement 3

**User Story:** As a developer using the coordinate token system, I want the loss extraction to work reliably across different model configurations, so that I can trust the training metrics.

#### Acceptance Criteria

1. WHEN model outputs contain coordinate loss attributes THEN they SHALL be reliably extracted
2. WHEN coordinate tokens are disabled THEN coordinate losses SHALL default to 0.0 without errors
3. WHEN model wrapper computes losses THEN they SHALL be properly attached to model outputs
4. WHEN loss manager processes outputs THEN it SHALL handle missing coordinate loss attributes gracefully