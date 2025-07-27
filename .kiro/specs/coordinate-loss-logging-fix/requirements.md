# Requirements Document

## Introduction

The coordinate token detection system is computing losses correctly (as evidenced by debug logs showing Detection loss: 76.333801, Focal loss: 0.281250, etc.), but these losses are not being properly extracted and passed to the HuggingFace trainer's logging system. The final logged output shows coordinate_loss: 0.0, focal_loss: 0.0, regular_loss: 0.0, indicating a disconnect between loss computation and loss logging.

## Requirements

### Requirement 1

**User Story:** As a developer training coordinate token detection, I want to see the actual computed coordinate losses in my training logs, so that I can monitor detection training progress.

#### Acceptance Criteria

1. WHEN coordinate losses are computed in the model wrapper THEN they SHALL be properly attached to model outputs
2. WHEN model outputs contain coordinate loss attributes THEN they SHALL be extracted by the loss manager
3. WHEN losses are accumulated and averaged THEN coordinate losses SHALL be included in the final logged metrics
4. WHEN training logs are generated THEN they SHALL show non-zero values for coordinate_loss, focal_loss, l1_loss, and giou_loss

### Requirement 2

**User Story:** As a developer debugging coordinate token training, I want separate logging for each loss component, so that I can understand which parts are contributing to the total loss.

#### Acceptance Criteria

1. WHEN coordinate losses are computed THEN focal_loss, l1_loss, and giou_loss SHALL be logged separately
2. WHEN loss accumulation occurs THEN each coordinate loss component SHALL be accumulated independently
3. WHEN losses are averaged THEN each component SHALL maintain its individual value
4. WHEN final logs are generated THEN all coordinate loss components SHALL appear with their actual computed values

### Requirement 3

**User Story:** As a developer using the coordinate token system, I want the loss logging to work reliably without requiring wrapper summation, so that I can trust the individual loss metrics.

#### Acceptance Criteria

1. WHEN coordinate losses are extracted THEN they SHALL be passed directly without additional wrapper summation
2. WHEN loss components are logged THEN they SHALL reflect the actual computed values from the model
3. WHEN the trainer logs losses THEN the coordinate loss values SHALL match the debug output values
4. WHEN gradient accumulation is used THEN coordinate losses SHALL be properly averaged across micro-batches