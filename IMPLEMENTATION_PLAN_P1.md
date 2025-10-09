# P1 User Stories Implementation Plan (T017-T043)

**Safety**: Commit 0518f72 created - can revert anytime
**Strategy**: TDD workflow, minimal production code changes, comprehensive testing

## Remaining Work: 27 tasks

### Phase 3: User Story 1 - Trust Region (4 tasks)
**T017**: Add diagnostic call in `completion_loss.py::_get_generation_logprobs`
- After ratio computation, call `compute_trust_region_diagnostic()`
- Store in trainer state, log warnings

**T018**: Verify generation_logps in `buffer.py::generate_and_score`
- Check tensor shape and non-zero values
- Log presence status

**T019**: TensorBoard logging in `grpo_trainer.py::train`
- Export diagnostic.to_tensorboard() after each step
- Export histogram every 10 steps

**T020**: Final test run
- Verify all 9 tests still pass

### Phase 4: User Story 2 - Multimodal (10 tasks)
**T021-T024**: Write 4 failing tests for multimodal alignment
**T025-T026**: Implement MultimodalAlignmentCheck + validator
**T027-T029**: Add validation at 3 pipeline stages (dataset, buffer, loss)
**T030**: Run tests

### Phase 5: User Story 3 - Rewards (13 tasks)
**T031-T035**: Write 5 failing tests for reward variance
**T036-T038**: Implement RewardProfile + CheckpointDiversityComparison
**T039-T040a**: Integrate with reward logging
**T041-T042**: Create diversity test script + config
**T043**: Run tests

## Execution Timeline
1. Complete US1 instrumentation (T017-T020): ~30 min
2. Complete US2 (T021-T030): ~45 min
3. Complete US3 (T031-T043): ~60 min

**Total**: ~2.5 hours for full P1 completion
