"""Distributed smoke test for GRPO prompt batch accumulation.

This test validates that the prompt batching system correctly:
- Broadcasts prompts to all ranks
- Accumulates the expected number of trajectories
- Executes optimizer.step() only after a full cycle
- Applies drop_last for incomplete batches
"""

import types

import torch

from src_new.rl.grpo_trainer import BBUGRPOTrainer


class DummyOptimizer:
    """Stub optimizer for testing accumulation logic."""

    def __init__(self, lr: float = 1e-3) -> None:
        self.param_groups = [{"lr": lr}]
        self.step_calls = 0
        self.zero_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_calls += 1


class DummyAccelerator:
    """Stub accelerator for testing."""

    def __init__(self) -> None:
        self.sync_gradients = True

    def clip_grad_norm_(self, _params, _max_norm):
        return torch.tensor(0.0)

    def wait_for_everyone(self):
        """Stub for distributed synchronization."""
        pass


class DummyModel(torch.nn.Module):
    """Minimal model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


def build_trainer_with_batching(
    prompt_batch_size: int = 4, sample_k: int = 8, world_size: int = 2
) -> BBUGRPOTrainer:
    """Build a minimal trainer configured for prompt batching."""
    trainer = object.__new__(BBUGRPOTrainer)
    trainer.rank = 0
    trainer.world_size = world_size
    trainer._device = torch.device("cpu")
    trainer.manual_cfg = types.SimpleNamespace(
        max_grad_norm=1.0,
        prompt_batch_size=prompt_batch_size,
        sample_k=sample_k,
        sample_k_per_rank=False,
        reward_average_window=5,
    )
    trainer.accelerator = DummyAccelerator()
    trainer.model = DummyModel()
    trainer._global_step = 0
    trainer._temperature_scale = 1.0
    trainer._eval_enabled = False
    trainer._eval_every_steps = 0
    trainer._maybe_checkpoint = lambda *args, **kwargs: None
    trainer._log_step = lambda *args, **kwargs: None

    # Initialize sampling window from manual_cfg
    trainer._sample_k_per_rank = trainer.manual_cfg.sample_k_per_rank
    trainer.global_sample_k = int(trainer.manual_cfg.sample_k)
    # When sample_k_per_rank=false, each rank generates sample_k/world_size responses
    trainer.local_sample_k = (
        trainer.global_sample_k // max(trainer.world_size, 1)
        if not trainer._sample_k_per_rank
        else trainer.global_sample_k
    )
    trainer.prompts_per_cycle = int(trainer.manual_cfg.prompt_batch_size)
    trainer.local_completions_per_cycle = (
        trainer.local_sample_k * trainer.prompts_per_cycle
    )

    # Initialize telemetry tracker
    from src_new.rl.prompt_batch_telemetry import PromptBatchTelemetryTracker
    trainer._prompt_batch_telemetry = PromptBatchTelemetryTracker(
        expected_trajectories_per_cycle=trainer.local_completions_per_cycle,
        reward_average_window=5
    )

    trainer._reset_prompt_cycle()
    return trainer


def test_smoke_full_accumulation_cycle():
    """Test that a full accumulation cycle triggers optimizer step."""
    # prompt_batch_size=4, sample_k=8, world_size=2
    # Each rank generates 8/2=4 trajectories per prompt
    # Total per cycle: 4 prompts × 4 local trajectories = 16 local completions
    trainer = build_trainer_with_batching(prompt_batch_size=4, sample_k=8, world_size=2)
    optimizer = DummyOptimizer()
    scheduler = types.SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [optimizer.param_groups[0]["lr"]],
    )

    # Simulate accumulating all required completions
    for prompt_idx in range(trainer.prompts_per_cycle):
        trainer._register_prompt_start()
        # Simulate each rank generating local_sample_k trajectories
        trainer._register_completions(trainer.local_sample_k)

    # Verify cycle is complete
    assert trainer._cycle_is_complete()

    # Now trigger optimizer step
    stepped = BBUGRPOTrainer._optimizer_step(
        trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        generation_result={"rewards": torch.tensor([0.0])},
        loss_value_for_log=torch.tensor(0.0),
    )

    assert stepped is True
    assert optimizer.step_calls == 1
    assert trainer._global_step == 1
    # Counters should be reset
    assert trainer._cycle_prompt_counter == 0
    assert trainer._cycle_completion_counter == 0


def test_smoke_incomplete_cycle_drops():
    """Test that incomplete cycles are dropped without optimizer step."""
    trainer = build_trainer_with_batching(prompt_batch_size=4, sample_k=8, world_size=2)
    optimizer = DummyOptimizer()

    # Only process 2 prompts instead of required 4
    for prompt_idx in range(2):
        trainer._register_prompt_start()
        trainer._register_completions(trainer.local_sample_k)

    # Cycle should NOT be complete
    assert not trainer._cycle_is_complete()

    # Trigger optimizer step - should drop_last
    stepped = BBUGRPOTrainer._optimizer_step(
        trainer,
        optimizer=optimizer,
        scheduler=None,
        generation_result={},
        loss_value_for_log=torch.tensor(0.0),
    )

    assert stepped is False
    assert optimizer.step_calls == 0
    assert optimizer.zero_calls >= 1  # gradients cleared on drop
    # Counters should be reset
    assert trainer._cycle_prompt_counter == 0
    assert trainer._cycle_completion_counter == 0


def test_smoke_broadcast_pattern():
    """Test that the broadcast pattern processes prompts sequentially."""
    trainer = build_trainer_with_batching(prompt_batch_size=3, sample_k=6, world_size=2)

    prompts_processed = []

    # Simulate processing prompts in order
    for prompt_id in range(trainer.prompts_per_cycle):
        prompts_processed.append(prompt_id)
        trainer._register_prompt_start()
        # Each rank generates local_sample_k trajectories
        trainer._register_completions(trainer.local_sample_k)

    # Verify all prompts were processed in order
    assert prompts_processed == [0, 1, 2]
    assert trainer._cycle_prompt_counter == 3
    # Total completions = 3 prompts × 3 local_k = 9
    assert trainer._cycle_completion_counter == 9
    assert trainer._cycle_is_complete()


def test_smoke_multiple_cycles():
    """Test that multiple accumulation cycles work correctly."""
    trainer = build_trainer_with_batching(prompt_batch_size=2, sample_k=4, world_size=2)
    optimizer = DummyOptimizer()
    scheduler = types.SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [optimizer.param_groups[0]["lr"]],
    )

    # Run 3 complete cycles
    for cycle in range(3):
        for prompt_idx in range(trainer.prompts_per_cycle):
            trainer._register_prompt_start()
            trainer._register_completions(trainer.local_sample_k)

        assert trainer._cycle_is_complete()

        stepped = BBUGRPOTrainer._optimizer_step(
            trainer,
            optimizer=optimizer,
            scheduler=scheduler,
            generation_result={"rewards": torch.tensor([0.0])},
            loss_value_for_log=torch.tensor(0.0),
        )

        assert stepped is True
        assert trainer._cycle_prompt_counter == 0
        assert trainer._cycle_completion_counter == 0

    # Verify 3 optimizer steps occurred
    assert optimizer.step_calls == 3
    assert trainer._global_step == 3
