import types

import torch

from src_new.rl.grpo_trainer import BBUGRPOTrainer


class DummyOptimizer:
    def __init__(self, lr: float = 1e-3) -> None:
        self.param_groups = [{"lr": lr}]
        self.step_calls = 0
        self.zero_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_calls += 1


class DummyAccelerator:
    def __init__(self) -> None:
        self.sync_gradients = True

    def clip_grad_norm_(self, _params, _max_norm):
        return torch.tensor(0.0)


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


def build_minimal_trainer() -> BBUGRPOTrainer:
    trainer = object.__new__(BBUGRPOTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer._device = torch.device("cpu")
    trainer.manual_cfg = types.SimpleNamespace(
        max_grad_norm=1.0, reward_average_window=5
    )
    trainer.accelerator = DummyAccelerator()
    trainer.model = DummyModel()
    trainer._global_step = 0
    trainer._temperature_scale = 1.0
    trainer._eval_enabled = False
    trainer._eval_every_steps = 0
    trainer._maybe_checkpoint = lambda *args, **kwargs: None
    trainer._log_step = lambda *args, **kwargs: None
    trainer.local_completions_per_cycle = 4
    trainer.prompts_per_cycle = 2

    # Initialize telemetry tracker
    from src_new.rl.prompt_batch_telemetry import PromptBatchTelemetryTracker

    trainer._prompt_batch_telemetry = PromptBatchTelemetryTracker(
        expected_trajectories_per_cycle=4, reward_average_window=5
    )

    trainer._reset_prompt_cycle()
    return trainer


def test_optimizer_step_executes_only_after_full_cycle():
    trainer = build_minimal_trainer()
    optimizer = DummyOptimizer()
    scheduler = types.SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [optimizer.param_groups[0]["lr"]],
    )
    trainer._cycle_prompt_counter = 2
    trainer._cycle_completion_counter = trainer.local_completions_per_cycle

    stepped = BBUGRPOTrainer._optimizer_step(
        trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        generation_result={"rewards": torch.tensor([0.0])},
        loss_value_for_log=torch.tensor(0.0),
    )

    assert stepped is True
    assert optimizer.step_calls == 1
    assert optimizer.zero_calls >= 1
    assert trainer._cycle_completion_counter == 0
    assert trainer._cycle_prompt_counter == 0
    assert trainer._global_step == 1


def test_optimizer_step_drops_incomplete_cycle():
    trainer = build_minimal_trainer()
    optimizer = DummyOptimizer()
    trainer._cycle_prompt_counter = 1
    trainer._cycle_completion_counter = 2  # below required 4 completions

    stepped = BBUGRPOTrainer._optimizer_step(
        trainer,
        optimizer=optimizer,
        scheduler=None,
        generation_result={},
        loss_value_for_log=torch.tensor(0.0),
    )

    assert stepped is False
    assert optimizer.step_calls == 0
    assert optimizer.zero_calls >= 1  # gradients cleared on drop_last
    assert trainer._cycle_completion_counter == 0
    assert trainer._cycle_prompt_counter == 0
