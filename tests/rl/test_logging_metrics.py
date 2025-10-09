import logging
from types import SimpleNamespace

import pytest
import torch

from src_new.rl.grpo_trainer import BBUGRPOTrainer


class DummyStateManager:
    def __init__(self):
        self._accum = {}

    def accumulate_loss_components(self, metrics):
        self._accum.update(metrics)

    def log_training_metrics(
        self, tr_loss, grad_norm, model, start_time, learning_rate, trainer_state
    ):
        # Echo basic logs back
        return {
            "loss": float(tr_loss.item() if torch.is_tensor(tr_loss) else tr_loss),
            "grad_norm": float(
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            ),
            "learning_rate": float(learning_rate),
        }

    def reset_metrics_state(self):
        self._accum.clear()


class DummyWriter:
    def __init__(self):
        self.records = []

    def add_scalar(self, tag, scalar_value, global_step):
        self.records.append((tag, float(scalar_value), int(global_step)))

    def close(self):
        pass


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


@pytest.mark.parametrize("world_size", [1])
def test_trainer_logs_tb_scalars_and_console(caplog, world_size):
    # Build a lightweight trainer via object.__new__ to avoid heavy init
    trainer = object.__new__(BBUGRPOTrainer)
    trainer.rank = 0
    trainer.world_size = world_size
    trainer._device = torch.device("cpu")
    trainer._state_manager = DummyStateManager()
    trainer._tb_writer = DummyWriter()
    trainer._global_step = 1
    trainer._start_time = 0.0
    trainer._last_grad_norm = 1.23
    trainer.manual_cfg = SimpleNamespace(logging_steps=1, max_steps=10)
    trainer.reward_func_names = ["wrappers", "bbox_giou"]
    trainer.model = DummyModel()

    # Initialize histories used by _log_step tail
    trainer._temperature_history = []
    trainer._reward_history = []
    trainer._clip_ratio_history = []
    trainer._adv_std_history = []
    trainer._adv_max_history = []
    trainer._reward_component_history = {"wrappers": [], "bbox_giou": []}

    # Prepare a generation_result with rewards and per-reward matrix
    generation_result = {
        "rewards": torch.tensor([0.5, 0.7], dtype=torch.float32),
        "advantages": torch.tensor([0.1, -0.2], dtype=torch.float32),
        "completion_lengths": torch.tensor([5, 7], dtype=torch.float32),
        "truncated_flags": torch.tensor([0, 1], dtype=torch.float32),
        "terminated_with_eos": torch.tensor([1, 1], dtype=torch.float32),
        "rewards_per_func": torch.tensor([[0.3, 0.2], [0.7, 0.8]], dtype=torch.float32),
        "reward_names": ["wrappers", "bbox_giou"],
        "temperature": 1.1,
        "beta": 0.0,
    }

    with caplog.at_level(logging.INFO):
        # Call the logging step directly
        BBUGRPOTrainer._log_step(
            trainer,
            loss_value=2.0,
            generation_result=generation_result,
            current_lr=5e-6,
        )

    # Assert console summary was formatted
    summary = trainer._last_console_summary
    assert summary is not None
    assert (
        "[step=1 epoch=" in summary and "loss=" in summary and "grad_norm=" in summary
    )

    # Assert TB scalars were written
    tags = {t for (t, _, _) in trainer._tb_writer.records}
    # Core
    assert "train/loss" in tags
    assert "train/learning_rate" in tags
    assert "train/grad_norm" in tags
    assert "train/epoch" in tags
    # RL-specific
    assert "reward" in tags and "reward_std" in tags
    assert "temperature" in tags and "eta_minutes" in tags and "step" in tags
    # Per-reward
    assert "rewards/wrappers/mean" in tags and "rewards/bbox_giou/mean" in tags
    assert "rewards/wrappers/std" in tags and "rewards/bbox_giou/std" in tags
    # Completion stats
    assert "completions/mean_length" in tags
    assert "completions/clipped_ratio" in tags
    assert "completions/terminated_ratio" in tags
    # Advantage stats
    assert "advantages/std" in tags and "advantages/max_abs" in tags
