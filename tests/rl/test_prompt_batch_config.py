from pathlib import Path
from typing import Any, Dict

import pytest

from src_new.rl.runner import _load_yaml
from src_new.config.rl_config_v2 import ConfigValidationError, RLConfig
from src_new.rl.grpo_trainer import BBUGRPOTrainer


def _load_debug_config_dict() -> Dict[str, Any]:
    """Load the canonical debug configuration used for smoke runs."""
    cfg_path = Path("configs/dense_rl/debug.yaml")
    assert cfg_path.exists(), "Expected configs/dense_rl/debug.yaml to exist"
    return _load_yaml(str(cfg_path))


def test_sampling_section_requires_explicit_keys() -> None:
    """YAML configs must explicitly define all sampling keys; no silent defaults."""
    cfg_dict = _load_debug_config_dict()
    sampling = cfg_dict.pop("sampling")

    # Entire sampling section missing
    with pytest.raises(ConfigValidationError):
        RLConfig.from_yaml_dict(cfg_dict)

    # Missing prompt_batch_size
    cfg_dict["sampling"] = dict(sampling)
    cfg_dict["sampling"].pop("prompt_batch_size")
    with pytest.raises(ConfigValidationError):
        RLConfig.from_yaml_dict(cfg_dict)

    # Missing sample_k
    cfg_dict["sampling"] = dict(sampling)
    cfg_dict["sampling"].pop("sample_k")
    with pytest.raises(ConfigValidationError):
        RLConfig.from_yaml_dict(cfg_dict)

    # Missing sample_k_per_rank
    cfg_dict["sampling"] = dict(sampling)
    cfg_dict["sampling"].pop("sample_k_per_rank")
    with pytest.raises(ConfigValidationError):
        RLConfig.from_yaml_dict(cfg_dict)


def test_sampling_values_round_trip_from_yaml() -> None:
    """Ensure prompt_batch_size and sample_k are surfaced from YAML config."""
    cfg_dict = _load_debug_config_dict()
    rl_config = RLConfig.from_yaml_dict(cfg_dict)

    assert rl_config.sampling.prompt_batch_size > 0
    assert rl_config.sampling.sample_k > 0
    assert isinstance(rl_config.sampling.sample_k_per_rank, bool)
    assert rl_config.sampling.reward_average_window > 0


def test_manual_trainer_config_respects_prompt_batch_product() -> None:
    """Manual trainer config should compute trajectories_per_cycle from sampling."""
    cfg_dict = _load_debug_config_dict()
    rl_config = RLConfig.from_yaml_dict(cfg_dict)

    class _DummyDataset:
        def __len__(self) -> int:
            return 128

    manual_cfg = BBUGRPOTrainer._build_manual_cfg(rl_config, _DummyDataset())
    assert (
        manual_cfg.trajectories_per_cycle
        == manual_cfg.prompt_batch_size * manual_cfg.sample_k
    )
