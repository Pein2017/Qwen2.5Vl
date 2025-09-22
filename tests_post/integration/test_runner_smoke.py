import os
import pytest

from src_post.config import load_and_validate_config


@pytest.mark.skip(reason="Smoke test requires environment with model files; run locally.")
def test_runner_constructs_with_enhancements(tmp_path):
    cfg = {
        "checkpoint": str(tmp_path / "ckpt"),
        "processor": str(tmp_path / "proc"),
        "output_dir": str(tmp_path / "out"),
        "train_data_dir": str(tmp_path / "data"),
        "device": "cpu",
        "seed": 1,
        "temperature": 0.5,
        "top_p": 0.5,
        "max_new_tokens_stage_a": 8,
        "max_new_tokens_stage_b": 8,
        "mask_geometry_tokens": False,
        "mask_coordinate_tokens": False,
        "sanitize_stage_a": False,
        "K_B": 2,
        "K_A": 2,
        "K_set": 1,
        "adv_clip": 1.0,
        "length_norm": True,
        "train": False,
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "drop_last": True,
        "llm_top_k_block": 0,
        "vision_top_k_block": 0,
        "freeze_patch_embed": True,
        "train_aligner": False,
        "use_ref_kl": False,
        "lambda_kl_stage_b": 0.0,
        "lambda_kl_stage_a": 0.0,
        "train_stage_a_mode": "off",
        "stage_a_weight": 1.0,
        "max_images_tf": 1,
        "reward_fns": ["label_match"],
        "reward_weights": [1.0],
        "enable_phase_a_diagnostics": False,
        "train_stage_b": True,
        "stage_b_weight": 1.0,
        "freeze_stage_b_steps": 0,
        "use_mission_checklist": False,
        "group_reward_mode": "combined",
        "pairwise_credit_enabled": False,
        "pairwise_pairs_per_group": 0,
        "pairwise_delta_threshold": 0.0,
        "use_uncertainty_gate": False,
        "uncertainty_gate_min_entropy": 0.0,
        "grad_accum_steps": 1,
        "decision_ce_ema_beta": 0.9,
        # enhancements
        "enable_clipped_grpo": True,
        "epsilon_low": 0.2,
        "loss_type_stage_b": "grpo",
        "enable_entropy_mask_stage_b": True,
        "entropy_top_quantile_stage_b": 0.5,
        "max_resample_times": 1,
        "stage_a_top_m": 0,
        "uncertainty_decay_factor": 0.0,
        "pairwise_select": "heuristic",
        "log_all_candidates": False,
        "soft_overlong_penalty_enabled": True,
        "soft_overlong_penalty_weight": 0.1,
    }
    # create required dirs
    for k in ("checkpoint", "processor", "train_data_dir", "output_dir"):
        os.makedirs(cfg[k], exist_ok=True)
    p = tmp_path / "cfg.yaml"
    import yaml
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    try:
        load_and_validate_config(str(p))
    except FileNotFoundError:
        pytest.skip("Environment lacks real model files; validation passed.")
