import pytest

from src_post.config import load_and_validate_config


def _minimal_cfg(**overrides):
    # Provide a minimal valid config stub for load_and_validate_config
    base = {
        "checkpoint": "outputs/7B-full-retrained/phase_3/9-12-phase_3-tok_k_8-vis_k_6/best-1000-eval_loss0.4479",
        "processor": "outputs/7B-full-retrained/phase_3/9-12-phase_3-tok_k_8-vis_k_6/best-1000-eval_loss0.4479",
        "output_dir": "output-post/test",
        "train_data_dir": "group_data/bbu_scene_2.0_order/BBU接地线检查",
        "device": "cuda",
        "seed": 1,
        "temperature": 0.5,
        "top_p": 0.5,
        "max_new_tokens_stage_a": 16,
        "max_new_tokens_stage_b": 32,
        "mask_geometry_tokens": False,
        "mask_coordinate_tokens": False,
        "sanitize_stage_a": False,
        "K_B": 2,
        "K_A": 2,
        "K_set": 1,
        "adv_clip": 1.0,
        "length_norm": True,
        "train": True,
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
    }
    base.update(overrides)
    return base


def test_enable_clipped_grpo_requires_epsilon_and_loss(tmp_path, monkeypatch):
    cfg = _minimal_cfg(enable_clipped_grpo=True)
    with pytest.raises(ValueError):
        load_and_validate_config(_write_yaml(tmp_path, cfg))
    cfg = _minimal_cfg(enable_clipped_grpo=True, epsilon_low=0.2, loss_type_stage_b="grpo")
    # should pass validation now (paths may still fail if not present on FS)
    p = _write_yaml(tmp_path, cfg)
    try:
        load_and_validate_config(p)
    except FileNotFoundError:
        pass


def test_entropy_mask_requires_exactly_one_threshold(tmp_path):
    cfg = _minimal_cfg(enable_entropy_mask_stage_b=True)
    with pytest.raises(ValueError):
        load_and_validate_config(_write_yaml(tmp_path, cfg))
    cfg = _minimal_cfg(enable_entropy_mask_stage_b=True, entropy_top_quantile_stage_b=0.2)
    p = _write_yaml(tmp_path, cfg)
    try:
        load_and_validate_config(p)
    except FileNotFoundError:
        pass


def test_soft_overlong_penalty_requires_weight(tmp_path):
    cfg = _minimal_cfg(soft_overlong_penalty_enabled=True)
    with pytest.raises(ValueError):
        load_and_validate_config(_write_yaml(tmp_path, cfg))
    cfg = _minimal_cfg(soft_overlong_penalty_enabled=True, soft_overlong_penalty_weight=0.1)
    p = _write_yaml(tmp_path, cfg)
    try:
        load_and_validate_config(p)
    except FileNotFoundError:
        pass


# helpers

def _write_yaml(tmp_path, data):
    import yaml, os
    p = tmp_path / "cfg.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    # Ensure checkpoint/processor paths exist check won't block unit; create dirs
    for k in ("checkpoint", "processor", "train_data_dir"):
        v = data.get(k)
        if isinstance(v, str):
            d = v if os.path.splitext(v)[1] == '' else os.path.dirname(v)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
    return str(p)
