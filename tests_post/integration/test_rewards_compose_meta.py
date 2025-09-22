import math
from src_post.rewards.compose import compose_reward


def test_compose_reward_with_soft_overlong_meta():
    names = ["label_match", "soft_overlong_penalty"]
    weights = [1.0, 1.0]
    # Base: label_match returns 1.0 if pred==gt else 0. Here pred==gt
    r = compose_reward(
        gt_label="pass",
        pred_label="pass",
        summary_lines=["x"],
        reason=None,
        checklist_lines=[],
        reward_names=names,
        reward_weights=weights,
        tf_p_pass=0.9,
        tf_p_fail=0.1,
        stage_b_text="...",
        mission=None,
        meta={"stage_b_hit_max": True, "soft_overlong_penalty_weight": 0.2},
    )
    # compose divides by sum(|w|)=2. Expected (1.0 + (-0.2))/2 = 0.4
    assert abs(r - 0.4) < 1e-6
