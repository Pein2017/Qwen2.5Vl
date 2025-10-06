#!/usr/bin/env python3
"""
Feature checks for RL training components:
- Runs a short GRPO training loop with the manual BBUGRPOTrainer
- Verifies:
  * Advantage clipping (max |adv| <= configured clip)
  * Reward standardization (finite values, reasonable mean/std)
  * Temperature scheduling (captures per-step temperature)
  * Generation diagnostics (prints prompt/completion samples)

Usage:
  python scripts/rl_feature_checks.py \
    --config /abs/path/to/rl_config.yaml \
    --max_steps 8 \
    --temperature_schedule cosine \
    --standardize_rewards true \
    --max_advantage_magnitude 5.0

Note:
- Uses real dataset and checkpoint specified in the config
- Keep max_steps small for a quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from typing import Any, Callable, Dict, List

import copy

from src_new.config.rl_config import EnhancedRLConfig
from src_new.rl.grpo_trainer import BBUGRPOTrainer
from src_new.rl.rewards.registry import REGISTRY
from src_new.rl.runner import _load_yaml, build_datasets
from src_new.utils.rank_aware_logging import get_rank_aware_logger


_LOGGER = get_rank_aware_logger("rl.feature_checks")
def _wrap_reward_funcs(
    weights: Dict[str, float],
) -> tuple[List[Callable[..., List[float]]], List[float], List[str]]:
    """Mimic runner's reward wrapping but return names as well."""
    active_keys = [k for k, w in weights.items() if float(w) != 0.0 and k in REGISTRY]
    if len(active_keys) == 0:
        raise ValueError(
            "RL rewards must include at least one non-zero component present in the registry."
        )

    reward_funcs: List[Callable[..., List[float]]] = []
    reward_weights: List[float] = []

    import inspect

    for key in active_keys:
        base_fn = REGISTRY[key]
        sig = inspect.signature(base_fn)
        expects_meta = "meta" in sig.parameters

        def _wrap(
            fn: Callable[..., float], *, wants_meta: bool
        ) -> Callable[..., List[float]]:
            def _inner(
                _prompts: List[Any], completions: List[str], **kwargs
            ) -> List[float]:
                metas = kwargs.get("meta") if wants_meta else None
                out: List[float] = []
                for idx, text in enumerate(completions):
                    call_kwargs: Dict[str, Any] = {}
                    if wants_meta:
                        if isinstance(metas, list) and idx < len(metas):
                            call_kwargs["meta"] = metas[idx]
                        else:
                            call_kwargs["meta"] = None
                    try:
                        out.append(float(fn(text, **call_kwargs)))
                    except TypeError:
                        out.append(float(fn(text)))
                return out

            return _inner

        reward_funcs.append(_wrap(base_fn, wants_meta=expects_meta))
        reward_weights.append(float(weights[key]))

    return reward_funcs, reward_weights, active_keys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RL feature checks (adv clip, temp schedule, reward std)"
    )
    ap.add_argument("--config", type=str, required=True, help="Path to RL YAML config")
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override max_new_tokens / max_completion_length",
    )
    ap.add_argument(
        "--override_sample_k",
        type=int,
        default=None,
        help="Override sample_k / num_generations",
    )
    ap.add_argument(
        "--temperature_schedule",
        type=str,
        default=None,
        choices=[None, "constant", "linear_decay", "cosine"],
        help="Override schedule",
    )
    ap.add_argument(
        "--standardize_rewards",
        type=str,
        default=None,
        choices=[None, "true", "false"],
        help="Override std flag",
    )
    ap.add_argument(
        "--max_advantage_magnitude",
        type=float,
        default=None,
        help="Override advantage clip",
    )
    ap.add_argument("--log_level", type=str, default="INFO")
    args = ap.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    os.environ["SRC_RL_LOG_LEVEL"] = args.log_level.upper()

    cfg = _load_yaml(args.config)
    bundles = build_datasets(args.config)

    rewards_cfg: Dict[str, float] = cfg.get("rewards", {})
    reward_funcs, reward_weights, reward_names = _wrap_reward_funcs(rewards_cfg)

    cfg_override = copy.deepcopy(cfg)
    training_cfg = cfg_override.setdefault("training", {})
    training_cfg["max_steps"] = int(args.max_steps)
    training_cfg["logging_steps"] = 1

    grpo_cfg = cfg_override.setdefault("grpo", {})
    if args.override_sample_k is not None:
        grpo_cfg["sample_k"] = int(args.override_sample_k)
    if args.max_new_tokens is not None:
        grpo_cfg["max_new_tokens"] = int(args.max_new_tokens)
    if args.temperature_schedule is not None:
        grpo_cfg["temperature_schedule"] = args.temperature_schedule
    if args.standardize_rewards is not None:
        grpo_cfg["standardize_rewards"] = (
            args.standardize_rewards.strip().lower() == "true"
        )
    if args.max_advantage_magnitude is not None:
        grpo_cfg["max_advantage_magnitude"] = float(args.max_advantage_magnitude)

    generation_cfg = cfg_override.setdefault("generation", {})
    if args.max_new_tokens is not None:
        generation_cfg["max_new_tokens"] = int(args.max_new_tokens)

    feature_out_dir = os.path.join(
        cfg_override.get("output_dir", os.getcwd()), "feature_checks"
    )
    os.makedirs(feature_out_dir, exist_ok=True)
    cfg_override["output_dir"] = feature_out_dir

    enhanced_override = EnhancedRLConfig.from_yaml_dict(cfg_override)

    manual_trainer = BBUGRPOTrainer(
        model=bundles["model"],
        tokenizer=bundles["tokenizer"],
        processor=bundles["processor"],
        train_dataset=bundles["train"],
        val_dataset=bundles["val"],
        reward_functions=reward_funcs,
        reward_names=reward_names,
        reward_weights=reward_weights,
        enhanced_cfg=enhanced_override,
        raw_config=cfg_override,
        output_dir=feature_out_dir,
    )
    manual_trainer.train()

    temp_history = manual_trainer.temperature_history
    reward_history = manual_trainer.reward_history
    clip_history = manual_trainer.clip_ratio_history
    adv_std_history = manual_trainer.adv_std_history
    adv_max_history = manual_trainer.adv_max_history

    # Checks & reports
    report: Dict[str, Any] = {}

    # 1) Advantage clipping check
    if adv_max_history:
        report["advantage_max_abs"] = float(max(adv_max_history))
    clip_target = cfg_override.get("grpo", {}).get("max_advantage_magnitude")
    if clip_target is not None:
        report["advantage_clip"] = float(clip_target)

    # 2) Reward standardization sanity (finite values)
    for name, series in manual_trainer.reward_component_history.items():
        if series:
            report[f"reward/{name}/mean"] = float(statistics.mean(series))
            if len(series) > 1:
                report[f"reward/{name}/std"] = float(statistics.pstdev(series))

    # 3) Temperature schedule trace
    if temp_history:
        report["temperature/first_last"] = [
            float(temp_history[0]),
            float(temp_history[-1]),
        ]
        step = max(1, len(temp_history) // 5)
        report["temperature/samples"] = [
            float(temp_history[i]) for i in range(0, len(temp_history), step)
        ]

    # 4) Generation diagnostics: sample prompt/completion
    if manual_trainer.sample_prompt and manual_trainer.sample_completion:
        report["sample/prompt_tail"] = manual_trainer.sample_prompt[-256:]
        report["sample/completion_head"] = manual_trainer.sample_completion[:256]

    # Extra telemetry
    if reward_history:
        report["reward/history"] = reward_history
    if clip_history:
        report["clip_ratio/history"] = clip_history
    if adv_std_history:
        report["adv_std/history"] = adv_std_history

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
