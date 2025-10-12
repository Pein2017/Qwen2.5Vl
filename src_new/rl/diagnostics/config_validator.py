"""
GRPO Config Sanity Checker

Feature: 004-grpo-post-training
Constitution: v4.1.1

Validates GRPO configuration for common mistakes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


logger = logging.getLogger(__name__)


@dataclass
class ConfigIssue:
    """A configuration issue found during validation."""

    severity: str  # "error", "warning", "info"
    category: str  # "rewards", "training", "model", etc.
    message: str
    fix_suggestion: Optional[str] = None


class GRPOConfigValidator:
    """Validates GRPO configuration files."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.issues: List[ConfigIssue] = []

    def validate_all(self) -> List[ConfigIssue]:
        """Run all validation checks."""
        self.issues = []

        self._validate_reward_weights()
        self._validate_training_params()
        self._validate_grpo_params()
        self._validate_paths()
        self._validate_model_settings()
        self._validate_loss_weights()

        return self.issues

    def _validate_reward_weights(self):
        """Check reward configuration."""
        rewards_config = self.config.get("rewards_config", {})

        # Check that at least one detection reward has non-zero weight
        detection_rewards = [
            "bbox_giou", "quad_l1", "line_l1", "coverage", "geometry_sanity"
        ]

        has_detection_reward = False
        for reward_name in detection_rewards:
            weight_key = f"{reward_name}_weight"
            if rewards_config.get(weight_key, 0.0) > 0:
                has_detection_reward = True
                break

        if not has_detection_reward:
            self.issues.append(ConfigIssue(
                severity="error",
                category="rewards",
                message="No detection rewards enabled (all weights are 0)",
                fix_suggestion="Set at least one of: bbox_giou_weight, quad_l1_weight, "
                              "line_l1_weight, coverage_weight, geometry_sanity_weight > 0"
            ))

        # Check formatting rewards
        formatting_rewards = ["wrappers", "coords", "separators"]
        total_formatting_weight = sum(
            rewards_config.get(f"{r}_weight", 0.0) for r in formatting_rewards
        )

        if total_formatting_weight == 0:
            self.issues.append(ConfigIssue(
                severity="warning",
                category="rewards",
                message="No formatting rewards enabled - model may produce malformed output",
                fix_suggestion="Consider enabling wrappers_weight, coords_weight, or separators_weight"
            ))

    def _validate_training_params(self):
        """Check training parameters."""
        training = self.config.get("training", {})

        # Check warmup steps
        warmup_steps = training.get("warmup_steps", 0)
        max_steps = training.get("max_steps", 1000)

        if warmup_steps > max_steps * 0.2:
            self.issues.append(ConfigIssue(
                severity="warning",
                category="training",
                message=f"Warmup steps ({warmup_steps}) > 20% of max_steps ({max_steps})",
                fix_suggestion=f"Consider reducing warmup_steps to ~{int(max_steps * 0.1)}"
            ))

        # Check logging frequency
        logging_steps = self.config.get("logging", {}).get("logging_steps", 10)
        if logging_steps > 50:
            self.issues.append(ConfigIssue(
                severity="info",
                category="training",
                message=f"logging_steps={logging_steps} may miss early issues",
                fix_suggestion="Consider logging_steps <= 10 for better monitoring"
            ))

    def _validate_grpo_params(self):
        """Check GRPO-specific parameters."""
        grpo = self.config.get("grpo", {})

        # Check sample_k
        sample_k = grpo.get("sample_k", 8)
        if sample_k < 4:
            self.issues.append(ConfigIssue(
                severity="warning",
                category="grpo",
                message=f"sample_k={sample_k} is very low - may have insufficient variance",
                fix_suggestion="Consider sample_k >= 8 for better advantage estimates"
            ))

        # Check temperature
        temperature = grpo.get("temperature", 1.0)
        if temperature < 0.7:
            self.issues.append(ConfigIssue(
                severity="warning",
                category="grpo",
                message=f"temperature={temperature} is low - may reduce diversity",
                fix_suggestion="Consider temperature >= 0.7 for GRPO"
            ))
        elif temperature > 1.5:
            self.issues.append(ConfigIssue(
                severity="warning",
                category="grpo",
                message=f"temperature={temperature} is high - may produce incoherent outputs",
                fix_suggestion="Consider temperature <= 1.3"
            ))

        # Check max_new_tokens
        max_new_tokens = grpo.get("max_new_tokens", 512)
        if max_new_tokens < 256:
            self.issues.append(ConfigIssue(
                severity="warning",
                category="grpo",
                message=f"max_new_tokens={max_new_tokens} may be too small for dense captioning",
                fix_suggestion="Consider max_new_tokens >= 512"
            ))

    def _validate_paths(self):
        """Check file paths exist."""
        # Check model path
        model_path = self.config.get("model_path")
        if model_path and not Path(model_path).exists():
            self.issues.append(ConfigIssue(
                severity="error",
                category="paths",
                message=f"model_path does not exist: {model_path}",
                fix_suggestion="Verify the checkpoint path is correct"
            ))

        # Check data paths
        for key in ["train_data_path", "val_data_path"]:
            data_path = self.config.get(key)
            if data_path and not Path(data_path).exists():
                self.issues.append(ConfigIssue(
                    severity="error",
                    category="paths",
                    message=f"{key} does not exist: {data_path}",
                    fix_suggestion=f"Verify {key} points to valid JSONL file"
                ))

    def _validate_model_settings(self):
        """Check model configuration."""
        model = self.config.get("model", {})

        # Check dtype
        torch_dtype = model.get("torch_dtype")
        if torch_dtype != "bfloat16":
            self.issues.append(ConfigIssue(
                severity="warning",
                category="model",
                message=f"torch_dtype={torch_dtype} - constitution mandates bfloat16",
                fix_suggestion="Set model.torch_dtype: bfloat16"
            ))

        # Check attention implementation
        attn_impl = model.get("attn_implementation")
        if attn_impl not in ["flash_attention_2", "sdpa", "eager"]:
            self.issues.append(ConfigIssue(
                severity="warning",
                category="model",
                message=f"attn_implementation={attn_impl} not recognized",
                fix_suggestion="Use one of: flash_attention_2, sdpa, eager"
            ))

    def _validate_loss_weights(self):
        """Check loss weight configuration."""
        loss = self.config.get("loss", {})

        # Check all weights are non-negative
        weight_keys = [
            "teacher_loss_weight",
            "student_loss_weight",
            "caption_loss_weight",
            "grounding_loss_weight",
            "formatting_loss_weight",
        ]

        for key in weight_keys:
            weight = loss.get(key, 0.0)
            if weight < 0:
                self.issues.append(ConfigIssue(
                    severity="error",
                    category="loss",
                    message=f"{key}={weight} is negative",
                    fix_suggestion=f"Set {key} >= 0"
                ))

        # Check at least student loss is enabled
        if loss.get("student_loss_weight", 0.0) == 0:
            self.issues.append(ConfigIssue(
                severity="error",
                category="loss",
                message="student_loss_weight=0 - no learning signal!",
                fix_suggestion="Set student_loss_weight > 0 (typically 1.0)"
            ))

    def print_report(self):
        """Print validation report to console."""
        if not self.issues:
            print(f"✅ Config validation passed: {self.config_path}")
            return

        print(f"\n{'='*70}")
        print(f"Config Validation Report: {self.config_path}")
        print(f"{'='*70}\n")

        # Group by severity
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        infos = [i for i in self.issues if i.severity == "info"]

        if errors:
            print(f"❌ ERRORS ({len(errors)}):")
            for issue in errors:
                print(f"  [{issue.category}] {issue.message}")
                if issue.fix_suggestion:
                    print(f"    → {issue.fix_suggestion}")
                print()

        if warnings:
            print(f"⚠️  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                print(f"  [{issue.category}] {issue.message}")
                if issue.fix_suggestion:
                    print(f"    → {issue.fix_suggestion}")
                print()

        if infos:
            print(f"ℹ️  INFO ({len(infos)}):")
            for issue in infos:
                print(f"  [{issue.category}] {issue.message}")
                if issue.fix_suggestion:
                    print(f"    → {issue.fix_suggestion}")
                print()

        print(f"{'='*70}\n")

        if errors:
            print("❌ Config has ERRORS - training may fail")
        elif warnings:
            print("⚠️  Config has warnings - review before training")
        else:
            print("ℹ️  Config is mostly good - minor suggestions only")


def validate_config_file(config_path: str) -> bool:
    """
    Validate a GRPO config file and print report.

    Returns:
        True if no errors, False if errors found
    """
    validator = GRPOConfigValidator(config_path)
    validator.validate_all()
    validator.print_report()

    errors = [i for i in validator.issues if i.severity == "error"]
    return len(errors) == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src_new.rl.diagnostics.config_validator <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    is_valid = validate_config_file(config_path)
    sys.exit(0 if is_valid else 1)
