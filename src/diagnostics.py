"""
Diagnostic utilities for identifying training instability issues.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class TrainingDiagnostics:
    """
    Comprehensive diagnostics for training instability issues.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.diagnostics_history = []

    def diagnose_training_failure(
        self,
        model,
        tokenizer,
        data_collator,
        train_dataset,
        eval_dataset=None,
        num_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Comprehensive diagnosis of training failure causes.

        Args:
            model: The model being trained
            tokenizer: The tokenizer
            data_collator: The data collator
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            num_samples: Number of samples to analyze

        Returns:
            Dictionary with diagnostic results
        """
        self.logger.info("🔍 Starting comprehensive training diagnostics...")

        diagnostics = {
            "model_health": self._diagnose_model_health(model),
            "data_health": self._diagnose_data_health(
                train_dataset, tokenizer, data_collator, num_samples
            ),
            "gradient_health": self._diagnose_gradient_health(model),
            "memory_health": self._diagnose_memory_health(),
            "recommendations": [],
        }

        if eval_dataset:
            diagnostics["eval_data_health"] = self._diagnose_data_health(
                eval_dataset, tokenizer, data_collator, num_samples
            )

        # Generate recommendations based on findings
        diagnostics["recommendations"] = self._generate_recommendations(diagnostics)

        # Save diagnostics to file
        self._save_diagnostics(diagnostics)

        return diagnostics

    def _diagnose_model_health(self, model) -> Dict[str, Any]:
        """Diagnose model parameter health."""
        self.logger.info("🔍 Diagnosing model health...")

        health = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "nan_parameters": 0,
            "inf_parameters": 0,
            "zero_parameters": 0,
            "parameter_ranges": {},
            "problematic_layers": [],
            "dtype_issues": [],
        }

        try:
            for name, param in model.named_parameters():
                if param.data is not None:
                    health["total_parameters"] += param.numel()

                    if param.requires_grad:
                        health["trainable_parameters"] += param.numel()

                    # Check for NaN/Inf
                    if torch.isnan(param.data).any():
                        nan_count = torch.isnan(param.data).sum().item()
                        health["nan_parameters"] += nan_count
                        health["problematic_layers"].append(
                            f"NaN in {name}: {nan_count} values"
                        )

                    if torch.isinf(param.data).any():
                        inf_count = torch.isinf(param.data).sum().item()
                        health["inf_parameters"] += inf_count
                        health["problematic_layers"].append(
                            f"Inf in {name}: {inf_count} values"
                        )

                    # Check for zero parameters
                    if (param.data == 0).all():
                        health["zero_parameters"] += param.numel()
                        health["problematic_layers"].append(f"All zeros in {name}")

                    # Parameter ranges
                    health["parameter_ranges"][name] = {
                        "min": param.data.min().item(),
                        "max": param.data.max().item(),
                        "mean": param.data.mean().item(),
                        "std": param.data.std().item(),
                    }

                    # Check dtype
                    if param.dtype not in [
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                    ]:
                        health["dtype_issues"].append(
                            f"Unexpected dtype in {name}: {param.dtype}"
                        )

            # Calculate percentages
            if health["total_parameters"] > 0:
                health["nan_percentage"] = (
                    health["nan_parameters"] / health["total_parameters"]
                ) * 100
                health["inf_percentage"] = (
                    health["inf_parameters"] / health["total_parameters"]
                ) * 100
                health["trainable_percentage"] = (
                    health["trainable_parameters"] / health["total_parameters"]
                ) * 100

            self.logger.info("✅ Model health check complete:")
            self.logger.info(f"   - Total parameters: {health['total_parameters']:,}")
            self.logger.info(
                f"   - Trainable: {health['trainable_parameters']:,} ({health.get('trainable_percentage', 0):.1f}%)"
            )
            self.logger.info(f"   - NaN parameters: {health['nan_parameters']}")
            self.logger.info(f"   - Inf parameters: {health['inf_parameters']}")

        except Exception as e:
            self.logger.error(f"❌ Error in model health check: {e}")
            health["error"] = str(e)

        return health

    def _diagnose_data_health(
        self, dataset, tokenizer, data_collator, num_samples: int
    ) -> Dict[str, Any]:
        """Diagnose data health."""
        self.logger.info(f"🔍 Diagnosing data health with {num_samples} samples...")

        health = {
            "dataset_size": len(dataset),
            "sample_analysis": [],
            "token_statistics": {},
            "label_statistics": {},
            "collation_issues": [],
            "data_issues": [],
        }

        try:
            # Analyze individual samples
            for i in range(min(num_samples, len(dataset))):
                try:
                    sample = dataset[i]
                    sample_info = {
                        "index": i,
                        "input_ids_shape": sample.get(
                            "input_ids", torch.tensor([])
                        ).shape,
                        "labels_shape": sample.get("labels", torch.tensor([])).shape,
                        "has_pixel_values": "pixel_values" in sample
                        and sample["pixel_values"] is not None,
                        "issues": [],
                    }

                    # Check input_ids
                    if "input_ids" in sample:
                        input_ids = sample["input_ids"]
                        if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                            sample_info["issues"].append("NaN/Inf in input_ids")

                        if input_ids.max() >= tokenizer.vocab_size:
                            sample_info["issues"].append(
                                f"Invalid token ID: {input_ids.max()} >= {tokenizer.vocab_size}"
                            )

                    # Check labels
                    if "labels" in sample:
                        labels = sample["labels"]
                        valid_labels = labels[labels != -100]
                        sample_info["valid_labels_count"] = len(valid_labels)

                        if len(valid_labels) == 0:
                            sample_info["issues"].append("No valid labels (all -100)")
                        elif (
                            torch.isnan(valid_labels).any()
                            or torch.isinf(valid_labels).any()
                        ):
                            sample_info["issues"].append("NaN/Inf in labels")

                    # Check pixel values
                    if "pixel_values" in sample and sample["pixel_values"] is not None:
                        pixel_values = sample["pixel_values"]
                        if (
                            torch.isnan(pixel_values).any()
                            or torch.isinf(pixel_values).any()
                        ):
                            sample_info["issues"].append("NaN/Inf in pixel_values")

                    health["sample_analysis"].append(sample_info)

                except Exception as e:
                    health["data_issues"].append(f"Error processing sample {i}: {e}")

            # Test data collation
            try:
                samples = [dataset[i] for i in range(min(3, len(dataset)))]
                batch = data_collator(samples)

                # Check batch health
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any() or torch.isinf(value).any():
                            health["collation_issues"].append(
                                f"NaN/Inf in batch['{key}']"
                            )

            except Exception as e:
                health["collation_issues"].append(f"Data collation failed: {e}")

            self.logger.info("✅ Data health check complete:")
            self.logger.info(f"   - Dataset size: {health['dataset_size']}")
            self.logger.info(f"   - Samples analyzed: {len(health['sample_analysis'])}")
            self.logger.info(f"   - Data issues found: {len(health['data_issues'])}")
            self.logger.info(
                f"   - Collation issues: {len(health['collation_issues'])}"
            )

        except Exception as e:
            self.logger.error(f"❌ Error in data health check: {e}")
            health["error"] = str(e)

        return health

    def _diagnose_gradient_health(self, model) -> Dict[str, Any]:
        """Diagnose gradient health."""
        self.logger.info("🔍 Diagnosing gradient health...")

        health = {
            "has_gradients": False,
            "nan_gradients": 0,
            "inf_gradients": 0,
            "zero_gradients": 0,
            "gradient_norms": {},
            "problematic_layers": [],
        }

        try:
            total_params_with_grad = 0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    health["has_gradients"] = True
                    total_params_with_grad += 1

                    # Check for NaN/Inf gradients
                    if torch.isnan(param.grad).any():
                        nan_count = torch.isnan(param.grad).sum().item()
                        health["nan_gradients"] += nan_count
                        health["problematic_layers"].append(
                            f"NaN gradient in {name}: {nan_count} values"
                        )

                    if torch.isinf(param.grad).any():
                        inf_count = torch.isinf(param.grad).sum().item()
                        health["inf_gradients"] += inf_count
                        health["problematic_layers"].append(
                            f"Inf gradient in {name}: {inf_count} values"
                        )

                    # Check for zero gradients
                    if (param.grad.abs() < 1e-10).all():
                        health["zero_gradients"] += 1
                        health["problematic_layers"].append(f"Zero gradient in {name}")

                    # Gradient norms
                    grad_norm = param.grad.norm().item()
                    health["gradient_norms"][name] = grad_norm

            health["total_params_with_gradients"] = total_params_with_grad

            if not health["has_gradients"]:
                health["problematic_layers"].append(
                    "No gradients found - model may not be in training mode"
                )

            self.logger.info("✅ Gradient health check complete:")
            self.logger.info(
                f"   - Parameters with gradients: {total_params_with_grad}"
            )
            self.logger.info(f"   - NaN gradients: {health['nan_gradients']}")
            self.logger.info(f"   - Inf gradients: {health['inf_gradients']}")
            self.logger.info(f"   - Zero gradients: {health['zero_gradients']}")

        except Exception as e:
            self.logger.error(f"❌ Error in gradient health check: {e}")
            health["error"] = str(e)

        return health

    def _diagnose_memory_health(self) -> Dict[str, Any]:
        """Diagnose memory health."""
        self.logger.info("🔍 Diagnosing memory health...")

        health = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_memory": {},
        }

        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_info = {
                        "total": torch.cuda.get_device_properties(i).total_memory,
                        "allocated": torch.cuda.memory_allocated(i),
                        "cached": torch.cuda.memory_reserved(i),
                    }
                    memory_info["free"] = (
                        memory_info["total"] - memory_info["allocated"]
                    )
                    memory_info["utilization"] = (
                        memory_info["allocated"] / memory_info["total"]
                    ) * 100

                    health["gpu_memory"][f"gpu_{i}"] = memory_info

                    self.logger.info(
                        f"   GPU {i}: {memory_info['utilization']:.1f}% used ({memory_info['allocated'] / 1e9:.1f}GB / {memory_info['total'] / 1e9:.1f}GB)"
                    )

        except Exception as e:
            self.logger.error(f"❌ Error in memory health check: {e}")
            health["error"] = str(e)

        return health

    def _generate_recommendations(self, diagnostics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []

        # Model health recommendations
        model_health = diagnostics.get("model_health", {})
        if model_health.get("nan_parameters", 0) > 0:
            recommendations.append(
                "❌ CRITICAL: NaN parameters detected - model is corrupted, restart training from a good checkpoint"
            )

        if model_health.get("inf_parameters", 0) > 0:
            recommendations.append(
                "❌ CRITICAL: Inf parameters detected - reduce learning rate significantly"
            )

        if model_health.get("trainable_percentage", 0) == 0:
            recommendations.append(
                "❌ CRITICAL: No trainable parameters - check model configuration"
            )

        # Data health recommendations
        data_health = diagnostics.get("data_health", {})
        if data_health.get("collation_issues"):
            recommendations.append(
                "❌ CRITICAL: Data collation issues detected - check data preprocessing"
            )

        if data_health.get("data_issues"):
            recommendations.append("⚠️  Data issues detected - validate your dataset")

        # Gradient health recommendations
        grad_health = diagnostics.get("gradient_health", {})
        if not grad_health.get("has_gradients", False):
            recommendations.append(
                "❌ CRITICAL: No gradients found - ensure model is in training mode"
            )

        if grad_health.get("nan_gradients", 0) > 0:
            recommendations.append(
                "❌ CRITICAL: NaN gradients detected - reduce learning rate or add gradient clipping"
            )

        if grad_health.get("inf_gradients", 0) > 0:
            recommendations.append(
                "❌ CRITICAL: Inf gradients detected - reduce learning rate significantly"
            )

        # Memory recommendations
        memory_health = diagnostics.get("memory_health", {})
        for gpu_id, gpu_info in memory_health.get("gpu_memory", {}).items():
            if gpu_info.get("utilization", 0) > 95:
                recommendations.append(
                    f"⚠️  High memory usage on {gpu_id} ({gpu_info['utilization']:.1f}%) - consider reducing batch size"
                )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "✅ No critical issues detected - training failure may be due to:"
            )
            recommendations.append("   1. Learning rate too high (try reducing by 10x)")
            recommendations.append("   2. Gradient accumulation issues")
            recommendations.append("   3. Data preprocessing problems")
            recommendations.append("   4. Model architecture incompatibility")

        return recommendations

    def _save_diagnostics(self, diagnostics: Dict[str, Any]):
        """Save diagnostics to file."""
        try:
            os.makedirs("logs", exist_ok=True)

            # Convert tensors to serializable format
            serializable_diagnostics = self._make_serializable(diagnostics)

            with open("logs/training_diagnostics.json", "w") as f:
                json.dump(serializable_diagnostics, f, indent=2)

            self.logger.info("💾 Diagnostics saved to logs/training_diagnostics.json")

        except Exception as e:
            self.logger.error(f"❌ Failed to save diagnostics: {e}")

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if obj.numel() < 100 else f"<tensor shape={obj.shape}>"
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, torch.dtype):
            return str(obj)
        else:
            return obj


def run_diagnostics(model, tokenizer, data_collator, train_dataset, eval_dataset=None):
    """
    Convenience function to run comprehensive diagnostics.

    Args:
        model: The model being trained
        tokenizer: The tokenizer
        data_collator: The data collator
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)

    Returns:
        Dictionary with diagnostic results
    """
    logger = logging.getLogger(__name__)
    diagnostics = TrainingDiagnostics(logger)

    return diagnostics.diagnose_training_failure(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
