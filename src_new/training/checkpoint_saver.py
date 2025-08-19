import glob
import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, Optional

import torch


class BestCheckpointManager:
    """
    Best-checkpoint management for eliminating redundant checkpoint operations.

    Tracks the best metric and provides descriptive names for best checkpoints.
    """

    def __init__(self, metric_name: str = "eval_loss", greater_is_better: bool = False):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.current_best_metric: Optional[float] = None
        self.current_best_dir: Optional[str] = None

    def is_new_best(self, current_metrics: Dict[str, float]) -> bool:
        if not current_metrics or self.metric_name not in current_metrics:
            return False
        value = current_metrics[self.metric_name]
        try:
            import math

            if math.isnan(value):
                return False
        except Exception:
            pass
        if self.current_best_metric is None:
            return True
        return (
            (value > self.current_best_metric)
            if self.greater_is_better
            else (value < self.current_best_metric)
        )

    def create_best_checkpoint_name(self, metrics: Dict[str, float], step: int) -> str:
        if self.metric_name not in metrics:
            raise ValueError(
                f"Metric '{self.metric_name}' not found in metrics: {list(metrics.keys())}"
            )
        metric_value = metrics[self.metric_name]
        metric_str = f"{metric_value:.4f}"
        metric_type = self.metric_name.replace("eval_", "")
        return f"best-{step}-{metric_type}{metric_str}"

    def update_best_checkpoint(
        self, metrics: Dict[str, float], checkpoint_path: str
    ) -> None:
        if self.metric_name not in metrics:
            raise ValueError(
                f"Metric '{self.metric_name}' not found in metrics: {list(metrics.keys())}"
            )
        self.current_best_metric = metrics[self.metric_name]
        self.current_best_dir = checkpoint_path


def _get_rank_aware_logger() -> logging.Logger:
    try:
        from ..utils.rank_aware_logging import get_rank_aware_logger

        return get_rank_aware_logger(__name__)
    except Exception:
        return logging.getLogger(__name__)


logger = _get_rank_aware_logger()


class CheckpointSaver:
    """
    Unified, reusable checkpoint saver for Qwen2.5‑VL.

    - Saves inference‑ready checkpoints (SafeTensors, sharded) on rank 0
    - Handles tokenizer, processor, generation config
    - Writes coordinate_config.json when coordinate mode is enabled
    - Integrates with UnifiedCheckpointManager for best checkpoints
    - Supports DeepSpeed and non-DeepSpeed models
    - Performs checkpoint rotation based on save_total_limit
    """

    def __init__(
        self,
        *,
        args: Any,
        checkpoint_manager: BestCheckpointManager,
    ):
        self.args = args
        self.checkpoint_manager = checkpoint_manager

        # Internal flag to avoid duplicate saves per step
        self._checkpoint_in_progress: bool = False

    def save_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        processing_class: Optional[Any],
        processor: Optional[Any],
        step: int,
        current_metrics: Optional[Dict[str, float]] = None,
        is_deepspeed_enabled: bool = False,
        training_start_time: Optional[float] = None,
    ) -> str:
        """
        Save an inference‑ready checkpoint for the given training step.

        Returns:
                Path to the checkpoint directory.
        """
        if self._checkpoint_in_progress:
            logger.debug("🔄 Checkpoint already in progress, skipping duplicate call")
            return os.path.join(self.args.output_dir, f"checkpoint-{step}")

        self._checkpoint_in_progress = True
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        should_log = bool(
            getattr(self.args, "should_save", False)
        )  # True only on rank 0

        try:
            # Skip re-save if folder already contains model files
            if os.path.isdir(checkpoint_dir):
                existing_model_shards = glob.glob(
                    os.path.join(checkpoint_dir, "model-*.safetensors")
                )
                has_index = os.path.isfile(
                    os.path.join(checkpoint_dir, "model.safetensors.index.json")
                )
                has_config = os.path.isfile(os.path.join(checkpoint_dir, "config.json"))
                if existing_model_shards or has_index or has_config:
                    if should_log:
                        logger.info(
                            f"🔄 Checkpoint for step {step} already exists at {checkpoint_dir} — ensuring auxiliary files (tokenizer/processor/configs) are present and skipping duplicate model save"
                        )
                    # Ensure auxiliary files exist even if model files are already present
                    try:
                        self._ensure_auxiliary_files(
                            checkpoint_dir=checkpoint_dir,
                            model=model,
                            processing_class=processing_class,
                            processor=processor,
                        )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to ensure auxiliary files in existing checkpoint: {e}"
                        )
                    # Best checkpoint + rotation still need to run
                    if current_metrics is not None:
                        self._maybe_update_best_and_rotate(
                            checkpoint_dir, current_metrics, step
                        )
                    return checkpoint_dir

            # Pre-save logs
            if should_log:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(
                    f"\n🚀 [FAST CHECKPOINT] Starting inference-ready checkpoint save at {current_time}"
                )
                logger.info(f"📁 Checkpoint location: {checkpoint_dir}")
                try:
                    logger.info(f"📊 Training step: {step}")
                except Exception:
                    pass

            save_start_time = time.time()
            use_fast_checkpoint = bool(getattr(self.args, "fast_checkpoint_mode", True))

            # Always create directory (idempotent)
            os.makedirs(checkpoint_dir, exist_ok=True)

            if use_fast_checkpoint:
                if is_deepspeed_enabled:
                    self._save_deepspeed_inference_checkpoint(
                        model, checkpoint_dir, processing_class, processor
                    )
                else:
                    self._save_inference_checkpoint(
                        model, checkpoint_dir, processing_class, processor
                    )
            else:
                # Fall back to HuggingFace full checkpoint mode (optimizer etc.)
                # We cannot call HF internals from here; expect caller to handle this path.
                if should_log:
                    logger.info(
                        "🐌 Full checkpoint mode requested but not handled by CheckpointSaver. Caller should save training state."
                    )

            # After saving, handle best checkpoint and rotation
            if current_metrics is not None:
                self._maybe_update_best_and_rotate(
                    checkpoint_dir, current_metrics, step
                )

            if should_log:
                dur = time.time() - save_start_time
                logger.info(
                    f"✅ [FAST INFERENCE CHECKPOINT] Completed successfully in {dur:.2f}s"
                )
                logger.info(f"💾 Checkpoint saved to: {checkpoint_dir}")

            return checkpoint_dir
        finally:
            self._checkpoint_in_progress = False

    def _maybe_update_best_and_rotate(
        self, checkpoint_dir: str, current_metrics: Dict[str, float], step: int
    ) -> None:
        should_log = bool(getattr(self.args, "should_save", False))
        if (
            should_log
            and current_metrics
            and self.checkpoint_manager.is_new_best(current_metrics)
        ):
            logger.info("🏆 Creating best checkpoint by direct folder copy")
            best_checkpoint_name = self.checkpoint_manager.create_best_checkpoint_name(
                current_metrics, step
            )
            best_checkpoint_path = os.path.join(
                self.args.output_dir, best_checkpoint_name
            )

            # Remember previous best before updating
            old_best_path = getattr(self.checkpoint_manager, "current_best_dir", None)

            # Replace existing best directory
            if os.path.exists(best_checkpoint_path):
                shutil.rmtree(best_checkpoint_path)
            shutil.copytree(checkpoint_dir, best_checkpoint_path)
            self.checkpoint_manager.update_best_checkpoint(
                current_metrics, best_checkpoint_path
            )

            # Remove old best if different
            if (
                old_best_path
                and old_best_path != best_checkpoint_path
                and os.path.exists(old_best_path)
            ):
                try:
                    shutil.rmtree(old_best_path)
                    logger.info(
                        f"🗑️ Removed old best: {os.path.basename(old_best_path)}"
                    )
                except (OSError, PermissionError) as e:
                    logger.warning(f"⚠️ Could not remove old best checkpoint: {e}")

        # Rotate checkpoints after best handling
        self._rotate_inference_checkpoints()

    def _rotate_inference_checkpoints(self) -> None:
        if not getattr(self.args, "should_save", False) or not hasattr(
            self.args, "save_total_limit"
        ):
            return
        if not self.args.output_dir:
            return
        checkpoint_pattern = os.path.join(self.args.output_dir, "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)

        def _step_num(path: str) -> int:
            try:
                return int(os.path.basename(path).split("-")[1])
            except Exception:
                return 0

        checkpoints.sort(key=_step_num)
        if self.args.save_total_limit and len(checkpoints) > self.args.save_total_limit:
            to_remove = checkpoints[: -self.args.save_total_limit]
            for p in to_remove:
                try:
                    shutil.rmtree(p)
                    logger.info(f"🗑️ [RANK 0] Removed old checkpoint: {p}")
                except Exception as e:
                    logger.warning(f"⚠️ [RANK 0] Failed to remove checkpoint {p}: {e}")

    def _get_unwrapped_model(self, model: torch.nn.Module) -> torch.nn.Module:
        unwrapped = model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        # DetectionModel wrapper exposes base_model
        if hasattr(unwrapped, "base_model"):
            unwrapped = unwrapped.base_model
        return unwrapped

    def _save_inference_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_dir: str,
        processing_class: Optional[Any],
        processor: Optional[Any],
    ) -> None:
        if not getattr(self.args, "should_save", False):
            return

        logger.info(
            f"🚀 [RANK 0] Creating inference-ready checkpoint at {checkpoint_dir}"
        )

        unwrapped_model = self._get_unwrapped_model(model)

        # 1) Model weights (SafeTensors)
        logger.info("💾 [RANK 0] Saving model weights (SafeTensors format)...")
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            safe_serialization=True,
            max_shard_size="5GB",
            push_to_hub=False,
        )
        logger.info("✅ [RANK 0] Model weights saved")

        # 2) Tokenizer (extended if applicable)
        if processing_class is not None:
            logger.info("🔤 [RANK 0] Saving tokenizer...")
            processing_class.save_pretrained(checkpoint_dir)
            logger.info("✅ [RANK 0] Tokenizer saved")

        # 3) Processor (tokenizer + image processor)
        processor_saved = False
        if processor is not None:
            try:
                logger.info("💾 [RANK 0] Saving processor configuration...")
                processor.save_pretrained(checkpoint_dir)
                processor_saved = True
                logger.info("✅ [RANK 0] Processor configuration saved")
            except Exception as e:
                logger.error(f"❌ Failed to save processor configuration: {e}")
                logger.info("🔄 [RANK 0] Attempting fallback processor saving...")

        if not processor_saved:
            # Fallback: save image processor (if available) and create minimal preprocessor_config.json
            image_processor = None
            try:
                if processor is not None and hasattr(processor, "image_processor"):
                    image_processor = processor.image_processor
                elif processing_class is not None and hasattr(
                    processing_class, "image_processor"
                ):
                    image_processor = processing_class.image_processor
            except Exception:
                image_processor = None

            if image_processor is not None:
                try:
                    image_processor.save_pretrained(checkpoint_dir)
                    logger.info("✅ [RANK 0] Image processor saved separately")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save image processor separately: {e}")

            # Minimal preprocessor_config.json to keep HF happy
            preprocessor_config = {
                "processor_class": "Qwen2VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessor",
            }
            with open(
                os.path.join(checkpoint_dir, "preprocessor_config.json"), "w"
            ) as f:
                json.dump(preprocessor_config, f, indent=2)
            logger.info("✅ [RANK 0] Minimal preprocessor_config.json created")

        # 4) Generation config
        if (
            hasattr(unwrapped_model, "generation_config")
            and unwrapped_model.generation_config is not None
        ):
            logger.info("⚙️ [RANK 0] Saving generation configuration...")
            unwrapped_model.generation_config.save_pretrained(checkpoint_dir)
            logger.info("✅ [RANK 0] Generation config saved")

        # 5) Coordinate token config (optional)
        self._maybe_write_coordinate_config(
            checkpoint_dir, unwrapped_model, processing_class
        )

    def _save_deepspeed_inference_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_dir: str,
        processing_class: Optional[Any],
        processor: Optional[Any],
    ) -> None:
        # Only rank 0 should perform any saving under DeepSpeed
        if not getattr(self.args, "should_save", False):
            return

        logger.info(f"🚀 Creating inference-ready checkpoint at {checkpoint_dir}")

        # Save model weights only (no optimizer)
        unwrapped_model = self._get_unwrapped_model(model)
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            safe_serialization=True,
            max_shard_size="5GB",
            push_to_hub=False,
        )
        logger.info("✅ Inference-ready model weights saved")

        # Save auxiliary files on rank 0
        self._ensure_auxiliary_files(
            checkpoint_dir=checkpoint_dir,
            model=model,
            processing_class=processing_class,
            processor=processor,
        )

        # Generation + coordinate config
        self._maybe_write_coordinate_config(
            checkpoint_dir, unwrapped_model, processing_class
        )
        if (
            hasattr(unwrapped_model, "generation_config")
            and unwrapped_model.generation_config is not None
        ):
            unwrapped_model.generation_config.save_pretrained(checkpoint_dir)

    def _maybe_write_coordinate_config(
        self,
        checkpoint_dir: str,
        unwrapped_model: torch.nn.Module,
        processing_class: Optional[Any],
    ) -> None:
        try:
            if hasattr(unwrapped_model, "training_config") and getattr(
                unwrapped_model.training_config, "coordinate_tokens_enabled", False
            ):
                # Try to derive coordinate token range from tokenizer
                coord_range = [None, None]
                try:
                    from src_new.processing.special_tokens import get_coord_token_range

                    if processing_class is not None:
                        rng = get_coord_token_range(processing_class)
                        if rng is not None and rng.end_exclusive > rng.start_id:
                            coord_range = [rng.start_id, rng.end_exclusive]
                except Exception:
                    coord_range = [None, None]

                coord_config = {
                    "coordinate_tokens_enabled": True,
                    "max_coord_value": getattr(
                        unwrapped_model.training_config, "max_coord_value", None
                    ),
                    "vocab_size_extended": len(processing_class.get_vocab())
                    if processing_class is not None
                    and hasattr(processing_class, "get_vocab")
                    else None,
                    "coordinate_token_range": coord_range,
                }
                with open(
                    os.path.join(checkpoint_dir, "coordinate_config.json"), "w"
                ) as f:
                    json.dump(coord_config, f, indent=2)
                logger.info("✅ [RANK 0] Coordinate config saved")
        except Exception as e:
            logger.warning(f"⚠️ Failed to write coordinate_config.json: {e}")

    def _ensure_auxiliary_files(
        self,
        *,
        checkpoint_dir: str,
        model: torch.nn.Module,
        processing_class: Optional[Any],
        processor: Optional[Any],
    ) -> None:
        """
        Ensure tokenizer, processor, and related auxiliary files are present in the checkpoint directory.

        Only runs on rank 0 (args.should_save == True). Safe to call multiple times.
        """
        if not getattr(self.args, "should_save", False):
            return

        # 1) Tokenizer
        if processing_class is not None:
            try:
                logger.info("🔤 [RANK 0] Ensuring tokenizer files are saved...")
                processing_class.save_pretrained(checkpoint_dir)
                logger.info("✅ [RANK 0] Tokenizer saved/updated")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save tokenizer: {e}")

        # 2) Processor (or fallback to image processor + minimal preprocessor_config)
        processor_saved = False
        if processor is not None:
            try:
                logger.info("💾 [RANK 0] Ensuring processor configuration is saved...")
                processor.save_pretrained(checkpoint_dir)
                processor_saved = True
                logger.info("✅ [RANK 0] Processor configuration saved/updated")
            except Exception as e:
                logger.error(f"❌ Failed to save processor configuration: {e}")
                logger.info("🔄 [RANK 0] Attempting fallback processor saving...")

        if not processor_saved:
            image_processor = None
            try:
                if processor is not None and hasattr(processor, "image_processor"):
                    image_processor = processor.image_processor
                elif processing_class is not None and hasattr(
                    processing_class, "image_processor"
                ):
                    image_processor = processing_class.image_processor
            except Exception:
                image_processor = None

            if image_processor is not None:
                try:
                    image_processor.save_pretrained(checkpoint_dir)
                    logger.info("✅ [RANK 0] Image processor saved separately")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save image processor separately: {e}")

            # Minimal preprocessor_config.json to keep HF happy
            preprocessor_config = {
                "processor_class": "Qwen2VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessor",
            }
            try:
                with open(
                    os.path.join(checkpoint_dir, "preprocessor_config.json"), "w"
                ) as f:
                    json.dump(preprocessor_config, f, indent=2)
                logger.info("✅ [RANK 0] Minimal preprocessor_config.json ensured")
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to write minimal preprocessor_config.json: {e}"
                )
