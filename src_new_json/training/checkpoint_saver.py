from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class BestCheckpointManager:
    """
    Best-checkpoint management for eliminating redundant checkpoint operations.

    Tracks the best metric and provides descriptive names for best checkpoints.
    """

    metric_name: str = "eval_loss"
    greater_is_better: bool = False

    current_best_value: Optional[float] = None
    current_best_dir: Optional[str] = None

    def is_new_best(self, current_metrics: Dict[str, float]) -> bool:
        value = current_metrics.get(self.metric_name)
        if value is None:
            return False
        if self.current_best_value is None:
            return True
        if self.greater_is_better:
            return value > self.current_best_value
        return value < self.current_best_value

    def create_best_checkpoint_name(self, metrics: Dict[str, float], step: int) -> str:
        val = metrics.get(self.metric_name)
        suffix = (
            f"{self.metric_name}{val:.4f}" if isinstance(val, (int, float)) else "best"
        )
        return f"best-{step}-{suffix}"

    def update_best_checkpoint(self, metrics: Dict[str, float], best_dir: str) -> None:
        self.current_best_value = metrics.get(self.metric_name)
        self.current_best_dir = best_dir


def _get_rank_aware_logger() -> logging.Logger:
    try:
        from src_new_json.utils.rank_aware_logging import get_rank_aware_logger as _get

        return _get("training.checkpoint_saver")
    except Exception:
        import logging as _logging

        return _logging.getLogger("training.checkpoint_saver")


logger = _get_rank_aware_logger()

# Files that are optional for inference and can be safely skipped during best-checkpoint copying
IGNORED_COPY_FILES = {
    "special_tokens_map.json",
}


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

    args: Any
    checkpoint_manager: BestCheckpointManager

    _checkpoint_in_progress: bool = False

    def __init__(
        self,
        *,
        args: Any,
        checkpoint_manager: BestCheckpointManager,
    ):
        self.args = args
        self.checkpoint_manager = checkpoint_manager

    def _copytree_atomic(self, src: str, dst: str) -> None:
        """Copy a directory tree atomically via a temporary directory.

        Falls back to file-by-file copy if the filesystem rejects copytree (e.g., Unknown error 524 on some mounts).
        Ensures partial artifacts are cleaned up on failure.
        """
        tmp_dst = f"{dst}.tmp"
        # Cleanup any previous tmp
        try:
            if os.path.exists(tmp_dst):
                shutil.rmtree(tmp_dst)
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to remove existing tmp best path '{tmp_dst}': {e}"
            )

        # First attempt: standard copytree into tmp
        try:
            shutil.copytree(
                src,
                tmp_dst,
                dirs_exist_ok=False,
                ignore=shutil.ignore_patterns(*IGNORED_COPY_FILES),
                copy_function=shutil.copy,
            )
            # Replace destination atomically
            try:
                if os.path.exists(dst):
                    shutil.rmtree(dst)
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to remove existing best path '{dst}' before rename: {e}"
                )
            os.replace(tmp_dst, dst)
            return
        except Exception as e:
            logger.warning(
                f"⚠️ copytree failed for best checkpoint (falling back to per-file copy): {e}"
            )

        # Fallback: file-by-file copy into tmp, then atomic replace
        try:
            # Ensure tmp destination exists and is empty
            try:
                if os.path.exists(tmp_dst):
                    shutil.rmtree(tmp_dst)
            except Exception:
                pass
            os.makedirs(tmp_dst, exist_ok=True)

            for root, dirs, files in os.walk(src):
                rel = os.path.relpath(root, src)
                target_dir = tmp_dst if rel == "." else os.path.join(tmp_dst, rel)
                os.makedirs(target_dir, exist_ok=True)
                for d in dirs:
                    os.makedirs(os.path.join(target_dir, d), exist_ok=True)
                for f in files:
                    if f in IGNORED_COPY_FILES:
                        # Skip optional files that are not required for inference
                        continue
                    src_f = os.path.join(root, f)
                    dst_f = os.path.join(target_dir, f)
                    # Retry a few times to work around transient FS errors (e.g., errno 524)
                    last_exc: Optional[Exception] = None
                    for attempt in range(3):
                        try:
                            shutil.copy(src_f, dst_f)
                            last_exc = None
                            break
                        except Exception as fe:
                            last_exc = fe
                            time.sleep(0.2)
                    if last_exc is not None:
                        logger.warning(f"⚠️ Failed to copy '{src_f}' → '{dst_f}': {last_exc}")

            # Replace destination atomically
            try:
                if os.path.exists(dst):
                    shutil.rmtree(dst)
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to remove existing best path '{dst}' before rename; proceeding with atomic replace: {e}"
                )
            os.replace(tmp_dst, dst)

            # Cleanup tmp if created (should be gone after replace, but be safe)
            try:
                if os.path.exists(tmp_dst):
                    shutil.rmtree(tmp_dst)
            except Exception:
                pass
            return
        except Exception as e2:
            # Final cleanup and propagate
            try:
                if os.path.exists(tmp_dst):
                    shutil.rmtree(tmp_dst)
            except Exception:
                pass
            raise RuntimeError(
                f"Best checkpoint copy failed (per-file copy also failed): {e2}"
            )

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
            logger.warning(
                "⚠️ Checkpoint already in progress; skipping concurrent save."
            )
            return ""
        self._checkpoint_in_progress = True
        try:
            save_start_time = time.time()
            should_log = bool(getattr(self.args, "should_save", False))

            # Determine final checkpoint directory (step-scoped)
            final_checkpoint_dir = os.path.join(
                self.args.output_dir, f"checkpoint-{step}"
            )

            # Save core model + processor/tokenizer files
            if is_deepspeed_enabled:
                self._save_deepspeed_inference_checkpoint(
                    model=model,
                    checkpoint_dir=final_checkpoint_dir,
                    processing_class=processing_class,
                    processor=processor,
                )
                checkpoint_dir = final_checkpoint_dir
                # Best checkpoint handling
                if current_metrics:
                    self._maybe_update_best_and_rotate(
                        checkpoint_dir, current_metrics, step
                    )
            else:
                checkpoint_dir = final_checkpoint_dir
                if os.path.exists(checkpoint_dir):
                    # Avoid overwriting; keep idempotent behavior across retries
                    logger.info(
                        f"🔄 Checkpoint for step {step} already exists at {checkpoint_dir} — ensuring auxiliary files (tokenizer/processor/configs) are present and skipping duplicate model save"
                    )
                    self._ensure_auxiliary_files(
                        checkpoint_dir=checkpoint_dir,
                        model=model,
                        processing_class=processing_class,
                        processor=processor,
                    )
                else:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    unwrapped = self._get_unwrapped_model(model)
                    logger.info(
                        "💾 [RANK 0] Saving model weights (SafeTensors format)..."
                    )
                    unwrapped.save_pretrained(
                        checkpoint_dir,
                        safe_serialization=True,
                        max_shard_size="5GB",
                        push_to_hub=False,
                    )
                    logger.info("✅ [RANK 0] Model weights saved")
                    self._ensure_auxiliary_files(
                        checkpoint_dir=checkpoint_dir,
                        model=model,
                        processing_class=processing_class,
                        processor=processor,
                    )

                # Best checkpoint handling
                if current_metrics:
                    self._maybe_update_best_and_rotate(
                        checkpoint_dir, current_metrics, step
                    )

            # Always rotate step checkpoints (regardless of metrics or deepspeed)
            self._rotate_inference_checkpoints()

            if should_log:
                dur = time.time() - save_start_time
                logger.info(
                    f"✅ [FAST INFERENCE CHECKPOINT] Completed successfully in {dur:.2f}s"
                )
                logger.info(f"💾 Checkpoint saved to: {final_checkpoint_dir}")

            return final_checkpoint_dir
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

            # Replace existing best directory using robust atomic copy
            try:
                if os.path.exists(best_checkpoint_path):
                    shutil.rmtree(best_checkpoint_path)
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not remove existing best checkpoint dir '{best_checkpoint_path}': {e}"
                )
            try:
                self._copytree_atomic(checkpoint_dir, best_checkpoint_path)
            except Exception as e:
                logger.warning(
                    f"⚠️ Best checkpoint copy encountered an error but training will continue: {e}"
                )
            else:
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
            video_processor = None
            try:
                if processor is not None and hasattr(processor, "image_processor"):
                    image_processor = processor.image_processor
                elif processing_class is not None and hasattr(
                    processing_class, "image_processor"
                ):
                    image_processor = processing_class.image_processor
                # Try to extract video processor too
                if processor is not None and hasattr(processor, "video_processor"):
                    video_processor = processor.video_processor
                elif processing_class is not None and hasattr(
                    processing_class, "video_processor"
                ):
                    video_processor = processing_class.video_processor
            except Exception:
                image_processor = None
                video_processor = None

            if image_processor is not None:
                try:
                    image_processor.save_pretrained(checkpoint_dir)
                    logger.info("✅ [RANK 0] Image processor saved separately")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save image processor separately: {e}")

            if video_processor is not None:
                try:
                    video_processor.save_pretrained(checkpoint_dir)
                    logger.info("✅ [RANK 0] Video processor saved separately")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save video processor separately: {e}")

            # Minimal preprocessor_config.json to keep HF happy
            preprocessor_config = {
                "processor_class": "Qwen2VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessor",
                "video_processor_type": "Qwen2VLVideoProcessor",
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
        # 6) Copy original training config (if provided) for inference auto-load
        self._maybe_copy_original_config(checkpoint_dir)

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
        # Copy original training config (if provided) for inference auto-load
        self._maybe_copy_original_config(checkpoint_dir)

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
                    from src_new_json.processing.special_tokens import get_coord_token_range

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
                "video_processor_type": "Qwen2VLVideoProcessor",
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

        # 3) Ensure original training config copied if available
        self._maybe_copy_original_config(checkpoint_dir)

    def _maybe_copy_original_config(self, checkpoint_dir: str) -> None:
        """Copy the original YAML config file into the checkpoint directory once.

        Looks for 'original_config_path' on training args and copies it as
        'training_config.yaml' if not already present. No rewriting or updates.
        """
        try:
            src = getattr(self.args, "original_config_path", None)
            if not src or not os.path.isfile(src):
                return
            dst = os.path.join(checkpoint_dir, "training_config.yaml")
            if not os.path.exists(dst):
                try:
                    shutil.copyfile(src, dst)
                    logger.info(
                        f"✅ [RANK 0] Copied original training config to checkpoint: {dst}"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to copy original config: {e}")
        except Exception:
            pass
