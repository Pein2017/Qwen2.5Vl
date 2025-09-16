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
        value = current_metrics[self.metric_name] if self.metric_name in current_metrics else None
        if value is None:
            return False
        if self.current_best_value is None:
            return True
        if self.greater_is_better:
            return value > self.current_best_value
        return value < self.current_best_value

    def create_best_checkpoint_name(self, metrics: Dict[str, float], step: int) -> str:
        val = metrics[self.metric_name] if self.metric_name in metrics else None
        suffix = (
            f"{self.metric_name}{val:.4f}" if isinstance(val, (int, float)) else "best"
        )
        return f"best-{step}-{suffix}"

    def update_best_checkpoint(self, metrics: Dict[str, float], best_dir: str) -> None:
        self.current_best_value = metrics[self.metric_name] if self.metric_name in metrics else None
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

        # Use safe per-file copy by default (copytree disabled)
        skip_copytree = True
        logger.info("🛡️ Using safe per-file copy for best checkpoint (copytree disabled by default)")

        # Source directory validation with retries (important for NFS sync delays)
        max_wait_attempts = 5
        for attempt in range(max_wait_attempts):
            if os.path.exists(src) and os.path.isdir(src):
                # Double-check that key files exist (not just the directory)
                try:
                    files_in_src = list(os.listdir(src))
                    if len(files_in_src) > 0:
                        break  # Source looks ready
                except Exception:
                    pass
            if attempt < max_wait_attempts - 1:
                logger.info(f"🕒 Waiting for source checkpoint directory to be ready (attempt {attempt + 1}/{max_wait_attempts})")
                time.sleep(1.0)  # Wait a bit for NFS sync
        else:
            raise Exception(f"Source checkpoint directory '{src}' not ready after {max_wait_attempts} attempts")

        # First attempt: standard copytree (skip if NFS detected)
        # copytree disabled; using safe per-file copy below

        # Safe per-file copy into tmp, then atomic replace
        try:
            # Ensure tmp destination exists and is empty
            try:
                if os.path.exists(tmp_dst):
                    shutil.rmtree(tmp_dst)
            except Exception:
                pass
            os.makedirs(tmp_dst, exist_ok=True)

            def _copy_with_fsync(src_path: str, dst_path: str) -> None:
                # Copy with explicit read/write and fsync to ensure durability on NFS
                import errno
                bufsize = 1024 * 1024
                with open(src_path, 'rb', buffering=0) as rf:
                    st = os.fstat(rf.fileno())
                    # Pre-create destination with same mode
                    with open(dst_path, 'wb', buffering=0) as wf:
                        try:
                            os.chmod(dst_path, st.st_mode & 0o777)
                        except Exception:
                            pass
                        while True:
                            chunk = rf.read(bufsize)
                            if not chunk:
                                break
                            wf.write(chunk)
                        try:
                            wf.flush()
                            os.fsync(wf.fileno())
                        except Exception as _:
                            pass
                # Preserve mtime/atime
                try:
                    os.utime(dst_path, (st.st_atime, st.st_mtime), follow_symlinks=False)
                except Exception:
                    pass

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
                            _copy_with_fsync(src_f, dst_f)
                            last_exc = None
                            break
                        except Exception as fe:
                            last_exc = fe
                            time.sleep(0.2)
                    if last_exc is not None:
                        logger.warning(f"⚠️ Failed to copy '{src_f}' → '{dst_f}': {last_exc}")

            # Ensure tmp dir visibility and durability (NFS): fsync tmp dir and parent
            def _fsync_dir(path: str) -> None:
                try:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                except Exception:
                    pass
            for _ in range(5):
                if os.path.isdir(tmp_dst):
                    break
                time.sleep(0.2)
            _fsync_dir(tmp_dst)
            _fsync_dir(os.path.dirname(tmp_dst))
            # Atomic replace (works if dst exists or not)
            os.replace(tmp_dst, dst)
            # Fsync parent of destination to flush rename
            _fsync_dir(os.path.dirname(dst))

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
                f"Best checkpoint copy failed: {e2}"
            )

    def save_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        processing_class: Optional[Any],
        processor: Optional[Any],
        step: int,
        current_metrics: Optional[Dict[str, float]] = None,
        is_eval_step: bool = False,
        force_step_save: bool = False,
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

            # Determine if we should create/update the step checkpoint directory on this call
            # Rules:
            # - If force_step_save=True (HF save_steps cadence), always save step checkpoint now (eval-gated by caller).
            # - Else, if is_eval_step and best-interval gate passes (checked below), we'll still need a step checkpoint to copy-from.
            # - Otherwise skip writing step checkpoints at every eval.
            should_write_step_ckpt = bool(force_step_save)

            # Compute best-save min_interval early for gating
            min_interval = 0
            cadence_steps = 0
            interval_multiplier = 0
            explicit_min = None
            try:
                cfg = getattr(getattr(self, "args", None), "training_config", None)
                eval_steps = 0
                if cfg is not None and hasattr(cfg, "eval_steps"):
                    eval_steps = int(getattr(cfg, "eval_steps") or 0)
                if (not isinstance(eval_steps, int)) or eval_steps <= 0:
                    eval_steps = int(getattr(self.args, "eval_steps", 0)) if hasattr(self.args, "eval_steps") else 0
                try:
                    interval_multiplier = int(getattr(cfg, "best_checkpoint_interval_multiplier")) if (cfg is not None and hasattr(cfg, "best_checkpoint_interval_multiplier")) else int(getattr(self.args, "best_checkpoint_interval_multiplier", 10))
                except Exception:
                    interval_multiplier = 10
                try:
                    explicit_min = getattr(cfg, "best_checkpoint_min_interval_steps") if (cfg is not None and hasattr(cfg, "best_checkpoint_min_interval_steps")) else getattr(self.args, "best_checkpoint_min_interval_steps", None)
                except Exception:
                    explicit_min = None
                cadence_steps = int(eval_steps) if int(eval_steps) > 0 else 0
                computed = (int(cadence_steps) * int(interval_multiplier)) if (int(cadence_steps) > 0 and int(interval_multiplier) > 0) else 0
                min_interval = int(explicit_min) if (explicit_min is not None) else int(computed)
            except Exception:
                min_interval = 0

            # If not forcing, allow step checkpoint only on best-save ticks so we can copy to best-*
            if (not should_write_step_ckpt) and is_eval_step and (min_interval > 0):
                if (int(step) % int(min_interval)) == 0:
                    should_write_step_ckpt = True

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
                if is_eval_step and current_metrics:
                    self._maybe_update_best_and_rotate(
                        checkpoint_dir, current_metrics, step
                    )
            else:
                checkpoint_dir = final_checkpoint_dir
                if should_write_step_ckpt:
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

                # Best checkpoint handling (only when we have metrics and a fresh step checkpoint)
                if is_eval_step and current_metrics and should_write_step_ckpt:
                    self._maybe_update_best_and_rotate(
                        checkpoint_dir, current_metrics, step
                    )

            # Rotate only if a step checkpoint was written (avoids repeated directory scans)
            if should_write_step_ckpt:
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
        # Enforce a minimum interval between best checkpoint saves
        min_interval = 0
        try:
            cfg = getattr(getattr(self, "args", None), "training_config", None)
            # Prefer values from training_config when available, otherwise fall back to args
            eval_steps = 0
            try:
                if cfg is not None and hasattr(cfg, "eval_steps"):
                    eval_steps = int(getattr(cfg, "eval_steps") or 0)
            except Exception:
                eval_steps = 0
            if (not isinstance(eval_steps, int)) or eval_steps <= 0:
                eval_steps = int(getattr(self.args, "eval_steps", 0)) if hasattr(self.args, "eval_steps") else 0

            try:
                interval_multiplier = int(getattr(cfg, "best_checkpoint_interval_multiplier")) if (cfg is not None and hasattr(cfg, "best_checkpoint_interval_multiplier")) else int(getattr(self.args, "best_checkpoint_interval_multiplier", 10))
            except Exception:
                interval_multiplier = 10

            try:
                explicit_min = getattr(cfg, "best_checkpoint_min_interval_steps") if (cfg is not None and hasattr(cfg, "best_checkpoint_min_interval_steps")) else getattr(self.args, "best_checkpoint_min_interval_steps", None)
            except Exception:
                explicit_min = None

            # Determine cadence steps for interval: prefer eval_steps, else save_steps, else logging_steps
            cadence_steps = 0
            if int(eval_steps) > 0:
                cadence_steps = int(eval_steps)
            else:
                try:
                    cadence_steps = int(getattr(self.args, "save_steps", 0) or 0)
                except Exception:
                    cadence_steps = 0
                if cadence_steps <= 0:
                    try:
                        cadence_steps = int(getattr(self.args, "logging_steps", 0) or 0)
                    except Exception:
                        cadence_steps = 0

            computed = (int(cadence_steps) * int(interval_multiplier)) if (int(cadence_steps) > 0 and int(interval_multiplier) > 0) else 0
            min_interval = int(explicit_min) if (explicit_min is not None) else int(computed)
        except Exception:
            min_interval = 0

        try:
            logger.debug(f"Best checkpoint min_interval computed: cadence_steps={cadence_steps}, multiplier={interval_multiplier}, explicit_min={explicit_min} -> min_interval={min_interval}")
        except Exception:
            pass

        # Dynamic eval-period detection and baseline skip if eval cadence not yet known
        try:
            last_eval_step = getattr(self.checkpoint_manager, "_last_eval_step", None)
            if last_eval_step is None and explicit_min is None:
                # Establish baseline on first eval; do not save best here
                setattr(self.checkpoint_manager, "_last_eval_step", int(step))
                logger.debug("⏭️ Skipping best checkpoint on first eval to establish baseline")
                return
            if last_eval_step is not None and int(step) > int(last_eval_step):
                detected_period = int(step) - int(last_eval_step)
                setattr(self.checkpoint_manager, "_last_eval_step", int(step))
                if (min_interval <= 0) and (int(interval_multiplier) > 0) and (explicit_min is None) and (detected_period > 0):
                    min_interval = int(detected_period) * int(interval_multiplier)
                    logger.debug(f"🧮 Derived min_interval dynamically: detected_period={detected_period}, multiplier={interval_multiplier} -> min_interval={min_interval}")
        except Exception:
            pass

        # Skip best checkpointing entirely during the initial warmup window
        if min_interval > 0 and int(step) < int(min_interval):
            logger.debug(
                f"⏭️ Skipping best checkpoint within warmup window: step {step} < min_interval {min_interval}"
            )
            return

        # Track last best save step on the manager for gating
        last_best_step = getattr(self.checkpoint_manager, "_last_best_save_step", None)
        if min_interval > 0:
            if last_best_step is None:
                # First best attempt after warmup: gate on absolute step distance as well
                # Only allow best save if we are at a multiple of min_interval from step 0
                if (int(step) % int(min_interval)) != 0:
                    return
            else:
                # Subsequent best saves must be at least min_interval apart
                if int(step) - int(last_best_step) < int(min_interval):
                    return

        if (
            current_metrics
            and self.checkpoint_manager.is_new_best(current_metrics)
        ):
            # Deduplication: prevent multiple best checkpoint attempts for same step
            metrics_hash = hash(str(sorted(current_metrics.items())))
            last_best_attempt = getattr(self.checkpoint_manager, "_last_best_attempt", None)
            if last_best_attempt and last_best_attempt == (step, metrics_hash):
                return
            
            # Record this attempt
            setattr(self.checkpoint_manager, "_last_best_attempt", (step, metrics_hash))
            
            logger.info("🏆 Creating best checkpoint by direct folder copy")
            best_checkpoint_name = self.checkpoint_manager.create_best_checkpoint_name(
                current_metrics, step
            )
            best_checkpoint_path = os.path.join(
                self.args.output_dir, best_checkpoint_name
            )

            # Remember previous best before updating
            old_best_path = getattr(self.checkpoint_manager, "current_best_dir", None)

            # Replace existing best directory using robust atomic copy (no symlinks)
            try:
                if os.path.islink(best_checkpoint_path) or os.path.exists(best_checkpoint_path):
                    try:
                        if os.path.islink(best_checkpoint_path):
                            os.remove(best_checkpoint_path)
                        else:
                            shutil.rmtree(best_checkpoint_path)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not remove existing best checkpoint path '{best_checkpoint_path}': {e}"
                        )
            except Exception:
                pass

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
                # Record the step for interval gating
                try:
                    setattr(self.checkpoint_manager, "_last_best_save_step", int(step))
                except Exception:
                    pass

                # Remove old best if different
                if (
                    old_best_path
                    and old_best_path != best_checkpoint_path
                    and os.path.exists(old_best_path)
                ):
                    try:
                        if os.path.islink(old_best_path):
                            os.remove(old_best_path)
                        else:
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
        # JSON mode: no coordinate config file
        return

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
