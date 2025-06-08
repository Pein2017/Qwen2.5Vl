"""
Training Monitor for Qwen2.5-VL Object Detection.

Handles monitoring and logging of model predictions, ground truth data,
and token-level analysis during training. Decoupled from loss computation
for clean separation of concerns.


"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.logger_utils import get_monitor_logger


class TokenStats:
    """Container for token-level statistics."""

    def __init__(self):
        self.input_lengths: List[int] = []
        self.prompt_lengths: List[int] = []
        self.generated_lengths: List[int] = []
        self.gt_lengths: List[int] = []
        self.total_output_lengths: List[int] = []

    def add_sample(
        self,
        input_length: int = 0,
        prompt_length: int = 0,
        generated_length: int = 0,
        gt_length: int = 0,
        total_output_length: int = 0,
    ):
        """Add token statistics for a single sample."""
        self.input_lengths.append(input_length)
        self.prompt_lengths.append(prompt_length)
        self.generated_lengths.append(generated_length)
        self.gt_lengths.append(gt_length)
        self.total_output_lengths.append(total_output_length)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""

        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "avg_input_length": safe_avg(self.input_lengths),
            "avg_prompt_length": safe_avg(self.prompt_lengths),
            "avg_generated_length": safe_avg(self.generated_lengths),
            "avg_gt_length": safe_avg(self.gt_lengths),
            "avg_total_output_length": safe_avg(self.total_output_lengths),
            "total_samples": len(self.input_lengths),
        }


class PredictionRecord:
    """Container for prediction and ground truth data."""

    def __init__(
        self,
        sample_idx: int,
        predicted_objects: List[Dict],
        ground_truth_objects: List[Dict],
        predicted_text: str = "",
        ground_truth_text: str = "",
        token_stats: Optional[Dict] = None,
    ):
        self.sample_idx = sample_idx
        self.predicted_objects = predicted_objects
        self.ground_truth_objects = ground_truth_objects
        self.predicted_text = predicted_text
        self.ground_truth_text = ground_truth_text
        self.token_stats = token_stats or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_idx": self.sample_idx,
            "timestamp": self.timestamp,
            "predicted_objects": self.predicted_objects,
            "ground_truth_objects": self.ground_truth_objects,
            "predicted_text": self.predicted_text,
            "ground_truth_text": self.ground_truth_text,
            "token_stats": self.token_stats,
            "object_counts": {
                "predicted": len(self.predicted_objects),
                "ground_truth": len(self.ground_truth_objects),
            },
        }


class TrainingMonitor:
    """
    Training monitor for object detection model.

    Handles all monitoring and logging functionality that was previously
    embedded in the loss computation module.
    """

    def __init__(
        self,
        log_dir: str = "logs/training_monitor",
        save_predictions: bool = True,
        save_token_analysis: bool = True,
        save_raw_text: bool = False,
        max_text_preview_length: int = 200,
    ):
        self.log_dir = Path(log_dir)
        self.save_predictions = save_predictions
        self.save_token_analysis = save_token_analysis
        self.save_raw_text = save_raw_text
        self.max_text_preview_length = max_text_preview_length

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = get_monitor_logger()

        # Initialize session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_count = 0
        self.total_samples = 0

        # File paths
        self.predictions_file = self.log_dir / f"predictions_{self.session_id}.jsonl"
        self.token_stats_file = self.log_dir / f"token_stats_{self.session_id}.jsonl"
        self.summary_file = self.log_dir / f"summary_{self.session_id}.json"

        # Session statistics
        self.session_stats = {
            "start_time": datetime.now().isoformat(),
            "total_batches": 0,
            "total_samples": 0,
            "total_predicted_objects": 0,
            "total_gt_objects": 0,
            "token_stats": TokenStats(),
        }

        self.logger.info(f"🔍 TrainingMonitor initialized:")
        self.logger.info(f"   Session ID: {self.session_id}")
        self.logger.info(f"   Log directory: {self.log_dir}")
        self.logger.info(f"   Save predictions: {save_predictions}")
        self.logger.info(f"   Save token analysis: {save_token_analysis}")
        self.logger.info(f"   Save raw text: {save_raw_text}")

    def log_batch_analysis(
        self,
        batch_idx: int,
        predicted_objects_batch: List[List[Dict]],
        ground_truth_objects_batch: List[List[Dict]],
        predicted_texts: Optional[List[str]] = None,
        ground_truth_texts: Optional[List[str]] = None,
        token_stats: Optional[TokenStats] = None,
        input_parts: Optional[List] = None,
    ):
        """
        Log comprehensive batch analysis.

        Args:
            batch_idx: Current batch index
            predicted_objects_batch: List of predicted objects for each sample
            ground_truth_objects_batch: List of ground truth objects for each sample
            predicted_texts: Optional list of raw predicted texts
            ground_truth_texts: Optional list of raw ground truth texts
            token_stats: Optional token statistics
            input_parts: Optional input token sequences
        """
        batch_size = len(predicted_objects_batch)
        self.batch_count += 1
        self.total_samples += batch_size

        # Log batch overview
        self.logger.info(f"📊 BATCH {batch_idx} ANALYSIS:")
        self.logger.info(f"   Batch size: {batch_size}")

        # Process each sample in the batch
        batch_records = []
        for i in range(batch_size):
            sample_idx = self.total_samples - batch_size + i

            pred_objects = (
                predicted_objects_batch[i] if i < len(predicted_objects_batch) else []
            )
            gt_objects = (
                ground_truth_objects_batch[i]
                if i < len(ground_truth_objects_batch)
                else []
            )
            pred_text = (
                predicted_texts[i]
                if predicted_texts and i < len(predicted_texts)
                else ""
            )
            gt_text = (
                ground_truth_texts[i]
                if ground_truth_texts and i < len(ground_truth_texts)
                else ""
            )

            # Create prediction record
            sample_token_stats = {}
            if token_stats and i < len(token_stats.input_lengths):
                sample_token_stats = {
                    "input_length": token_stats.input_lengths[i]
                    if i < len(token_stats.input_lengths)
                    else 0,
                    "prompt_length": token_stats.prompt_lengths[i]
                    if i < len(token_stats.prompt_lengths)
                    else 0,
                    "generated_length": token_stats.generated_lengths[i]
                    if i < len(token_stats.generated_lengths)
                    else 0,
                    "gt_length": token_stats.gt_lengths[i]
                    if i < len(token_stats.gt_lengths)
                    else 0,
                    "total_output_length": token_stats.total_output_lengths[i]
                    if i < len(token_stats.total_output_lengths)
                    else 0,
                }

            record = PredictionRecord(
                sample_idx=sample_idx,
                predicted_objects=pred_objects,
                ground_truth_objects=gt_objects,
                predicted_text=pred_text if self.save_raw_text else "",
                ground_truth_text=gt_text if self.save_raw_text else "",
                token_stats=sample_token_stats,
            )

            batch_records.append(record)

            # Log sample details
            self.logger.info(
                f"   Sample {i}: Pred={len(pred_objects)}, GT={len(gt_objects)} objects"
            )

            # Update session statistics
            self.session_stats["total_predicted_objects"] += len(pred_objects)
            self.session_stats["total_gt_objects"] += len(gt_objects)

        # Save batch records
        if self.save_predictions:
            self._save_batch_predictions(batch_records)

        # Log token statistics
        if token_stats:
            self._log_token_statistics(batch_idx, token_stats)
            # Update session token stats
            for i in range(len(token_stats.input_lengths)):
                self.session_stats["token_stats"].add_sample(
                    input_length=token_stats.input_lengths[i]
                    if i < len(token_stats.input_lengths)
                    else 0,
                    prompt_length=token_stats.prompt_lengths[i]
                    if i < len(token_stats.prompt_lengths)
                    else 0,
                    generated_length=token_stats.generated_lengths[i]
                    if i < len(token_stats.generated_lengths)
                    else 0,
                    gt_length=token_stats.gt_lengths[i]
                    if i < len(token_stats.gt_lengths)
                    else 0,
                    total_output_length=token_stats.total_output_lengths[i]
                    if i < len(token_stats.total_output_lengths)
                    else 0,
                )

        # Update session stats
        self.session_stats["total_batches"] = self.batch_count
        self.session_stats["total_samples"] = self.total_samples

        # Log batch summary
        total_pred = sum(len(record.predicted_objects) for record in batch_records)
        total_gt = sum(len(record.ground_truth_objects) for record in batch_records)

        self.logger.info(
            f"   Total objects - Predicted: {total_pred}, Ground Truth: {total_gt}"
        )

        # Save session summary periodically
        if self.batch_count % 10 == 0:
            self._save_session_summary()

    def log_token_analysis(
        self,
        sample_idx: int,
        input_length: int = 0,
        prompt_length: int = 0,
        generated_length: int = 0,
        gt_length: int = 0,
        total_output_length: int = 0,
        generated_text: str = "",
        gt_text: str = "",
    ):
        """Log detailed token analysis for a single sample."""
        self.logger.info(f"📊 TOKEN ANALYSIS - Sample {sample_idx}:")
        self.logger.info(f"   Input length: {input_length} tokens")
        self.logger.info(f"   Prompt length: {prompt_length} tokens")
        self.logger.info(f"   Generated length: {generated_length} tokens")
        self.logger.info(f"   GT length: {gt_length} tokens")
        self.logger.info(f"   Total output length: {total_output_length} tokens")

        if generated_text:
            preview = self._get_text_preview(generated_text)
            self.logger.info(f"   Generated text preview: '{preview}'")

        if gt_text:
            preview = self._get_text_preview(gt_text)
            self.logger.info(f"   GT text preview: '{preview}'")

    def log_generation_analysis(
        self,
        sample_idx: int,
        prompt_length: int,
        max_new_tokens: int,
        actual_generated_tokens: int,
        total_output_length: int,
        generated_text: str,
        parsed_objects_count: int,
    ):
        """Log generation-specific analysis."""
        self.logger.info(f"📊 GENERATION ANALYSIS - Sample {sample_idx}:")
        self.logger.info(f"   Prompt length: {prompt_length} tokens")
        self.logger.info(f"   Max new tokens: {max_new_tokens}")
        self.logger.info(f"   Generated tokens: {actual_generated_tokens}")
        self.logger.info(f"   Total output length: {total_output_length} tokens")
        self.logger.info(f"   Generated text length: {len(generated_text)} chars")
        self.logger.info(f"   Parsed objects count: {parsed_objects_count}")

        if generated_text:
            preview = self._get_text_preview(generated_text)
            self.logger.info(f"   Generated text preview: '{preview}'")
        else:
            self.logger.warning(f"   ⚠️ Empty generated text for sample {sample_idx}")

    def _log_token_statistics(self, batch_idx: int, token_stats: TokenStats):
        """Log token statistics for the batch."""
        summary = token_stats.get_summary()

        self.logger.info(f"📊 BATCH {batch_idx} TOKEN STATISTICS:")
        self.logger.info(f"   Samples: {summary['total_samples']}")
        self.logger.info(
            f"   Avg input length: {summary['avg_input_length']:.1f} tokens"
        )
        self.logger.info(
            f"   Avg prompt length: {summary['avg_prompt_length']:.1f} tokens"
        )
        self.logger.info(
            f"   Avg generated length: {summary['avg_generated_length']:.1f} tokens"
        )
        self.logger.info(f"   Avg GT length: {summary['avg_gt_length']:.1f} tokens")
        self.logger.info(
            f"   Avg total output length: {summary['avg_total_output_length']:.1f} tokens"
        )

        # Save token statistics
        if self.save_token_analysis:
            token_record = {
                "batch_idx": batch_idx,
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "detailed_stats": {
                    "input_lengths": token_stats.input_lengths,
                    "prompt_lengths": token_stats.prompt_lengths,
                    "generated_lengths": token_stats.generated_lengths,
                    "gt_lengths": token_stats.gt_lengths,
                    "total_output_lengths": token_stats.total_output_lengths,
                },
            }

            with open(self.token_stats_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(token_record) + "\n")

    def _save_batch_predictions(self, batch_records: List[PredictionRecord]):
        """Save batch prediction records to file."""
        with open(self.predictions_file, "a", encoding="utf-8") as f:
            for record in batch_records:
                f.write(json.dumps(record.to_dict()) + "\n")

    def _save_session_summary(self):
        """Save session summary to file."""
        summary = {
            "session_id": self.session_id,
            "end_time": datetime.now().isoformat(),
            "total_batches": self.session_stats["total_batches"],
            "total_samples": self.session_stats["total_samples"],
            "total_predicted_objects": self.session_stats["total_predicted_objects"],
            "total_gt_objects": self.session_stats["total_gt_objects"],
            "avg_objects_per_sample": {
                "predicted": self.session_stats["total_predicted_objects"]
                / max(1, self.session_stats["total_samples"]),
                "ground_truth": self.session_stats["total_gt_objects"]
                / max(1, self.session_stats["total_samples"]),
            },
            "token_statistics": self.session_stats["token_stats"].get_summary(),
        }

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _get_text_preview(self, text: str) -> str:
        """Get preview of text with length limit."""
        if len(text) <= self.max_text_preview_length:
            return text
        return text[: self.max_text_preview_length] + "..."

    def finalize_session(self):
        """Finalize the monitoring session and save final summary."""
        self.logger.info("🏁 Finalizing training monitor session...")

        # Save final session summary
        self._save_session_summary()

        # Log final statistics
        self.logger.info(f"📊 FINAL SESSION STATISTICS:")
        self.logger.info(f"   Session ID: {self.session_id}")
        self.logger.info(f"   Total batches: {self.session_stats['total_batches']}")
        self.logger.info(f"   Total samples: {self.session_stats['total_samples']}")
        self.logger.info(
            f"   Total predicted objects: {self.session_stats['total_predicted_objects']}"
        )
        self.logger.info(
            f"   Total GT objects: {self.session_stats['total_gt_objects']}"
        )

        token_summary = self.session_stats["token_stats"].get_summary()
        self.logger.info(f"   Average token lengths:")
        self.logger.info(f"     Input: {token_summary['avg_input_length']:.1f}")
        self.logger.info(f"     Prompt: {token_summary['avg_prompt_length']:.1f}")
        self.logger.info(f"     Generated: {token_summary['avg_generated_length']:.1f}")
        self.logger.info(f"     Ground Truth: {token_summary['avg_gt_length']:.1f}")

        self.logger.info(f"   Log files saved to: {self.log_dir}")
        self.logger.info("✅ Training monitor session finalized")


def create_training_monitor(
    log_dir: str = "logs/training_monitor",
    save_predictions: bool = True,
    save_token_analysis: bool = True,
    save_raw_text: bool = False,
) -> TrainingMonitor:
    """
    Factory function to create a training monitor instance.

    Args:
        log_dir: Directory for log files
        save_predictions: Whether to save prediction records
        save_token_analysis: Whether to save token analysis
        save_raw_text: Whether to save raw text content

    Returns:
        Configured TrainingMonitor instance
    """
    return TrainingMonitor(
        log_dir=log_dir,
        save_predictions=save_predictions,
        save_token_analysis=save_token_analysis,
        save_raw_text=save_raw_text,
    )
