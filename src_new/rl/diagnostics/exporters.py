"""
Diagnostic Exporters Implementation

Provides concrete implementations of DiagnosticExporter protocol for:
- Standard JSON/PNG exports to disk
- TensorBoard scalar/histogram logging

Feature: 004-grpo-post-training
Constitution: v4.1.1
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib


matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class StandardDiagnosticExporter:
    """Standard implementation of DiagnosticExporter for JSON and PNG exports."""

    def __init__(self, enabled: bool = True):
        """
        Initialize exporter.

        Args:
            enabled: If False, export_artifacts becomes a no-op
        """
        self.enabled = enabled

    def export_artifacts(
        self,
        step: int,
        output_dir: Path,
        rank: int = 0,
    ) -> List[Path]:
        """
        Export diagnostic artifacts for a training step (rank-0 only).

        Args:
            step: Training step number
            output_dir: Base output directory
            rank: Process rank (only rank 0 exports)

        Returns:
            List of exported file paths
        """
        if not self.enabled or rank != 0:
            return []

        # Create step-specific directory
        step_dir = output_dir / "diagnostics" / f"{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Created diagnostic export directory: {step_dir}")
        return [step_dir]

    def export_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """
        Export dictionary as formatted JSON file.

        Args:
            data: Dictionary to serialize
            filepath: Target file path
        """
        if not self.enabled:
            return

        # Convert any tensor values to Python types
        def convert_value(v):
            if isinstance(v, torch.Tensor):
                return v.item() if v.numel() == 1 else v.tolist()
            elif isinstance(v, (np.ndarray, np.number)):
                return v.tolist() if hasattr(v, "tolist") else float(v)
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [convert_value(val) for val in v]
            return v

        serializable_data = convert_value(data)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Exported JSON to {filepath}")

    def export_histogram(
        self,
        values: torch.Tensor,
        filepath: Path,
        title: str,
        xlabel: str = "Value",
        ylabel: str = "Count",
        bins: int = 50,
        range_limits: Optional[tuple] = None,
        reference_line: Optional[float] = None,
        reference_label: Optional[str] = None,
    ) -> None:
        """
        Export histogram as PNG image.

        Args:
            values: Tensor of values to plot
            filepath: Target file path
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            bins: Number of histogram bins
            range_limits: Optional (min, max) tuple for histogram range
            reference_line: Optional vertical line to draw (e.g., identity at 1.0)
            reference_label: Label for reference line
        """
        if not self.enabled:
            return

        # Convert to numpy
        values_np = values.detach().cpu().numpy().flatten()

        # Create figure
        plt.figure(figsize=(10, 6))
        plt.hist(values_np, bins=bins, range=range_limits, alpha=0.7, edgecolor="black")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # Add reference line if specified
        if reference_line is not None:
            plt.axvline(
                reference_line,
                color="r",
                linestyle="--",
                linewidth=2,
                label=reference_label or "Reference",
            )
            plt.legend()

        # Save figure
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug(f"Exported histogram to {filepath}")

    def export_heatmap(
        self,
        data: torch.Tensor,
        filepath: Path,
        title: str,
        xlabel: str = "X",
        ylabel: str = "Y",
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        cmap: str = "viridis",
        log_scale: bool = False,
    ) -> None:
        """
        Export 2D heatmap as PNG image.

        Args:
            data: 2D tensor to visualize
            filepath: Target file path
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            row_labels: Optional row labels
            col_labels: Optional column labels
            cmap: Matplotlib colormap name
            log_scale: If True, use log color scale
        """
        if not self.enabled:
            return

        # Convert to numpy
        data_np = data.detach().cpu().numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Apply log scale if requested
        if log_scale and (data_np > 0).all():
            data_np = np.log10(data_np + 1e-10)
            title = f"{title} (log scale)"

        # Create heatmap
        im = ax.imshow(data_np, cmap=cmap, aspect="auto")

        # Set labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Set tick labels if provided
        if row_labels:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, rotation=0)

        if col_labels:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=90)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Save figure
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug(f"Exported heatmap to {filepath}")


class TensorBoardExporter:
    """Wrapper for TensorBoard logging with rank-awareness."""

    def __init__(
        self,
        log_dir: Path,
        enabled: bool = True,
        rank: int = 0,
    ):
        """
        Initialize TensorBoard writer.

        Args:
            log_dir: TensorBoard log directory
            enabled: If False, all logging becomes no-op
            rank: Process rank (only rank 0 logs)
        """
        self.enabled = enabled
        self.rank = rank
        self.writer: Optional[SummaryWriter] = None

        if self.enabled and self.rank == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"Initialized TensorBoard writer at {log_dir}")

    def log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        """
        Log multiple scalar metrics at once.

        Args:
            scalars: Dict mapping metric name to value
            step: Training step
        """
        if not self.enabled or self.rank != 0 or self.writer is None:
            return

        for name, value in scalars.items():
            # Convert tensor to scalar if needed
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.mean().item()
            elif isinstance(value, (np.ndarray, np.number)):
                value = float(value)

            self.writer.add_scalar(name, value, global_step=step)

        logger.debug(f"Logged {len(scalars)} scalars to TensorBoard at step {step}")

    def log_histogram(self, name: str, values: torch.Tensor, step: int) -> None:
        """
        Log histogram of tensor values.

        Args:
            name: Metric name
            values: Tensor to histogram
            step: Training step
        """
        if not self.enabled or self.rank != 0 or self.writer is None:
            return

        self.writer.add_histogram(name, values, global_step=step)
        logger.debug(f"Logged histogram '{name}' to TensorBoard at step {step}")

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            logger.info("Closed TensorBoard writer")
