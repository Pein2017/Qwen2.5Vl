"""
Diagnostic Export Interface

Defines the contract for exporting diagnostic artifacts (histograms, heatmaps, JSON)
to disk at specified training steps.
"""

from pathlib import Path
from typing import Any, Dict, List, Protocol

import torch


class DiagnosticExporter(Protocol):
    """Interface for exporting diagnostic artifacts to disk."""

    def export_artifacts(
        self,
        step: int,
        output_dir: Path,
        rank: int = 0,
    ) -> List[Path]:
        """
        Export diagnostic artifacts for a training step.

        Only executes on rank 0 (main process) for distributed training.

        Args:
            step: Training step number (used in output path)
            output_dir: Base output directory (artifacts go to output_dir/diagnostics/{step:06d}/)
            rank: Process rank (default 0; export only if rank == 0)

        Returns:
            List of Path objects for all exported files

        Raises:
            OSError: If output directory cannot be created
        """
        ...

    def export_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """
        Export dictionary as formatted JSON file.

        Args:
            data: Dictionary to serialize (must be JSON-compatible)
            filepath: Target file path
        """
        ...

    def export_histogram(
        self,
        values: torch.Tensor,
        filepath: Path,
        title: str,
        xlabel: str = "Value",
        ylabel: str = "Count",
        bins: int = 50,
    ) -> None:
        """
        Export tensor values as histogram PNG.

        Args:
            values: 1D tensor of values to plot
            filepath: Target PNG file path
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            bins: Number of histogram bins
        """
        ...

    def export_heatmap(
        self,
        values: torch.Tensor,
        row_labels: List[str],
        col_labels: List[str],
        filepath: Path,
        title: str,
        cmap: str = "viridis",
    ) -> None:
        """
        Export 2D tensor as heatmap PNG.

        Args:
            values: 2D tensor of shape [rows, cols]
            row_labels: Labels for rows (e.g., layer names)
            col_labels: Labels for columns (e.g., training steps)
            filepath: Target PNG file path
            title: Plot title
            cmap: Matplotlib colormap name
        """
        ...


class StandardDiagnosticExporter:
    """
    Concrete implementation of DiagnosticExporter using matplotlib and JSON.

    Usage:
        exporter = StandardDiagnosticExporter()
        if accelerator.is_main_process:
            paths = exporter.export_artifacts(
                step=100,
                output_dir=Path("outputs/grpo_run"),
                diagnostics=[trust_region_diag, reward_profile, ...],
            )
            logger.info(f"Exported {len(paths)} artifacts")
    """

    def __init__(self):
        self.rank = 0  # Set by caller

    def export_artifacts(
        self,
        step: int,
        output_dir: Path,
        rank: int = 0,
    ) -> List[Path]:
        """Export all diagnostic artifacts for a step (rank-0 only)."""
        if rank != 0:
            return []

        step_dir = output_dir / "diagnostics" / f"{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        exported_paths = []
        # Subclasses implement actual export logic
        return exported_paths

    def export_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """Export dictionary as JSON."""
        import json
        from dataclasses import asdict

        # Handle dataclass instances
        if hasattr(data, "__dataclass_fields__"):
            data = asdict(data)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def export_histogram(
        self,
        values: torch.Tensor,
        filepath: Path,
        title: str,
        xlabel: str = "Value",
        ylabel: str = "Count",
        bins: int = 50,
    ) -> None:
        """Export histogram as PNG."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.hist(values.cpu().numpy(), bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

    def export_heatmap(
        self,
        values: torch.Tensor,
        row_labels: List[str],
        col_labels: List[str],
        filepath: Path,
        title: str,
        cmap: str = "viridis",
    ) -> None:
        """Export heatmap as PNG."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(values.cpu().numpy(), aspect="auto", cmap=cmap)
        plt.colorbar()
        plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
        plt.yticks(range(len(row_labels)), row_labels)
        plt.xlabel("Steps")
        plt.ylabel("Layers")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()


class TensorBoardExporter:
    """
    Helper to export diagnostic entities to TensorBoard SummaryWriter.

    Usage:
        exporter = TensorBoardExporter(writer=tb_writer)
        exporter.log_diagnostic(diagnostic, step=100)
    """

    def __init__(self, writer):
        """
        Args:
            writer: torch.utils.tensorboard.SummaryWriter instance
        """
        self.writer = writer

    def log_diagnostic(self, diagnostic, step: int) -> None:
        """
        Log diagnostic entity to TensorBoard.

        Calls diagnostic.to_tensorboard() and logs all scalars.

        Args:
            diagnostic: Diagnostic entity with to_tensorboard() method
            step: Global training step
        """
        if not hasattr(diagnostic, "to_tensorboard"):
            raise TypeError(
                f"{type(diagnostic).__name__} must implement to_tensorboard()"
            )

        scalar_dict = diagnostic.to_tensorboard()
        for key, value in scalar_dict.items():
            self.writer.add_scalar(key, value, global_step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log tensor histogram to TensorBoard."""
        self.writer.add_histogram(tag, values, global_step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text to TensorBoard."""
        self.writer.add_text(tag, text, global_step=step)
