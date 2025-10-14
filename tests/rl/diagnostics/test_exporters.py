"""
Unit tests for diagnostic exporters.

Feature: 004-grpo-post-training
Constitution: v4.1.1
"""

import json

import torch

from src_new.rl.diagnostics.exporters import (
    StandardDiagnosticExporter,
    TensorBoardExporter,
)


class TestStandardDiagnosticExporter:
    """Tests for StandardDiagnosticExporter."""

    def test_export_artifacts_creates_directory(self, tmp_output_dir):
        """Test that export_artifacts creates step-specific directory."""
        exporter = StandardDiagnosticExporter(enabled=True)

        step = 100
        paths = exporter.export_artifacts(step=step, output_dir=tmp_output_dir, rank=0)

        assert len(paths) == 1
        expected_dir = tmp_output_dir / "diagnostics" / "000100"
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    def test_export_artifacts_rank_filtered(self, tmp_output_dir):
        """Test that export_artifacts only runs on rank 0."""
        exporter = StandardDiagnosticExporter(enabled=True)

        # Rank 1 should not export
        paths = exporter.export_artifacts(step=50, output_dir=tmp_output_dir, rank=1)
        assert len(paths) == 0

        # Rank 0 should export
        paths = exporter.export_artifacts(step=50, output_dir=tmp_output_dir, rank=0)
        assert len(paths) == 1

    def test_export_json_saves_dict(self, tmp_output_dir):
        """Test that export_json correctly serializes dictionaries."""
        exporter = StandardDiagnosticExporter(enabled=True)

        data = {
            "step": 10,
            "ratio_mean": 1.05,
            "is_degenerate": False,
            "nested": {"value": 42},
        }

        filepath = tmp_output_dir / "test_data.json"
        exporter.export_json(data, filepath)

        assert filepath.exists()

        # Verify content
        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded == data

    def test_export_json_converts_tensors(self, tmp_output_dir):
        """Test that export_json converts tensors to Python types."""
        exporter = StandardDiagnosticExporter(enabled=True)

        data = {
            "scalar_tensor": torch.tensor(1.5),
            "vector_tensor": torch.tensor([1, 2, 3]),
            "nested": {
                "tensor_value": torch.tensor(99.0),
            }
        }

        filepath = tmp_output_dir / "test_tensors.json"
        exporter.export_json(data, filepath)

        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded["scalar_tensor"] == 1.5
        assert loaded["vector_tensor"] == [1, 2, 3]
        assert loaded["nested"]["tensor_value"] == 99.0

    def test_export_histogram_creates_png(self, tmp_output_dir, mock_ratio_tensor):
        """Test that export_histogram creates PNG file."""
        exporter = StandardDiagnosticExporter(enabled=True)

        filepath = tmp_output_dir / "test_histogram.png"
        exporter.export_histogram(
            values=mock_ratio_tensor,
            filepath=filepath,
            title="Test Histogram",
            xlabel="Ratio",
            ylabel="Frequency",
            bins=30,
            reference_line=1.0,
            reference_label="Identity",
        )

        assert filepath.exists()
        assert filepath.suffix == ".png"
        assert filepath.stat().st_size > 1000  # PNG should be non-trivial size

    def test_export_heatmap_creates_png(self, tmp_output_dir):
        """Test that export_heatmap creates PNG file."""
        exporter = StandardDiagnosticExporter(enabled=True)

        # Create 2D tensor for heatmap
        data = torch.randn(10, 8)

        filepath = tmp_output_dir / "test_heatmap.png"
        exporter.export_heatmap(
            data=data,
            filepath=filepath,
            title="Test Heatmap",
            xlabel="Columns",
            ylabel="Rows",
            row_labels=[f"Layer{i}" for i in range(10)],
            log_scale=False,
        )

        assert filepath.exists()
        assert filepath.suffix == ".png"
        assert filepath.stat().st_size > 1000

    def test_disabled_exporter_no_op(self, tmp_output_dir):
        """Test that disabled exporter performs no operations."""
        exporter = StandardDiagnosticExporter(enabled=False)

        # These should all be no-ops
        paths = exporter.export_artifacts(step=1, output_dir=tmp_output_dir, rank=0)
        assert len(paths) == 0

        filepath = tmp_output_dir / "test.json"
        exporter.export_json({"key": "value"}, filepath)
        assert not filepath.exists()

        png_path = tmp_output_dir / "test.png"
        exporter.export_histogram(
            values=torch.randn(10),
            filepath=png_path,
            title="Test",
        )
        assert not png_path.exists()


class TestTensorBoardExporter:
    """Tests for TensorBoardExporter."""

    def test_initialization_creates_writer(self, tmp_output_dir):
        """Test that TensorBoard writer is initialized on rank 0."""
        log_dir = tmp_output_dir / "tb_logs"

        exporter = TensorBoardExporter(log_dir=log_dir, enabled=True, rank=0)

        assert exporter.writer is not None
        assert log_dir.exists()

        exporter.close()

    def test_initialization_skipped_on_rank_1(self, tmp_output_dir):
        """Test that TensorBoard writer is not initialized on rank > 0."""
        log_dir = tmp_output_dir / "tb_logs_rank1"

        exporter = TensorBoardExporter(log_dir=log_dir, enabled=True, rank=1)

        assert exporter.writer is None

    def test_log_scalars_accepts_dict(self, tmp_output_dir):
        """Test that log_scalars handles dictionary of scalars."""
        log_dir = tmp_output_dir / "tb_logs"
        exporter = TensorBoardExporter(log_dir=log_dir, enabled=True, rank=0)

        scalars = {
            "trust_region/ratio_mean": 1.05,
            "trust_region/ratio_std": 0.18,
            "trust_region/is_degenerate": 0.0,
        }

        # Should not raise
        exporter.log_scalars(scalars, step=10)

        exporter.close()

    def test_log_scalars_converts_tensors(self, tmp_output_dir):
        """Test that log_scalars converts tensor values to floats."""
        log_dir = tmp_output_dir / "tb_logs"
        exporter = TensorBoardExporter(log_dir=log_dir, enabled=True, rank=0)

        scalars = {
            "metric_a": torch.tensor(1.5),
            "metric_b": torch.tensor([2.0, 3.0, 4.0]),  # Should take mean
        }

        # Should not raise
        exporter.log_scalars(scalars, step=5)

        exporter.close()

    def test_log_histogram_accepts_tensor(self, tmp_output_dir, mock_ratio_tensor):
        """Test that log_histogram handles tensor values."""
        log_dir = tmp_output_dir / "tb_logs"
        exporter = TensorBoardExporter(log_dir=log_dir, enabled=True, rank=0)

        # Should not raise
        exporter.log_histogram("trust_region/ratios", mock_ratio_tensor, step=20)

        exporter.close()

    def test_disabled_exporter_no_op(self, tmp_output_dir):
        """Test that disabled TensorBoard exporter performs no operations."""
        log_dir = tmp_output_dir / "tb_logs_disabled"
        exporter = TensorBoardExporter(log_dir=log_dir, enabled=False, rank=0)

        assert exporter.writer is None

        # These should all be no-ops
        exporter.log_scalars({"metric": 1.0}, step=1)
        exporter.log_histogram("metric", torch.randn(10), step=1)
        exporter.close()  # Should not raise

    def test_rank_1_no_logging(self, tmp_output_dir, mock_ratio_tensor):
        """Test that rank > 0 does not log to TensorBoard."""
        log_dir = tmp_output_dir / "tb_logs_rank1"
        exporter = TensorBoardExporter(log_dir=log_dir, enabled=True, rank=1)

        # These should all be no-ops
        exporter.log_scalars({"metric": 1.0}, step=1)
        exporter.log_histogram("metric", mock_ratio_tensor, step=1)

        assert exporter.writer is None
