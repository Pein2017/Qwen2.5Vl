from typing import Any, Dict


def conversation_output_stats(inputs: Dict[str, Any]) -> Dict[str, Any]:
	stats: Dict[str, Any] = {}
	if "input_ids" in inputs and hasattr(inputs["input_ids"], "shape"):
		ids = inputs["input_ids"]
		stats["seq_len"] = int(ids.shape[1]) if ids.dim() == 2 else int(ids.shape[0])
		stats["batch"] = int(ids.shape[0]) if ids.dim() == 2 else 1
	if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "shape"):
		pv = inputs["pixel_values"]
		stats["num_patches"] = int(pv.shape[0])
		stats["patch_dim"] = int(pv.shape[1]) if pv.dim() >= 2 else None
	if "image_grid_thw" in inputs and hasattr(inputs["image_grid_thw"], "shape"):
		grid = inputs["image_grid_thw"]
		stats["num_images"] = int(grid.shape[0])
	return stats

__all__ = ["conversation_output_stats"]
