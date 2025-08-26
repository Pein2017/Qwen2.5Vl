from __future__ import annotations

import copy
from typing import Any, Dict

from PIL import Image

from src_new.augmentation.base import AugmentationPipeline
from src_new.config.augmentation_config import (
    AngleRotateConfig,
    AugmentationConfig,
)


def make_sample() -> Dict[str, Any]:
    # Provided sample
    return {
        "images": ["images/QC-20230217-0000279_19621.jpeg"],
        "objects": [
            {
                "bbox_2d": [264, 144, 326, 201],
                "desc": "螺丝、光纤插头/BBU安装螺丝,显示完整,符合要求",
            },
            {
                "quad": [704, 487, 670, 554, 973, 644, 993, 590],
                "desc": "标签/4G-RRU3-光纤",
            },
            {
                "line": [
                    614,
                    1271,
                    498,
                    1179,
                    419,
                    1216,
                    280,
                    1280,
                    117,
                    1456,
                    3,
                    1721,
                ],
                "desc": "光纤/有遮挡,有保护措施,弯曲半径合理/蛇形管",
            },
        ],
        "width": 532,
        "height": 728,
    }


def make_image(w: int, h: int) -> Image.Image:
    # Create a blank RGB image; actual content is irrelevant for geometry checks
    return Image.new("RGB", (w, h), color=(10, 20, 30))


def run_case_identity_skip_lines() -> None:
    sample = make_sample()
    img = make_image(sample["width"], sample["height"])

    aug_cfg = AugmentationConfig(
        enabled=True,
        rng_seed=123,
        apply_to_teachers=False,
        lines_policy="identity",  # skip augmentation if lines present
        debug_visualization=False,
        debug_output_dir=None,
        op=AngleRotateConfig(
            sample_mode="fixed",
            fixed_angle_deg=30.0,  # would rotate, but lines cause skip
            angle_min_deg=None,
            angle_max_deg=None,
            angles_set_deg=None,
            expand=True,
            interpolation="bilinear",
            fill_color=(0, 0, 0),
        ),
    )

    pipeline = AugmentationPipeline.from_config(aug_cfg)
    imgs_out, sample_out = pipeline.apply(
        sample=copy.deepcopy(sample), images=[img], sample_index=0
    )

    assert (
        sample_out["width"] == sample["width"]
        and sample_out["height"] == sample["height"]
    ), "identity policy should keep W/H"
    assert sample_out["objects"] == sample["objects"], (
        "identity policy should keep objects unchanged"
    )
    assert imgs_out[0].size == img.size, "identity policy should keep image size"
    print("OK: identity_skip_lines")


def run_case_drop_lines_expand() -> None:
    sample = make_sample()
    img = make_image(sample["width"], sample["height"])  # 532x728

    aug_cfg = AugmentationConfig(
        enabled=True,
        rng_seed=123,
        apply_to_teachers=False,
        lines_policy="drop_objects",  # drop lines, then rotate others
        debug_visualization=False,
        debug_output_dir=None,
        op=AngleRotateConfig(
            sample_mode="fixed",
            fixed_angle_deg=30.0,  # non-zero -> bbox becomes quad
            angle_min_deg=None,
            angle_max_deg=None,
            angles_set_deg=None,
            expand=True,  # canvas expands
            interpolation="bilinear",
            fill_color=(0, 0, 0),
        ),
    )

    pipeline = AugmentationPipeline.from_config(aug_cfg)
    imgs_out, sample_out = pipeline.apply(
        sample=copy.deepcopy(sample), images=[img], sample_index=0
    )

    # No line objects remain
    assert not any("line" in o for o in sample_out["objects"]), (
        "lines should be dropped"
    )
    # bbox should be converted to quad
    assert any("quad" in o for o in sample_out["objects"]), (
        "bbox should convert to quad under non-zero angle"
    )
    # canvas should expand (size change)
    assert imgs_out[0].size != img.size, "expand=True should change image size"
    assert (sample_out["width"], sample_out["height"]) == imgs_out[0].size, (
        "sample W/H should match image size"
    )
    print("OK: drop_lines_expand")


def run_case_keep_size_fixed_zero() -> None:
    sample = make_sample()
    img = make_image(sample["width"], sample["height"])  # 532x728

    aug_cfg = AugmentationConfig(
        enabled=True,
        rng_seed=999,
        apply_to_teachers=False,
        lines_policy="drop_objects",  # drop lines allowed
        debug_visualization=False,
        debug_output_dir=None,
        op=AngleRotateConfig(
            sample_mode="fixed",
            fixed_angle_deg=0.0,  # identity angle
            angle_min_deg=None,
            angle_max_deg=None,
            angles_set_deg=None,
            expand=False,  # keep canvas
            interpolation="nearest",
            fill_color=(0, 0, 0),
        ),
    )

    pipeline = AugmentationPipeline.from_config(aug_cfg)
    imgs_out, sample_out = pipeline.apply(
        sample=copy.deepcopy(sample), images=[img], sample_index=42
    )

    # Without lines, objects retained; bbox remains bbox (identity transform)
    assert any("bbox_2d" in o for o in sample_out["objects"]), (
        "bbox should remain bbox for 0 angle"
    )
    # Size unchanged
    assert imgs_out[0].size == img.size, "expand=False and 0° should keep size"
    assert (sample_out["width"], sample_out["height"]) == img.size
    print("OK: keep_size_fixed_zero")


if __name__ == "__main__":
    run_case_identity_skip_lines()
    run_case_drop_lines_expand()
    run_case_keep_size_fixed_zero()
    print("All augmentation tests passed.")
