from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw

from src_new.augmentation import ObjectAwareAugmentationPipeline
from src_new.config.augmentation_config import (
    AugmentationConfig,
    CriteriaConfig,
    ImageGeomConfig,
    LineAugConfig,
    OcclusionCriterionConfig,
    PhotometricConfig,
)


def _make_image(size_wh: Tuple[int, int] = (160, 120)) -> Image.Image:
    w, h = size_wh
    img = Image.new("RGB", (w, h), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 60, 40], outline=(200, 0, 0), width=2)
    draw.polygon([(80, 20), (120, 20), (120, 60), (80, 60)], outline=(0, 200, 0))
    draw.line([(20, 80), (100, 100)], fill=(0, 0, 200), width=2)
    return img


def _base_sample(size_wh: Tuple[int, int] = (160, 120)) -> Dict[str, Any]:
    w, h = size_wh
    return {
        "width": w,
        "height": h,
        "objects": [
            {"bbox_2d": [12, 12, 58, 38], "desc": "rect"},
            {"quad": [82, 22, 118, 22, 118, 58, 82, 58], "desc": "box"},
            {"line": [22, 82, 98, 98], "desc": "wire"},
        ],
    }


class TestObjectAwareAug(unittest.TestCase):
    def test_image_geom_rotate_invariants(self):
        sample = _base_sample()
        img = _make_image()

        cfg = AugmentationConfig(
            enabled=True,
            rng_seed=123,
            apply_to_teachers=False,
            lines_policy="transform",
            debug_visualization=False,
            debug_output_dir=None,
            image_geom=ImageGeomConfig(
                rotate_deg_range=(30.0, 30.0),
                translate_pct=0.0,
                scale_range=(1.0, 1.0),
                perspective_pct=0.0,
                crop_pct=0.0,
                multiscale_short_edges=None,
            ),
            photometric=None,
            lines=None,
            type_policies=None,
            ocr=None,
            criteria=CriteriaConfig(occlusion=None),
        )

        pipeline = ObjectAwareAugmentationPipeline.from_config(cfg)
        out_imgs, out_sample = pipeline.apply(
            copy.deepcopy(sample), [img], sample_index=42
        )

        # Image size should change with 30° rotation
        self.assertNotEqual(
            (img.width, img.height), (out_imgs[0].width, out_imgs[0].height)
        )
        # All coordinates must be within bounds; quads must have positive area
        W, H = out_sample["width"], out_sample["height"]
        for obj in out_sample["objects"]:
            if "quad" in obj:
                q = obj["quad"]
                xs = q[0::2]
                ys = q[1::2]
                self.assertTrue(all(0 <= x < W for x in xs))
                self.assertTrue(all(0 <= y < H for y in ys))
                area2 = 0
                pts = [(q[i], q[i + 1]) for i in range(0, 8, 2)]
                for i in range(4):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % 4]
                    area2 += x1 * y2 - x2 * y1
                self.assertGreater(abs(area2) * 0.5, 0.0)
            if "line" in obj:
                l = obj["line"]
                xs = l[0::2]
                ys = l[1::2]
                self.assertTrue(all(0 <= x < W for x in xs))
                self.assertTrue(all(0 <= y < H for y in ys))

    def test_photometric_keeps_geometry(self):
        sample = _base_sample()
        img = _make_image()

        cfg = AugmentationConfig(
            enabled=True,
            rng_seed=321,
            apply_to_teachers=False,
            lines_policy="transform",
            debug_visualization=False,
            debug_output_dir=None,
            image_geom=None,
            photometric=PhotometricConfig(
                enabled=True,
                apply_prob=1.0,
                num_ops=2,
                magnitude=0.5,
                ocr_safe_pool=True,
            ),
            lines=None,
            type_policies=None,
            ocr=None,
            criteria=CriteriaConfig(occlusion=None),
        )

        pipeline = ObjectAwareAugmentationPipeline.from_config(cfg)
        out_imgs, out_sample = pipeline.apply(
            copy.deepcopy(sample), [img], sample_index=7
        )
        self.assertEqual(out_sample["objects"], sample["objects"])  # geometry preserved
        self.assertEqual(
            (out_imgs[0].width, out_imgs[0].height), (img.width, img.height)
        )

    def test_line_ops_resample_and_jitter(self):
        sample = _base_sample()
        img = _make_image()

        cfg = AugmentationConfig(
            enabled=True,
            rng_seed=999,
            apply_to_teachers=False,
            lines_policy="transform",
            debug_visualization=False,
            debug_output_dir=None,
            image_geom=None,
            photometric=None,
            lines=LineAugConfig(
                enabled=True,
                jitter_px_minmax=(2, 4),
                resample_points=32,
                min_length_px=10,
            ),
            type_policies=None,
            ocr=None,
            criteria=CriteriaConfig(occlusion=None),
        )
        pipe = ObjectAwareAugmentationPipeline.from_config(cfg)
        _, out_sample = pipe.apply(copy.deepcopy(sample), [img], sample_index=1)
        # Ensure line has the expected number of points after resample
        line_objs = [o for o in out_sample["objects"] if "line" in o]
        self.assertTrue(len(line_objs) >= 1)
        for o in line_objs:
            l = o["line"]
            self.assertEqual(len(l), 2 * 32)

        

if __name__ == "__main__":
    unittest.main()
