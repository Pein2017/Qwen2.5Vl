import json
import random
import unittest

from src_new.processing.coordinate_converter import CoordinateTokenConverter
from src_new.processing.variants import TEXT_ONLY_USER_NOISE_ENABLED, TextOnlyHandler


class TestTextOnlyUserNoise(unittest.TestCase):
	def setUp(self) -> None:
		# Deterministic randomness for repeatable assertions
		random.seed(1337)
		self.coord_limit = 4096
		self.converter = CoordinateTokenConverter()
		self.handler = TextOnlyHandler(self.converter)
		# Simple three-object sample covering bbox, quad, and line
		self.objects = [
			{
				"desc": "BBU设备/华为,显示完整,无需安装",
				"bbox_2d": [10, 12, 110, 120],
			},
			{
				"desc": "挡风板/华为,显示完整,安装方向正确",
				"quad": [0, 0, 30, 0, 30, 30, 0, 30],
			},
			{
				"desc": "光纤/无遮挡,有保护措施,弯曲半径合理/蛇形管",
				"line": [5, 5, 15, 5, 20, 20, 5, 20],
			},
		]

	def _parse_user_objects(self, user_text: str) -> list[dict]:
		parsed: list[dict] = []
		for line in user_text.splitlines():
			line = line.strip()
			if not line.startswith("Object "):
				continue
			# Expect format: "Object k: {json}"
			parts = line.split(": ", 1)
			self.assertEqual(len(parts), 2, f"Unexpected user line format: {line}")
			obj = json.loads(parts[1])
			self.assertIsInstance(obj, dict)
			parsed.append(obj)
		return parsed

	def test_text_only_user_noise_shuffles_and_noisifies_geometry_preserves_desc(self) -> None:
		# Ensure feature toggle is enabled by default
		self.assertTrue(TEXT_ONLY_USER_NOISE_ENABLED)

		user_spec = self.handler.build_user_text(self.objects)
		self.assertIsInstance(user_spec, dict)
		self.assertTrue(user_spec.get("include_image", False))
		user_text: str = user_spec.get("text", "")
		self.assertIsInstance(user_text, str)
		self.assertGreater(len(user_text), 0)

		# Parse back the JSON objects from the user text
		user_objs = self._parse_user_objects(user_text)
		self.assertEqual(len(user_objs), len(self.objects))

		# Descriptions should be preserved exactly (order may change)
		orig_descs = sorted([o["desc"] for o in self.objects])
		user_descs = sorted([o.get("desc", "") for o in user_objs])
		self.assertListEqual(user_descs, orig_descs)

		# Geometry should be noisy (at least one object differs from any original geometry)
		def _geom(o: dict) -> tuple[str, list[int]]:
			for k in ("bbox_2d", "quad", "line"):
				if k in o:
					return k, [int(v) for v in o[k]]
			return "", []

		original_geoms = [
			_geom(o) for o in self.objects
		]
		user_geoms = [
			_geom(o) for o in user_objs
		]

		# Check ranges and shapes for noisy geometries
		for gtype, coords in user_geoms:
			self.assertIn(gtype, {"bbox_2d", "quad", "line"})
			self.assertTrue(all(isinstance(c, int) for c in coords))
			self.assertTrue(all(0 <= int(c) <= self.coord_limit for c in coords))
			if gtype == "bbox_2d":
				self.assertEqual(len(coords), 4)
			elif gtype == "quad":
				self.assertEqual(len(coords), 8)
			elif gtype == "line":
				self.assertTrue(len(coords) >= 4 and len(coords) % 2 == 0)

		# It's possible (though unlikely) that some geometry equals original by chance; require at least one differs
		any_diff = False
		for ug in user_geoms:
			if ug not in original_geoms:
				any_diff = True
				break
		self.assertTrue(any_diff, "Expected at least one user geometry to differ from original")

		# Assistant text must equal canonical conversion from original (unchanged)
		assistant_text = self.handler.build_assistant_text(self.objects)
		expected_assistant = self.converter.convert_objects_to_tokens(self.objects)
		self.assertEqual(assistant_text, expected_assistant)


if __name__ == "__main__":
	unittest.main()
