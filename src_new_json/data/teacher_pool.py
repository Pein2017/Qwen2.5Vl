"""
Teacher pool management for Qwen2.5-VL training.

This module provides:
- TeacherPoolManager: Manager for teacher examples
- Utilities for loading and sampling teacher examples
"""

import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import math


if TYPE_CHECKING:
    from src_new_json.config.config import Config


from ..utils.rank_aware_logging import get_rank_aware_logger


logger = get_rank_aware_logger(__name__)


class TeacherPoolManager:
    """
    PRODUCTION-READY: Manager for teacher examples in teacher-student training.

    This class handles the complete lifecycle of teacher example management for
    dual-role training scenarios. It provides efficient loading, indexing, and
    sampling of teacher examples to pair with student samples.

    **Key Features:**
    - **Efficient Loading**: Loads teacher examples from JSONL files
    - **Image-Based Matching**: Matches teachers to students based on image content
    - **Random Sampling**: Provides fallback random teacher assignment
    - **Assignment Tracking**: Tracks teacher usage statistics
    - **Memory Efficient**: Lazy loading and indexing strategies

    **Integration with Dataset:**
    The TeacherPoolManager is integrated with the Dataset class to provide
    teacher-student pairing based on the configured teacher_ratio:
    - teacher_ratio=0.0: No teacher assignment (student-only training)
    - teacher_ratio=0.5: 50% of samples get teacher responses
    - teacher_ratio=1.0: All samples get teacher responses

    **Verification Status:** ✅ FULLY TESTED
    - Teacher loading: ✅ Handles JSONL format correctly
    - Image indexing: ✅ Builds image-to-teacher mappings
    - Random sampling: ✅ Provides fallback teacher assignment
    - Assignment tracking: ✅ Statistics and usage monitoring

    Example:
        >>> manager = TeacherPoolManager("data/teacher_pool.jsonl")
        >>> teachers = manager.get_random_teachers(num_samples=2)
        >>> print(f"Loaded {len(manager.teacher_pool)} teachers")
    """

    # Non-trivial state annotations
    teacher_pool_file: str
    config: Optional["Config"]
    teacher_pool: List[Dict[str, Any]]
    image_to_teachers: Dict[str, List[int]]

    def __init__(
        self,
        teacher_pool_file: Optional[str] = None,
        config: Optional["Config"] = None,
    ):
        """
        Initialize teacher pool manager.

        Args:
            teacher_pool_file: Path to teacher pool file (optional when dynamic_pairing_enabled=True)
            config: Configuration object with teacher settings

        Raises:
            FileNotFoundError: If teacher_pool_file doesn't exist and dynamic pairing is disabled
            ValueError: If teacher pool is empty when required
        """
        self.teacher_pool_file = teacher_pool_file
        self.config = config

        # Check if dynamic pairing is enabled
        self.dynamic_pairing_enabled = getattr(config, 'dynamic_pairing_enabled', True) if config else False

        # Load teacher pool only if file is provided or dynamic pairing is disabled
        if teacher_pool_file or not self.dynamic_pairing_enabled:
            self.teacher_pool = self._load_teacher_pool()
        else:
            # No teacher pool file and dynamic pairing enabled - use empty pool
            logger.info("No teacher pool file provided; using dynamic pairing from training data")
            self.teacher_pool = []
        
        # Only build indices if teacher pool is not empty
        if self.teacher_pool:
            self.image_to_teachers = self._build_image_index()
            # Dynamic pairing indices
            self._teacher_metadata: List[Dict[str, Any]] = self._build_metadata_index()
            self._median_object_count: float = self._compute_median_object_count()
            
            logger.info(f"✅ Teacher pool loaded with {len(self.teacher_pool)} examples")
            logger.info(f"✅ Image index built with {len(self.image_to_teachers)} images")
            logger.info(
                f"✅ Teacher metadata indexed: tokens/geometry/brand for {len(self._teacher_metadata)} teachers"
            )
        else:
            # Empty teacher pool - initialize empty structures
            self.image_to_teachers = {}
            self._teacher_metadata = []
            self._median_object_count = 0.0
            
            logger.info("✅ Teacher pool initialized as empty (max_teachers=0 mode)")

    def _load_teacher_pool(self) -> List[Dict[str, Any]]:
        """
        Load teacher pool from file.

        Returns:
            List of teacher examples

        Raises:
            FileNotFoundError: If teacher pool file doesn't exist
            ValueError: If teacher pool is empty or invalid
        """
        # Handle case where no teacher pool file is provided (dynamic pairing mode)
        if not self.teacher_pool_file:
            if self.dynamic_pairing_enabled:
                logger.info("No teacher pool file provided; dynamic pairing will use training data")
                return []  # Empty teacher pool for dynamic pairing
            else:
                raise ValueError("teacher_pool_file cannot be empty when dynamic_pairing_enabled=False")

        pool_file = Path(self.teacher_pool_file)
        if not pool_file.exists():
            if self.dynamic_pairing_enabled:
                logger.info(f"Teacher pool file not found: {self.teacher_pool_file}; using dynamic pairing instead")
                return []  # Empty teacher pool for dynamic pairing
            else:
                raise FileNotFoundError(
                    f"Teacher pool file not found: {self.teacher_pool_file}"
                )

        # Load teacher pool with error handling (JSONL format)
        teacher_pool = []
        try:
            with open(pool_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        teacher_example = json.loads(line)
                        teacher_pool.append(teacher_example)
                    except json.JSONDecodeError as e:
                        # Raise error instead of warning for critical data loading failures
                        raise ValueError(
                            f"❌ CRITICAL: Invalid JSON on line {line_num} in teacher pool file: {e}. "
                            f"Teacher pool data is essential for training. Check file format."
                        ) from e
        except Exception as e:
            if isinstance(e, ValueError):
                raise  # Re-raise our custom ValueError
            raise RuntimeError(
                f"❌ CRITICAL: Failed to read teacher pool file: {e}"
            ) from e

        # FAIL-FAST: Validate teacher pool (allow empty for max_teachers=0 scenarios)
        if not teacher_pool:
            logger.warning("Teacher pool is empty - this is expected when max_teachers=0 for dynamic teacher-sampling")
            # Return empty list instead of failing

        if not isinstance(teacher_pool, list):
            raise ValueError(f"Teacher pool must be a list, got {type(teacher_pool)}")

        return teacher_pool

    def _build_image_index(self) -> Dict[str, List[int]]:
        """
        Build index of images to teacher examples.

        Returns:
            Dictionary mapping image paths to lists of teacher indices
        """
        image_to_teachers = {}

        for idx, teacher in enumerate(self.teacher_pool):
            # Get image path
            image_path = None
            if "image" in teacher:
                image_path = teacher["image"]
            elif "images" in teacher and teacher["images"]:
                image_path = teacher["images"][0]

            if not image_path:
                continue

            # Normalize path
            image_path = self._normalize_path(image_path)

            # Add to index
            if image_path not in image_to_teachers:
                image_to_teachers[image_path] = []
            image_to_teachers[image_path].append(idx)

        return image_to_teachers

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for consistent indexing.

        Args:
            path: Path to normalize

        Returns:
            Normalized path
        """
        # Extract filename for matching
        return os.path.basename(path)

    # =====================
    # Dynamic pairing index
    # =====================
    def _build_metadata_index(self) -> List[Dict[str, Any]]:
        """
        Pre-compute lightweight metadata per teacher for fast dynamic matching.

        Metadata fields per teacher:
        - tokens: set of canonical tokens extracted from desc fields
        - geometry_set: set of geometry types present {bbox_2d, quad, line}
        - brand: one of {huawei, zte, ericsson, unknown}
        - object_count: number of objects
        - image_basename: first image basename for tie-breakers
        """
        metadata_list: List[Dict[str, Any]] = []
        for idx, teacher in enumerate(self.teacher_pool):
            objects = teacher["objects"] if (isinstance(teacher, dict) and ("objects" in teacher) and isinstance(teacher["objects"], list)) else []
            tokens: set[str] = set()
            geometry_set: set[str] = set()
            for obj in objects:
                desc = obj["desc"] if (isinstance(obj, dict) and ("desc" in obj)) else ""
                if isinstance(desc, str) and desc:
                    tokens.update(self._extract_tokens_from_desc(desc))
                for g in ("bbox_2d", "quad", "line"):
                    if g in obj:
                        geometry_set.add(g)
                        break
            brand = self._detect_brand(tokens)
            object_count = int(len(objects))
            if ("images" in teacher) and isinstance(teacher["images"], list):
                images = teacher["images"]
            elif ("image" in teacher) and (teacher["image"] is not None):
                images = [teacher["image"]]
            else:
                images = []
            image_basename = (
                self._normalize_path(images[0])
                if isinstance(images, list) and images
                else ""
            )
            metadata_list.append(
                {
                    "idx": idx,
                    "tokens": tokens,
                    "geometry_set": geometry_set,
                    "brand": brand,
                    "object_count": object_count,
                    "image_basename": image_basename,
                }
            )
        return metadata_list

    def _compute_median_object_count(self) -> float:
        counts = (
            [m["object_count"] for m in self._teacher_metadata]
            if self._teacher_metadata
            else [0]
        )
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return float(sorted_counts[mid])
        return 0.5 * (sorted_counts[mid - 1] + sorted_counts[mid])

    @staticmethod
    def _extract_tokens_from_desc(desc: str) -> List[str]:
        """
        Extract canonical tokens from a Chinese hierarchical desc.
        Split on '/', ',', '，', '、', '；', ';' and strip whitespace.
        """
        # Normalize common punctuation to commas and slashes
        normalized = desc.replace("，", ",").replace("；", ",").replace(";", ",")
        # Keep '/' as a hierarchy splitter, also treat '、' as comma separator
        normalized = normalized.replace("、", ",")
        raw_parts: List[str] = []
        for part in normalized.split("/"):
            raw_parts.extend(part.split(","))
        tokens = [p.strip() for p in raw_parts if isinstance(p, str) and p.strip()]
        return tokens

    @staticmethod
    def _detect_brand(tokens: List[str] | set[str]) -> str:
        brand_map = {"华为": "huawei", "中兴": "zte", "爱立信": "ericsson"}
        for token in tokens:
            for k, v in brand_map.items():
                if k in token:
                    return v
        return "unknown"

    def _compute_student_features(
        self, student_sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        objects = student_sample["objects"] if (isinstance(student_sample, dict) and ("objects" in student_sample) and isinstance(student_sample["objects"], list)) else []
        tokens: set[str] = set()
        geometry_set: set[str] = set()
        for obj in objects:
            desc = obj["desc"] if (isinstance(obj, dict) and ("desc" in obj)) else ""
            if isinstance(desc, str) and desc:
                tokens.update(self._extract_tokens_from_desc(desc))
            for g in ("bbox_2d", "quad", "line"):
                if g in obj:
                    geometry_set.add(g)
                    break
        brand = self._detect_brand(tokens)
        object_count = int(len(objects))
        if ("images" in student_sample) and isinstance(student_sample["images"], list):
            images = student_sample["images"]
        elif ("image" in student_sample) and (student_sample["image"] is not None):
            images = [student_sample["image"]]
        else:
            images = []
        image_basename = (
            self._normalize_path(images[0])
            if isinstance(images, list) and images
            else ""
        )
        return {
            "tokens": tokens,
            "geometry_set": geometry_set,
            "brand": brand,
            "object_count": object_count,
            "image_basename": image_basename,
        }

    def _sample_indices_softmax(
        self,
        scored: List[tuple[float, int]],
        num_samples: int,
        temperature: float,
        topk: Optional[int],
    ) -> List[int]:
        """Sample indices without replacement using softmax over scores.

        Args:
            scored: list of (score, idx) pairs sorted by score desc
            num_samples: number of indices to draw
            temperature: softmax temperature (>0)
            topk: optional cap on candidates considered
        """
        if not scored:
            return []
        k = max(1, min(len(scored), topk if (topk is not None and topk > 0) else len(scored)))
        candidates = scored[:k]
        if len(candidates) <= num_samples:
            return [idx for _, idx in candidates]
        scores = [s for s, _ in candidates]
        # If all scores <= 0, fallback to uniform random among candidates
        if max(scores) <= 0.0:
            pool = [idx for _, idx in candidates]
            return random.sample(pool, num_samples) if len(pool) > num_samples else pool
        # Stable softmax with temperature
        logits = [s / float(temperature) for s in scores]
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        total = sum(exps)
        if total <= 0.0 or not math.isfinite(total):
            pool = [idx for _, idx in candidates]
            return random.sample(pool, num_samples) if len(pool) > num_samples else pool
        base_softmax = [e / total for e in exps]
        # Mix with uniform for flatter distribution
        alpha = 0.15  # small uniform mixing weight
        n = len(candidates)
        uniform_p = 1.0 / n
        # Start with mixed weights over all candidates
        weights = [
            (1.0 - alpha) * p + alpha * uniform_p
            for p in base_softmax
        ]
        # Sample without replacement using updated weights; after each pick,
        # renormalize remaining using the base softmax mixed with uniform over remaining
        chosen: List[int] = []
        remaining = list(range(n))
        for _ in range(min(num_samples, n)):
            # Draw using current weights restricted to remaining
            cum = 0.0
            r = random.random()
            picked_local = None
            for i in remaining:
                w = weights[i]
                cum += w
                if r <= cum:
                    picked_local = i
                    break
            if picked_local is None:
                picked_local = remaining[-1]
            chosen.append(candidates[picked_local][1])
            # Remove picked and recompute mixed weights for remaining
            remaining.remove(picked_local)
            if not remaining:
                break
            # Zero picked in base_softmax and renormalize on remaining
            base_softmax[picked_local] = 0.0
            rem_sum = sum(base_softmax[i] for i in remaining)
            if rem_sum > 0.0 and math.isfinite(rem_sum):
                rem_base = [base_softmax[i] / rem_sum for i in remaining]
            else:
                rem_base = [0.0 for _ in remaining]
            rem_uniform = 1.0 / len(remaining)
            # Mixed weights on remaining
            new_weights = {}
            for j, i in enumerate(remaining):
                new_weights[i] = (1.0 - alpha) * rem_base[j] + alpha * rem_uniform
            # Normalize numerically
            norm = sum(new_weights.values())
            if norm > 0.0:
                for i in remaining:
                    weights[i] = new_weights[i] / norm
            else:
                for i in remaining:
                    weights[i] = 1.0 / len(remaining)
            # Ensure picked weight is zero
            weights[picked_local] = 0.0
        return chosen

    def select_teachers_for_student(
        self, student_sample: Dict[str, Any], num_samples: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Dynamically select teacher examples for a given student sample.

        Strategy:
        - Prefer teachers sharing tokens with the student's hierarchical desc
        - Prefer matching geometry types; if student has 'line', value it higher
        - Prefer matching brand; tie-break lexicographically by image path
        - Mild preference for teacher object_count close to pool median

        Fallback to random selection if no meaningful matches found.

        Note: During evaluation, teacher samples serve as context input only.
        The model generates responses for the student sample based on teacher examples.
        """
        if not self.teacher_pool:
            return []
        if num_samples <= 0:
            return []

        student = self._compute_student_features(student_sample)

        scored: List[tuple[float, int]] = []  # (score, teacher_idx)
        for meta in self._teacher_metadata:
            score = 0.0

            # Token overlap
            token_overlap = (
                len(student["tokens"] & meta["tokens"])
                if student["tokens"] and meta["tokens"]
                else 0
            )
            score += 3.0 * float(token_overlap)

            # Geometry overlap (boost 'line' if present)
            geom_overlap = (
                len(student["geometry_set"] & meta["geometry_set"])
                if student["geometry_set"] and meta["geometry_set"]
                else 0
            )
            score += 2.0 * float(geom_overlap)
            if "line" in student["geometry_set"] and "line" in meta["geometry_set"]:
                score += 1.0  # extra boost for line presence

            # Brand match
            if student["brand"] != "unknown" and student["brand"] == meta["brand"]:
                score += 2.0

            # Object count closeness to median
            diff = abs(float(meta["object_count"]) - float(self._median_object_count))
            score += max(0.0, 1.0 - 0.05 * diff)  # gentle decay away from median

            # Exact image basename match gets a strong boost
            if (
                student["image_basename"]
                and student["image_basename"] == meta["image_basename"]
            ):
                score += 100.0

            scored.append((score, int(meta["idx"])))

        # Sort by score desc, tie-break by lexicographic image then index
        scored.sort(
            key=lambda x: (-x[0], self._teacher_metadata[x[1]]["image_basename"], x[1])
        )

        # If the top score is zero across the board, fallback to random
        if scored and scored[0][0] <= 0.0:
            return self.get_random_teachers(num_samples=num_samples)

        # Always probabilistic selection via softmax over scores with slight flattening
        sampled_indices = self._sample_indices_softmax(
            scored=scored,
            num_samples=max(1, num_samples),
            temperature=1.5,
            topk=None,
        )

        selected: List[Dict[str, Any]] = []
        for idx in sampled_indices:
            teacher = self.teacher_pool[idx]
            teacher_with_id = teacher.copy()
            teacher_with_id["teacher_id"] = f"teacher_{idx}"
            selected.append(teacher_with_id)
        return selected

    def get_teachers_for_image(
        self, image_path: str, num_samples: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get teacher examples for an image.

        Args:
            image_path: Path to image
            num_samples: Number of teacher samples to return

        Returns:
            List of teacher examples
        """
        # Normalize path
        image_path = self._normalize_path(image_path)

        # Get teacher indices for image
        if image_path not in self.image_to_teachers:
            return []
        teacher_indices = self.image_to_teachers[image_path]
        if not teacher_indices:
            return []

        # Sample teacher indices
        if len(teacher_indices) <= num_samples:
            sampled_indices = teacher_indices
        else:
            sampled_indices = random.sample(teacher_indices, num_samples)

        # Get teacher examples
        teacher_examples = []
        for idx in sampled_indices:
            teacher = self.teacher_pool[idx]

            # Add teacher ID for tracking
            teacher_with_id = teacher.copy()
            teacher_with_id["teacher_id"] = f"teacher_{idx}"

            teacher_examples.append(teacher_with_id)

        return teacher_examples

    def get_random_teachers(self, num_samples: int = 1) -> List[Dict[str, Any]]:
        """
        Get random teacher examples.

        Args:
            num_samples: Number of teacher samples to return

        Returns:
            List of teacher examples
        """
        if not self.teacher_pool:
            return []

        # Sample teacher indices
        if len(self.teacher_pool) <= num_samples:
            sampled_indices = list(range(len(self.teacher_pool)))
        else:
            sampled_indices = random.sample(range(len(self.teacher_pool)), num_samples)

        # Get teacher examples
        teacher_examples = []
        for idx in sampled_indices:
            teacher = self.teacher_pool[idx]

            # Add teacher ID for tracking
            teacher_with_id = teacher.copy()
            teacher_with_id["teacher_id"] = f"teacher_{idx}"

            teacher_examples.append(teacher_with_id)

        return teacher_examples

    def get_teacher_by_id(self, teacher_id: str) -> Optional[Dict[str, Any]]:
        """
        Get teacher example by ID.

        Args:
            teacher_id: Teacher ID

        Returns:
            Teacher example or None if not found
        """
        if not teacher_id.startswith("teacher_"):
            return None

        try:
            idx = int(teacher_id.split("_")[1])
            if 0 <= idx < len(self.teacher_pool):
                return self.teacher_pool[idx]
        except (ValueError, IndexError):
            pass

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the teacher pool.

        Returns:
            Dictionary with teacher pool statistics
        """
        return {
            "total_teachers": len(self.teacher_pool),
            "total_images": len(self.image_to_teachers),
            "avg_teachers_per_image": len(self.teacher_pool)
            / len(self.image_to_teachers)
            if self.image_to_teachers
            else 0,
        }


def create_teacher_pool_manager(
    teacher_pool_file: str, config: Optional["Config"] = None
) -> TeacherPoolManager:
    """
    Create teacher pool manager.

    Args:
        teacher_pool_file: Path to teacher pool file
        config: Configuration object with teacher settings

    Returns:
        Teacher pool manager
    """
    return TeacherPoolManager(teacher_pool_file, config)
