#!/usr/bin/env python3
"""
Teacher Selector for Diverse Teacher Pool Creation

Selects representative teacher samples that maximize label coverage
and scene diversity (sparse/medium/dense objects, spatial distribution).
"""

import logging
import random
import sys
from typing import Dict, List, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class TeacherSelector:
    """Selects diverse teacher samples covering all labels and scene types."""
    
    def __init__(self, label_hierarchy: Dict[str, List[str]], max_teachers: int = 10, seed: int = 42):
        self.label_hierarchy = label_hierarchy
        self.max_teachers = max_teachers
        self.seed = seed
        
        # Extract all possible labels from hierarchy
        self.all_labels = set()
        for obj_type, props in label_hierarchy.items():
            self.all_labels.add(obj_type)
            self.all_labels.update(props)
        
        logger.info(f"Initialized TeacherSelector with {len(self.all_labels)} labels, max_teachers={max_teachers}")
    
    def _extract_sample_labels(self, sample: Dict) -> Set[str]:
        """Extract all labels present in a sample."""
        labels_in_sample = set()
        objects = sample.get("objects", [])
        
        for obj in objects:
            desc = obj.get("desc", "")
            # Check which labels appear in the description
            for label in self.all_labels:
                if label in desc:
                    labels_in_sample.add(label)
        
        return labels_in_sample
    
    def _get_object_count_bucket(self, sample: Dict) -> str:
        """Categorize sample by object density."""
        object_count = len(sample.get("objects", []))
        
        if object_count <= 3:
            return "sparse"
        elif object_count <= 10:
            return "medium"
        else:
            return "dense"
    
    def _get_spatial_bucket(self, sample: Dict) -> str:
        """Categorize sample by spatial distribution of objects."""
        objects = sample.get("objects", [])
        width = sample.get("width", 1)
        height = sample.get("height", 1)
        
        if not objects or not width or not height:
            return "unknown"
        
        # Calculate center points of all bboxes
        centers = []
        for obj in objects:
            bbox = obj.get("bbox_2d", [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0 / width
            cy = (y1 + y2) / 2.0 / height
            centers.append((cx, cy))
        
        if not centers:
            return "unknown"
        
        # Calculate average center position
        avg_x = sum(cx for cx, cy in centers) / len(centers)
        avg_y = sum(cy for cx, cy in centers) / len(centers)
        
        # Categorize by position in image
        h_pos = "left" if avg_x < 1/3 else ("center" if avg_x < 2/3 else "right")
        v_pos = "top" if avg_y < 1/3 else ("middle" if avg_y < 2/3 else "bottom")
        
        return f"{v_pos}-{h_pos}"
    
    def _get_size_bucket(self, sample: Dict) -> str:
        """Categorize sample by average object size."""
        objects = sample.get("objects", [])
        width = sample.get("width", 1)
        height = sample.get("height", 1)
        
        if not objects or not width or not height:
            return "unknown"
        
        total_area = width * height
        size_fractions = []
        
        for obj in objects:
            bbox = obj.get("bbox_2d", [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            obj_width = max(0, x2 - x1)
            obj_height = max(0, y2 - y1)
            obj_area = obj_width * obj_height
            
            size_fractions.append(obj_area / total_area)
        
        if not size_fractions:
            return "unknown"
        
        avg_fraction = sum(size_fractions) / len(size_fractions)
        
        if avg_fraction < 0.05:
            return "small"
        elif avg_fraction < 0.2:
            return "medium"
        else:
            return "large"
    
    def _compute_sample_metadata(self, samples: List[Dict]) -> List[Dict]:
        """Compute metadata for all samples for selection algorithm."""
        metadata_list = []
        
        for idx, sample in enumerate(samples):
            labels = self._extract_sample_labels(sample)
            count_bucket = self._get_object_count_bucket(sample)
            spatial_bucket = self._get_spatial_bucket(sample)
            size_bucket = self._get_size_bucket(sample)
            
            metadata_list.append({
                "idx": idx,
                "labels": labels,
                "count_bucket": count_bucket,
                "spatial_bucket": spatial_bucket,
                "size_bucket": size_bucket,
                "object_count": len(sample.get("objects", []))
            })
        
        return metadata_list
    
    def _greedy_selection(self, metadata_list: List[Dict]) -> List[int]:
        """Greedy selection algorithm for maximum coverage."""
        selected_indices = []
        
        # Track what we still need to cover
        needed_labels = self.all_labels.copy()
        needed_count_buckets = set(item["count_bucket"] for item in metadata_list)
        needed_spatial_buckets = set(
            item["spatial_bucket"] for item in metadata_list 
            if item["spatial_bucket"] != "unknown"
        )
        needed_size_buckets = set(
            item["size_bucket"] for item in metadata_list 
            if item["size_bucket"] != "unknown"
        )
        
        logger.debug(f"Initial coverage needs: {len(needed_labels)} labels, "
                    f"{len(needed_count_buckets)} count buckets, "
                    f"{len(needed_spatial_buckets)} spatial buckets, "
                    f"{len(needed_size_buckets)} size buckets")
        
        # Greedy selection loop
        while (needed_labels or needed_count_buckets or needed_spatial_buckets or needed_size_buckets) and \
              len(selected_indices) < self.max_teachers:
            
            best_score = -1
            best_metadata = None
            
            for metadata in metadata_list:
                if metadata["idx"] in selected_indices:
                    continue
                
                # Calculate gain for each dimension
                label_gain = len(metadata["labels"] & needed_labels)
                count_gain = 1 if metadata["count_bucket"] in needed_count_buckets else 0
                spatial_gain = 1 if metadata["spatial_bucket"] in needed_spatial_buckets else 0
                size_gain = 1 if metadata["size_bucket"] in needed_size_buckets else 0
                
                # Weighted score (prioritize label coverage)
                score = 10 * label_gain + 3 * count_gain + 2 * spatial_gain + 1 * size_gain
                
                if score > best_score:
                    best_score = score
                    best_metadata = metadata
            
            if not best_metadata or best_score <= 0:
                logger.debug("No more beneficial samples found, stopping greedy selection")
                break
            
            # Add best sample and update needs
            selected_indices.append(best_metadata["idx"])
            needed_labels -= best_metadata["labels"]
            needed_count_buckets.discard(best_metadata["count_bucket"])
            needed_spatial_buckets.discard(best_metadata["spatial_bucket"])
            needed_size_buckets.discard(best_metadata["size_bucket"])
            
            logger.debug(f"Selected sample {best_metadata['idx']} "
                        f"(score={best_score}, labels={len(best_metadata['labels'])}, "
                        f"remaining_labels={len(needed_labels)})")
        
        return selected_indices
    
    def _random_fill(self, selected_indices: List[int], metadata_list: List[Dict]) -> List[int]:
        """Fill remaining slots with random diverse samples."""
        if len(selected_indices) >= self.max_teachers:
            return selected_indices
        
        remaining_metadata = [m for m in metadata_list if m["idx"] not in selected_indices]
        random.seed(self.seed)
        
        while len(selected_indices) < self.max_teachers and remaining_metadata:
            choice = random.choice(remaining_metadata)
            selected_indices.append(choice["idx"])
            remaining_metadata = [m for m in remaining_metadata if m["idx"] != choice["idx"]]
            logger.debug(f"Random fill: added sample {choice['idx']}")
        
        return selected_indices
    
    def select_teachers(self, samples: List[Dict]) -> Tuple[List[Dict], List[int]]:
        """
        Select teacher samples using multi-objective optimization.
        
        Returns:
            Tuple of (selected_teacher_samples, selected_indices)
        """
        if len(samples) <= self.max_teachers:
            logger.info(f"Total samples ({len(samples)}) <= max_teachers ({self.max_teachers}), selecting all")
            return samples, list(range(len(samples)))
        
        logger.info(f"Selecting {self.max_teachers} teachers from {len(samples)} samples")
        
        # Compute metadata for all samples
        metadata_list = self._compute_sample_metadata(samples)
        
        # Phase 1: Greedy selection for coverage
        selected_indices = self._greedy_selection(metadata_list)
        logger.info(f"Greedy selection: {len(selected_indices)} samples")
        
        # Phase 2: Random fill for additional diversity
        selected_indices = self._random_fill(selected_indices, metadata_list)
        logger.info(f"Final selection: {len(selected_indices)} teacher samples")
        
        # Extract selected samples
        teacher_samples = [samples[i] for i in selected_indices]
        
        # Log selection statistics
        total_labels_covered = set()
        for idx in selected_indices:
            total_labels_covered.update(metadata_list[idx]["labels"])
        
        logger.info(f"Teacher pool covers {len(total_labels_covered)}/{len(self.all_labels)} labels")
        
        return teacher_samples, selected_indices