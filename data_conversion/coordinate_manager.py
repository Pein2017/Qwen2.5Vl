#!/usr/bin/env python3
"""
Centralized Coordinate Management System

This module provides a unified approach to handling all coordinate transformations
including EXIF orientation, dimension rescaling, and smart resize operations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageOps


logger = logging.getLogger(__name__)


class CoordinateManager:
    """
    Centralized coordinate transformation and geometry processing manager.

    Handles all coordinate transformations in proper order:
    1. EXIF orientation compensation
    2. Dimension mismatch rescaling
    3. Smart resize scaling

    Also provides unified geometry processing for all coordinate types:
    - Simple bbox: [x1, y1, x2, y2]
    - ExtentPolygon: GeoJSON-style with coordinates array
    - Square: Four-point polygon (四边形)
    - LineString: Multi-point line annotation
    """

    @staticmethod
    def get_exif_transform_matrix(image_path: Path) -> Tuple[bool, int, int, int, int]:
        """
        Analyze EXIF orientation and return transformation info.

        Returns:
            (is_transformed, original_width, original_height, new_width, new_height)
        """
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            # Apply EXIF orientation to get transformed dimensions
            transformed_img = ImageOps.exif_transpose(img)
            if transformed_img is None:
                transformed_img = img
            new_width, new_height = transformed_img.size

            is_transformed = (
                original_width != new_width or original_height != new_height
            )

            return (
                is_transformed,
                original_width,
                original_height,
                new_width,
                new_height,
            )

    @staticmethod
    def apply_exif_orientation_to_bbox(
        bbox: List[float],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
        exif_orientation: Optional[int] = None,
    ) -> List[float]:
        """
        Transform bbox coordinates to account for EXIF orientation changes.

        Args:
            bbox: [x1, y1, x2, y2] in original coordinate system
            original_width, original_height: Image dimensions before EXIF transform
            new_width, new_height: Image dimensions after EXIF transform
            exif_orientation: EXIF orientation value (optional, can detect from dimensions)

        Returns:
            Transformed bbox coordinates
        """
        if original_width == new_width and original_height == new_height:
            return bbox  # No transformation needed

        x1, y1, x2, y2 = bbox

        # Detect transformation type from dimension changes
        if original_width == new_height and original_height == new_width:
            # 90° or 270° rotation
            if new_width > new_height:
                # Likely 270° rotation (landscape from portrait)
                # Transform: (x,y) -> (y, original_width - x)
                new_x1 = y1
                new_y1 = original_width - x2
                new_x2 = y2
                new_y2 = original_width - x1
            else:
                # Likely 90° rotation (portrait from landscape)
                # Transform: (x,y) -> (original_height - y, x)
                new_x1 = original_height - y2
                new_y1 = x1
                new_x2 = original_height - y1
                new_y2 = x2
        elif original_width == new_width and original_height == new_height:
            # 180° rotation
            # Transform: (x,y) -> (original_width - x, original_height - y)
            new_x1 = original_width - x2
            new_y1 = original_height - y2
            new_x2 = original_width - x1
            new_y2 = original_height - y1
        else:
            # Complex transformation or no transformation
            logger.warning(
                f"Unexpected EXIF transformation: {original_width}x{original_height} -> {new_width}x{new_height}"
            )
            return bbox

        # Ensure coordinates are in correct order
        final_x1 = min(new_x1, new_x2)
        final_y1 = min(new_y1, new_y2)
        final_x2 = max(new_x1, new_x2)
        final_y2 = max(new_y1, new_y2)

        # Clamp to image bounds
        final_x1 = max(0, min(final_x1, new_width))
        final_y1 = max(0, min(final_y1, new_height))
        final_x2 = max(0, min(final_x2, new_width))
        final_y2 = max(0, min(final_y2, new_height))

        logger.debug(
            f"EXIF bbox transform: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] -> "
            f"[{final_x1:.1f},{final_y1:.1f},{final_x2:.1f},{final_y2:.1f}]"
        )

        return [final_x1, final_y1, final_x2, final_y2]

    @staticmethod
    def apply_dimension_rescaling(
        bbox: List[float],
        json_width: int,
        json_height: int,
        actual_width: int,
        actual_height: int,
    ) -> List[float]:
        """
        Rescale bbox coordinates when JSON dimensions differ from actual image dimensions.

        This typically happens when:
        1. EXIF orientation was applied to image but not to JSON coordinates
        2. Image was preprocessed but JSON coordinates weren't updated

        Note: Returns float coordinates to preserve precision for subsequent transformations.
        Final integer conversion happens in smart_resize_scaling.
        """
        if json_width == actual_width and json_height == actual_height:
            return bbox  # No rescaling needed

        scale_x = actual_width / json_width
        scale_y = actual_height / json_height

        x1, y1, x2, y2 = bbox

        new_x1 = x1 * scale_x
        new_y1 = y1 * scale_y
        new_x2 = x2 * scale_x
        new_y2 = y2 * scale_y

        # Clamp to image bounds (keep as float for precision)
        new_x1 = max(0.0, min(new_x1, float(actual_width)))
        new_y1 = max(0.0, min(new_y1, float(actual_height)))
        new_x2 = max(0.0, min(new_x2, float(actual_width)))
        new_y2 = max(0.0, min(new_y2, float(actual_height)))

        logger.debug(
            f"Dimension rescale: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] "
            f"({json_width}x{json_height}) -> [{new_x1:.1f},{new_y1:.1f},{new_x2:.1f},{new_y2:.1f}] "
            f"({actual_width}x{actual_height})"
        )

        return [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    def apply_smart_resize_scaling(
        bbox: List[float],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
    ) -> List[int]:
        """
        Scale bbox coordinates for smart resize operation.
        Returns integer coordinates suitable for pixel-based operations.
        """
        if original_width == new_width and original_height == new_height:
            # Convert to integers even if no scaling needed
            x1, y1, x2, y2 = bbox
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        x1, y1, x2, y2 = bbox

        # Apply scaling and round to integers
        new_x1 = int(round(x1 * scale_x))
        new_y1 = int(round(y1 * scale_y))
        new_x2 = int(round(x2 * scale_x))
        new_y2 = int(round(y2 * scale_y))

        # Clamp to image bounds and ensure valid bbox
        new_x1 = max(0, min(new_x1, new_width - 1))
        new_y1 = max(0, min(new_y1, new_height - 1))
        new_x2 = max(new_x1 + 1, min(new_x2, new_width))  # Ensure x2 > x1
        new_y2 = max(new_y1 + 1, min(new_y2, new_height))  # Ensure y2 > y1

        logger.debug(
            f"Smart resize scale: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] "
            f"({original_width}x{original_height}) -> [{new_x1},{new_y1},{new_x2},{new_y2}] "
            f"({new_width}x{new_height})"
        )

        return [new_x1, new_y1, new_x2, new_y2]

    @classmethod
    def transform_bbox_complete(
        cls,
        bbox: List[float],
        image_path: Path,
        json_width: int,
        json_height: int,
        enable_smart_resize: bool = True,
        smart_resize_factor: int = 28,
    ) -> Tuple[List[float], int, int]:
        """
        Apply complete bbox transformation pipeline.

        Pipeline:
        1. Apply EXIF orientation compensation
        2. Apply dimension mismatch rescaling
        3. Apply smart resize scaling (if enabled)

        Returns:
            (transformed_bbox, final_width, final_height)
        """
        # Step 1: Get EXIF transformation info
        is_exif_transformed, orig_w, orig_h, exif_w, exif_h = (
            CoordinateManager.get_exif_transform_matrix(image_path)
        )

        current_bbox = bbox
        current_width, current_height = exif_w, exif_h

        # Step 2: Apply EXIF orientation compensation if needed
        if is_exif_transformed:
            current_bbox = CoordinateManager.apply_exif_orientation_to_bbox(
                current_bbox, orig_w, orig_h, exif_w, exif_h
            )
            logger.debug(
                f"Applied EXIF orientation: {orig_w}x{orig_h} -> {exif_w}x{exif_h}"
            )

        # Step 3: Apply dimension mismatch rescaling if needed
        if json_width != current_width or json_height != current_height:
            current_bbox = CoordinateManager.apply_dimension_rescaling(
                current_bbox, json_width, json_height, current_width, current_height
            )
            logger.debug(
                f"Applied dimension rescaling: {json_width}x{json_height} -> {current_width}x{current_height}"
            )

        # Step 4: Apply smart resize scaling if enabled
        if enable_smart_resize:
            # Use the proper smart_resize function from vision_process.py that respects MAX_PIXELS
            from data_conversion.vision_process import (
                MAX_PIXELS,
                MIN_PIXELS,
                smart_resize,
            )

            resize_h, resize_w = smart_resize(
                height=current_height,
                width=current_width,
                factor=smart_resize_factor,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )

            if resize_w != current_width or resize_h != current_height:
                current_bbox = CoordinateManager.apply_smart_resize_scaling(
                    current_bbox, current_width, current_height, resize_w, resize_h
                )
                current_width, current_height = resize_w, resize_h
                logger.debug(
                    f"Applied smart resize: {current_width}x{current_height} (within MAX_PIXELS={MAX_PIXELS})"
                )

        return [float(x) for x in current_bbox], current_width, current_height

    @staticmethod
    def validate_bbox_bounds(bbox: List[float], width: int, height: int) -> bool:
        """
        Validate that bbox coordinates are within image bounds.

        Args:
            bbox: [x1, y1, x2, y2] (can be int or float)
            width, height: Image dimensions

        Returns:
            True if bbox is valid, False otherwise
        """
        x1, y1, x2, y2 = bbox

        # Check coordinate order
        if x1 >= x2 or y1 >= y2:
            return False

        # Check bounds (allow small floating point tolerance)
        tolerance = 0.1
        if (
            x1 < -tolerance
            or y1 < -tolerance
            or x2 > width + tolerance
            or y2 > height + tolerance
        ):
            return False

        return True

    @classmethod
    def process_sample_coordinates(
        cls,
        sample_data: Dict,
        image_path: Path,
        json_width: int,
        json_height: int,
        enable_smart_resize: bool = True,
    ) -> Tuple[Dict, int, int]:
        """
        Process all object coordinates in a sample.

        Args:
            sample_data: Sample with 'objects' list containing 'bbox_2d' fields
            image_path: Path to the image file
            json_width, json_height: Dimensions from JSON metadata
            enable_smart_resize: Whether to apply smart resize

        Returns:
            (updated_sample_data, final_width, final_height)
        """
        if "objects" not in sample_data or not sample_data["objects"]:
            # No objects to process, just get final dimensions
            if enable_smart_resize:
                from data_conversion.vision_process import (
                    MAX_PIXELS,
                    MIN_PIXELS,
                    smart_resize,
                )

                _, _, _, final_w, final_h = CoordinateManager.get_exif_transform_matrix(
                    image_path
                )
                resize_h, resize_w = smart_resize(
                    height=final_h,
                    width=final_w,
                    factor=28,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                )
                return sample_data, resize_w, resize_h
            else:
                _, _, _, final_w, final_h = CoordinateManager.get_exif_transform_matrix(
                    image_path
                )
                return sample_data, final_w, final_h

        # Process first object to get final dimensions
        first_bbox = sample_data["objects"][0]["bbox_2d"]
        _, final_width, final_height = CoordinateManager.transform_bbox_complete(
            first_bbox, image_path, json_width, json_height, enable_smart_resize
        )

        # Process all objects
        updated_objects = []
        for obj in sample_data["objects"]:
            if "bbox_2d" not in obj:
                updated_objects.append(obj)
                continue

            transformed_bbox, _, _ = CoordinateManager.transform_bbox_complete(
                obj["bbox_2d"], image_path, json_width, json_height, enable_smart_resize
            )

            # Validate transformed coordinates
            if CoordinateManager.validate_bbox_bounds(
                transformed_bbox, final_width, final_height
            ):
                updated_obj = obj.copy()
                updated_obj["bbox_2d"] = transformed_bbox
                updated_objects.append(updated_obj)
            else:
                logger.warning(
                    f"Dropping invalid bbox after transformation: {transformed_bbox}"
                )

        updated_sample = sample_data.copy()
        updated_sample["objects"] = updated_objects

        return updated_sample, final_width, final_height

    @staticmethod
    def round_coordinates_to_int(coords: Union[List[float], List[int]]) -> List[int]:
        """
        Round all coordinates to integers for model compatibility.

        Args:
            coords: List of coordinate values (float or int)

        Returns:
            List of integer coordinates
        """
        return [int(round(float(coord))) for coord in coords]

    @staticmethod
    def apply_exif_orientation_to_geometry(
        geometry_input: Union[List, Dict],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
    ) -> Union[List, Dict]:
        """
        Apply EXIF orientation transformation to any geometry type.

        Args:
            geometry_input: Geometry data (bbox list, GeoJSON geometry, etc.)
            original_width, original_height: Image dimensions before EXIF transform
            new_width, new_height: Image dimensions after EXIF transform

        Returns:
            Transformed geometry maintaining original structure
        """
        if original_width == new_width and original_height == new_height:
            return geometry_input  # No transformation needed

        # For simple bbox, use existing method
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            return CoordinateManager.apply_exif_orientation_to_bbox(
                geometry_input, original_width, original_height, new_width, new_height
            )

        # For complex geometries, transform all coordinate points
        if isinstance(geometry_input, dict):
            all_points = CoordinateManager.get_all_coordinate_points(geometry_input)

            if not all_points:
                return geometry_input

            # Transform each point using the EXIF transformation logic
            transformed_points = []
            for x, y in all_points:
                # Apply same transformation logic as apply_exif_orientation_to_bbox
                if original_width == new_height and original_height == new_width:
                    # 90° or 270° rotation
                    if new_width > new_height:
                        # 270° rotation
                        new_x, new_y = y, original_width - x
                    else:
                        # 90° rotation
                        new_x, new_y = original_height - y, x
                elif original_width == new_width and original_height == new_height:
                    # 180° rotation
                    new_x, new_y = original_width - x, original_height - y
                else:
                    # No transformation or complex case
                    new_x, new_y = x, y

                # Clamp to bounds
                new_x = max(0, min(new_x, new_width))
                new_y = max(0, min(new_y, new_height))
                transformed_points.append((new_x, new_y))

            # Reconstruct geometry with transformed points
            return CoordinateManager._reconstruct_geometry_from_points(
                geometry_input, transformed_points
            )

        return geometry_input

    @staticmethod
    def apply_dimension_rescaling_to_geometry(
        geometry_input: Union[List, Dict],
        json_width: int,
        json_height: int,
        actual_width: int,
        actual_height: int,
    ) -> Union[List, Dict]:
        """
        Apply dimension rescaling to any geometry type.

        Args:
            geometry_input: Geometry data
            json_width, json_height: Dimensions from JSON metadata
            actual_width, actual_height: Actual image dimensions

        Returns:
            Rescaled geometry
        """
        if json_width == actual_width and json_height == actual_height:
            return geometry_input  # No rescaling needed

        scale_x = actual_width / json_width
        scale_y = actual_height / json_height

        return CoordinateManager.scale_all_coordinates(geometry_input, scale_x, scale_y)

    @staticmethod
    def apply_smart_resize_to_geometry(
        geometry_input: Union[List, Dict],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
    ) -> Union[List, Dict]:
        """
        Apply smart resize scaling to any geometry type.

        Args:
            geometry_input: Geometry data
            original_width, original_height: Original image dimensions
            new_width, new_height: Target dimensions after smart resize

        Returns:
            Scaled geometry
        """
        if original_width == new_width and original_height == new_height:
            return geometry_input  # No scaling needed

        scale_x = new_width / original_width
        scale_y = new_height / original_height

        return CoordinateManager.scale_all_coordinates(geometry_input, scale_x, scale_y)

    @classmethod
    def transform_geometry_complete(
        cls,
        geometry_input: Union[List, Dict],
        image_path: Path,
        json_width: int,
        json_height: int,
        enable_smart_resize: bool = True,
        smart_resize_factor: int = 28,
    ) -> Tuple[List[float], Union[List, Dict], int, int]:
        """
        Apply complete geometry transformation pipeline for any geometry type.

        This is the unified replacement for transform_bbox_complete that works
        with both simple bbox and complex geometries.

        Pipeline:
        1. Apply EXIF orientation compensation
        2. Apply dimension mismatch rescaling
        3. Apply smart resize scaling (if enabled)

        Args:
            geometry_input: Any geometry type (bbox list, GeoJSON geometry, etc.)
            image_path: Path to image file
            json_width, json_height: Dimensions from JSON metadata
            enable_smart_resize: Whether to apply smart resize
            smart_resize_factor: Factor for smart resize calculation

        Returns:
            (bbox_for_compatibility, transformed_geometry, final_width, final_height)
        """
        # Step 1: Get EXIF transformation info
        is_exif_transformed, orig_w, orig_h, exif_w, exif_h = (
            CoordinateManager.get_exif_transform_matrix(image_path)
        )

        current_geometry = geometry_input
        current_width, current_height = exif_w, exif_h

        # Step 2: Apply EXIF orientation compensation if needed
        if is_exif_transformed:
            current_geometry = CoordinateManager.apply_exif_orientation_to_geometry(
                current_geometry, orig_w, orig_h, exif_w, exif_h
            )
            logger.debug(
                f"Applied EXIF orientation: {orig_w}x{orig_h} -> {exif_w}x{exif_h}"
            )

        # Step 3: Apply dimension mismatch rescaling if needed
        if json_width != current_width or json_height != current_height:
            current_geometry = CoordinateManager.apply_dimension_rescaling_to_geometry(
                current_geometry, json_width, json_height, current_width, current_height
            )
            logger.debug(
                f"Applied dimension rescaling: {json_width}x{json_height} -> {current_width}x{current_height}"
            )

        # Step 4: Apply smart resize scaling if enabled
        if enable_smart_resize:
            from data_conversion.vision_process import (
                MAX_PIXELS,
                MIN_PIXELS,
                smart_resize,
            )

            resize_h, resize_w = smart_resize(
                height=current_height,
                width=current_width,
                factor=smart_resize_factor,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )

            if resize_w != current_width or resize_h != current_height:
                current_geometry = CoordinateManager.apply_smart_resize_to_geometry(
                    current_geometry, current_width, current_height, resize_w, resize_h
                )
                current_width, current_height = resize_w, resize_h
                logger.debug(
                    f"Applied smart resize: {current_width}x{current_height} (within MAX_PIXELS={MAX_PIXELS})"
                )

        # Extract bbox for compatibility
        final_bbox = CoordinateManager.extract_bbox_from_geometry(current_geometry)

        return final_bbox, current_geometry, current_width, current_height

    @staticmethod
    def _reconstruct_geometry_from_points(
        original_geometry: Dict, transformed_points: List[Tuple[float, float]]
    ) -> Dict:
        """
        Reconstruct geometry object with transformed coordinate points.

        This maintains the original structure while updating coordinate values.
        """
        if not original_geometry or not transformed_points:
            return original_geometry

        geometry_type = original_geometry.get("type", "")
        result_geometry = original_geometry.copy()

        # Convert points back to original coordinate structure
        if geometry_type in ["ExtentPolygon", "LineString"]:
            # Direct 2D array format
            result_geometry["coordinates"] = [[x, y] for x, y in transformed_points]

        elif geometry_type == "Square":
            # 3D array format with rings
            if len(transformed_points) > 0:
                result_geometry["coordinates"] = [
                    [[x, y] for x, y in transformed_points]
                ]

        else:
            # Generic fallback - try to preserve original structure
            logger.warning(f"Reconstructing unknown geometry type: {geometry_type}")
            result_geometry["coordinates"] = [[x, y] for x, y in transformed_points]

        return result_geometry

    # ============================================================================
    # GEOMETRY PROCESSING METHODS (merged from GeometryProcessor)
    # ============================================================================

    @staticmethod
    def extract_bbox_from_geometry(geometry_input: Union[List, Dict]) -> List[float]:
        """
        Extract bounding box from any geometry type.

        Args:
            geometry_input: Geometry data (bbox list, GeoJSON geometry object, etc.)

        Returns:
            [x1, y1, x2, y2] bounding box coordinates (floats)
        """
        if not geometry_input:
            return [0.0, 0.0, 0.0, 0.0]

        # Handle simple bbox list
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            return [float(x) for x in geometry_input]

        # Handle GeoJSON-style geometry object
        if isinstance(geometry_input, dict):
            geometry_type = geometry_input.get("type", "")
            coordinates = geometry_input.get("coordinates", [])

            if not coordinates:
                logger.warning(f"Empty coordinates in geometry type: {geometry_type}")
                return [0.0, 0.0, 0.0, 0.0]

            # Extract all coordinate points
            all_points = CoordinateManager.get_all_coordinate_points(geometry_input)

            if not all_points:
                logger.warning(f"No valid coordinate points found in {geometry_type}")
                return [0.0, 0.0, 0.0, 0.0]

            # Calculate bounding box from all points
            x_coords = [point[0] for point in all_points]
            y_coords = [point[1] for point in all_points]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            bbox = [
                float(round(x_min)),
                float(round(y_min)),
                float(round(x_max)),
                float(round(y_max)),
            ]
            logger.debug(
                f"Extracted bbox {bbox} from {geometry_type} with {len(all_points)} points"
            )
            return bbox

        logger.warning(f"Unknown geometry input type: {type(geometry_input)}")
        return [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def get_all_coordinate_points(
        geometry_input: Union[List, Dict],
    ) -> List[Tuple[float, float]]:
        """
        Extract all coordinate points from any geometry type.

        Args:
            geometry_input: Geometry data

        Returns:
            List of (x, y) coordinate tuples
        """
        if not geometry_input:
            return []

        # Handle simple bbox list
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            x1, y1, x2, y2 = geometry_input
            return [(float(x1), float(y1)), (float(x2), float(y2))]

        # Handle GeoJSON-style geometry object
        if isinstance(geometry_input, dict):
            coordinates = geometry_input.get("coordinates", [])
            geometry_type = geometry_input.get("type", "")

            return CoordinateManager._extract_points_from_coordinates(
                coordinates, geometry_type
            )

        return []

    @staticmethod
    def _extract_points_from_coordinates(
        coordinates: List, geometry_type: str
    ) -> List[Tuple[float, float]]:
        """
        Extract coordinate points from GeoJSON coordinates array.

        Handles various coordinate structures:
        - ExtentPolygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x1,y1]]
        - Square: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x1,y1]]]
        - LineString: [[x1,y1], [x2,y2], [x3,y3], ...]
        """
        points = []

        try:
            if geometry_type in ["ExtentPolygon", "LineString"]:
                # Direct 2D array: [[x1,y1], [x2,y2], ...]
                for coord in coordinates:
                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        points.append((float(coord[0]), float(coord[1])))

            elif geometry_type == "Square":
                # 3D array with lineType: [[[x1,y1], [x2,y2], ...]]
                if coordinates and isinstance(coordinates[0], list):
                    for ring in coordinates:
                        for coord in ring:
                            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                                points.append((float(coord[0]), float(coord[1])))

            else:
                # Generic fallback - flatten all coordinate data
                points = CoordinateManager._flatten_coordinates_recursive(coordinates)

        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"Error extracting points from {geometry_type}: {e}")

        return points

    @staticmethod
    def _flatten_coordinates_recursive(coords: Any) -> List[Tuple[float, float]]:
        """Recursively flatten coordinate structures to extract all points."""
        points = []

        if isinstance(coords, (list, tuple)):
            # Check if this is a coordinate pair [x, y]
            if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
                points.append((float(coords[0]), float(coords[1])))
            else:
                # Recurse into nested structures
                for item in coords:
                    points.extend(
                        CoordinateManager._flatten_coordinates_recursive(item)
                    )

        return points

    @staticmethod
    def scale_all_coordinates(
        geometry_input: Union[List, Dict], scale_x: float, scale_y: float
    ) -> Union[List, Dict]:
        """
        Scale all coordinate points in any geometry type.

        Args:
            geometry_input: Original geometry data
            scale_x: Horizontal scaling factor
            scale_y: Vertical scaling factor

        Returns:
            Scaled geometry maintaining original structure
        """
        if not geometry_input:
            return geometry_input

        # Handle simple bbox list
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            x1, y1, x2, y2 = geometry_input
            return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

        # Handle GeoJSON-style geometry object
        if isinstance(geometry_input, dict):
            scaled_geometry = geometry_input.copy()
            coordinates = geometry_input.get("coordinates", [])
            geometry_type = geometry_input.get("type", "")

            if coordinates:
                scaled_coords = CoordinateManager._scale_coordinates_recursive(
                    coordinates, scale_x, scale_y
                )
                scaled_geometry["coordinates"] = scaled_coords

                logger.debug(
                    f"Scaled {geometry_type} coordinates by ({scale_x:.3f}, {scale_y:.3f})"
                )

            return scaled_geometry

        return geometry_input

    @staticmethod
    def _scale_coordinates_recursive(
        coords: Any, scale_x: float, scale_y: float
    ) -> Any:
        """Recursively scale coordinate structures."""
        if isinstance(coords, (list, tuple)):
            # Check if this is a coordinate pair [x, y]
            if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
                return [coords[0] * scale_x, coords[1] * scale_y]
            else:
                # Recurse into nested structures
                return [
                    CoordinateManager._scale_coordinates_recursive(
                        item, scale_x, scale_y
                    )
                    for item in coords
                ]

        return coords

    @staticmethod
    def validate_geometry_bounds(
        geometry_input: Union[List, Dict],
        width: int,
        height: int,
        tolerance: float = 0.1,
    ) -> bool:
        """
        Validate that all coordinate points are within image bounds.

        Args:
            geometry_input: Geometry data to validate
            width, height: Image dimensions
            tolerance: Floating point tolerance for bounds checking

        Returns:
            True if all coordinates are valid, False otherwise
        """
        all_points = CoordinateManager.get_all_coordinate_points(geometry_input)

        if not all_points:
            return False

        for x, y in all_points:
            if (
                x < -tolerance
                or y < -tolerance
                or x > width + tolerance
                or y > height + tolerance
            ):
                return False

        return True

    @staticmethod
    def extract_hierarchical_geometry(
        geometry_input: Union[List, Dict], preferred_format: str = "auto"
    ) -> Dict[str, Union[List[float], Dict, Any]]:
        """
        Extract geometry in hierarchical learning format with multiple annotation types.

        Args:
            geometry_input: Geometry data
            preferred_format: Preferred output format ("bbox_2d", "square", "line", or "auto")

        Returns:
            Dictionary with annotation format and coordinates
        """
        if not geometry_input:
            return {"bbox_2d": [0.0, 0.0, 0.0, 0.0]}

        # Handle simple bbox list
        if isinstance(geometry_input, list) and len(geometry_input) == 4:
            return {"bbox_2d": [float(x) for x in geometry_input]}

        # Handle GeoJSON-style geometry object
        if isinstance(geometry_input, dict):
            geometry_type = geometry_input.get("type", "")
            coordinates = geometry_input.get("coordinates", [])

            if not coordinates:
                logger.warning(f"Empty coordinates in geometry type: {geometry_type}")
                return {"bbox_2d": [0.0, 0.0, 0.0, 0.0]}

            # Extract bbox for all types (always available)
            bbox = CoordinateManager.extract_bbox_from_geometry(geometry_input)
            result: Dict[str, Union[List[float], Dict, Any]] = {"bbox_2d": bbox}

            # Determine appropriate annotation format based on geometry type and preference
            if geometry_type == "LineString" and (preferred_format in ["line", "auto"]):
                # Line annotation: flatten coordinates for line format
                line_coords = CoordinateManager._extract_line_coordinates(
                    geometry_input
                )
                if line_coords:
                    result["line"] = line_coords
                    if preferred_format == "line":
                        # Remove bbox_2d if line is preferred format
                        result = {"line": line_coords, "bbox_2d": bbox}

            elif geometry_type == "Square" and (preferred_format in ["square", "auto"]):
                # Square annotation: extract 4-point polygon
                square_coords = CoordinateManager._extract_square_coordinates(
                    geometry_input
                )
                if square_coords:
                    result["square"] = square_coords
                    if preferred_format == "square":
                        # Remove bbox_2d if square is preferred format
                        result = {"square": square_coords, "bbox_2d": bbox}

            # Always preserve full geometry for reference
            result["geometry"] = geometry_input
            return result

        logger.warning(f"Unknown geometry input type: {type(geometry_input)}")
        # Return a dictionary with a valid bbox_2d list to match expected type
        return {"bbox_2d": [0.0, 0.0, 0.0, 0.0]}

    @staticmethod
    def _extract_line_coordinates(geometry: Dict) -> List[float]:
        """Extract flattened line coordinates from LineString geometry."""
        coordinates = geometry.get("coordinates", [])
        line_coords = []

        for coord in coordinates:
            if isinstance(coord, list) and len(coord) >= 2:
                line_coords.extend(
                    [int(round(float(coord[0]))), int(round(float(coord[1])))]
                )

        return line_coords

    @staticmethod
    def _extract_square_coordinates(geometry: Dict) -> List[float]:
        """Extract 4-point square coordinates from Square/Polygon geometry."""
        coordinates = geometry.get("coordinates", [])

        # Handle nested coordinate structure
        if coordinates and isinstance(coordinates[0], list):
            points = (
                coordinates[0] if isinstance(coordinates[0][0], list) else coordinates
            )
        else:
            points = coordinates

        # Extract first 4 points for square format
        square_coords = []
        for _, point in enumerate(points[:4]):  # Take first 4 points
            if isinstance(point, list) and len(point) >= 2:
                square_coords.extend(
                    [int(round(float(point[0]))), int(round(float(point[1])))]
                )

        # Ensure we have exactly 8 coordinates (4 points)
        if len(square_coords) == 8:
            return square_coords
        else:
            logger.warning(
                f"Invalid square coordinates: expected 8 values, got {len(square_coords)}"
            )
            return []


# Merged from utils/transformations.py - FormatConverter class
class FormatConverter:
    """Handles conversion between different data formats."""

    @staticmethod
    def format_description(
        content_dict: Dict[str, str],
        response_types: List[str],
        language: str = "chinese",
    ) -> str:
        """Format content dictionary to Chinese description string."""
        parts = []
        for resp_type in response_types:
            value = content_dict.get(resp_type, "")
            if value:
                parts.append(value)

        if not parts:
            return ""

        # Use compact format for Chinese
        result = "/".join(parts)
        return result.replace(", ", "/").replace(",", "/")

    @staticmethod
    def parse_description_string(description: str) -> Dict[str, str]:
        """Parse Chinese description string back into components."""
        components = {"object_type": "", "property": "", "extra_info": ""}

        if not description:
            return components

        # Parse compact format (Chinese)
        parts = description.split("/")
        if len(parts) >= 1:
            components["object_type"] = parts[0].strip()
        if len(parts) >= 2:
            components["property"] = parts[1].strip()
        if len(parts) >= 3:
            components["extra_info"] = "/".join(parts[2:]).strip()

        return components

    @staticmethod
    def clean_annotation_content(data: Dict) -> Dict:
        """Clean annotation content preserving essential Chinese structure only."""
        cleaned_data = {}

        # Preserve essential metadata
        essential_keys = ["info", "tagInfo", "version"]
        for key in essential_keys:
            if key in data:
                cleaned_data[key] = data[key]

        # Clean features in markResult - Chinese only
        if "markResult" in data and "features" in data["markResult"]:
            cleaned_features = []

            for feature in data["markResult"]["features"]:
                properties = {}
                original_properties = feature.get("properties", {})

                # Keep only Chinese content
                properties["contentZh"] = original_properties.get("contentZh", {})

                cleaned_features.append(
                    {
                        "type": feature.get("type", "Feature"),
                        "geometry": feature.get("geometry", {}),
                        "properties": properties,
                    }
                )

            cleaned_data["markResult"] = {
                "features": cleaned_features,
                "type": data["markResult"].get("type", "FeatureCollection"),
            }

            # Preserve other markResult fields
            for key in data["markResult"]:
                if key not in ["features", "type"]:
                    cleaned_data["markResult"][key] = data["markResult"][key]

        return cleaned_data


# Merged from utils/validators.py - DataValidator and StructureValidator classes
class DataValidator:
    """Validates data structures and content."""

    @staticmethod
    def validate_bbox(
        bbox: List[float],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> bool:
        """
        Validate a single bounding box with enhanced checks.

        Args:
            bbox: A list of 4 numbers [x_min, y_min, x_max, y_max]
            image_width: Optional width to check bounds
            image_height: Optional height to check bounds

        Raises:
            ValueError: If the bounding box is invalid
        """
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Bbox must be a list of 4 elements, got: {bbox}")

        if not all(isinstance(coord, (int, float)) for coord in bbox):
            raise ValueError(f"Bbox coordinates must be numbers, got: {bbox}")

        x_min, y_min, x_max, y_max = bbox

        # Ensure correct ordering
        if x_min > x_max:
            x_min, x_max = x_max, x_min
            bbox = [x_min, y_min, x_max, y_max]
        if y_min > y_max:
            y_min, y_max = y_max, y_min
            bbox = [x_min, y_min, x_max, y_max]

        # Allow zero-width or zero-height bboxes (for lines)
        if x_min > x_max or y_min > y_max:
            raise ValueError(
                f"Invalid bbox: x_min <= x_max and y_min <= y_max required, got: {bbox}"
            )

        if x_min < 0 or y_min < 0 or x_max < 0 or y_max < 0:
            raise ValueError(f"Bbox coordinates cannot be negative, got: {bbox}")

        if image_width is not None and image_height is not None:
            if x_max > image_width or y_max > image_height:
                raise ValueError(
                    f"Bbox {bbox} exceeds image dimensions ({image_width}x{image_height})"
                )

        return True

    @staticmethod
    def validate_square(square) -> bool:
        """Validate square format [x1, y1, x2, y2, x3, y3, x4, y4]."""
        if not isinstance(square, list) or len(square) != 8:
            raise ValueError(f"Square must be list of 8 numbers, got {square}")

        for i, coord in enumerate(square):
            if not isinstance(coord, (int, float)):
                raise ValueError(
                    f"Square coordinate {i} must be number, got {type(coord)}"
                )

        return True

    @staticmethod
    def validate_line(line) -> bool:
        """Validate line format [x1, y1, x2, y2, ..., xn, yn]."""
        if not isinstance(line, list) or len(line) < 4 or len(line) % 2 != 0:
            raise ValueError(
                f"Line must be list of even number of coordinates (>=4), got {line}"
            )

        for i, coord in enumerate(line):
            if not isinstance(coord, (int, float)):
                raise ValueError(
                    f"Line coordinate {i} must be number, got {type(coord)}"
                )

        return True

    @staticmethod
    def validate_sample_structure(sample: Dict[str, Any]) -> bool:
        """Validate basic sample structure for training data."""
        required_fields = ["images", "objects"]

        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Missing required field: {field}")

        # Validate images field
        images = sample.get("images")
        if not isinstance(images, list) or len(images) == 0:
            raise ValueError("Field 'images' must be a non-empty list")

        # Validate objects field
        objects = sample.get("objects")
        if not isinstance(objects, list):
            raise ValueError("Field 'objects' must be a list")

        # Validate each object
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f"Object {i} must be a dictionary")

            # Check for required desc field
            if "desc" not in obj:
                raise ValueError(f"Object {i} missing 'desc'")

            # Check for at least one geometry type
            geometry_types = ["bbox_2d", "square", "line"]
            if not any(geom_type in obj for geom_type in geometry_types):
                raise ValueError(
                    f"Object {i} missing geometry type (bbox_2d, square, or line)"
                )

            # Validate geometry coordinates
            if "bbox_2d" in obj:
                DataValidator.validate_bbox(obj["bbox_2d"])
            elif "square" in obj:
                DataValidator.validate_square(obj["square"])
            elif "line" in obj:
                DataValidator.validate_line(obj["line"])

            # Validate description
            desc = obj["desc"]
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(f"Object {i} 'desc' must be non-empty string")

        return True


class StructureValidator:
    """Validates pipeline structures and outputs."""

    @staticmethod
    def validate_pipeline_output(
        train_samples: List[Dict], val_samples: List[Dict], teacher_samples: List[Dict]
    ) -> bool:
        """Validate complete pipeline output."""
        if not train_samples:
            raise ValueError("Training samples cannot be empty")

        # For small datasets, validation samples can be empty
        if not val_samples and len(train_samples) > 1:
            raise ValueError(
                "Validation samples cannot be empty when multiple training samples exist"
            )

        # For very small datasets, teacher samples can be empty
        if not teacher_samples and len(train_samples) + len(val_samples) > 2:
            raise ValueError(
                "Teacher samples cannot be empty when sufficient samples exist"
            )

        # Validate sample structures
        for i, sample in enumerate(train_samples):
            try:
                DataValidator.validate_sample_structure(sample)
            except ValueError as e:
                raise ValueError(f"Train sample {i}: {e}")

        for i, sample in enumerate(val_samples):
            try:
                DataValidator.validate_sample_structure(sample)
            except ValueError as e:
                raise ValueError(f"Validation sample {i}: {e}")

        for i, sample in enumerate(teacher_samples):
            try:
                DataValidator.validate_sample_structure(sample)
            except ValueError as e:
                raise ValueError(f"Teacher sample {i}: {e}")

        # Check for overlap between sets
        train_images = {sample["images"][0] for sample in train_samples}
        val_images = {sample["images"][0] for sample in val_samples}
        teacher_images = {sample["images"][0] for sample in teacher_samples}

        if train_images & val_images:
            raise ValueError("Training and validation sets have overlapping images")

        if train_images & teacher_images:
            raise ValueError("Training and teacher sets have overlapping images")

        if val_images & teacher_images:
            raise ValueError("Validation and teacher sets have overlapping images")

        logger.info(
            f"Pipeline output validation passed: "
            f"{len(train_samples)} train, {len(val_samples)} val, {len(teacher_samples)} teacher"
        )

        return True
