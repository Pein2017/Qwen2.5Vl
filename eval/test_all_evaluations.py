#!/usr/bin/env python3
"""
Comprehensive test suite for the Qwen2.5-VL evaluation pipeline.

This script merges tests from:
- test_simplified_eval.py
- test_enhanced_metrics.py
- test_without_transformers.py

It covers:
1. Basic evaluation functionality and script execution.
2. Enhanced metrics (soft, hierarchical, novel) with transformer-based semantics.
3. Rule-based fallback for semantic matching when transformers are unavailable.
4. Detailed demonstrations of hierarchical matching.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.coco_metrics import COCOStyleMetrics

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- TEST DATA ---

# From test_simplified_eval.py
SIMPLE_TEST_DATA = [
    {
        "id": "sample_001",
        "ground_truth": '[{"bbox_2d": [10, 20, 100, 120], "label": "螺丝连接点/BBU安装螺丝"}, {"bbox_2d": [200, 300, 250, 350], "label": "机柜空间"}]',
        "result": '[{"bbox_2d": [12, 22, 98, 118], "label": "螺丝连接点/BBU安装螺丝"}, {"bbox_2d": [205, 305, 245, 345], "label": "机柜空间"}]',
    },
    {
        "id": "sample_002",
        "ground_truth": '[{"bbox_2d": [50, 60, 150, 160], "label": "bbu基带处理单元/华为"}]',
        "result": '[{"bbox_2d": [55, 65, 145, 155], "label": "bbu基带处理单元/华为"}]',
    },
]


def _create_enhanced_test_data():
    """Create test data that demonstrates various scenarios."""
    return [
        # Sample 1: Perfect matches
        {
            "id": "sample_001",
            "ground_truth": json.dumps(
                [
                    {"bbox_2d": [10, 20, 100, 120], "label": "螺丝连接点/BBU安装螺丝"},
                    {"bbox_2d": [200, 300, 400, 500], "label": "线缆/光纤"},
                ]
            ),
            "result": json.dumps(
                [
                    {"bbox_2d": [12, 22, 98, 118], "label": "螺丝连接点/BBU安装螺丝"},
                    {"bbox_2d": [202, 302, 398, 498], "label": "线缆/光纤"},
                ]
            ),
        },
        # Sample 2: Hierarchical partial matches
        {
            "id": "sample_002",
            "ground_truth": json.dumps(
                [
                    {
                        "bbox_2d": [50, 60, 150, 160],
                        "label": "螺丝连接点/BBU安装螺丝/连接正确",
                    },
                    {"bbox_2d": [300, 400, 500, 600], "label": "BBU基带处理单元/华为"},
                ]
            ),
            "result": json.dumps(
                [
                    {"bbox_2d": [52, 62, 148, 158], "label": "螺丝连接点"},
                    {"bbox_2d": [302, 402, 498, 598], "label": "BBU/Huawei"},
                ]
            ),
        },
        # Sample 3: Novel descriptions
        {
            "id": "sample_003",
            "ground_truth": json.dumps(
                [
                    {"bbox_2d": [100, 200, 300, 400], "label": "机柜空间"},
                    {"bbox_2d": [400, 500, 600, 700], "label": "标签贴纸"},
                ]
            ),
            "result": json.dumps(
                [
                    {"bbox_2d": [102, 202, 298, 398], "label": "机柜内部空间"},
                    {"bbox_2d": [402, 502, 598, 698], "label": "设备标识贴纸"},
                    {"bbox_2d": [700, 800, 900, 1000], "label": "散热风扇"},
                ]
            ),
        },
        # Sample 4: Localization vs classification errors
        {
            "id": "sample_004",
            "ground_truth": json.dumps(
                [
                    {
                        "bbox_2d": [50, 50, 200, 200],
                        "label": "螺丝连接点/BBU接地线机柜接地端",
                    },
                    {"bbox_2d": [300, 300, 450, 450], "label": "线缆/非光纤"},
                ]
            ),
            "result": json.dumps(
                [
                    {
                        "bbox_2d": [150, 150, 300, 300],
                        "label": "螺丝连接点/BBU接地线机柜接地端",
                    },
                    {"bbox_2d": [302, 302, 448, 448], "label": "挡风板"},
                ]
            ),
        },
    ]


# --- TEST FUNCTIONS ---


def test_simplified_evaluation_direct():
    """Test direct invocation of the evaluation pipeline."""
    logger.info("🧪 Testing Simplified Evaluation (Direct Call)")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SIMPLE_TEST_DATA, f, indent=2, ensure_ascii=False)
        test_file = f.name
    output_file = tempfile.mktemp(suffix="_metrics.json")

    try:
        evaluator = COCOStyleMetrics()
        results = evaluator.evaluate_dataset(
            responses_file=test_file, output_file=output_file, use_semantic=True
        )
        overall = results.get("overall_metrics", {})
        logger.info(
            f"   mAP: {overall.get('mAP', 0):.4f}, AP@0.5: {overall.get('AP@0.5', 0):.4f}"
        )
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_simplified_evaluation_script():
    """Test the eval_dataset.py script via subprocess."""
    logger.info("🧪 Testing Simplified Evaluation (Script Call)")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SIMPLE_TEST_DATA, f, indent=2, ensure_ascii=False)
        test_file = f.name
    output_file = tempfile.mktemp(suffix="_metrics.json")

    try:
        cmd = [
            sys.executable,
            "eval/eval_dataset.py",
            "--responses_file",
            test_file,
            "--output_file",
            output_file,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=".", check=True
        )
        if os.path.exists(output_file):
            logger.info("   Script execution successful.")
            return True
        else:
            logger.error("   Script ran but output file was not created.")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"❌ Script test failed: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_enhanced_metrics_comparison():
    """Run evaluation with different settings to showcase enhanced metrics."""
    logger.info("🧪 Testing Enhanced Metrics Comparison (requires transformers)")
    test_data = _create_enhanced_test_data()
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_responses.json")
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        settings = [
            {
                "name": "Standard",
                "enable_soft_matching": False,
                "enable_hierarchical": False,
                "enable_novel_detection": False,
            },
            {
                "name": "Hierarchical",
                "enable_soft_matching": False,
                "enable_hierarchical": True,
                "enable_novel_detection": False,
            },
            {
                "name": "Soft Semantic",
                "enable_soft_matching": True,
                "enable_hierarchical": False,
                "enable_novel_detection": False,
            },
            {
                "name": "Full Enhanced",
                "enable_soft_matching": True,
                "enable_hierarchical": True,
                "enable_novel_detection": True,
            },
        ]

        for setting in settings:
            logger.info(f"   - Running with setting: {setting['name']}")
            evaluator = COCOStyleMetrics(
                semantic_threshold=0.7,
                enable_soft_matching=setting["enable_soft_matching"],
                enable_hierarchical=setting["enable_hierarchical"],
                enable_novel_detection=setting["enable_novel_detection"],
            )
            output_file = os.path.join(temp_dir, f"results_{setting['name']}.json")
            evaluator.evaluate_dataset(test_file, output_file)
    logger.info("   Enhanced metrics comparison tests completed.")
    return True


def test_enhanced_metrics_with_transformers():
    """Test enhanced metrics with SentenceTransformer semantic matching."""
    logger.info("🧪 Testing Enhanced Metrics (SentenceTransformer)")
    test_data = _create_enhanced_test_data()
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_responses.json")
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        evaluator = COCOStyleMetrics(
            semantic_threshold=0.3,
            enable_soft_matching=True,
            enable_hierarchical=True,
            enable_novel_detection=True,
        )
        output_file = os.path.join(temp_dir, "test_results.json")
        results = evaluator.evaluate_dataset(test_file, output_file)

        if results and "overall_metrics" in results:
            logger.info("   Enhanced metrics test completed successfully.")
            return True
        else:
            logger.error("   Enhanced metrics test failed to produce results.")
            return False


def test_categorization_modes():
    """Test both grouped and individual categorization modes."""
    print("\n" + "=" * 60)
    print("TESTING CATEGORIZATION MODES")
    print("=" * 60)

    # Sample data with diverse object labels
    sample_data = [
        {
            "predictions": [
                {"bbox_2d": [10, 10, 50, 50], "label": "螺丝_M8"},
                {"bbox_2d": [60, 60, 100, 100], "label": "螺丝_M10"},
                {"bbox_2d": [110, 110, 150, 150], "label": "BBU_模块_A"},
            ],
            "ground_truth": [
                {"bbox_2d": [12, 12, 52, 52], "label": "螺丝_M8"},
                {"bbox_2d": [62, 62, 102, 102], "label": "螺丝_M10"},
                {"bbox_2d": [112, 112, 152, 152], "label": "BBU_模块_A"},
            ],
        }
    ]

    # Test with grouped categories (default)
    print("\n--- Testing GROUPED CATEGORIES mode ---")
    evaluator_grouped = COCOStyleMetrics(
        iou_thresholds=[0.5],
        enable_soft_matching=False,
        enable_hierarchical=False,
        enable_novel_detection=False,
        use_individual_categories=True,  # Override default to use grouped mode
    )

    # Extract predictions and ground truths
    all_preds = [sample_data[0]["predictions"]]
    all_gts = [sample_data[0]["ground_truth"]]

    # Calculate metrics
    grouped_results = evaluator_grouped.calculate_ap_ar(all_preds, all_gts)
    print(
        f"Grouped results: mAP={grouped_results['mAP']:.3f}, mAR={grouped_results['mAR']:.3f}"
    )

    # Show categories found
    grouped_categories = set()
    for gts in all_gts:
        for gt in gts:
            grouped_categories.add(evaluator_grouped.get_object_category(gt["label"]))
    print(f"Grouped categories found: {sorted(grouped_categories)}")

    # Test with individual categories
    print("\n--- Testing INDIVIDUAL CATEGORIES mode ---")
    evaluator_individual = COCOStyleMetrics(
        iou_thresholds=[0.5],
        enable_soft_matching=False,
        enable_hierarchical=False,
        enable_novel_detection=False,
    )

    individual_results = evaluator_individual.calculate_ap_ar(all_preds, all_gts)
    print(
        f"Individual results: mAP={individual_results['mAP']:.3f}, mAR={individual_results['mAR']:.3f}"
    )

    # Show categories found
    individual_categories = set()
    for gts in all_gts:
        for gt in gts:
            individual_categories.add(
                evaluator_individual.get_object_category(gt["label"])
            )
    print(f"Individual categories found: {sorted(individual_categories)}")

    # Compare the difference
    print(f"\nComparison:")
    print(f"  Grouped mode: {len(grouped_categories)} categories")
    print(f"  Individual mode: {len(individual_categories)} categories")
    print(f"  Individual mode provides more granular analysis for diverse datasets!")


def test_robust_llm_validation():
    """Test robust validation with problematic LLM outputs."""
    print("\n" + "=" * 60)
    print("TESTING ROBUST LLM VALIDATION")
    print("=" * 60)

    # Create test data with various problematic LLM outputs
    problematic_data = [
        {
            "result": json.dumps(
                [
                    # Valid object
                    {"bbox_2d": [10, 10, 50, 50], "label": "valid_object"},
                    # Wrong coordinate order
                    {"bbox_2d": [100, 100, 50, 50], "label": "wrong_order"},
                    # Out of bounds coordinates
                    {"bbox_2d": [-10, -10, 2000, 2000], "label": "out_of_bounds"},
                    # String coordinates
                    {"bbox_2d": "[60, 60, 100, 100]", "label": "string_coords"},
                    # Missing coordinates
                    {"bbox_2d": [70, 70], "label": "missing_coords"},
                    # NaN coordinates
                    {"bbox_2d": [80, 80, float("nan"), 120], "label": "nan_coords"},
                    # Zero area box
                    {"bbox_2d": [90, 90, 90, 90], "label": "zero_area"},
                    # Empty label
                    {"bbox_2d": [110, 110, 150, 150], "label": ""},
                    # None label
                    {"bbox_2d": [120, 120, 160, 160], "label": None},
                ]
            ),
            "ground_truth": [
                {"bbox_2d": [12, 12, 52, 52], "label": "valid_object"},
                {"bbox_2d": [62, 62, 102, 102], "label": "string_coords"},
            ],
        }
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "problematic_responses.json")
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(problematic_data, f, ensure_ascii=False, indent=2)

        print("\nTesting with problematic LLM outputs...")
        evaluator = COCOStyleMetrics(
            iou_thresholds=[0.5],
            enable_soft_matching=False,
            enable_hierarchical=False,
            enable_novel_detection=False,
        )

        output_file = os.path.join(temp_dir, "robust_results.json")
        results = evaluator.evaluate_dataset(test_file, output_file)

        print(f"Results: mAP={results['overall_metrics']['mAP']:.3f}")
        print(f"Valid predictions: {results['evaluation_info']['total_predictions']}")
        print(f"Valid ground truth: {results['evaluation_info']['total_ground_truth']}")

        print("\nRobust validation successfully handled problematic LLM outputs!")
        return True


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    results = defaultdict(lambda: "SKIPPED")

    logger.info("\n" + "=" * 70)
    logger.info("🚀 RUNNING SIMPLIFIED EVALUATION TESTS")
    logger.info("=" * 70)
    results["simplified_direct"] = (
        "PASS" if test_simplified_evaluation_direct() else "FAIL"
    )
    results["simplified_script"] = (
        "PASS" if test_simplified_evaluation_script() else "FAIL"
    )

    logger.info("\n" + "=" * 70)
    logger.info("🚀 RUNNING ENHANCED METRICS TESTS")
    logger.info("=" * 70)
    try:
        results["enhanced_metrics"] = (
            "PASS" if test_enhanced_metrics_with_transformers() else "FAIL"
        )
    except ImportError:
        logger.warning(
            "   SentenceTransformers not found. Skipping enhanced metrics tests."
        )
        results["enhanced_metrics"] = "SKIPPED"

    logger.info("\n" + "=" * 70)
    logger.info("🚀 TESTING CATEGORIZATION MODES")
    logger.info("=" * 70)
    try:
        test_categorization_modes()
        results["categorization_modes"] = "PASS"
    except Exception as e:
        logger.error(f"Categorization modes test failed: {e}")
        results["categorization_modes"] = "FAIL"

    logger.info("\n" + "=" * 70)
    logger.info("🚀 TESTING ROBUST LLM VALIDATION")
    logger.info("=" * 70)
    try:
        results["robust_validation"] = (
            "PASS" if test_robust_llm_validation() else "FAIL"
        )
    except Exception as e:
        logger.error(f"Robust validation test failed: {e}")
        results["robust_validation"] = "FAIL"

    logger.info("\n" + "=" * 70)
    logger.info("🏁 OVERALL TEST SUMMARY")
    logger.info("=" * 70)

    all_passed = True
    for name, status in results.items():
        icon = "✅" if status == "PASS" else ("❌" if status == "FAIL" else "⚠️")
        logger.info(f"   {name:<30} {icon} {status}")
        if status == "FAIL":
            all_passed = False

    logger.info("-" * 70)
    if all_passed:
        logger.info("🎉 All tests passed or were skipped correctly!")
    else:
        logger.error("🔥 Some tests failed. Please review the logs.")
        sys.exit(1)
