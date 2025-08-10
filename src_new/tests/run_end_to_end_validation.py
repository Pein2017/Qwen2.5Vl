#!/usr/bin/env python3
"""
Comprehensive end-to-end validation runner for the BBU inference pipeline.

This script runs all end-to-end validation tests including:
- Real data format processing
- Eval script integration 
- Performance benchmarks
- Error handling validation
- Memory usage analysis
"""

import os
import sys
import subprocess
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src_new to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from tests.integration.inference.test_end_to_end_pipeline import (
    TestEndToEndInferencePipeline,
    TestEvalScriptIntegration
)


class EndToEndValidationRunner:
    """Main runner for comprehensive end-to-end validation."""
    
    def __init__(self, verbose: bool = True, fast_mode: bool = False):
        self.verbose = verbose
        self.fast_mode = fast_mode
        self.results = {}
        
    def run_validation_suite(self) -> Dict[str, Any]:
        """Run complete end-to-end validation suite."""
        print("🚀 Starting End-to-End BBU Inference Pipeline Validation")
        print("=" * 70)
        
        validation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_suites": {},
            "summary": {},
            "errors": []
        }
        
        # Test suites to run
        test_suites = [
            {
                "name": "Real Data Processing",
                "description": "Validate processing of actual dataset formats",
                "test_class": "TestEndToEndInferencePipeline",
                "tests": [
                    "test_complete_pipeline_with_real_data_format",
                    "test_eval_script_parameter_compatibility",
                    "test_performance_benchmarks_with_real_constraints"
                ]
            },
            {
                "name": "Eval Script Integration", 
                "description": "Test integration with eval/infer_dataset.sh",
                "test_class": "TestEvalScriptIntegration",
                "tests": [
                    "test_eval_script_command_compatibility",
                    "test_eval_script_file_structure_compatibility"
                ]
            }
        ]
        
        if not self.fast_mode:
            test_suites.append({
                "name": "Performance & Memory",
                "description": "Memory usage and performance validation",
                "test_class": "TestEndToEndInferencePipeline", 
                "tests": [
                    "test_memory_usage_with_coordinate_tokens",
                    "test_error_handling_and_recovery"
                ]
            })
        
        # Run each test suite
        for suite in test_suites:
            print(f"\n📋 Running {suite['name']} Tests")
            print(f"   {suite['description']}")
            print("-" * 50)
            
            suite_results = self._run_test_suite(suite)
            validation_results["test_suites"][suite["name"]] = suite_results
            
            if self.verbose:
                self._print_suite_results(suite["name"], suite_results)
        
        # Generate summary
        validation_results["summary"] = self._generate_summary(validation_results)
        
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)
        
        self._print_summary(validation_results["summary"])
        
        # Save detailed results
        self._save_results(validation_results)
        
        return validation_results
    
    def _run_test_suite(self, suite: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test suite."""
        suite_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": {},
            "execution_time": 0
        }
        
        start_time = time.time()
        
        # Build pytest command for specific tests
        test_file = "src_new/tests/integration/inference/test_end_to_end_pipeline.py"
        
        for test_name in suite["tests"]:
            test_pattern = f"{suite['test_class']}::{test_name}"
            
            print(f"  🔍 Running {test_name}...")
            
            test_result = self._run_single_test(test_file, test_pattern)
            
            suite_results["tests_run"] += 1
            suite_results["test_details"][test_name] = test_result
            
            if test_result["passed"]:
                suite_results["tests_passed"] += 1
                print(f"    ✅ PASSED ({test_result['duration']:.2f}s)")
            else:
                suite_results["tests_failed"] += 1
                print(f"    ❌ FAILED ({test_result['duration']:.2f}s)")
                if self.verbose and test_result["error"]:
                    print(f"       Error: {test_result['error']}")
        
        suite_results["execution_time"] = time.time() - start_time
        return suite_results
    
    def _run_single_test(self, test_file: str, test_pattern: str) -> Dict[str, Any]:
        """Run a single test and capture results."""
        test_result = {
            "passed": False,
            "duration": 0,
            "error": None,
            "output": ""
        }
        
        start_time = time.time()
        
        try:
            # Run pytest with specific test
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "-k", test_pattern,
                "-v",
                "--tb=short",
                "--capture=no" if self.verbose else "--capture=sys"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            test_result["duration"] = time.time() - start_time
            test_result["output"] = result.stdout + result.stderr
            test_result["passed"] = result.returncode == 0
            
            if result.returncode != 0:
                test_result["error"] = f"Exit code: {result.returncode}"
                
        except subprocess.TimeoutExpired:
            test_result["duration"] = time.time() - start_time
            test_result["error"] = "Test timed out after 5 minutes"
            
        except Exception as e:
            test_result["duration"] = time.time() - start_time
            test_result["error"] = str(e)
        
        return test_result
    
    def _print_suite_results(self, suite_name: str, results: Dict[str, Any]):
        """Print detailed results for a test suite."""
        total_tests = results["tests_run"]
        passed_tests = results["tests_passed"]
        failed_tests = results["tests_failed"]
        
        print(f"\n📈 {suite_name} Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ✅")
        print(f"   Failed: {failed_tests} ❌")
        print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%")
        print(f"   Execution Time: {results['execution_time']:.2f}s")
        
        if failed_tests > 0 and self.verbose:
            print(f"\n❌ Failed Tests in {suite_name}:")
            for test_name, details in results["test_details"].items():
                if not details["passed"]:
                    print(f"   • {test_name}: {details['error']}")
    
    def _generate_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall validation summary."""
        summary = {
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
            "overall_success_rate": 0,
            "total_execution_time": 0,
            "suite_summaries": {}
        }
        
        for suite_name, suite_results in validation_results["test_suites"].items():
            summary["total_tests"] += suite_results["tests_run"]
            summary["total_passed"] += suite_results["tests_passed"]
            summary["total_failed"] += suite_results["tests_failed"]
            summary["total_execution_time"] += suite_results["execution_time"]
            
            summary["suite_summaries"][suite_name] = {
                "tests": suite_results["tests_run"],
                "passed": suite_results["tests_passed"],
                "success_rate": (suite_results["tests_passed"] / suite_results["tests_run"] * 100) if suite_results["tests_run"] > 0 else 0
            }
        
        if summary["total_tests"] > 0:
            summary["overall_success_rate"] = (summary["total_passed"] / summary["total_tests"]) * 100
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print validation summary."""
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['total_passed']} ✅")
        print(f"   Failed: {summary['total_failed']} ❌")
        print(f"   Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"   Total Time: {summary['total_execution_time']:.2f}s")
        
        print(f"\n📋 Suite Breakdown:")
        for suite_name, suite_summary in summary["suite_summaries"].items():
            status_icon = "✅" if suite_summary["success_rate"] == 100 else "⚠️" if suite_summary["success_rate"] >= 80 else "❌"
            print(f"   {status_icon} {suite_name}: {suite_summary['passed']}/{suite_summary['tests']} ({suite_summary['success_rate']:.1f}%)")
        
        # Overall status
        if summary["overall_success_rate"] == 100:
            print(f"\n🎉 ALL TESTS PASSED - Pipeline is production ready!")
        elif summary["overall_success_rate"] >= 80:
            print(f"\n⚠️  MOSTLY PASSING - Some issues need attention")
        else:
            print(f"\n❌ SIGNIFICANT FAILURES - Pipeline needs fixes before production")
    
    def _save_results(self, validation_results: Dict[str, Any]):
        """Save validation results to file."""
        # Create results directory
        results_dir = Path(__file__).parent / "validation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        timestamp = validation_results["timestamp"].replace(" ", "_").replace(":", "-")
        results_file = results_dir / f"end_to_end_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {results_file}")


def main():
    """Main entry point for end-to-end validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive end-to-end validation for BBU inference pipeline"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Run in fast mode (skip memory/performance tests)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true", 
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Create and run validation
    runner = EndToEndValidationRunner(
        verbose=not args.quiet,
        fast_mode=args.fast
    )
    
    try:
        results = runner.run_validation_suite()
        
        # Exit with appropriate code
        if results["summary"]["overall_success_rate"] == 100:
            sys.exit(0)  # All tests passed
        elif results["summary"]["overall_success_rate"] >= 80:
            sys.exit(1)  # Some failures but mostly working
        else:
            sys.exit(2)  # Significant failures
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Validation interrupted by user")
        sys.exit(3)
        
    except Exception as e:
        print(f"\n\n💥 Validation runner failed: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()