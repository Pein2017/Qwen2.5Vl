#!/usr/bin/env python3
"""
Data Processor - Main Orchestrator

Coordinates all data processing components to convert raw JSON/image files
into train.jsonl, val.jsonl, and teacher.jsonl files in a single pipeline.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from core_modules import TokenMapper
from data_loader import SampleLoader
from data_splitter import DataSplitter
from sample_processor import SampleProcessor
from teacher_selector import TeacherSelector

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class DataProcessor:
    """Main orchestrator for the data processing pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self._validate_config()
        
        # Initialize components
        self.sample_loader = SampleLoader(self.config["input_dir"])
        self.token_mapper = self._init_token_mapper()
        self.label_hierarchy = self._load_label_hierarchy()
        self.sample_processor = self._init_sample_processor()
        self.teacher_selector = TeacherSelector(
            label_hierarchy=self.label_hierarchy,
            max_teachers=self.config.get("max_teachers", 10),
            seed=self.config.get("seed", 42)
        )
        self.data_splitter = DataSplitter(
            val_ratio=self.config.get("val_ratio", 0.1),
            seed=self.config.get("seed", 42)
        )
        
        logger.info("DataProcessor initialized successfully")
    
    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_fields = ["input_dir", "output_dir", "language"]
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Validate paths
        input_dir = Path(self.config["input_dir"])
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Validate language
        if self.config["language"] not in ["chinese", "english"]:
            raise ValueError(f"Unsupported language: {self.config['language']}")
        
        # Create output directories
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.get("resize", False):
            output_image_dir = Path(self.config.get("output_image_dir", "ds_rescaled"))
            output_image_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_token_mapper(self) -> Optional[TokenMapper]:
        """Initialize token mapper if token map file is provided."""
        token_map_path = self.config.get("token_map_path")
        if not token_map_path:
            if self.config["language"] == "english":
                raise ValueError("token_map_path is required for English language mode")
            return None
        
        token_map_path = Path(token_map_path)
        if not token_map_path.exists():
            raise FileNotFoundError(f"Token map file not found: {token_map_path}")
        
        return TokenMapper(token_map_path)
    
    def _load_label_hierarchy(self) -> Dict[str, List[str]]:
        """Load label hierarchy for filtering."""
        hierarchy_path = self.config.get("hierarchy_path", "data_conversion/label_hierarchy.json")
        hierarchy_path = Path(hierarchy_path)
        
        if not hierarchy_path.exists():
            logger.warning(f"Label hierarchy file not found: {hierarchy_path}, using empty hierarchy")
            return {}
        
        with open(hierarchy_path, "r", encoding="utf-8") as f:
            raw_hierarchy = json.load(f)
        
        # Normalize hierarchy format
        if isinstance(raw_hierarchy, list):
            hierarchy = {entry["object_type"]: entry.get("property", []) for entry in raw_hierarchy}
        elif isinstance(raw_hierarchy, dict):
            if all(isinstance(v, list) for v in raw_hierarchy.values()):
                hierarchy = raw_hierarchy
            else:
                hierarchy = {k: v.get("property", []) for k, v in raw_hierarchy.items()}
        else:
            logger.warning("Invalid hierarchy format, using empty hierarchy")
            hierarchy = {}
        
        logger.info(f"Loaded hierarchy with {len(hierarchy)} object types")
        return hierarchy
    
    def _init_sample_processor(self) -> SampleProcessor:
        """Initialize sample processor with configuration."""
        response_types = set(self.config.get("response_types", ["object_type", "property"]))
        
        return SampleProcessor(
            language=self.config["language"],
            response_types=response_types,
            label_hierarchy=self.label_hierarchy,
            token_mapper=self.token_mapper,
            resize_enabled=self.config.get("resize", False),
            output_image_dir=self.config.get("output_image_dir"),
            input_dir=self.config["input_dir"]
        )
    
    def _write_jsonl(self, samples: List[Dict], output_path: Path) -> None:
        """Write samples to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logger.info(f"Written {len(samples)} samples to {output_path}")
    
    def process(self) -> Dict[str, int]:
        """
        Execute the complete data processing pipeline.
        
        Returns:
            Dictionary with counts of train/val/teacher samples
        """
        logger.info("🚀 Starting data processing pipeline")
        
        # Step 1: Find and load all JSON files
        json_files = self.sample_loader.find_json_files()
        logger.info(f"📁 Found {len(json_files)} JSON files")
        
        # Step 2: Process all samples
        logger.info("🔄 Processing samples...")
        all_samples = []
        processed_count = 0
        skipped_count = 0
        
        output_base_dir = Path(self.config["output_dir"])
        
        for json_file in json_files:
            try:
                # Load JSON and image data
                json_data, image_path, width, height = self.sample_loader.load_sample_pair(json_file)
                
                # Process the sample
                processed_sample = self.sample_processor.process_sample(
                    json_data, image_path, width, height, output_base_dir
                )
                
                if processed_sample:
                    all_samples.append(processed_sample)
                    processed_count += 1
                else:
                    skipped_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} samples...")
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                skipped_count += 1
                # Fail-fast: re-raise the exception
                raise
        
        logger.info(f"✅ Sample processing complete: {processed_count} processed, {skipped_count} skipped")
        
        if not all_samples:
            raise ValueError("No valid samples were processed")
        
        # Step 3: Select teacher samples
        logger.info("📊 Selecting teacher samples...")
        teacher_samples, teacher_indices = self.teacher_selector.select_teachers(all_samples)
        
        # Step 4: Remove teacher samples from student pool
        teacher_image_paths = {sample["images"][0] for sample in teacher_samples}
        student_samples = [
            sample for sample in all_samples 
            if sample["images"][0] not in teacher_image_paths
        ]
        
        logger.info(f"📚 Teacher pool: {len(teacher_samples)} samples")
        logger.info(f"🎓 Student pool: {len(student_samples)} samples")
        
        # Step 5: Split student samples into train/val
        logger.info("🔧 Splitting train/validation data...")
        train_samples, val_samples = self.data_splitter.split(student_samples)
        
        # Step 6: Write output files
        output_dir = Path(self.config["output_dir"])
        
        self._write_jsonl(train_samples, output_dir / "train.jsonl")
        self._write_jsonl(val_samples, output_dir / "val.jsonl")
        self._write_jsonl(teacher_samples, output_dir / "teacher.jsonl")
        
        # Report token mapping issues if applicable
        if self.token_mapper:
            missing_tokens = self.token_mapper.get_missing_tokens()
            if missing_tokens:
                logger.warning(f"Found {len(missing_tokens)} missing tokens: {sorted(missing_tokens)}")
            else:
                logger.info("All tokens successfully mapped")
        
        # Final summary
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📊 Final Output:")
        logger.info(f"   Training: {len(train_samples)} samples → train.jsonl")
        logger.info(f"   Validation: {len(val_samples)} samples → val.jsonl")
        logger.info(f"   Teacher: {len(teacher_samples)} samples → teacher.jsonl")
        
        return {
            "train": len(train_samples),
            "val": len(val_samples),
            "teacher": len(teacher_samples),
            "total_processed": len(all_samples)
        }


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Processor for Qwen2.5-VL")
    
    # Required arguments
    parser.add_argument("--input_dir", required=True, help="Input directory with JSON/image files")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--language", choices=["chinese", "english"], required=True, help="Language mode")
    
    # Optional arguments
    parser.add_argument("--output_image_dir", help="Output directory for processed images")
    parser.add_argument("--token_map_path", help="Path to token mapping file")
    parser.add_argument("--hierarchy_path", help="Path to label hierarchy file")
    parser.add_argument("--response_types", nargs="+", default=["object_type", "property"],
                       help="Response types to include")
    parser.add_argument("--resize", action="store_true", help="Enable image resizing")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--max_teachers", type=int, default=10, help="Maximum teacher samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8"
    )
    
    # Build configuration
    config = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "language": args.language,
        "output_image_dir": args.output_image_dir,
        "token_map_path": args.token_map_path,
        "hierarchy_path": args.hierarchy_path,
        "response_types": args.response_types,
        "resize": args.resize,
        "val_ratio": args.val_ratio,
        "max_teachers": args.max_teachers,
        "seed": args.seed
    }
    
    # Create and run processor
    processor = DataProcessor(config)
    result = processor.process()
    
    print(f"\n✅ Processing complete: {result}")


if __name__ == "__main__":
    main()