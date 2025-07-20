#!/usr/bin/env python3
"""
Production Training Example with Coordinate Token Support

This example demonstrates how to integrate coordinate token soft expectation
regression into your existing training pipeline.

Usage:
    # Standard JSON format (current)
    python production_training_example.py --mode json
    
    # Coordinate token format (new soft expectation)
    python production_training_example.py --mode coordinate
    
    # Comparison training (both modes)
    python production_training_example.py --mode comparison
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.wrapper import Qwen25VLWithDetection, CoordinateConfig
from chat_processor import ChatProcessor
from transformers import AutoTokenizer
from config import config as global_config


class ProductionTrainingExample:
    """
    Example showing production integration of coordinate token training.
    """
    
    def __init__(self, mode: str = "coordinate"):
        self.mode = mode
        self.model_path = "/data3/Qwen2.5-VL-main/model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
        
        # Configure coordinate tokens based on mode
        self.use_coordinate_tokens = mode in ["coordinate", "comparison"]
        
        print(f"🎯 Production Training Example - Mode: {mode}")
        print(f"📍 Coordinate tokens: {'Enabled' if self.use_coordinate_tokens else 'Disabled'}")
        
    def setup_model_and_processor(self):
        """Setup model and data processor with appropriate configuration."""
        print("\n🔧 Setting up model and data processor...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print(f"   Original tokenizer vocab size: {len(tokenizer)}")
        
        # Configure coordinate tokens
        coordinate_config = CoordinateConfig(
            enable_coordinate_tokens=self.use_coordinate_tokens,
            max_coord_value=2048,
            coordinate_loss_weight=1.0,
            regular_loss_weight=1.0,
            soft_expectation_temperature=1.0,
            focal_loss_alpha=0.25,
            focal_loss_gamma=2.0,
        ) if self.use_coordinate_tokens else CoordinateConfig(enable_coordinate_tokens=False)
        
        # Create model
        model = Qwen25VLWithDetection(
            base_model_path=self.model_path,
            num_queries=100,
            max_caption_length=50,
            tokenizer=tokenizer,
            coordinate_config=coordinate_config
        )
        
        # Create chat processor
        chat_processor = ChatProcessor(
            tokenizer=tokenizer,
            image_processor=None,  # Will be set up by training pipeline
            enable_coordinate_tokens=self.use_coordinate_tokens,
            max_coord_value=2048,
            language="chinese",
            use_training_prompts=True,
        )
        
        print(f"   Model vocab size: {model.config.vocab_size}")
        print(f"   Coordinate tokens: {model.coordinate_tokens_enabled}")
        print(f"   Chat processor format: {'Coordinate' if chat_processor.coordinate_processor.enabled else 'JSON'}")
        
        return model, chat_processor, tokenizer
        
    def create_sample_training_data(self, chat_processor):
        """Create sample training data in the appropriate format."""
        print("\n📋 Creating sample training data...")
        
        # Sample BBU detection data
        sample_data = {
            "image_path": "sample_image.jpg",
            "objects": [
                {"bbox_2d": [0.1, 0.2, 0.4, 0.6], "desc": "螺丝连接器"},
                {"bbox_2d": [0.5, 0.3, 0.8, 0.7], "desc": "电缆接头"},
                {"bbox_2d": [0.2, 0.7, 0.6, 0.9], "desc": "标签贴纸"}
            ]
        }
        
        # Format objects using chat processor
        formatted_response = chat_processor._format_objects_response(sample_data["objects"])
        
        print(f"   Sample objects: {len(sample_data['objects'])}")
        print(f"   Formatted response:")
        print(f"   {formatted_response}")
        
        return sample_data, formatted_response
        
    def demonstrate_training_loop(self, model, chat_processor, sample_data):
        """Demonstrate a training step with the configured format."""
        print(f"\n🎓 Demonstrating training loop ({self.mode} mode)...")
        
        # Create training input
        system_prompt = "你是专业的BBU基站设备检测专家。"
        user_prompt = "请检测图像中的设备和部件:"
        
        # Format response
        formatted_response = chat_processor._format_objects_response(sample_data["objects"])
        
        # Create full conversation
        conversation_text = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant: {formatted_response}"
        
        # Tokenize
        inputs = model.tokenizer(
            conversation_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        print(f"   Input shape: {inputs['input_ids'].shape}")
        print(f"   Sample tokens: {inputs['input_ids'][0][:20].tolist()}")
        
        # Training forward pass
        model.train()
        outputs = model(
            input_ids=inputs['input_ids'].to(model.device),
            labels=inputs['input_ids'].to(model.device)  # Causal language modeling
        )
        
        loss = outputs.loss
        print(f"   Training loss: {loss:.6f}")
        
        # Demonstrate gradient computation
        loss.backward()
        
        # Check gradient flow to coordinate tokens (if enabled)
        if self.use_coordinate_tokens and model.extended_lm_head is not None:
            coord_grads = model.extended_lm_head.weight[model.original_vocab_size:].grad
            if coord_grads is not None:
                grad_norm = coord_grads.norm().item()
                print(f"   Coordinate token gradient norm: {grad_norm:.6f}")
                print(f"   ✅ Gradients flowing to coordinate tokens")
            
        print(f"   ✅ Training step completed successfully")
        
        return loss.item()
        
    def compare_formats(self):
        """Compare JSON vs coordinate token formats."""
        print("\n🔍 Comparing JSON vs Coordinate Token Formats")
        
        sample_objects = [{"bbox_2d": [0.1, 0.2, 0.8, 0.9], "desc": "螺丝连接器"}]
        
        # JSON format
        print("   JSON Format:")
        json_processor = ChatProcessor(
            tokenizer=AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True),
            image_processor=None,
            enable_coordinate_tokens=False
        )
        json_response = json_processor._format_objects_response(sample_objects)
        print(f"   {json_response}")
        
        # Coordinate token format
        print("   Coordinate Token Format:")
        coord_processor = ChatProcessor(
            tokenizer=AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True),
            image_processor=None,
            enable_coordinate_tokens=True
        )
        coord_response = coord_processor._format_objects_response(sample_objects)
        print(f"   {coord_response}")
        
        # Token count comparison
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        json_tokens = len(tokenizer(json_response)['input_ids'])
        coord_tokens = len(tokenizer(coord_response)['input_ids'])
        
        print(f"   Token counts - JSON: {json_tokens}, Coordinate: {coord_tokens}")
        
    def integration_with_existing_pipeline(self):
        """Show how to integrate with existing training configuration."""
        print("\n🔗 Integration with Existing Training Pipeline")
        
        # Show configuration updates needed
        config_updates = {
            "coordinate_tokens": {
                "enable_coordinate_tokens": True,
                "max_coord_value": 2048,
                "coordinate_loss_weight": 1.0,
                "regular_loss_weight": 1.0,
                "soft_expectation_temperature": 1.0,
            },
            "chat_processor": {
                "enable_coordinate_tokens": True,
                "max_coord_value": 2048,
            },
            "model": {
                "coordinate_config": "CoordinateConfig(enable_coordinate_tokens=True)",
            }
        }
        
        print("   Required configuration updates:")
        for section, settings in config_updates.items():
            print(f"   [{section}]")
            for key, value in settings.items():
                print(f"      {key} = {value}")
            
        # Show trainer integration
        print("\n   Trainer Integration:")
        print("   1. Pass coordinate_config to Qwen25VLWithDetection")
        print("   2. Pass enable_coordinate_tokens=True to ChatProcessor") 
        print("   3. No changes needed to training loop - loss computed automatically")
        print("   4. Model handles both regular and coordinate tokens seamlessly")
        
    def run_example(self):
        """Run the complete production example."""
        print("🚀 Production Training Example with Coordinate Tokens")
        print("=" * 60)
        
        if self.mode == "comparison":
            self.compare_formats()
            
        # Setup components
        model, chat_processor, tokenizer = self.setup_model_and_processor()
        
        # Create sample data
        sample_data, formatted_response = self.create_sample_training_data(chat_processor)
        
        # Demonstrate training
        loss = self.demonstrate_training_loop(model, chat_processor, sample_data)
        
        # Show integration guidance
        self.integration_with_existing_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 PRODUCTION EXAMPLE COMPLETED SUCCESSFULLY!")
        print(f"✅ Mode: {self.mode}")
        print(f"✅ Training loss: {loss:.6f}")
        print(f"✅ Coordinate tokens: {'Enabled' if self.use_coordinate_tokens else 'Disabled'}")
        
        if self.use_coordinate_tokens:
            print("\n🎯 Next Steps for Coordinate Token Training:")
            print("1. Update your training config to enable coordinate tokens")
            print("2. Run comparative training (JSON vs Coordinate) to measure improvement")
            print("3. Evaluate localization accuracy with soft expectation regression")
            print("4. Monitor coordinate token gradient flow during training")
            
        return model, loss


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Production training example with coordinate tokens")
    parser.add_argument(
        "--mode", 
        choices=["json", "coordinate", "comparison"],
        default="coordinate",
        help="Training mode: json (standard), coordinate (soft expectation), or comparison"
    )
    
    args = parser.parse_args()
    
    # Run example
    example = ProductionTrainingExample(mode=args.mode)
    model, loss = example.run_example()
    
    print(f"\n📊 Final Results:")
    print(f"   Mode: {args.mode}")
    print(f"   Model vocab size: {model.config.vocab_size}")
    print(f"   Training loss: {loss:.6f}")
    print(f"   Success: ✅")


if __name__ == "__main__":
    main()