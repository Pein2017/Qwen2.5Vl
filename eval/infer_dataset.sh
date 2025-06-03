#!/bin/bash

# Pure inference script for Qwen2.5-VL model
# Role: ONLY generate raw responses (no evaluation)
export CUDA_VISIBLE_DEVICES=6

# Configuration
script='eval/infer_dataset.py'
model_path='output/qwen_multi_image_7B_lr2e-7_bs16_ep20_20250527_040517/checkpoint-340'
max_new_tokens=2048
output_file=eval_results/val_multi_image_${max_new_tokens}.json
val_jsonl_path='521_qwen_val_multi_image.jsonl'
max_samples=5 

# Create output directory
mkdir -p eval_results

echo "🚀 Running pure inference (training-aligned)..."
echo "📁 Model: $model_path"
echo "📊 Validation data: $val_jsonl_path"
echo "💾 Output file: $output_file"
echo "🔧 Max new tokens: $max_new_tokens"
echo "🧪 Max samples: $max_samples (testing mode)"
echo "🎯 Role: ONLY generate raw responses (no evaluation)"

# Build command
cmd="python $script \
    --model_path $model_path \
    --validation_jsonl $val_jsonl_path \
    --output_file $output_file \
    --max_new_tokens $max_new_tokens"

# Add max_samples if specified
if [ ! -z "$max_samples" ]; then
    cmd="$cmd --max_samples $max_samples"
fi

echo "📝 Command: $cmd"
echo ""

# Run inference
eval $cmd

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Pure inference completed successfully!"
    echo "📄 Raw responses saved to: $output_file"
    echo "📊 Next step: Run eval_dataset.py to calculate metrics"
    echo ""
    echo "Example next command:"
    echo "python eval/eval_dataset.py \\"
    echo "    --responses_file $output_file \\"
    echo "    --validation_jsonl $val_jsonl_path \\"
    echo "    --output_dir eval_results"
else
    echo "❌ Pure inference failed!"
    exit 1
fi
