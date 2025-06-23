python data_analysis/create_teacher_pool.py \
  --data_path data_conversion/qwen_combined.jsonl \
  --hierarchy data_conversion/label_hierarchy.json \
  --max_teachers 10 \
  --output data_analysis/teacher_pool.json