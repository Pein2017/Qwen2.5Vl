#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config loading and validation tests for Qwen2.5-VL pipeline.
"""

from pathlib import Path

from src_new.config.config import load_config


def test_config_yaml_loading_and_coordinate_settings():
    cfg_path = Path("/data3/Qwen2.5-VL-main/configs/bbu_v2_use_coord.yaml")
    cfg = load_config(cfg_path)

    # Fail-fast explicit expectations
    assert cfg is not None
    assert cfg.coordinate_tokens_enabled is True
    assert cfg.max_coord_value == 1024

    # Critical training args set for SafeTensors
    assert cfg.save_total_limit >= 0
    assert cfg.save_strategy is not None


def test_training_arguments_factory_safetensors():
    # Verify scripts.train_new creates TrainingArguments with save_safetensors=True
    from scripts.train_new import create_training_arguments_with_deepspeed

    class Dummy:
        output_dir = "/tmp/out"
        run_name = "t"
        num_train_epochs = 1
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 1
        gradient_accumulation_steps = 1
        learning_rate = 1e-4
        warmup_ratio = 0.0
        weight_decay = 0.0
        max_grad_norm = 1.0
        lr_scheduler_type = "linear"
        gradient_checkpointing = False
        bf16 = False
        fp16 = False
        eval_strategy = "no"
        eval_steps = 0
        save_strategy = "steps"
        save_steps = 1
        save_total_limit = 1
        load_best_model_at_end = False
        metric_for_best_model = "eval_loss"
        greater_is_better = False
        logging_steps = 1
        logging_dir = "/tmp/log"
        report_to = []
        disable_tqdm = True
        dataloader_num_workers = 0
        pin_memory = False
        prefetch_factor = None
        remove_unused_columns = True
        save_on_each_node = False

    args = create_training_arguments_with_deepspeed(Dummy())
    assert args.save_safetensors is True
