#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Qwen2VLProcessor

from src_new.models.patches import apply_comprehensive_qwen25_fixes
from src_new.models.wrapper import DetectionModel
from src_new.training.phase_freeze_manager import PhaseFreezeManager


logger = logging.getLogger("src_post.models")


def load_processor(path: str) -> Qwen2VLProcessor:
    proc = Qwen2VLProcessor.from_pretrained(path, use_fast=False)
    # Preserve tokenizer chat template if available
    try:
        tok = getattr(proc, "tokenizer", None)
        tmpl = getattr(tok, "chat_template", None)
        if tmpl:
            proc.chat_template = tmpl
    except Exception:
        pass
    return proc


def _build_minimal_training_config(base_model) -> Any:
    from types import SimpleNamespace

    merge_size = 14
    try:
        if (
            hasattr(base_model.config, "vision_config")
            and getattr(base_model.config.vision_config, "spatial_merge_size", None)
            is not None
        ):
            merge_size = int(base_model.config.vision_config.spatial_merge_size)
    except Exception:
        merge_size = 14
    cfg = SimpleNamespace(
        coordinate_tokens_enabled=False,
        coordinate_init_mode="fourier_ramp",
        new_geometry_tokens=[],
        max_coord_value=2047,
        merge_size=merge_size,
        use_cache=True,
        # ---- Required by LossManager (nonnegative weights; RL path won't use them) ----
        teacher_loss_weight=0.0,
        student_loss_weight=0.0,
        caption_loss_weight=0.0,
        grounding_loss_weight=0.0,
        formatting_loss_weight=0.0,
    )
    return cfg


def load_detection_model(
    checkpoint: str, processor: Qwen2VLProcessor, device: str
) -> DetectionModel:
    apply_comprehensive_qwen25_fixes()
    from transformers import Qwen2_5_VLForConditionalGeneration

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    try:
        base_model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass
    train_cfg = _build_minimal_training_config(base_model)
    model = DetectionModel(
        base_model=base_model,
        config=train_cfg,
        tokenizer=processor.tokenizer,
        skip_expansion=True,
    )
    model.eval().to(device)
    try:
        model.base_model.config.use_cache = True
    except Exception:
        pass
    return model


def _get_lang_layers(module: torch.nn.Module):
    for attr in ["model", "language_model", "transformer"]:
        sub = getattr(module, attr, None)
        if sub is None:
            continue
        layers = (
            getattr(sub, "layers", None)
            or getattr(sub, "h", None)
            or getattr(sub, "decoder", None)
        )
        if layers is not None and hasattr(layers, "__len__"):
            return layers
    return None


def _get_vision_blocks(module: torch.nn.Module):
    base = module
    for attr in ["visual", "vision_tower", "vision_model"]:
        sub = getattr(base, attr, None)
        if sub is not None:
            blocks = (
                getattr(sub, "blocks", None)
                or getattr(sub, "layers", None)
                or getattr(sub, "encoder", None)
            )
            if blocks is not None and hasattr(blocks, "__len__"):
                return blocks
    model_sub = getattr(base, "model", None)
    if model_sub is not None:
        visual_sub = getattr(model_sub, "visual", None)
        if visual_sub is not None:
            blocks = getattr(visual_sub, "blocks", None)
            if blocks is not None and hasattr(blocks, "__len__"):
                return blocks
    return None


def _discover_aligner_module(policy: DetectionModel) -> Optional[torch.nn.Module]:
    candidate_paths = [
        "visual.merger",
        "visual.mlp_align",
        "multi_modal_projector",
        "mm_projector",
        "visual_projector",
        "image_projector",
        "vision_tower.mlp",
        "vision_tower.proj",
        "vision_tower.projection",
    ]
    for path in candidate_paths:
        cur = policy.base_model
        ok = True
        for name in path.split("."):
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    return None


def apply_freeze_and_param_groups(
    policy: DetectionModel, tokenizer, cfg
) -> List[Dict[str, Any]]:
    # Freeze all first
    for p in policy.parameters():
        p.requires_grad = False
    pfm = PhaseFreezeManager()
    pfm.apply_phase(
        model=policy,
        tokenizer=tokenizer,
        phase="phase_3",
        llm_top_k_block=int(getattr(cfg, "llm_top_k_block", 0) or 0),
        vision_top_k_block=int(getattr(cfg, "vision_top_k_block", 0) or 0),
        freeze_patch_embed=bool(getattr(cfg, "freeze_patch_embed", True)),
    )
    params_groups: List[Dict[str, Any]] = []
    # Aligner group
    aligner_module = _discover_aligner_module(policy)
    if aligner_module is not None:
        # Respect explicit toggle; freeze aligner if disabled
        if not bool(getattr(cfg, "train_aligner", True)):
            for p in aligner_module.parameters():
                p.requires_grad = False
        else:
            aligner_params = [p for p in aligner_module.parameters() if p.requires_grad]
            if aligner_params:
                params_groups.append(
                    {
                        "params": aligner_params,
                        "lr": (cfg.aligner_lr or cfg.learning_rate),
                        "name": "aligner",
                    }
                )
    # LLM top-K group
    llm_params: List[torch.nn.Parameter] = []
    if (
        getattr(cfg, "llm_top_k_block", 0)
        and int(getattr(cfg, "llm_top_k_block", 0)) > 0
    ):
        layers = _get_lang_layers(policy.base_model)
        if layers is not None and hasattr(layers, "__len__"):
            for layer in list(layers)[-int(getattr(cfg, "llm_top_k_block", 0)) :]:
                for p in layer.parameters():
                    if p.requires_grad:
                        llm_params.append(p)
    if llm_params:
        params_groups.append(
            {
                "params": llm_params,
                "lr": (cfg.llm_lr or cfg.learning_rate),
                "name": "llm_topk",
            }
        )
    # Vision top-K group
    vision_params: List[torch.nn.Parameter] = []
    if (
        getattr(cfg, "vision_top_k_block", 0)
        and int(getattr(cfg, "vision_top_k_block", 0)) > 0
    ):
        vblocks = _get_vision_blocks(policy.base_model)
        if vblocks is not None and hasattr(vblocks, "__len__"):
            for block in list(vblocks)[-int(getattr(cfg, "vision_top_k_block", 0)) :]:
                for p in block.parameters():
                    if p.requires_grad:
                        vision_params.append(p)
    if vision_params:
        params_groups.append(
            {
                "params": vision_params,
                "lr": (cfg.vision_lr or cfg.learning_rate),
                "name": "vision_topk",
            }
        )
    # Fallback: if nothing selected, attach any trainables
    if len(params_groups) == 0:
        fallback = [p for p in policy.parameters() if p.requires_grad]
        if fallback:
            params_groups = [
                {"params": fallback, "lr": cfg.learning_rate, "name": "default"}
            ]
    return params_groups


def wrap_ddp_if_needed(model: torch.nn.Module, device: str) -> Optional[DDP]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
