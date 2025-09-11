#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import Qwen2VLProcessor
from transformers.generation import GenerationConfig, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from src_new.models.wrapper import DetectionModel


def to_device_and_cast(enc: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            t = v.to(device)
            if k == "pixel_values" and t.dtype == torch.float32 and torch.cuda.is_available():
                t = t.to(torch.bfloat16)
            out[k] = t
        else:
            out[k] = v
    return out


def decode_to_text(processor: Qwen2VLProcessor, token_ids: List[int]) -> str:
    try:
        return processor.tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception:
        try:
            toks = processor.tokenizer.convert_ids_to_tokens(token_ids)
            toks = [t for t in toks if isinstance(t, str)]
            return processor.tokenizer.convert_tokens_to_string(toks)
        except Exception:
            return ""


def sft_style_preprocess_image(img: Any) -> Image.Image:
    from data_conversion.vision_process import smart_resize, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS
    from data_conversion.utils.exif_utils import apply_exif_orientation
    if isinstance(img, Image.Image):
        pil = img
    elif isinstance(img, str):
        pil = Image.open(img)
    else:
        return img
    pil = apply_exif_orientation(pil)
    w, h = pil.size
    new_h, new_w = smart_resize(height=h, width=w, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    if (new_w, new_h) != (w, h):
        try:
            pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        except Exception:
            pil = pil.resize((new_w, new_h))
    return pil


class _StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: List[int]) -> None:
        super().__init__()
        self.stop_ids = set(int(s) for s in stop_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if input_ids is None or input_ids.size(0) == 0 or input_ids.size(1) == 0:
            return False
        last_id = int(input_ids[0, -1].item())
        return last_id in self.stop_ids


def build_stage_a_stopping(processor: Qwen2VLProcessor) -> Optional[StoppingCriteriaList]:
    newline_stop_ids: List[int] = []
    try:
        for s in ["\n", "\r\n", "。"]:
            ids = processor.tokenizer.encode(s, add_special_tokens=False)
            if isinstance(ids, list) and len(ids) == 1:
                newline_stop_ids.append(int(ids[0]))
    except Exception:
        pass
    if len(newline_stop_ids) == 0:
        return None
    return StoppingCriteriaList([_StopOnTokens(newline_stop_ids)])


def build_stage_a_context_lines(
    policy: DetectionModel,
    processor: Qwen2VLProcessor,
    images: List[Image.Image],
    gen_cfg: GenerationConfig,
    logits_processors: LogitsProcessorList,
    stopping: Optional[StoppingCriteriaList],
    sanitize: bool,
) -> List[str]:
    from src_post.conversation import GroupQCConversationBuilder
    lines: List[str] = []
    conv_builder = GroupQCConversationBuilder(processor=processor)
    for img in images:
        img_proc = sft_style_preprocess_image(img)
        messages = conv_builder.build_stage_a_messages(None)
        messages = [messages[0], {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "请只输出一行摘要"}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], images=[img_proc], padding=True, return_tensors="pt")
        enc = to_device_and_cast(enc, next(policy.parameters()).device)
        with torch.no_grad():
            out = policy.generate(
                **enc,
                generation_config=gen_cfg,
                logits_processor=logits_processors,
                stopping_criteria=stopping,
            )
        new_ids = out[:, enc["input_ids"].size(1):]
        toks = new_ids[0].tolist()
        raw = decode_to_text(processor, toks).strip()
        if sanitize:
            # Inline sanitized variant mirroring current runner behavior
            import re as _re
            text_s = raw.replace("Ġ", " ").replace("Ċ", " ")
            text_s = _re.sub(r"<\|[^|>]+?\|>", "", text_s)
            text_s = _re.sub(r"\[\s*(?:-?\d+\s*(?:,\s*)?)+\s*\]?", "", text_s)
            for ch in ["<", ">", "[", "]", "'", '"', "“", "”", "‘", "’"]:
                text_s = text_s.replace(ch, "")
            text_s = _re.sub(r"[A-Za-z0-9]+", "", text_s)
            text_s = _re.sub(r"\s+", " ", text_s)
            text_s = _re.sub(r"[,，；;]\s*[,，；;]+", ",", text_s)
            parts = _re.split(r"[，,；;、]", text_s)
            seen: List[str] = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(seen) == 0 or p != seen[-1]:
                    seen.append(p)
            text_s = "，".join(seen)
            text_s = text_s.strip("，,；;、")
            if len(text_s) > 80:
                text_s = text_s[:80]
            lines.append(text_s)
        else:
            lines.append(raw)
    return lines
