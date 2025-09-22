#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable

import torch
import logging
from PIL import Image
from transformers import Qwen2VLProcessor
from transformers.generation import GenerationConfig, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from src_new.models.wrapper import DetectionModel
from src_new.processing.special_tokens import IMAGE_PAD

logger = logging.getLogger("src_post.generation")


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


def build_decision_prefix_constraint(processor: Qwen2VLProcessor, prompt_len: int) -> Optional[Callable[[int, torch.Tensor], List[int]]]:
    try:
        tkn = processor.tokenizer
        ids_pass = tkn.encode("总评: 通过", add_special_tokens=False)
        ids_fail = tkn.encode("总评: 不通过", add_special_tokens=False)
        def _prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
            # HuggingFace passes a 1D tensor here: shape [seq_len]; handle both 1D/2D defensively
            try:
                seq_len = int(input_ids.size(-1))
            except Exception:
                return list(range(len(tkn.get_vocab())))
            generated_len = int(seq_len - int(prompt_len))
            if generated_len < 0:
                generated_len = 0
            alts = [ids_pass, ids_fail]
            compatible: List[List[int]] = []
            for alt in alts:
                if generated_len <= len(alt):
                    ok = True
                    for m in range(generated_len):
                        cur_id = int(input_ids[prompt_len + m]) if input_ids.dim() == 1 else int(input_ids[0, prompt_len + m])
                        if cur_id != int(alt[m]):
                            ok = False
                            break
                    if ok:
                        compatible.append(alt)
            max_prefix_len = max(len(a) for a in alts)
            if 0 <= generated_len < max_prefix_len:
                base = compatible if compatible else alts
                next_ids = {int(a[generated_len]) for a in base if generated_len < len(a)}
                if len(next_ids) == 0:
                    return list(range(len(tkn.get_vocab())))
                return list(next_ids)
            return list(range(len(tkn.get_vocab())))
        return _prefix_allowed_tokens_fn
    except Exception:
        return None


def build_stage_a_context_lines(
    policy: DetectionModel,
    processor: Qwen2VLProcessor,
    images: List[Image.Image],
    gen_cfg: GenerationConfig,
    logits_processors: LogitsProcessorList,
    stopping: Optional[StoppingCriteriaList],
    sanitize: bool,
    mission: Optional[str] = None,
    conv_builder: Optional[object] = None,
) -> List[str]:
    from src_post.prompting.conversation import GroupQCConversationBuilder
    lines: List[str] = []
    conv = conv_builder if conv_builder is not None else GroupQCConversationBuilder(processor=processor)
    for img in images:
        img_proc = sft_style_preprocess_image(img)
        messages = conv.build_stage_a_messages(mission)
        # Enforce typed message carries exactly one image
        GroupQCConversationBuilder.validate_typed_image_count(messages, 1)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[text], images=[img_proc], padding=True, return_tensors="pt")
        # Internal check: ensure prompt has image tokens and grid count matches images
        try:
            decoded_prompt = processor.tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)
            num_image_tokens = int(decoded_prompt.count(IMAGE_PAD))
            num_grids = int(enc["image_grid_thw"].size(0)) if "image_grid_thw" in enc else 0
            if num_image_tokens == 0:
                logger.warning("[stage-a] No <|image_pad|> tokens found in decoded prompt (expected >0)")
            if num_grids != 1:
                logger.warning(f"[stage-a] image_grid_thw count mismatch: got {num_grids}, expected 1")
            else:
                logger.debug(f"[stage-a] image_pad_tokens={num_image_tokens}, image_grids={num_grids}")
        except Exception:
            # Non-fatal logging only
            logger.debug("[stage-a] Prompt/image grid validation skipped due to decoding error")
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
            # Keep digits to preserve '×N' counting; strip only Latin letters
            text_s = _re.sub(r"[A-Za-z]+", "", text_s)
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
