#!/usr/bin/env python3
"""demo_single_hardcoded_sample.py

Run a single-image QA round-trip on the fine-tuned Qwen2.5-VL checkpoint using the
hard-coded sample provided by the user.  The purpose is to visually inspect what
the model "sees" when asked a simple question.

This script is **not** an automated unit test (no assert) – it prints the model
output so we can qualitatively judge language & vision fidelity.

Usage (from project root):

    python tests/demo_single_hardcoded_sample.py --model_path /path/to/checkpoint

The checkpoint may contain LoRA / PEFT adapters; the script will attempt to load
via `peft.AutoPeftModelForCausalLM` first and fall back to the full model if no
adapter config is present.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import sys
from pathlib import Path

import src.logger_utils as _logger_utils


class _NoOpLogger:
    def debug(self, *_, **__): ...
    def info(self, *_, **__): ...
    def warning(self, *_, **__): ...
    def error(self, *_, **__): ...
    def critical(self, *_, **__): ...


# Silence logging early
_logger_utils.get_logger = lambda *_, **__: _NoOpLogger()  # type: ignore
_logger_utils.get_patches_logger = lambda *_, **__: _NoOpLogger()  # type: ignore

# Initialise config before patches import
from pathlib import Path as _Path  # noqa: E402

from src.config import init_config as _init_config  # noqa: E402


_default_cfg_path = _Path(__file__).resolve().parents[1] / "configs" / "base_flat.yaml"
_init_config(str(_default_cfg_path))

import json  # for post-processing model output

import torch
from PIL import Image
from transformers import AutoProcessor

# ----------------------------------------------------------------------------
# Patch Qwen2.5-VL first (mRoPE, vision fixes)
# ----------------------------------------------------------------------------
from src.models.patches import apply_comprehensive_qwen25_fixes


apply_comprehensive_qwen25_fixes()

# ----------------------------------------------------------------------------
# Constants & hard-coded sample (supplied by user)
# ----------------------------------------------------------------------------

SAMPLE = {
    "images": ["ds_rescaled/QC-TEMP-20250103-0015787_4438755.jpeg"],
    "objects": [
        {"bbox_2d": [6, 4, 416, 835], "desc": "机柜空间"},
        {"bbox_2d": [0, 28, 399, 510], "desc": "挡风板"},
        {"bbox_2d": [325, 249, 401, 327], "desc": "螺丝连接点/BBU安装螺丝"},
        {"bbox_2d": [43, 585, 106, 638], "desc": "螺丝连接点/BBU安装螺丝"},
    ],
    "height": 896,
    "width": 420,
}


QUESTION_CN = """
你是专业的通信机房BBU设备检测AI助手，负责精确识别和定位图像中的所有相关设备和部件。

**检测任务**:识别图像中所有目标对象的位置和类型。

**目标对象分类**:

1. **BBU设备**
   - bbu基带处理单元/华为
   - bbu基带处理单元/中兴
   - bbu基带处理单元/爱立信

2. **螺丝连接点**
   - 螺丝连接点/BBU安装螺丝
   - 螺丝连接点/CPRI光缆和BBU连接点
   - 螺丝连接点/地排处螺丝
   - 螺丝连接点/BBU接地线机柜接地端
   - 螺丝连接点/BBU尾纤和ODF连接点

3. **线缆**
   - 线缆/光纤
   - 线缆/非光纤

4. **机柜部件**
   - 机柜空间
   - 挡风板
   - 标签贴纸

**检测要求**:
1. 仔细观察图像中的每个区域，识别所有目标对象
2. 准确标注边界框，确保完全包含目标对象
3. 使用上述标准分类标签，保持格式一致性
4. 忽略图像中的水印、时间戳等叠加信息
5. 对于部分遮挡的对象，标注可见部分的边界

**输出格式**:严格按照JSON数组格式输出:
```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "标准分类标签"}
]
```

坐标说明:(x1,y1)为左上角，(x2,y2)为右下角，使用绝对像素坐标。
"""
QUESTION_EN = "What do you see?"

# ----------------------------------------------------------------------------
# Helper – adapter-aware model loading
# ----------------------------------------------------------------------------


def load_model(model_path: Path):
    """Attempt PEFT adapter load first; fallback to full model."""
    # 1) If checkpoint already contains full fused weights -> normal load
    adapter_config_file = model_path / "adapter_config.json"

    # Prefer adapter path if adapter_config.json exists
    if adapter_config_file.exists():
        try:
            from peft import PeftConfig, PeftModel  # type: ignore
        except ImportError:
            print(
                "❌ PEFT not installed – cannot load adapter. Falling back to full weight load."
            )
        else:
            peft_cfg = PeftConfig.from_pretrained(model_path)
            base_model_path = Path(peft_cfg.base_model_name_or_path or "")

            if not base_model_path.exists():
                # Assume base model was saved inside the same dir (common when using `peft.save_pretrained`) – fallback to model_path
                base_model_path = model_path

            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

            print(f"🔧 Loading base model from {base_model_path}")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(base_model_path),
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

            print(f"🔧 Loading PEFT adapter weights from {model_path}")
            model = PeftModel.from_pretrained(base_model, str(model_path))

            # Optionally merge LoRA weights for inference efficiency
            try:
                model = model.merge_and_unload()
                print("✅ LoRA adapter merged into base model for inference.")
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Could not merge adapter (continuing with PEFT model): {exc}")

            return model

    # 2) If no adapter config, attempt AutoPeftModelForCausalLM (covers cases where adapter is *inside* dir with full weights)
    try:
        from peft import AutoPeftModelForCausalLM  # type: ignore

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        print("✅ Loaded model via AutoPeftModelForCausalLM (adapter already fused).")
        return model
    except (ImportError, ValueError, OSError):
        pass  # Fall back to plain model

    # 3) Plain full-weights load
    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print("✅ Loaded full Qwen2.5-VL model (no adapter detected).")
    return model


# ----------------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------------


def main(model_path: Path):
    if not model_path.exists():
        sys.exit(f"❌ Model path does not exist: {model_path}")

    # Load processor first (slow tokenizer, vision preproc)
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=False
    )

    # Override chat_template with training template via collator
    try:
        from src.reference.qwen2_5vl_collator import Qwen2_5VLCollator

        _ = Qwen2_5VLCollator(processor)  # mutates processor.tokenizer.chat_template
        print("✅ Chat template overridden with training collator template.")
    except Exception as exc:  # noqa: BLE001
        print(
            "⚠️  Could not import training collator – proceeding with checkpoint template."
        )
        print(f"   Reason: {exc}")

    model = load_model(model_path)
    model.eval()

    # ------------------------------------------------------------------
    # Prepare prompt & vision inputs
    # ------------------------------------------------------------------
    img_path = Path(SAMPLE["images"][0])
    if not img_path.exists():
        sys.exit(f"❌ Image file not found: {img_path}")

    image = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION_CN},
            ],
        }
    ]

    prompt_str = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=[prompt_str], images=[image], return_tensors="pt")
    inputs = {
        k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()
    }

    # ------------------------------------------------------------------
    # Timed generation
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    import time  # local import to avoid adding to global imports

    start_time = time.perf_counter()

    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ),
    ):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # deterministic to preserve exact keys/labels
            repetition_penalty=1.01,  # mild repeat penalty without n-gram ban
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    prompt_tokens = inputs["input_ids"].shape[1]
    total_tokens = output_ids.shape[1]
    generated_tokens = max(total_tokens - prompt_tokens, 1)

    tokens_per_second = generated_tokens / elapsed if elapsed > 0 else float("inf")
    seconds_per_token = elapsed / generated_tokens if generated_tokens else float("inf")

    response_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    try:
        assistant_part = response_text.split("assistant\n", 1)[1].strip()
    except IndexError:
        assistant_part = response_text

    print("\n================= MODEL RESPONSE =================")
    print(assistant_part)
    print("==================================================")

    print(
        f"\n⏱️  Generation took {elapsed:.2f} s for {generated_tokens} new tokens → "
        f"{tokens_per_second:.2f} tokens/s ( {seconds_per_token:.3f} s/token )"
    )

    # ------------------------------------------------------------------
    # Deduplicate identical predictions (exact bbox + label match)
    # ------------------------------------------------------------------
    def _salvage_json(txt: str):
        """Best-effort extraction of first [...] block and JSON decode."""
        import json
        import re

        # 1) take substring between first '[' and last ']'
        start = txt.find("[")
        end = txt.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = txt[start : end + 1]

        # 2) remove any trailing commas like {..},]
        candidate = re.sub(r",\s*]$", "]", candidate)

        try:
            return json.loads(candidate)
        except Exception:
            return None

    try:
        parsed = json.loads(assistant_part)
    except Exception as exc:  # noqa: BLE001
        parsed = _salvage_json(assistant_part)
        if parsed is None:
            print(
                f"⚠️  Could not parse assistant output as JSON – skipping dedup. Reason: {exc}"
            )
    else:
        if isinstance(parsed, list):
            unique_items = []
            seen_keys = set()
            for item in parsed:
                bbox = tuple(item.get("bbox_2d", []))
                label = item.get("label") or item.get("desc")
                key = (bbox, label)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_items.append(item)

            removed = len(parsed) - len(unique_items)
            if removed > 0:
                print(f"\n🧹 Removed {removed} duplicate entries. Deduplicated result:")
                print(json.dumps(unique_items, ensure_ascii=False, indent=2))
            else:
                print("\n✅ No duplicates detected in model output.")
        else:
            print("⚠️  Parsed JSON is not a list – skipping dedup step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-image demo with hard-coded sample."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=Path,
        help="Path to fine-tuned checkpoint directory (may contain adapters)",
    )
    args = parser.parse_args()

    main(args.model_path.resolve())
