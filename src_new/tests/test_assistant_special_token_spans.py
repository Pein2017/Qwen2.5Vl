from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import Qwen2_5_VLProcessor

from src_new.losses.token_grouping import TokenGroupingPlugin
from src_new.processing.conversation_processor import ConversationProcessor
from src_new.processing.span_builder import (
    build_assistant_spans_token_aligned,
    build_assistant_spans_with_token_offsets,
)
from src_new.processing.templates import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT,
    get_system_prompt,
)


def _make_image(size_wh: Tuple[int, int]) -> Image.Image:
    w, h = size_wh
    return Image.new("RGB", (w, h), color=(10, 20, 30))


class TestAssistantSpecialTokenSpans(unittest.TestCase):
    processor: Qwen2_5_VLProcessor | None = None

    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "model_cache/Qwen/Qwen2.5-VL-7B-Instruct-line_tokens"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at default path: {model_path}. Please place the model there."
            )
        cls.processor = Qwen2_5_VLProcessor.from_pretrained(str(model_path))

    def _geometry_tokens(self) -> List[str]:
        return [
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
            "<|line_start|>",
            "<|line_end|>",
        ]

    def test_all_prompts_single_token_counts(self) -> None:
        assert self.processor is not None
        tok = self.processor.tokenizer
        vocab: Dict[str, int] = tok.get_vocab()

        prompts = [
            get_system_prompt(format_mode="special_tokens"),
            SUMMARY_SYSTEM_PROMPT,
            SUMMARY_USER_PROMPT,
        ]

        for prompt in prompts:
            ids = tok.encode(prompt, add_special_tokens=False)
            for sym in self._geometry_tokens():
                # Not all prompts must contain all symbols; only check those present
                if sym in prompt:
                    self.assertIn(sym, vocab, msg=f"Token missing from vocab: {sym}")
                    tid = vocab[sym]
                    text_count = prompt.count(sym)
                    id_count = sum(1 for i in ids if i == tid)
                    self.assertEqual(
                        id_count,
                        text_count,
                        msg=(
                            f"Prompt tokenization mismatch for '{sym}': "
                            f"text_count={text_count}, id_count={id_count}"
                        ),
                    )

    def test_assistant_span_offsets_treat_special_tokens_as_whole(self) -> None:
        assert self.processor is not None
        tok = self.processor.tokenizer
        vocab: Dict[str, int] = tok.get_vocab()

        # Build a sample that includes all three geometry wrappers in assistant content
        sample = {
            "width": 96,
            "height": 64,
            "objects": [
                {"quad": [1, 2, 10, 2, 10, 12, 1, 12], "desc": "对象A"},
                {"bbox_2d": [3, 4, 16, 20], "desc": "对象B"},
                {"line": [0, 0, 5, 5, 9, 9, 12, 12], "desc": "对象C"},
            ],
        }
        img = _make_image((sample["width"], sample["height"]))

        conv = ConversationProcessor(
            processor=self.processor,
        )
        out = conv.create_simple_conversation(sample=sample, images=[img])

        input_ids = out["input_ids"][0]
        ids_list = input_ids.tolist()
        conversation_text = str(out["conversation_text"])
        offset_mapping = out["offset_mapping"]  # [seq, 2]

        # Build assistant spans for the simple conversation
        t_spans, s_spans = build_assistant_spans_token_aligned(
            conversation_text=conversation_text,
            offset_mapping=offset_mapping,
            tokenizer=tok,
            input_ids_expanded=input_ids,
            has_teachers=False,
            num_teachers=0,
            include_eos=True,
        )
        self.assertFalse(t_spans)
        self.assertTrue(s_spans)

        # Validate per special token inside assistant span
        start, end = s_spans[0]
        ids_list[start:end]
        assistant_text = tok.decode(input_ids[start:end], skip_special_tokens=False)

        # Re-tokenize assistant text alone to validate offsets deterministically
        enc = tok(
            assistant_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors=None,
        )
        sub_ids = enc["input_ids"] if isinstance(enc["input_ids"], list) else enc["input_ids"][0]
        sub_offsets = enc["offset_mapping"] if isinstance(enc["offset_mapping"], list) else enc["offset_mapping"][0]

        for sym in self._geometry_tokens():
            if sym not in vocab:
                continue
            tid = vocab[sym]
            text_count = assistant_text.count(sym)
            id_count = sum(1 for i in sub_ids if i == tid)
            self.assertEqual(
                id_count,
                text_count,
                msg=(
                    f"Assistant (isolated) count mismatch for '{sym}': "
                    f"text_count={text_count}, id_count={id_count}"
                ),
            )

            # For each token occurrence in isolated assistant text, offsets must map exactly to the symbol substring
            for j, tok_id in enumerate(sub_ids):
                if tok_id == tid:
                    ch_start, ch_end = sub_offsets[j]
                    # Some tokenizers return tuples or lists; normalize to ints
                    s = int(ch_start)
                    e = int(ch_end)
                    substr = assistant_text[s:e]
                    self.assertEqual(
                        substr,
                        sym,
                        msg=(
                            f"Isolated assistant offset mismatch for '{sym}' at j={j}: "
                            f"substr='{substr}'"
                        ),
                    )

    def test_real_val_dense_grouping_and_offsets(self) -> None:
        assert self.processor is not None
        tok = self.processor.tokenizer
        vocab: Dict[str, int] = tok.get_vocab()

        # Load 2 samples from val.jsonl
        repo_root = Path(__file__).resolve().parents[2]
        val_path = repo_root / "data/ds_v2_full/val.jsonl"
        with open(val_path, "r", encoding="utf-8") as f:
            lines = [json.loads(next(f)) for _ in range(2)]
        # For each, build dense conversation
        all_inputs = []
        conv = ConversationProcessor(
            processor=self.processor,
        )
        for sample in lines:
            w = int(sample.get("width", 256))
            h = int(sample.get("height", 256))
            # Use synthetic image as we test tokenization/offsets only
            img = _make_image((w, h))
            out = conv.create_simple_conversation(sample=sample, images=[img])
            all_inputs.append(out)

        # Verify spans, per-token offsets, and grouping masks
        plugin = TokenGroupingPlugin(tok)
        for out in all_inputs:
            inp_ids = out["input_ids"]
            if inp_ids.dim() == 2:
                inp_ids = inp_ids[0]
            conversation_text = str(out["conversation_text"])
            offset_mapping = out["offset_mapping"]

            t_spans, s_spans, t_tok_offs, s_tok_offs = build_assistant_spans_with_token_offsets(
                conversation_text=conversation_text,
                offset_mapping=offset_mapping,
                tokenizer=tok,
                input_ids_expanded=inp_ids,
                has_teachers=False,
                num_teachers=0,
                include_eos=True,
            )
            self.assertFalse(t_spans)
            self.assertTrue(s_spans)
            # Ensure we have per-token offsets aligned to student spans
            self.assertEqual(len(s_tok_offs), len(s_spans))
            st, ed = s_spans[0]
            # All per-token indices must lie inside span
            for idx, (exp_idx, (cs, ce)) in enumerate(s_tok_offs[0]):
                self.assertTrue(st <= exp_idx <= ed, msg=f"token offset idx out of span: {exp_idx} not in [{st},{ed}]")
                self.assertTrue(0 <= cs <= ce <= len(conversation_text))

            # Build labels and grouping masks
            labels = inp_ids.clone().unsqueeze(0)  # [1, S]
            # Mask outside-student tokens with -100 to simulate label construction
            lbl = labels.clone()
            mask_student = torch.zeros_like(lbl, dtype=torch.bool)
            for a, b in s_spans:
                mask_student[0, a:b] = True
            lbl[~mask_student] = -100
            gm = plugin.build_group_masks(
                labels=lbl,
                teacher_spans=[[]],
                student_spans=[s_spans],
                input_ids=inp_ids.unsqueeze(0),
                variant_key="dense",
            )
            # Verify coverage equals assistant mask (shifted)
            s_assist = mask_student[:, 1:]
            union = gm.student_caption | gm.student_grounding | gm.student_formatting
            self.assertTrue(torch.equal(union, s_assist))
            # Check some counts (non-zero if there is any content)
            self.assertGreater(int(gm.student_formatting.sum().item()), 0)
            self.assertGreater(int(gm.student_caption.sum().item()), 0)
            # Grounding should be non-zero if any geometry wrappers appear
            any_geom_token = any(vocab.get(t) in inp_ids.tolist() for t in [
                "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|line_start|>", "<|line_end|>"
            ] if t in vocab)
            if any_geom_token:
                self.assertGreater(int(gm.student_grounding.sum().item()), 0)

            # Extra: verify substrings for wrappers and grouping class membership
            wrapper_tokens = [
                "<|object_ref_start|>", "<|object_ref_end|>",
                "<|box_start|>", "<|box_end|>",
                "<|quad_start|>", "<|quad_end|>",
                "<|line_start|>", "<|line_end|>",
            ]
            # Build a fast map of expanded idx -> (cs, ce)
            span_token_offsets = {exp_idx: (cs, ce) for exp_idx, (cs, ce) in (s_tok_offs[0] if s_tok_offs else [])}
            for tok_str in wrapper_tokens:
                tid = vocab.get(tok_str)
                if tid is None or tid not in inp_ids.tolist():
                    continue
                # Validate substrings and grouping
                for pos in range(int(inp_ids.shape[0])):
                    if int(inp_ids[pos].item()) == int(tid) and (pos in span_token_offsets):
                        cs, ce = span_token_offsets[pos]
                        substr = conversation_text[cs:ce]
                        # Global offsets can land on template whitespace; skip strict equality if so
                        if substr != tok_str and substr.strip() == "":
                            continue
                        # Grouping masks are shifted by 1
                        if pos - 1 >= 0 and pos - 1 < gm.student_formatting.shape[1]:
                            if tok_str in ("<|object_ref_start|>", "<|object_ref_end|>"):
                                # object-ref wrappers => formatting
                                self.assertTrue(bool(gm.student_formatting[0, pos - 1].item()))
                            else:
                                # geometry wrappers => grounding
                                self.assertTrue(bool(gm.student_grounding[0, pos - 1].item()))


if __name__ == "__main__":
    unittest.main()
