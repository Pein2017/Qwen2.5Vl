from __future__ import annotations

import types
import unittest

import torch

from src_new.data.dataset import Dataset
from src_new.processing.special_tokens import IM_END, IM_START, IMAGE_PAD


class _FakeTokenizer:
    def __init__(self, vocab_map: dict[str, int]):
        self._vocab = dict(vocab_map)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, -1)


def _build_full_text_and_offsets(tokens: list[str]):
    parts = []
    offsets = []
    pos = 0
    for t in tokens:
        parts.append(t)
        start = pos
        pos += len(t)
        offsets.append((start, pos))
    return "".join(parts), torch.tensor(offsets, dtype=torch.long)


class TestDatasetTeacherStudentSpansSimple(unittest.TestCase):
    def _make_dummy_dataset(self):
        ds = object.__new__(Dataset)
        ds.config = types.SimpleNamespace(span_include_im_end_in_labels=True)
        return ds

    def test_teacher_student_spans(self):
        TOK_ASSIST = IM_START + "assistant\n"
        tokens = [
            # Teacher turn
            IM_START, "user ", IMAGE_PAD, IM_END,
            TOK_ASSIST, "T", IM_END,
            # Student turn
            IM_START, "user ", IMAGE_PAD, IM_END,
            TOK_ASSIST, "S", IM_END,
        ]
        vocab = {IM_START: 1, IM_END: 2, IMAGE_PAD: 3, TOK_ASSIST: 4, "user ": 5, "T": 6, "S": 7}
        tok = _FakeTokenizer(vocab)
        text, offsets = _build_full_text_and_offsets(tokens)
        input_ids = torch.tensor([vocab[t] for t in tokens], dtype=torch.long)

        ds = self._make_dummy_dataset()
        labels, t_spans, s_spans = ds._create_masked_labels_with_spans(
            input_ids=input_ids,
            tokenizer=tok,
            has_teachers=True,
            conversation_text=text,
            offset_mapping=offsets,
            num_teachers=1,
        )

        # Expect one teacher and one student span
        self.assertEqual(len(t_spans), 1)
        self.assertEqual(len(s_spans), 1)
        t_st, t_ed = t_spans[0]
        s_st, s_ed = s_spans[0]
        self.assertEqual(t_st, tokens.index("T"))
        self.assertEqual(s_st, tokens.index("S"))
        # Edges include their respective <|im_end|>
        self.assertGreater(t_ed, t_st)
        self.assertGreater(s_ed, s_st)

        # Image pads are masked
        for pos in [tokens.index(IMAGE_PAD), tokens.index(IMAGE_PAD, tokens.index(IMAGE_PAD) + 1)]:
            self.assertEqual(int(labels[pos].item()), -100)

        # Assistant tokens are unmasked in both spans
        self.assertEqual(int(labels[tokens.index("T")].item()), vocab["T"])
        self.assertEqual(int(labels[tokens.index("S")].item()), vocab["S"])


if __name__ == "__main__":
    unittest.main()
