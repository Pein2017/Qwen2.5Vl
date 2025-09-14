from typing import List, Optional, TypedDict

import torch


class MultimodalBatch(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    teacher_assistant_spans: Optional[List[List[int]]]
    student_assistant_spans: Optional[List[List[int]]]


__all__ = ["MultimodalBatch"]
