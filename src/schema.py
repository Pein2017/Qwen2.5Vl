from __future__ import annotations

"""schema.py – Centralised tensor & structure definitions

This module formalises **all** intermediate data structures used in the
Qwen-BBU dense-object captioning pipeline.  It provides:

1.   `@dataclass` wrappers with explicit **PyTorch tensor ranks** and    
     descriptive docstrings.
2.   Light-weight `__post_init__` runtime assertions (fail-fast) to catch
     shape regressions early during development.  These checks add
     negligible overhead and are skipped by TorchScript at export time.
3.   Convenience shim functions (`assert_chat_processor_output`,
     `assert_collated_batch`, …) that can be invoked inside training /
     evaluation loops without importing the entire set of dataclasses.

Notation (used in docstrings):
    B  – batch size
    S  – text sequence length (after vision-token expansion)
    I  – number of images in the sample / batch
    N  – number of object queries
    L  – maximum caption length
    D  – hidden size of the base LLM
    C  – image channels (always 3 – RGB)
    H,W – spatial resolution of pre-scaled JPEG (pixels)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple, TypeVar

import torch
from torchtyping import TensorType, patch_typeguard  # type: ignore
from typeguard import typechecked

# Enable automatic runtime checks globally (no-op if called multiple times)
patch_typeguard()

Tensor = torch.Tensor


if TYPE_CHECKING:  # pragma: no cover – executed by type checkers only
    B = TypeVar("B")
    S = TypeVar("S")
    IMG = TypeVar("IMG")
    H = TypeVar("H")
    W = TypeVar("W")
    D = TypeVar("D")
    N = TypeVar("N")
    L = TypeVar("L")
    V = TypeVar("V")
    SV = TypeVar("SV")
    C = TypeVar("C")
    IMG_COUNT = TypeVar("IMG_COUNT")  # number of flattened images (older alias `I`)
    B_IMAGES = TypeVar("B_IMAGES")

# ---------------------------------------------------------------------------
# 🔢 TensorType aliases for explicit shape validation
# ---------------------------------------------------------------------------

# B = batch size, S = sequence length, D = hidden dimension, SV = vision patch count
LLMTokenType = TensorType["B", "S", "D"]
AttentionMaskType = TensorType["B", "S"]
VisionFeatType = TensorType["B", "SV", "D"]  # Raw vision features per batch

# NEW: flattened vision features directly from Qwen2.5-VL visual encoder (no batch dim)
FlattenVisionFeatType = TensorType["SV", "D"]

# Example for object query features: B × N_queries × D
ObjectQueriesType = TensorType["B", "N", "D"]

# Caption logits: B × N_queries × L_caption × Vocab
CaptionLogitsType = TensorType["B", "N", "L", "V"]

# Scalar tensor (0-D) – used for indiv. loss values like lm_loss or bbox_loss
LossScalarType = TensorType[()]

# Tuple of per-layer hidden states (each B × S × D)
HiddenStatesTupleType = Tuple[LLMTokenType, ...]

# Dictionary mapping loss component names → scalar tensors
LossDictType = Dict[str, torch.Tensor]

# ---------------------------------------------------------------------------
# 🆕 Common low-level Tensor aliases
# ---------------------------------------------------------------------------
# 2-D bounding box \[x1, y1, x2, y2] in **absolute pixel** coordinates
BBox2DType = TensorType[4]

# ---------------------------------------------------------------------------
# 📜 Conversation & sample-level structures
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:  # noqa: D401 – basic conversation unit
    """Single message inside a multi-turn conversation.

    Attributes
    ----------
    role
        Must be one of {"system", "user", "assistant"}.
    content
        Raw text **after** special-token replacement (e.g. vision tokens).
    """

    role: str
    content: str

    def __post_init__(self) -> None:  # noqa: D401
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"ChatMessage.role must be system|user|assistant, got {self.role}"
            )
        if not isinstance(self.content, str):
            raise TypeError("ChatMessage.content must be str")


@dataclass
class ImageSample:  # noqa: D401 – one image + objects pair (teacher OR student)
    """Atomic sample consisting of **at least one image** and its object list."""

    images: list[str]
    objects: list["GroundTruthObject"]

    def __post_init__(self) -> None:  # noqa: D401
        if len(self.images) == 0:
            raise AssertionError("ImageSample must contain ≥1 image path")
        for obj in self.objects:
            if not isinstance(obj, GroundTruthObject):
                raise TypeError(
                    "ImageSample.objects items must be GroundTruthObject instances"
                )


@dataclass
class MultiChatSample:  # noqa: D401 – full teacher/student bundle
    """Input structure expected by :py:meth:`ChatProcessor.process_sample`.

    It mirrors the *teacher / student* JSON layout produced by our data-
    preparation pipeline and consumed throughout the code-base.
    """

    teachers: list[ImageSample]
    student: ImageSample

    def __post_init__(self) -> None:  # noqa: D401
        if any(not isinstance(t, ImageSample) for t in self.teachers):
            raise TypeError("All teachers must be ImageSample instances")
        if not isinstance(self.student, ImageSample):
            raise TypeError("student must be an ImageSample instance")


# ---------------------------------------------------------------------------
# Helper to assert tensor shapes via typeguard
# ---------------------------------------------------------------------------


def assert_tensor_shape(fn):
    """Decorator to apply runtime shape checks from type annotations"""
    return typechecked(fn)


# ---------------------------------------------------------------------------
# Base structures
# ---------------------------------------------------------------------------


@dataclass
class ChatProcessorOutput:
    """Output of ChatProcessor.process_sample (single sample)."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    ground_truth_objects: list["GroundTruthObject"]
    position_ids: Optional[torch.Tensor] = None  # (3,1,S) optional from ChatProcessor

    def __post_init__(self):
        if (
            self.input_ids.shape != self.labels.shape
            or self.input_ids.shape != self.attention_mask.shape
        ):
            raise AssertionError(
                f"ChatProcessorOutput: mismatched text shapes {self.input_ids.shape}, {self.labels.shape}, {self.attention_mask.shape}"
            )
        # Validate optional position_ids (should be 3×1×S or 3×S)
        if self.position_ids is not None:
            if self.position_ids.ndim == 3:
                if (
                    self.position_ids.shape[0] != 3
                    or self.position_ids.shape[2] != self.input_ids.shape[-1]
                ):
                    raise AssertionError(
                        f"ChatProcessorOutput: position_ids must be (3,1,S) got {self.position_ids.shape} with S={self.input_ids.shape[-1]}"
                    )
            elif self.position_ids.ndim == 2:
                if (
                    self.position_ids.shape[0] != 3
                    or self.position_ids.shape[1] != self.input_ids.shape[-1]
                ):
                    raise AssertionError(
                        f"ChatProcessorOutput: position_ids must be (3,S) got {self.position_ids.shape} with S={self.input_ids.shape[-1]}"
                    )
            else:
                raise AssertionError(
                    f"ChatProcessorOutput: position_ids must have 2 or 3 dims, got {self.position_ids.ndim}"
                )
        if self.pixel_values is not None:
            if self.image_grid_thw is None:
                raise AssertionError(
                    "ChatProcessorOutput: pixel_values present but image_grid_thw is None"
                )
            # Qwen2-VL image processor flattens *all* spatial-temporal patches
            # so the first dimension of `pixel_values` equals the product
            # ``t * h * w`` for **each** image.  Therefore the correct sanity
            # check is:  \sum_i t_i*h_i*w_i == pixel_values.shape[0].  We keep
            # this strict equality to fail fast while allowing arbitrary per-
            # image patch counts (which was previously flagged as an error).

            # Ensure shape compatibility
            if self.image_grid_thw.ndim != 2 or self.image_grid_thw.shape[1] != 3:
                raise AssertionError(
                    f"ChatProcessorOutput: image_grid_thw must have shape (N,3), got {self.image_grid_thw.shape}"
                )

            tokens_expected = int(
                (
                    self.image_grid_thw[:, 0]
                    * self.image_grid_thw[:, 1]
                    * self.image_grid_thw[:, 2]
                )
                .sum()
                .item()
            )

            if self.pixel_values.shape[0] != tokens_expected:
                raise AssertionError(
                    f"ChatProcessorOutput: pixel_values first dimension ({self.pixel_values.shape[0]}) does not match the total number of flattened patches ({tokens_expected}) derived from image_grid_thw"
                )


@dataclass
class CollatedBatch:
    """Output of StandardDataCollator.__call__."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    image_counts_per_sample: list[int]
    ground_truth_objects: list[list["GroundTruthObject"]]
    position_ids: Optional[torch.Tensor] = None

    def __post_init__(self):
        B, S = self.input_ids.shape
        if self.labels.shape != (B, S) or self.attention_mask.shape != (B, S):
            raise AssertionError(
                f"CollatedBatch: mismatched shapes input_ids {self.input_ids.shape}, labels {self.labels.shape}, attention_mask {self.attention_mask.shape}"
            )
        if self.position_ids is not None:
            # Accept either (B,S) or (3,B,S)
            ok = False
            if self.position_ids.shape == (B, S):
                ok = True
            elif self.position_ids.shape == (3, B, S):
                ok = True
            if not ok:
                raise AssertionError(
                    f"CollatedBatch: position_ids shape {self.position_ids.shape} must be (B,S) or (3,B,S) with B={B}, S={S}"
                )
        total_images = sum(self.image_counts_per_sample)
        if self.pixel_values is not None:
            if self.image_grid_thw is None:
                raise AssertionError(
                    "CollatedBatch: pixel_values present but image_grid_thw is None"
                )
            if self.image_grid_thw.ndim != 2 or self.image_grid_thw.shape[1] != 3:
                raise AssertionError(
                    f"CollatedBatch: image_grid_thw must have shape (N,3), got {self.image_grid_thw.shape}"
                )

            # Compute expected number of flattened patches across **all** images
            tokens_expected = int(
                (
                    self.image_grid_thw[:, 0]
                    * self.image_grid_thw[:, 1]
                    * self.image_grid_thw[:, 2]
                )
                .sum()
                .item()
            )

            if self.pixel_values.shape[0] != tokens_expected:
                raise AssertionError(
                    f"CollatedBatch: pixel_values first dimension ({self.pixel_values.shape[0]}) does not match total flattened patches ({tokens_expected}) from image_grid_thw"
                )
        else:
            if total_images != 0:
                raise AssertionError(
                    f"CollatedBatch: no pixel_values but image_counts_per_sample sum is {total_images}"
                )
        if len(self.ground_truth_objects) != B:
            raise AssertionError(
                f"CollatedBatch: ground_truth_objects length {len(self.ground_truth_objects)} must equal batch size {B}"
            )


@dataclass
class LLMHiddenStates:
    """Final hidden states from Qwen model."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor

    def __post_init__(self):
        if self.hidden_states.shape[:2] != self.attention_mask.shape:
            raise AssertionError(
                f"LLMHiddenStates: hidden_states batch and sequence dims {self.hidden_states.shape[:2]} must equal attention_mask shape {self.attention_mask.shape}"
            )


@dataclass
class DetectionHeadOutputs:
    """Output of DetectionHead.forward."""

    pred_boxes: torch.Tensor
    pred_boxes_raw: torch.Tensor
    pred_objectness: torch.Tensor
    caption_logits: torch.Tensor
    object_features: torch.Tensor

    def __post_init__(self):
        B, N = self.pred_boxes.shape[:2]
        if self.pred_boxes_raw.shape[:2] != (B, N):
            raise AssertionError(
                f"DetectionHeadOutputs: pred_boxes_raw first dims {self.pred_boxes_raw.shape[:2]} must match pred_boxes {(B, N)}"
            )
        if self.pred_objectness.shape != (B, N):
            raise AssertionError(
                f"DetectionHeadOutputs: pred_objectness shape {self.pred_objectness.shape} must be {(B, N)}"
            )
        if self.caption_logits.shape[:2] != (B, N):
            raise AssertionError(
                f"DetectionHeadOutputs: caption_logits first dims {self.caption_logits.shape[:2]} must be {(B, N)}"
            )
        if self.object_features.shape[:2] != (B, N):
            raise AssertionError(
                f"DetectionHeadOutputs: object_features first dims {self.object_features.shape[:2]} must be {(B, N)}"
            )


@dataclass
class VisionFeatures:
    """Vision features returned by Qwen2.5-VL visual tower.

    Accepts either
      • (SV, D)  – flattened sequence for ALL images across the *whole* batch,
      • (B, SV, D) – pre-expanded with explicit batch dimension.

    Down-stream components can check `.ndim` to differentiate the cases.
    """

    vision_feats: torch.Tensor

    def __post_init__(self) -> None:  # noqa: D401
        if self.vision_feats.ndim not in {2, 3}:
            raise AssertionError(
                f"VisionFeatures: vision_feats must be 2-D (SV, D) or 3-D (B, SV, D); got {self.vision_feats.shape}"
            )


# ---------------------------------------------------------------------------
# Shorthand validation helpers
# ---------------------------------------------------------------------------


def assert_chat_processor_output(sample):
    ChatProcessorOutput(**sample)


def assert_collated_batch(batch):
    CollatedBatch(**batch)


def assert_detection_head_outputs(outputs):
    DetectionHeadOutputs(
        pred_boxes=outputs["pred_boxes"],
        pred_boxes_raw=outputs.get("pred_boxes_raw", outputs["pred_boxes"]),
        pred_objectness=outputs["pred_objectness"],
        caption_logits=outputs["caption_logits"],
        object_features=outputs["object_features"],
    )


def assert_llm_hidden_states(hidden_states, attention_mask):
    LLMHiddenStates(hidden_states=hidden_states, attention_mask=attention_mask)


def assert_vision_features(vision_feats):
    """Assert correct shape for vision features.
    Permits both flattened (SV, D) and batched (B, SV, D) tensors.
    """
    VisionFeatures(vision_feats=vision_feats)


# ---------------------------------------------------------------------------
# Top-level asset registry (config, tokenizer, processors)
# ---------------------------------------------------------------------------


if TYPE_CHECKING:  # Only imported for static type checkers / IDEs
    from transformers import PretrainedConfig as _HFPretrainedConfig
    from transformers import PreTrainedTokenizerBase as _HFPreTrainedTokenizerBase

    # Config class (exists in the 2_5 namespace)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor as HFQwen25VLImageProcessor,
    )


@dataclass
class ModelAssets:  # noqa: D401 – composite holder
    """Convenience container bundling all core HF components.

    Useful for IDE navigation (*jump-to-definition*) and for explicit type
    signatures inside the training / inference helpers.
    """

    config: "_HFPretrainedConfig"
    tokenizer: "_HFPreTrainedTokenizerBase"
    image_processor: "HFQwen25VLImageProcessor"
    model: torch.nn.Module

    def __post_init__(self) -> None:  # noqa: D401
        assert hasattr(self.config, "model_type"), "config appears invalid."
        # Minimal attribute sanity checks
        for attr in [
            (self.tokenizer, "pad_token_id"),
            (self.image_processor, "size"),
        ]:
            obj, name = attr
            if not hasattr(obj, name):
                raise AttributeError(f"{obj.__class__.__name__} missing '{name}'")


# ---------------------------------------------------------------------------
# 🔢 Additional TensorType aliases for key data structures
# ---------------------------------------------------------------------------
# Pixel values: I = total number of images (flattened across batch), C = channels, H/W = spatial dims
PixelValuesType = TensorType["I", "C", "H", "W"]
# Image grid info: I images × (token_count, height, width)
ImageGridThwType = TensorType["I", 3]
# Input token sequences: B batch size, S sequence length
InputIdsType = TensorType["B", "S"]
LabelsType = TensorType["B", "S"]
# Model predictions: N = num_queries, L = caption length, V = vocab size
PredBoxesType = TensorType["B", "N", 4]
PredBoxesRawType = TensorType["B", "N", 4]
PredObjectnessType = TensorType["B", "N"]
ObjectFeaturesType = TensorType["B", "N", "D"]

# Collated batch pixel and grid shapes
CollatedPixelType = PixelValuesType
CollatedGridThwType = TensorType["B_IMAGES", 3]  # B_IMAGES = total images across batch


# ---------------------------------------------------------------------------
# 🔢 Model I/O TensorType definitions
# ---------------------------------------------------------------------------
@dataclass
class ModelInputs:
    """Structure for input dict into Qwen2.5-VL forward pass."""

    input_ids: InputIdsType
    attention_mask: AttentionMaskType
    position_ids: Optional[InputIdsType] = None
    pixel_values: Optional[PixelValuesType] = None
    image_grid_thw: Optional[ImageGridThwType] = None
    labels: Optional[LabelsType] = None
    # Additional keys (after collation)
    image_counts_per_sample: Optional[list] = None
    ground_truth_objects: Optional[list] = None

    def __post_init__(self):
        B, S = self.input_ids.shape
        if self.attention_mask.shape != (B, S):
            raise AssertionError(
                f"ModelInputs: attention_mask shape {self.attention_mask.shape} must be {(B, S)}"
            )
        if self.labels is not None and self.labels.shape != (B, S):
            raise AssertionError(
                f"ModelInputs: labels shape {self.labels.shape} must be {(B, S)}"
            )
        # Delegated pixel and grid validation
        if self.pixel_values is not None or self.image_grid_thw is not None:
            if self.pixel_values is None or self.image_grid_thw is None:
                raise AssertionError(
                    "ModelInputs: pixel_values and image_grid_thw must be both present or both None"
                )
            assert_collated_batch(
                CollatedBatch(
                    input_ids=self.input_ids,
                    labels=self.labels if self.labels is not None else self.input_ids,
                    attention_mask=self.attention_mask,
                    pixel_values=self.pixel_values,
                    image_grid_thw=self.image_grid_thw,
                    image_counts_per_sample=self.image_counts_per_sample or [],
                    ground_truth_objects=self.ground_truth_objects or [],
                )
            )


@dataclass
class ModelOutput:
    """Structure of Qwen2.5-VL forward output (before detection head)."""

    logits: TensorType["B", "S", "V"]
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    loss: Optional[torch.Tensor] = None  # Scalar
    rope_deltas: Optional[torch.Tensor] = None

    def __post_init__(self):
        B, S, V = self.logits.shape
        if self.hidden_states is not None:
            for h in self.hidden_states:
                if h.shape[:2] != (B, S):
                    raise AssertionError(
                        f"ModelOutput: hidden_states each must have shape (B,S,*) got {h.shape}"
                    )
        if self.attentions is not None:
            for att in self.attentions:
                if att.ndim != 4 or att.shape[0] != B or att.shape[-1] != S:
                    raise AssertionError(
                        f"ModelOutput: attentions must be (B,h,S,S), got {att.shape}"
                    )


def assert_model_inputs(inputs: ModelInputs):
    """Assert correctness of a model input bundle."""
    ModelInputs(**vars(inputs))


def assert_model_output(output: ModelOutput):
    """Assert correctness of a model forward output."""
    ModelOutput(
        logits=output.logits,
        hidden_states=output.hidden_states,
        attentions=output.attentions,
        loss=output.loss,
        rope_deltas=output.rope_deltas,
    )


# ---------------------------------------------------------------------------
# 🗄️ Ground-Truth object structure
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthObject:  # noqa: D401 – simple container
    """Single object annotation used throughout the dense-caption pipeline.

    Attributes
    ----------
    bbox_2d
        Tensor containing \[x1, y1, x2, y2] in **absolute pixel** coordinates.
    desc
        Natural-language description / caption of the object.
    """

    bbox_2d: torch.Tensor  # TensorType[4]
    desc: str

    def __post_init__(self) -> None:  # noqa: D401
        if isinstance(self.bbox_2d, (list, tuple)):
            self.bbox_2d = torch.tensor(self.bbox_2d, dtype=torch.float32)
        if self.bbox_2d.shape[-1] != 4:
            raise AssertionError(
                f"GroundTruthObject.bbox_2d must have 4 elements, got {self.bbox_2d.shape}"
            )

    # ------------------------------------------------------------------
    # Compatibility shim – allow legacy dict-style access so that existing
    # code using obj["bbox_2d"] or obj["desc"] keeps working until fully
    # migrated to attribute access.
    # ------------------------------------------------------------------
    def __getitem__(self, key: str):
        if key == "bbox_2d":
            return self.bbox_2d
        if key == "desc":
            return self.desc
        raise KeyError(key)


# Nested alias helpers
GroundTruthObjectsPerSample = list[GroundTruthObject]
GroundTruthBatchType = list[GroundTruthObjectsPerSample]


def ensure_batched_vision_feats(
    vision_feats: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Ensure vision features have an explicit batch dimension (B, SV, D).

    The Qwen2.5-VL vision tower returns either:
      • (SV, D)  for single-sample calls, or
      • (B, SV, D) when called with `batch=True`.

    This helper adds the missing batch dim and repeats if necessary so that
    down-stream modules can rely on a consistent 3-D layout.
    """
    if vision_feats.ndim == 2:
        vision_feats = vision_feats.unsqueeze(0)  # (1, SV, D)
    if vision_feats.size(0) != batch_size:
        vision_feats = vision_feats.expand(batch_size, -1, -1)
    VisionFeatures(vision_feats=vision_feats)  # Validation
    return vision_feats


# ---------------------------------------------------------------------------
# 🔮 Model prediction & loss component structures
# ---------------------------------------------------------------------------


@dataclass
class DetectionPredictions:  # noqa: D401
    """Output bundle from `DetectionHead.forward`. All tensors validated.

    Shapes follow the aliases in this module.
    """

    pred_boxes: TensorType["B", "N", 4]
    pred_boxes_raw: TensorType["B", "N", 4]
    pred_objectness: TensorType["B", "N"]
    caption_logits: CaptionLogitsType
    object_features: ObjectQueriesType

    def __post_init__(self) -> None:  # noqa: D401
        B, N = self.pred_boxes.shape[:2]
        if self.pred_boxes_raw.shape[:2] != (B, N):
            raise AssertionError("pred_boxes_raw dims mismatch pred_boxes")
        if self.pred_objectness.shape != (B, N):
            raise AssertionError("pred_objectness shape mismatch (B,N)")
        if self.caption_logits.shape[:2] != (B, N):
            raise AssertionError("caption_logits first dims mismatch (B,N)")
        if self.object_features.shape[:2] != (B, N):
            raise AssertionError("object_features first dims mismatch (B,N)")

    # Back-compat for modules still expecting a dict --------------------
    def as_dict(self) -> dict:
        return {
            "pred_boxes": self.pred_boxes,
            "pred_boxes_raw": self.pred_boxes_raw,
            "pred_objectness": self.pred_objectness,
            "caption_logits": self.caption_logits,
            "object_features": self.object_features,
        }


# Loss components returned by DetectionLoss ---------------------------------


@dataclass
class DetectionLossComponents:  # noqa: D401
    total_loss: LossScalarType
    bbox_loss: LossScalarType
    caption_loss: LossScalarType
    objectness_loss: LossScalarType

    def as_dict(self) -> dict:
        return {
            "total_loss": self.total_loss,
            "bbox_loss": self.bbox_loss,
            "caption_loss": self.caption_loss,
            "objectness_loss": self.objectness_loss,
        }
