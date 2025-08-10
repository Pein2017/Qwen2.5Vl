import pytest
import torch
from transformers import AutoTokenizer, Qwen2VLImageProcessor

from src_new.data.collator import create_data_collator
from src_new.data.dataset import Dataset


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("collator_type", ["packed"])
def test_packed_collator_multiround_with_fa2_masks_and_batching(
    real_test_config, batch_size, collator_type
):
    # Ensure FA2 is requested in config like bbu_v2_use_coord.yaml
    real_test_config.attn_implementation = "flash_attention_2"
    real_test_config.per_device_train_batch_size = batch_size
    real_test_config.collator_type = collator_type
    real_test_config.teacher_ratio = 1.0  # force teacher-student multi-round

    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(
        real_test_config.model_path, trust_remote_code=True, use_fast=True
    )
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        real_test_config.model_path, trust_remote_code=True
    )

    # Teacher pool manager to enable multi-round
    from src_new.data.teacher_pool import TeacherPoolManager

    teacher_pool_manager = TeacherPoolManager(
        teacher_pool_file=real_test_config.teacher_pool_file
    )

    # Build dataset
    dataset = Dataset(
        data_path=real_test_config.train_data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        teacher_pool_manager=teacher_pool_manager,
        config=real_test_config,
    )

    # Provide HF processor to dataset with chat_template
    from transformers import Qwen2VLProcessor

    processor_from_pretrained = Qwen2VLProcessor.from_pretrained(
        real_test_config.model_path, trust_remote_code=True
    )
    processor = Qwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=processor_from_pretrained.chat_template,
    )
    dataset.set_processor(processor)

    # Collect batch_size samples
    features = [dataset[i] for i in range(batch_size)]

    # Create packed collator and collate
    collator = create_data_collator(
        collator_type=real_test_config.collator_type,
        tokenizer=tokenizer,
        config=real_test_config,
    )
    batch = collator(features)

    # Assertions for FA2 compatibility and multi-round labels
    assert "input_ids" in batch and "attention_mask" in batch and "labels" in batch
    assert batch["input_ids"].ndim == 2 and batch["labels"].ndim == 2

    # attention_mask must be bool and contiguous
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["attention_mask"].is_contiguous()

    # Right padding check: last positions for each sample with attention_mask False should have labels -100
    B, S = batch["input_ids"].shape
    assert B == batch_size
    for i in range(B):
        mask = batch["attention_mask"][i]
        # positions where mask is False should be padding
        if (~mask).any():
            pad_idx = (~mask).nonzero(as_tuple=True)[0]
            assert torch.all(batch["labels"][i, pad_idx] == -100)

    # Ensure there is at least one unmasked span per sample (assistant content)
    for i in range(B):
        assert (batch["labels"][i] != -100).any()

    # If images exist in features, ensure shapes are batched consistently
    if "pixel_values" in features[0]:
        assert "pixel_values" in batch
        # Qwen2.5-VL may concatenate patches; only check tensor existence and dims >= 2
        assert isinstance(batch["pixel_values"], torch.Tensor)

    # image_grid_thw should be present when provided and stacked per image
    if "image_grid_thw" in features[0]:
        assert "image_grid_thw" in batch
        assert (
            batch["image_grid_thw"].ndim == 2 and batch["image_grid_thw"].shape[-1] == 3
        )

    # Sanity: labels dtype is long, -100 padding present
    assert batch["labels"].dtype == torch.long
    assert (batch["labels"] == -100).any()
