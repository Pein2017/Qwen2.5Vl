"""
Official RoPE 2D position encoding for Qwen2.5VL - exact copy from qwen-vl-finetune.

This module handles 3D rotary position embeddings for vision-language models,
using the exact implementation from the official qwen-vl-finetune repository.
"""

from typing import Optional, Tuple

import torch


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            # STRICT PRE-VALIDATION: Check all indices before any tensor operations
            if i >= len(attention_mask):
                raise RuntimeError(
                    f"❌ CRITICAL INDEX ERROR: Batch index {i} >= attention_mask length {len(attention_mask)}!\n"
                    f"   This will cause index out of bounds in CUDA operations!"
                )

            # Extract valid tokens with strict bounds checking
            attention_valid = attention_mask[i] == 1
            if attention_valid.sum() == 0:
                raise RuntimeError(
                    f"❌ CRITICAL DATA ERROR: No valid tokens in sequence {i}!\n"
                    f"   Attention mask: {attention_mask[i]}\n"
                    f"   This indicates completely masked sequence!"
                )

            input_ids = input_ids[attention_valid]

            # STRICT VALIDATION: Check input_ids bounds
            if input_ids.numel() == 0:
                raise RuntimeError(
                    f"❌ CRITICAL DATA ERROR: Empty input_ids after attention masking for sequence {i}!"
                )

            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)

            # STRICT VALIDATION: Fail fast if no vision tokens found when images are expected
            if vision_start_indices.numel() == 0:
                raise RuntimeError(
                    f"❌ CRITICAL DATA ERROR: No vision start tokens found in sequence but image_grid_thw provided!\n"
                    f"   Sequence {i} length: {len(input_ids)}\n"
                    f"   Input tokens: {input_ids.tolist()[:20]}{'...' if len(input_ids) > 20 else ''}\n"
                    f"   Expected vision_start_token_id: {vision_start_token_id}\n"
                    f"   This indicates a problem in the data preprocessing pipeline.\n"
                    f"   Every sample with images must contain proper vision tokens!"
                )

            # STRICT VALIDATION: Ensure vision_start_indices + 1 doesn't go out of bounds
            max_valid_idx = len(input_ids) - 1

            # Check each vision start index individually
            for idx_pos, vision_idx in enumerate(vision_start_indices):
                if vision_idx >= max_valid_idx:
                    raise RuntimeError(
                        f"❌ CRITICAL INDEX ERROR: Vision start token at position {vision_idx} >= max valid index {max_valid_idx}!\n"
                        f"   Sequence {i}, vision token {idx_pos}\n"
                        f"   This will cause index out of bounds when accessing input_ids[{vision_idx} + 1]!\n"
                        f"   Sequence length: {len(input_ids)}\n"
                        f"   Vision start indices: {vision_start_indices.tolist()}"
                    )

                # STRICT VALIDATION: Check the next token exists and is valid
                next_token_idx = vision_idx + 1
                if next_token_idx >= len(input_ids):
                    raise RuntimeError(
                        f"❌ CRITICAL INDEX ERROR: Next token index {next_token_idx} >= sequence length {len(input_ids)}!\n"
                        f"   Sequence {i}, vision token {idx_pos} at position {vision_idx}\n"
                        f"   This will cause index out of bounds in CUDA operations!"
                    )

            # Use only valid indices for vision token detection
            valid_indices = vision_start_indices[vision_start_indices < max_valid_idx]

            if valid_indices.numel() != vision_start_indices.numel():
                raise RuntimeError(
                    f"❌ CRITICAL DATA ERROR: Some vision start tokens are at sequence end!\n"
                    f"   Sequence {i}: Total vision start tokens: {vision_start_indices.numel()}\n"
                    f"   Valid vision start tokens: {valid_indices.numel()}\n"
                    f"   Invalid indices: {vision_start_indices[vision_start_indices >= max_valid_idx].tolist()}\n"
                    f"   This indicates truncated or malformed vision sequences!"
                )

            # STRICT VALIDATION: Check vision token access before doing it
            for idx_pos, vision_idx in enumerate(valid_indices):
                access_idx = vision_idx + 1
                if access_idx >= len(input_ids):
                    raise RuntimeError(
                        f"❌ CRITICAL INDEX ERROR: About to access input_ids[{access_idx}] but sequence length is {len(input_ids)}!\n"
                        f"   Sequence {i}, vision token {idx_pos}\n"
                        f"   This is the exact cause of the CUDA index out of bounds error!"
                    )

            # Now safe to access vision tokens
            vision_tokens = input_ids[valid_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    # STRICT VALIDATION: Check image_index bounds before access
                    if image_index >= len(image_grid_thw):
                        raise RuntimeError(
                            f"❌ CRITICAL INDEX ERROR: image_index {image_index} >= image_grid_thw length {len(image_grid_thw)}!\n"
                            f"   Sequence {i}, processing image token\n"
                            f"   This will cause index out of bounds when accessing image_grid_thw[{image_index}]!\n"
                            f"   Total images expected: {image_nums}, processed so far: {image_index}"
                        )

                    # STRICT VALIDATION: Check image_grid_thw structure
                    if len(image_grid_thw[image_index]) < 3:
                        raise RuntimeError(
                            f"❌ CRITICAL DATA ERROR: image_grid_thw[{image_index}] has insufficient dimensions!\n"
                            f"   Expected 3 dimensions (t, h, w), got {len(image_grid_thw[image_index])}\n"
                            f"   Value: {image_grid_thw[image_index]}\n"
                            f"   This will cause index error when accessing t, h, w components!"
                        )

                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )

                    # STRICT VALIDATION: Check spatial_merge_size division
                    if (
                        h.item() % spatial_merge_size != 0
                        or w.item() % spatial_merge_size != 0
                    ):
                        raise RuntimeError(
                            f"❌ CRITICAL DATA ERROR: Image dimensions not divisible by spatial_merge_size!\n"
                            f"   Image {image_index}: h={h.item()}, w={w.item()}\n"
                            f"   spatial_merge_size: {spatial_merge_size}\n"
                            f"   h % spatial_merge_size = {h.item() % spatial_merge_size}\n"
                            f"   w % spatial_merge_size = {w.item() % spatial_merge_size}\n"
                            f"   This will cause incorrect grid calculations!"
                        )

                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    # STRICT VALIDATION: Check video bounds
                    if video_grid_thw is None:
                        raise RuntimeError(
                            f"❌ CRITICAL DATA ERROR: video_grid_thw is None but video token found!\n"
                            f"   Sequence {i}, video_index {video_index}\n"
                            f"   This indicates video tokens without corresponding video grid data!"
                        )

                    if video_index >= len(video_grid_thw):
                        raise RuntimeError(
                            f"❌ CRITICAL INDEX ERROR: video_index {video_index} >= video_grid_thw length {len(video_grid_thw)}!\n"
                            f"   Sequence {i}, processing video token\n"
                            f"   This will cause index out of bounds when accessing video_grid_thw[{video_index}]!"
                        )

                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        if video_index >= len(second_per_grid_ts):
                            raise RuntimeError(
                                f"❌ CRITICAL INDEX ERROR: video_index {video_index} >= second_per_grid_ts length {len(second_per_grid_ts)}!\n"
                                f"   This will cause index out of bounds when accessing second_per_grid_ts[{video_index}]!"
                            )
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            # Handle case where no position IDs were generated
            if not llm_pos_ids_list:
                seq_len = len(input_tokens)
                llm_positions = torch.arange(seq_len).view(1, -1).expand(3, -1)
            else:
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            # Ensure llm_positions matches the expected sequence length
            expected_len = (attention_mask[i] == 1).sum().item()
            if llm_positions.shape[-1] != expected_len:
                # Truncate or pad to match expected length
                if llm_positions.shape[-1] > expected_len:
                    llm_positions = llm_positions[:, :expected_len]
                else:
                    # Pad with sequential values
                    pad_len = expected_len - llm_positions.shape[-1]
                    start_val = (
                        llm_positions.max() + 1 if llm_positions.numel() > 0 else 0
                    )
                    pad_positions = (
                        torch.arange(start_val, start_val + pad_len)
                        .view(1, -1)
                        .expand(3, -1)
                    )
                    llm_positions = torch.cat([llm_positions, pad_positions], dim=-1)

            # STRICT VALIDATION: Validate position_ids assignment before CUDA operation
            attention_valid_mask = attention_mask[i] == 1
            expected_len = attention_valid_mask.sum().item()

            # Check position_ids tensor bounds
            if i >= position_ids.shape[-2]:
                raise RuntimeError(
                    f"❌ CRITICAL INDEX ERROR: Batch index {i} >= position_ids batch dimension {position_ids.shape[-2]}!\n"
                    f"   Position IDs shape: {position_ids.shape}\n"
                    f"   This will cause index out of bounds in position_ids[..., {i}, :]!"
                )

            # Check attention mask bounds
            if attention_valid_mask.shape[0] != position_ids.shape[-1]:
                raise RuntimeError(
                    f"❌ CRITICAL DIMENSION ERROR: Attention mask length {attention_valid_mask.shape[0]} != position_ids sequence length {position_ids.shape[-1]}!\n"
                    f"   Position IDs shape: {position_ids.shape}\n"
                    f"   Attention mask shape: {attention_mask[i].shape}\n"
                    f"   This will cause dimension mismatch in CUDA operations!"
                )

            # Check llm_positions dimensions
            if llm_positions.shape[-1] != expected_len:
                raise RuntimeError(
                    f"❌ CRITICAL DIMENSION ERROR: llm_positions length {llm_positions.shape[-1]} != expected length {expected_len}!\n"
                    f"   llm_positions shape: {llm_positions.shape}\n"
                    f"   Expected length from attention mask: {expected_len}\n"
                    f"   Attention valid positions: {attention_valid_mask.sum().item()}\n"
                    f"   This will cause dimension mismatch in position assignment!"
                )

            # Check if any indices in attention_valid_mask are out of bounds
            valid_indices = torch.where(attention_valid_mask)[0]
            if len(valid_indices) > 0:
                max_valid_idx = valid_indices.max().item()
                if max_valid_idx >= position_ids.shape[-1]:
                    raise RuntimeError(
                        f"❌ CRITICAL INDEX ERROR: Max valid attention index {max_valid_idx} >= position_ids sequence length {position_ids.shape[-1]}!\n"
                        f"   Valid indices: {valid_indices.tolist()[:10]}{'...' if len(valid_indices) > 10 else ''}\n"
                        f"   Position IDs shape: {position_ids.shape}\n"
                        f"   This is the exact cause of the CUDA index out of bounds error!"
                    )

            # Validate the assignment dimensions match exactly
            target_slice_shape = position_ids[..., i, attention_valid_mask].shape
            source_shape = llm_positions.shape

            if target_slice_shape != source_shape:
                raise RuntimeError(
                    f"❌ CRITICAL SHAPE ERROR: Target slice shape {target_slice_shape} != source shape {source_shape}!\n"
                    f"   Target: position_ids[..., {i}, attention_mask[{i}] == 1]\n"
                    f"   Source: llm_positions\n"
                    f"   Position IDs shape: {position_ids.shape}\n"
                    f"   Attention mask sum: {attention_valid_mask.sum().item()}\n"
                    f"   This will cause assignment error in CUDA!"
                )

            # Now safe to assign
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
