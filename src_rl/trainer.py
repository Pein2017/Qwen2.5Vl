#!/usr/bin/env python3
"""
VisionGRPOTrainer adapter (TRL) for multimodal GRPO on Qwen2.5-VL.

Responsibilities:
- Forward text + vision tensors (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`) to model.generate
- Preserve vision tensors when concatenating queries and responses
- Provide strict input validation and actionable errors

Note: This module does not start training; the RL loop wiring will import
this class and call its helper methods.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from accelerate.utils import gather_object

from trl import GRPOTrainer
from trl.data_utils import maybe_apply_chat_template, is_conversational
from trl.trainer.grpo_trainer import nanmax, nanmin, nanstd

from src_new.processing.special_tokens import IMAGE_PAD


_LOGGER = logging.getLogger("src_rl.trainer")


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor

class VisionGRPOTrainer(GRPOTrainer): 
    """
    Minimal GRPO adapter that is aware of vision tensors.

    This class focuses on the generation plumbing. The reward wiring and full
    training loop are implemented elsewhere and will call these helpers.
    """

    def __init__(self, *args, **kwargs) -> None:  
        super().__init__(*args, **kwargs)
        self._eos_token_id: Optional[int] = None
        try:
            tok = getattr(self, "tokenizer", None)
            if tok is not None:
                self._eos_token_id = int(getattr(tok, "eos_token_id", -1))
        except Exception:
            self._eos_token_id = None

    # -------- Generation helpers (multimodal) --------
    def prepare_generate_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize inputs for model.generate and move tensors to model device.
        Required keys: input_ids, attention_mask
        Optional keys: pixel_values, image_grid_thw
        """
        if not isinstance(batch, dict):
            raise ValueError("batch must be a dict of tensors")
        if "input_ids" not in batch or "attention_mask" not in batch:
            raise ValueError("batch missing required keys: 'input_ids' and 'attention_mask'")

        device = next(self.model.parameters()).device  # type: ignore[attr-defined]
        allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        gen_inputs: Dict[str, Any] = {}
        for k, v in batch.items():
            if k in allowed and torch.is_tensor(v):
                gen_inputs[k] = _to_device(v, device)
        # Enforce 2D text tensors
        for key in ("input_ids", "attention_mask"):
            if key in gen_inputs:
                t = gen_inputs[key]
                if t.dim() == 1:
                    gen_inputs[key] = t.unsqueeze(0)
                elif t.dim() != 2:
                    raise ValueError(f"{key} must be 1D or 2D tensor; got shape={tuple(t.shape)}")
        return gen_inputs

    def generate_completions(self, batch: Dict[str, Any], **gen_kwargs) -> torch.Tensor:
        """
        Generate assistant tokens for the given batch using model.generate.
        Ensures eos=<|im_end|> is respected when available.
        Returns tensor of shape [B, S_total].
        """
        gen_inputs = self.prepare_generate_inputs(batch)
        if self._eos_token_id is not None and self._eos_token_id >= 0:
            gen_kwargs = {**gen_kwargs, "eos_token_id": self._eos_token_id}
        try:
            with torch.no_grad():
                outputs = self.accelerator.unwrap_model(self.model).generate(  # type: ignore[attr-defined]
                    **gen_inputs, **gen_kwargs
                )
        except Exception as e:
            # Provide helpful diagnostics on common multimodal mismatches
            shapes = {k: tuple(v.shape) for k, v in gen_inputs.items() if torch.is_tensor(v)}
            _LOGGER.error("Generation failed: %s | shapes=%s", e, shapes)
            raise
        if isinstance(outputs, torch.Tensor):
            return outputs
        # Some HF versions might return a ModelOutput-like object
        ids = getattr(outputs, "sequences", None)
        if isinstance(ids, torch.Tensor):
            return ids
        raise RuntimeError("Unexpected return from model.generate; expected Tensor or sequences field")

    def sample_k(self, batch: Dict[str, Any], *, k: int, max_new_tokens: int, temperature: float = 0.9, repetition_penalty: float = 1.0) -> torch.Tensor:
        """
        Convenience sampler: generate K completions by looping generate() with sampling.
        Returns a tensor stacked along batch dim [K, seq].
        """
        outputs: List[torch.Tensor] = []  # type: ignore[name-defined]
        for _ in range(int(k)):
            seq = self.generate_completions(
                batch,
                do_sample=True,
                temperature=float(temperature),
                repetition_penalty=float(repetition_penalty),
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
            )
            outputs.append(seq)
        return torch.stack(outputs, dim=0)

    # -------- Concatenation helper --------
    def concatenate_queries_and_responses(self, queries: Dict[str, Any], responses: torch.Tensor) -> Dict[str, Any]:
        """
        Concatenate query input_ids with generated responses, preserving vision tensors.
        Assumes batch size 1 for GRPO step by step (standard for VLMs).
        """
        if not isinstance(queries, dict):
            raise ValueError("queries must be a dict")
        if not torch.is_tensor(responses):
            raise ValueError("responses must be a torch.Tensor")

        out: Dict[str, Any] = {}
        for key, value in queries.items():
            if key in ("pixel_values", "image_grid_thw"):
                out[key] = value  # unchanged
        # Text concat
        q_ids = queries.get("input_ids")
        q_mask = queries.get("attention_mask")
        if not (torch.is_tensor(q_ids) and torch.is_tensor(q_mask)):
            raise ValueError("queries must contain input_ids and attention_mask tensors")
        # Normalize to 2D
        if q_ids.dim() == 1:
            q_ids = q_ids.unsqueeze(0)
        if q_mask.dim() == 1:
            q_mask = q_mask.unsqueeze(0)
        # Concatenate along sequence dimension
        cat_ids = torch.cat([q_ids, responses[:, q_ids.shape[1]:]], dim=1)
        # Rebuild mask: ones for newly generated tokens
        new_len = cat_ids.shape[1]
        old_len = q_mask.shape[1]
        if new_len < old_len:
            raise RuntimeError("New sequence length shorter than original attention mask length")
        pad = torch.ones((q_mask.shape[0], new_len - old_len), dtype=q_mask.dtype, device=q_mask.device)
        cat_mask = torch.cat([q_mask, pad], dim=1)
        out["input_ids"] = cat_ids
        out["attention_mask"] = cat_mask
        return out

    # -------- Overrides integrating vision tensors --------
    def _stack_optional(self, tensors: List[Optional[torch.Tensor]], *, device: torch.device) -> Optional[torch.Tensor]:
        valid = [t for t in tensors if t is not None]
        if not valid:
            return None
        base_shape = valid[0].shape
        for t in valid[1:]:
            if t.shape != base_shape:
                raise ValueError(f"Inconsistent tensor shapes encountered: {t.shape} vs {base_shape}")
        stacked = torch.stack(valid, dim=0)
        return stacked.to(device)

    def _debug_validate_image_alignment(self, prompt_ids: torch.Tensor, image_grid_thw: Optional[torch.Tensor]) -> None:
        """Decode prompt_ids and validate <|image_pad|> count vs image_grid_thw per sample.

        Logs warnings/errors only (debug utility). Never raises in normal flow.
        """
        try:
            if image_grid_thw is None:
                return
            tokenizer = self.processing_class
            batch_size = int(prompt_ids.size(0)) if prompt_ids.dim() == 2 else 1
            for b in range(batch_size):
                ids_row = prompt_ids[b] if batch_size > 1 else prompt_ids
                decoded_text = tokenizer.decode(ids_row, skip_special_tokens=False)
                image_token_count = decoded_text.count(IMAGE_PAD)
                if image_grid_thw.dim() == 3:  # [B, num_images, 3]
                    num_images = int(image_grid_thw[b].shape[0])
                elif image_grid_thw.dim() == 2:  # [num_images, 3] (unlikely after stacking)
                    num_images = int(image_grid_thw.shape[0])
                else:
                    num_images = 0
                if image_token_count == 0 and num_images > 0:
                    _LOGGER.error("Image token mismatch: no %s tokens but %d image grids present (sample %d)", IMAGE_PAD, num_images, b)
                elif image_token_count > 0 and num_images == 0:
                    _LOGGER.error("Image token mismatch: found %d %s tokens but no image grids (sample %d)", image_token_count, IMAGE_PAD, b)
                else:
                    if num_images > 0:
                        tokens_per_image = float(image_token_count) / float(num_images)
                        if tokens_per_image < 50.0:
                            _LOGGER.warning("Very low image tokens per image: %.1f (sample %d) — check processor/chat template/image size", tokens_per_image, b)
                        elif tokens_per_image > 2000.0:
                            _LOGGER.warning("Very high image tokens per image: %.1f (sample %d) — check processor/chat template", tokens_per_image, b)
        except Exception as e:
            _LOGGER.debug("Alignment debug check skipped due to: %s", e)

    def _generate_and_score_completions(
        self, inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:  # type: ignore[override]
        if self.use_vllm:
            raise NotImplementedError("VisionGRPOTrainer does not yet support vLLM generation")

        device = self.accelerator.device

        prompts = [example["prompt"] for example in inputs]
        # Prefer the exact conversation text built by the SFT-aligned processor
        prompts_text = []
        for example in inputs:
            ct = example.get("conversation_text")
            if isinstance(ct, str) and ct:
                prompts_text.append(ct)
            else:
                # Fallback: decode from provided input_ids
                ids = example.get("input_ids")
                if not torch.is_tensor(ids):
                    ids = torch.tensor(ids)
                if ids.dim() == 2:
                    ids_for_decode = ids[0]
                else:
                    ids_for_decode = ids
                prompts_text.append(self.processing_class.decode(ids_for_decode, skip_special_tokens=False))

        # Build prompt tensors directly from dataset-provided input_ids/attention_mask (HF-first parity)
        ids_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        for example in inputs:
            ex_ids = example.get("input_ids")
            ex_mask = example.get("attention_mask")
            if not torch.is_tensor(ex_ids):
                ex_ids = torch.tensor(ex_ids)
            if not torch.is_tensor(ex_mask):
                ex_mask = torch.tensor(ex_mask)
            if ex_ids.dim() != 1 or ex_mask.dim() != 1:
                # Normalize trivial batch dim if present
                if ex_ids.dim() == 2 and ex_ids.size(0) == 1:
                    ex_ids = ex_ids[0]
                if ex_mask.dim() == 2 and ex_mask.size(0) == 1:
                    ex_mask = ex_mask[0]
            ids_list.append(ex_ids.long())
            mask_list.append(ex_mask.long())

        # Left-pad to max length to form a batch
        max_len = max(int(t.size(0)) for t in ids_list)
        pad_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_id is None or pad_id < 0:
            pad_id = getattr(getattr(self, "model", None), "config", None)
            pad_id = getattr(pad_id, "pad_token_id", None)
        if pad_id is None or pad_id < 0:
            raise ValueError("Tokenizer must define pad_token_id for left padding")

        padded_ids: List[torch.Tensor] = []
        padded_masks: List[torch.Tensor] = []
        for ids, m in zip(ids_list, mask_list):
            pad_len = max_len - int(ids.size(0))
            if pad_len > 0:
                ids = torch.cat([torch.full((pad_len,), int(pad_id), dtype=ids.dtype), ids], dim=0)
                m = torch.cat([torch.zeros((pad_len,), dtype=m.dtype), m], dim=0)
            padded_ids.append(ids)
            padded_masks.append(m)

        prompt_ids = torch.stack(padded_ids, dim=0).to(device)
        prompt_mask = torch.stack(padded_masks, dim=0).to(device)

        # Collect vision tensors (per-sample)
        pixel_values_list: List[Optional[torch.Tensor]] = []
        image_grid_thw_list: List[Optional[torch.Tensor]] = []
        for example in inputs:
            pv = example.get("pixel_values")
            if isinstance(pv, torch.Tensor):
                tensor = pv
            elif pv is None:
                tensor = None
            else:
                tensor = torch.tensor(pv)
            if tensor is not None and tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            pixel_values_list.append(tensor)

            grid = example.get("image_grid_thw")
            if isinstance(grid, torch.Tensor):
                g_tensor = grid
            elif grid is None:
                g_tensor = None
            else:
                g_tensor = torch.tensor(grid)
            image_grid_thw_list.append(g_tensor)

        pixel_values = self._stack_optional(pixel_values_list, device=device)
        image_grid_thw = self._stack_optional(image_grid_thw_list, device=device)

        # Debug-only alignment validation before generation
        if _LOGGER.isEnabledFor(logging.DEBUG):
            self._debug_validate_image_alignment(prompt_ids, image_grid_thw)

        generate_kwargs = {
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_completion_length,
            "use_cache": True,
        }
        if self.top_k is not None:
            generate_kwargs["top_k"] = self.top_k
        if self.min_p is not None:
            generate_kwargs["min_p"] = self.min_p
        if self.args.generation_kwargs is not None:
            generate_kwargs.update(self.args.generation_kwargs)
        if self._eos_token_id is not None and self._eos_token_id >= 0:
            generate_kwargs.setdefault("eos_token_id", self._eos_token_id)

        model_inputs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
        }
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw

        with torch.no_grad():
            sequences = self.accelerator.unwrap_model(self.model).generate(  # type: ignore[attr-defined]
                **model_inputs,
                repetition_penalty=self.repetition_penalty,
                num_return_sequences=1,
                **generate_kwargs,
            )

        if not isinstance(sequences, torch.Tensor):
            sequences = sequences.sequences

        if sequences.dim() != 2:
            raise RuntimeError(f"Unexpected generated tensor shape: {tuple(sequences.shape)}")

        prompt_length = prompt_ids.size(1)
        completion_ids = sequences[:, prompt_length:]

        # Build completion mask based on EOS
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        if self.mask_truncated_completions:
            truncated = ~has_eos
            completion_mask = completion_mask * (~truncated).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if self.model.training else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    torch.cat([prompt_ids, completion_ids], dim=1),
                    attention_mask,
                    logits_to_keep,
                    pixel_values,
                    image_grid_thw,
                    batch_size,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                ref_model = self.ref_model if self.ref_model is not None else self.model
                ref_per_token_logps = self._get_per_token_logps(
                    ref_model,
                    torch.cat([prompt_ids, completion_ids], dim=1),
                    attention_mask,
                    logits_to_keep,
                    pixel_values,
                    image_grid_thw,
                )
            else:
                ref_per_token_logps = None

        completion_lengths = completion_mask.sum(1)
        completion_ids_list = [
            [token.item() for token, mask in zip(row, mask_row) if mask]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt[-1]["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if self.model.training:
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics["train" if self.model.training else "eval"]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather(has_eos)
        term_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_ratio = 1 - len(term_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_ratio)
        if len(term_lengths) == 0:
            term_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_lengths.float().max().item())

        for i, reward_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def _get_per_token_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            ids_batch = input_ids[start:end]
            mask_batch = attention_mask[start:end]
            kwargs: Dict[str, torch.Tensor] = {}
            if pixel_values is not None:
                kwargs["pixel_values"] = pixel_values[start:end]
            if image_grid_thw is not None:
                kwargs["image_grid_thw"] = image_grid_thw[start:end]
            logits = model(
                input_ids=ids_batch,
                attention_mask=mask_batch,
                logits_to_keep=logits_to_keep + 1,
                **kwargs,
            ).logits
            logits = logits[:, :-1, :]
            ids_slice = ids_batch[:, -logits_to_keep:]
            logits = logits / self.temperature
            logps = torch.log_softmax(logits, dim=-1).gather(-1, ids_slice.unsqueeze(-1)).squeeze(-1)
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _compute_loss(self, model, inputs):  # type: ignore[override]
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            pixel_values,
            image_grid_thw,
        )

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
        else:
            per_token_kl = None

        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if model.training else "eval"
        if per_token_kl is not None:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

        gathered_low_clip = self.accelerator.gather(low_clip)
        gathered_high_clip = self.accelerator.gather(high_clip)
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)

        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss


__all__ = ["VisionGRPOTrainer"]
