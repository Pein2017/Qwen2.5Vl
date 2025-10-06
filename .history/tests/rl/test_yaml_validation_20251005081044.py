import pytest

from src_new.rl.runner import RLLoaderConfig


def _load_yaml_str(text: str):
    import yaml

    return yaml.safe_load(text)


def test_missing_attn_implementation_raises():
    cfg = _load_yaml_str(
        """
        model_path: /abs/model
        bf16: true
        model:
          image_max_pixels: 401408
        loss:
          teacher_loss_weight: 0.5
          student_loss_weight: 1.0
          caption_loss_weight: 1.0
          grounding_loss_weight: 1.0
          formatting_loss_weight: 1.0
        per_device_train_batch_size: 1
        update_steps: 1
        learning_rate: 1e-5
        weight_decay: 0.0
        max_steps: 1
        warmup_steps: 0
        logging_steps: 1
        save_steps: 1
        seed: 0
        sample_k: 1
        max_new_tokens: 8
        temperature: 1.0
        top_p: 0.95
        repetition_penalty: 1.1
        """
    )
    with pytest.raises(ValueError) as ei:
        RLLoaderConfig.from_yaml_dict(cfg)
    assert "model.attn_implementation" in str(ei.value)


def test_missing_image_max_pixels_raises():
    cfg = _load_yaml_str(
        """
        model_path: /abs/model
        bf16: true
        model:
          attn_implementation: flash_attention_2
        loss:
          teacher_loss_weight: 0.5
          student_loss_weight: 1.0
          caption_loss_weight: 1.0
          grounding_loss_weight: 1.0
          formatting_loss_weight: 1.0
        per_device_train_batch_size: 1
        update_steps: 1
        learning_rate: 1e-5
        weight_decay: 0.0
        max_steps: 1
        warmup_steps: 0
        logging_steps: 1
        save_steps: 1
        seed: 0
        sample_k: 1
        max_new_tokens: 8
        temperature: 1.0
        top_p: 0.95
        repetition_penalty: 1.1
        """
    )
    with pytest.raises(ValueError) as ei:
        RLLoaderConfig.from_yaml_dict(cfg)
    assert "model.image_max_pixels" in str(ei.value)


def test_missing_loss_key_raises():
    cfg = _load_yaml_str(
        """
        model_path: /abs/model
        bf16: true
        model:
          attn_implementation: sdpa
          image_max_pixels: 401408
        loss:
          teacher_loss_weight: 0.5
          student_loss_weight: 1.0
          caption_loss_weight: 1.0
          grounding_loss_weight: 1.0
        per_device_train_batch_size: 1
        update_steps: 1
        learning_rate: 1e-5
        weight_decay: 0.0
        max_steps: 1
        warmup_steps: 0
        logging_steps: 1
        save_steps: 1
        seed: 0
        sample_k: 1
        max_new_tokens: 8
        temperature: 1.0
        top_p: 0.95
        repetition_penalty: 1.1
        """
    )
    with pytest.raises(ValueError) as ei:
        RLLoaderConfig.from_yaml_dict(cfg)
    assert "loss.formatting_loss_weight" in str(ei.value)


def test_bf16_required_true():
    cfg = _load_yaml_str(
        """
        model_path: /abs/model
        bf16: false
        model:
          attn_implementation: eager
          image_max_pixels: 401408
        loss:
          teacher_loss_weight: 0.5
          student_loss_weight: 1.0
          caption_loss_weight: 1.0
          grounding_loss_weight: 1.0
          formatting_loss_weight: 1.0
        per_device_train_batch_size: 1
        update_steps: 1
        learning_rate: 1e-5
        weight_decay: 0.0
        max_steps: 1
        warmup_steps: 0
        logging_steps: 1
        save_steps: 1
        seed: 0
        sample_k: 1
        max_new_tokens: 8
        temperature: 1.0
        top_p: 0.95
        repetition_penalty: 1.1
        """
    )
    with pytest.raises(ValueError) as ei:
        RLLoaderConfig.from_yaml_dict(cfg)
    assert "bf16 is mandatory" in str(ei.value)
