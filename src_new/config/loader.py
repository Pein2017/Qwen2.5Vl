"""Layered configuration loader for Qwen2.5-VL src_new training."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from src_new.augmentation.presets import (
    PresetOptions,
    build_augmentation_config_from_preset,
)
from src_new.utils.data_resolver import DataResolver
from src_new.utils.validation import normalize_path_input

from .augmentation_config import (
    AugmentationConfig,
    CriteriaConfig,
    ImageGeomConfig,
    LineAugConfig,
    OcclusionCriterionConfig,
    OCRPolicyConfig,
    PhotometricConfig,
    SmartResizeConfig,
    TypePolicyConfig,
    validate_augmentation_config,
)
from .schema import (
    AdvancedConfig,
    AugmentationFeatures,
    AugmentationScheduleItem,
    CheckpointConfig,
    DataConfig,
    EvaluationConfig,
    FeaturesConfig,
    GroupLossWeights,
    LoggingConfig,
    LossConfig,
    ModelConfig,
    OutputConfig,
    PhaseConfig,
    RuntimeConfig,
    SchemaError,
    TeacherAugmentationConfig,
    TeacherPairingConfig,
    TrainingConfig,
    TrainingConfigSection,
)
from .validators import (
    requires_if,
)


class LayerResolutionError(ValueError):
    """Raised when the layered configuration graph is invalid."""


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _configs_dir(root: Path) -> Path:
    return root / "configs"


def _resolve_config_path(
    name: str,
    configs_dir: Path,
    *,
    base_dir: Optional[Path] = None,
) -> Path:
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate

    search_paths: List[Path] = []

    def _append_with_suffix(base: Path, rel: Path, suffix: str) -> None:
        search_paths.append(base / Path(f"{rel}{suffix}"))

    if base_dir is not None:
        search_paths.append(base_dir / candidate)
        if not candidate.suffix:
            _append_with_suffix(base_dir, candidate, ".yaml")
            _append_with_suffix(base_dir, candidate, ".yml")

    if candidate.suffix:
        search_paths.append(configs_dir / candidate)
    else:
        _append_with_suffix(configs_dir, candidate, ".yaml")
        _append_with_suffix(configs_dir, candidate, ".yml")
        search_paths.append(configs_dir / candidate)

    for path in search_paths:
        if path.exists():
            return path

    raise FileNotFoundError(f"Unable to resolve configuration path '{name}'")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise LayerResolutionError(f"Configuration file must contain a mapping: {path}")
    # Remove extends (handled separately)
    data = copy.deepcopy(data)
    data.pop("extends", None)
    return data


def _gather_layers(
    path: Path,
    configs_dir: Path,
    *,
    visited: Optional[Dict[Path, Dict[str, Any]]] = None,
    stack: Optional[List[Path]] = None,
) -> List[Tuple[Path, Dict[str, Any]]]:
    visited = visited or {}
    stack = stack or []

    if path in stack:
        cycle = " -> ".join(str(p) for p in stack + [path])
        raise LayerResolutionError(f"Detected cyclic extends chain: {cycle}")

    if path in visited:
        return []

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise LayerResolutionError(f"Configuration file must contain a mapping: {path}")

    extends = raw.get("extends", [])
    if extends and not isinstance(extends, list):
        raise LayerResolutionError(f"extends must be a list when provided ({path})")

    stack.append(path)
    layers: List[Tuple[Path, Dict[str, Any]]] = []
    base_dir = path.parent

    for ref in extends or []:
        if not isinstance(ref, str):
            raise LayerResolutionError(
                f"extends entries must be strings (file paths). Invalid entry in {path}: {ref!r}"
            )
        resolved = _resolve_config_path(ref, configs_dir, base_dir=base_dir)
        layers.extend(
            _gather_layers(resolved, configs_dir, visited=visited, stack=stack)
        )

    stack.pop()

    data = copy.deepcopy(raw)
    data.pop("extends", None)
    visited[path] = data
    layers.append((path, data))
    return layers


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _derive_phase_name(config: Mapping[str, Any]) -> Optional[str]:
    features = config.get("features")
    if not isinstance(features, Mapping):
        return None
    phase = features.get("phase")
    if not isinstance(phase, Mapping):
        return None
    name = phase.get("name")
    if not isinstance(name, str) or not name:
        return None
    return name


def _validate_raw_config(config: Dict[str, Any]) -> None:
    errors: List[str] = []

    # Basic validation
    errors.extend(
        requires_if(
            config,
            "features.augmentation.enabled",
            True,
            [
                "features.augmentation.config",
                "features.augmentation.config.rng_seed",
                "features.augmentation.config.smart_resize",
                "features.augmentation.config.geometric_transforms",
                "features.augmentation.config.photometric_transforms",
                "features.augmentation.config.line_operations",
            ],
        )
    )

    errors.extend(
        requires_if(
            config,
            "features.teacher_pairing.enabled",
            True,
            [
                "features.teacher_pairing.teacher_pool_file",
                "features.teacher_pairing.teacher_ratio",
            ],
        )
    )

    if errors:
        raise SchemaError("\n".join(errors))


def _normalize_paths(config: Dict[str, Any]) -> None:
    model_section = config.get("model", {})
    if "model_path" in model_section:
        model_section["model_path"] = str(
            normalize_path_input(model_section["model_path"])
        )

    data_section = config.get("data", {})
    data_root = data_section.get("data_root")
    if not data_root:
        raise SchemaError("data.data_root must be provided")
    data_section["data_root"] = str(normalize_path_input(data_root))

    if not data_section.get("train_data_path") or not data_section.get("val_data_path"):
        resolved = DataResolver.resolve_dataset_paths(data_section["data_root"])
        data_section.setdefault("train_data_path", str(resolved.train_data_path))
        data_section.setdefault("val_data_path", str(resolved.val_data_path))
        data_section.setdefault("teacher_pool_file", str(resolved.teacher_pool_file))

    for key in ("train_data_path", "val_data_path", "teacher_pool_file"):
        if data_section.get(key):
            data_section[key] = str(normalize_path_input(data_section[key]))

    output_section = config.get("output", {})
    for key in ("output_dir", "tb_dir"):
        if key in output_section and output_section[key]:
            output_section[key] = str(normalize_path_input(output_section[key]))


def _build_smart_resize(
    mapping: Optional[Mapping[str, Any]], path: str
) -> Optional[SmartResizeConfig]:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise SchemaError(f"{path} must be a mapping")
    return SmartResizeConfig(
        enabled=bool(mapping["enabled"]),
        factor=int(mapping["factor"]),
        min_pixels=int(mapping["min_pixels"]),
        max_pixels=int(mapping["max_pixels"]),
        max_ratio=float(mapping["max_ratio"]),
    )


def _build_image_geom(
    mapping: Optional[Mapping[str, Any]], path: str
) -> Optional[ImageGeomConfig]:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise SchemaError(f"{path} must be a mapping")
    return ImageGeomConfig(
        rotate_deg_range=tuple(mapping["rotate_deg_range"]),
        translate_pct=float(mapping["translate_pct"]),
        scale_range=tuple(mapping["scale_range"]),
        perspective_pct=float(mapping["perspective_pct"]),
        crop_pct=float(mapping["crop_pct"]),
        multiscale_short_edges=mapping.get("multiscale_short_edges"),
    )


def _build_photometric(
    mapping: Optional[Mapping[str, Any]], path: str
) -> Optional[PhotometricConfig]:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise SchemaError(f"{path} must be a mapping")
    cfg = PhotometricConfig(
        enabled=bool(mapping["enabled"]),
        apply_prob=float(mapping["apply_prob"]),
        num_ops=int(mapping["num_ops"]),
        magnitude=float(mapping["magnitude"]),
        ocr_safe_pool=bool(mapping["ocr_safe_pool"]),
    )
    return cfg


def _build_lines(
    mapping: Optional[Mapping[str, Any]], path: str
) -> Optional[LineAugConfig]:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise SchemaError(f"{path} must be a mapping")
    return LineAugConfig(
        enabled=bool(mapping["enabled"]),
        jitter_px_minmax=tuple(mapping["jitter_px_minmax"]),
        resample_points=int(mapping["resample_points"]),
        min_length_px=int(mapping["min_length_px"]),
    )


def _build_type_policies(
    mapping: Optional[Mapping[str, Any]], path: str
) -> Optional[Dict[str, TypePolicyConfig]]:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise SchemaError(f"{path} must be a mapping")
    policies: Dict[str, TypePolicyConfig] = {}
    for key, value in mapping.items():
        if not isinstance(value, Mapping):
            raise SchemaError(f"{path}.{key} must be a mapping")
        policies[key] = TypePolicyConfig(
            allow_move=bool(value["allow_move"]),
            allow_copy_paste=bool(value["allow_copy_paste"]),
            allow_blur=bool(value["allow_blur"]),
            occluder_prob=float(value["occluder_prob"]),
            inpaint_source=bool(value["inpaint_source"]),
            max_iou_with_existing=value.get("max_iou_with_existing"),
            max_occ_fraction=value.get("max_occ_fraction"),
            occ_grid_downscale=value.get("occ_grid_downscale"),
            occ_margin_px=value.get("occ_margin_px"),
            same_plane_constraint=bool(value.get("same_plane_constraint", True)),
            copy_paste_attempts=value.get("copy_paste_attempts"),
            alpha_feather_px=value.get("alpha_feather_px"),
            allowed_copy_types=value.get("allowed_copy_types"),
        )
    return policies


def _build_criteria(
    mapping: Optional[Mapping[str, Any]], path: str
) -> Optional[CriteriaConfig]:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise SchemaError(f"{path} must be a mapping")
    occlusion_cfg = None
    occlusion = mapping.get("occlusion")
    if occlusion is not None:
        if not isinstance(occlusion, Mapping):
            raise SchemaError(f"{path}.occlusion must be a mapping")
        occlusion_cfg = OcclusionCriterionConfig(
            enabled=bool(occlusion["enabled"]),
            min_overlap_fraction_bbox=float(occlusion["min_overlap_fraction_bbox"]),
            min_overlap_fraction_line=float(occlusion["min_overlap_fraction_line"]),
            mask_downscale=int(occlusion["mask_downscale"]),
            line_width_px=int(occlusion["line_width_px"]),
        )
    return CriteriaConfig(occlusion=occlusion_cfg)


def _build_ocr(
    mapping: Optional[Mapping[str, Any]], path: str
) -> Optional[OCRPolicyConfig]:
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise SchemaError(f"{path} must be a mapping")
    return OCRPolicyConfig(
        label_protect=bool(mapping["label_protect"]),
        force_unreadable_on_strong_distortion=bool(
            mapping["force_unreadable_on_strong_distortion"]
        ),
    )


def _build_augmentation_config(
    mapping: Mapping[str, Any], path: str
) -> AugmentationConfig:
    if "preset" in mapping and mapping["preset"] is not None:
        preset = str(mapping["preset"])
        smart_resize = mapping.get("smart_resize")
        if preset != "off" and smart_resize is None:
            raise SchemaError(
                f"{path}.smart_resize must be provided when preset is '{preset}'"
            )
        opts = PresetOptions(
            preset=preset,  # type: ignore[arg-type]
            rng_seed=int(mapping.get("rng_seed", 12345)),
            apply_to_teachers=bool(mapping.get("apply_to_teachers", False)),
            lines_policy=str(mapping.get("lines_policy", "transform")),
            debug_visualization=bool(mapping.get("debug_visualization", False)),
            debug_output_dir=mapping.get("debug_output_dir"),
            smart_resize_enabled=(
                bool(smart_resize.get("enabled"))
                if isinstance(smart_resize, Mapping)
                else None
            ),
            smart_resize_factor=(
                int(smart_resize.get("factor"))
                if isinstance(smart_resize, Mapping)
                else None
            ),
            smart_resize_min_pixels=(
                int(smart_resize.get("min_pixels"))
                if isinstance(smart_resize, Mapping)
                else None
            ),
            smart_resize_max_pixels=(
                int(smart_resize.get("max_pixels"))
                if isinstance(smart_resize, Mapping)
                else None
            ),
            smart_resize_max_ratio=(
                float(smart_resize.get("max_ratio"))
                if isinstance(smart_resize, Mapping)
                else None
            ),
        )
        cfg = build_augmentation_config_from_preset(opts)
        validate_augmentation_config(cfg)
        return cfg

    if "enabled" not in mapping:
        raise SchemaError(f"{path} must include either 'preset' or 'enabled'")

    smart_resize = _build_smart_resize(
        mapping.get("smart_resize"), f"{path}.smart_resize"
    )
    image_geom = _build_image_geom(mapping.get("image_geom"), f"{path}.image_geom")
    photometric = _build_photometric(mapping.get("photometric"), f"{path}.photometric")
    lines = _build_lines(mapping.get("lines"), f"{path}.lines")
    type_policies = _build_type_policies(
        mapping.get("type_policies"), f"{path}.type_policies"
    )
    ocr = _build_ocr(mapping.get("ocr"), f"{path}.ocr")
    criteria = _build_criteria(mapping.get("criteria"), f"{path}.criteria")

    cfg = AugmentationConfig(
        enabled=bool(mapping.get("enabled", False)),
        rng_seed=int(mapping["rng_seed"]),
        apply_to_teachers=bool(mapping["apply_to_teachers"]),
        lines_policy=str(mapping["lines_policy"]),
        debug_visualization=bool(mapping.get("debug_visualization", False)),
        debug_output_dir=mapping.get("debug_output_dir"),
        smart_resize=smart_resize,
        image_geom=image_geom,
        photometric=photometric,
        lines=lines,
        type_policies=type_policies,
        ocr=ocr,
        criteria=criteria,
    )
    validate_augmentation_config(cfg)
    return cfg


def _build_schedule(
    sequence: Optional[Sequence[Any]],
) -> Tuple[AugmentationScheduleItem, ...]:
    if sequence is None:
        return tuple()
    items: List[AugmentationScheduleItem] = []
    for idx, item in enumerate(sequence):
        if not isinstance(item, Mapping):
            raise SchemaError(
                f"features.augmentation.schedule[{idx}] must be a mapping"
            )
        smart_resize = _build_smart_resize(
            item.get("smart_resize"),
            f"features.augmentation.schedule[{idx}].smart_resize",
        )
        items.append(
            AugmentationScheduleItem(
                start_epoch=int(item["start_epoch"]),
                preset=str(item["preset"]),
                smart_resize=smart_resize,
            )
        )
    return tuple(items)


def _build_features(config: Dict[str, Any]) -> FeaturesConfig:
    features = config.get("features", {})
    augmentation_payload = features.get("augmentation", {})
    teacher_aug_payload = features.get("teacher_augmentation", {})
    teacher_pairing_payload = features.get("teacher_pairing", {})
    phase_payload = features.get("phase", {})
    checkpoint_payload = features.get("checkpoint", {})
    logging_payload = features.get("logging", {})
    evaluation_payload = features.get("evaluation", {})

    augmentation_config = None
    if augmentation_payload.get("config") is not None:
        augmentation_config = _build_augmentation_config(
            augmentation_payload["config"], "features.augmentation.config"
        )

    schedule = _build_schedule(augmentation_payload.get("schedule"))
    smart_resize_defaults = _build_smart_resize(
        augmentation_payload.get("smart_resize_defaults"),
        "features.augmentation.smart_resize_defaults",
    )

    augmentation = AugmentationFeatures(
        enabled=bool(augmentation_payload.get("enabled", False)),
        config=augmentation_config,
        schedule=schedule,
        smart_resize_defaults=smart_resize_defaults,
    )

    teacher_aug_config = None
    if teacher_aug_payload.get("config") is not None:
        teacher_aug_mapping = copy.deepcopy(teacher_aug_payload["config"])
        teacher_aug_mapping.setdefault("enabled", True)
        teacher_aug_mapping.setdefault("apply_to_teachers", True)
        teacher_aug_mapping.setdefault(
            "lines_policy", teacher_aug_mapping.get("lines_policy", "transform")
        )
        teacher_aug_mapping.setdefault(
            "debug_visualization", teacher_aug_mapping.get("debug_visualization", False)
        )
        teacher_aug_mapping.setdefault(
            "rng_seed",
            teacher_aug_mapping.get(
                "rng_seed", teacher_aug_payload.get("rng_seed", 12345)
            ),
        )
        teacher_aug_config = _build_augmentation_config(
            teacher_aug_mapping, "features.teacher_augmentation.config"
        )

    teacher_augmentation = TeacherAugmentationConfig(
        enabled=bool(teacher_aug_payload.get("enabled", False)),
        config=teacher_aug_config,
    )

    teacher_pairing = TeacherPairingConfig(
        enabled=bool(teacher_pairing_payload.get("enabled", False)),
        teacher_ratio=teacher_pairing_payload.get("teacher_ratio"),
        dynamic_pairing_enabled=bool(
            teacher_pairing_payload.get("dynamic_pairing_enabled", False)
        ),
        dynamic_pair_cross_bucket_explore_prob=teacher_pairing_payload.get(
            "dynamic_pair_cross_bucket_explore_prob"
        ),
    )

    phase = PhaseConfig(
        enabled=bool(phase_payload.get("enabled", False)),
        name=phase_payload.get("name"),
        llm_top_k_block=phase_payload.get("llm_top_k_block"),
        vision_top_k_block=phase_payload.get("vision_top_k_block"),
        freeze_patch_embed=phase_payload.get("freeze_patch_embed"),
    )

    checkpoint = CheckpointConfig(
        enabled=bool(checkpoint_payload.get("enabled", False)),
        save_steps=checkpoint_payload.get("save_steps"),
        save_total_limit=checkpoint_payload.get("save_total_limit"),
        metric_for_best_model=checkpoint_payload.get("metric_for_best_model"),
        greater_is_better=checkpoint_payload.get("greater_is_better"),
        save_strategy=checkpoint_payload.get("save_strategy"),
        load_best_model_at_end=checkpoint_payload.get("load_best_model_at_end"),
        min_interval_steps=checkpoint_payload.get("min_interval_steps"),
        interval_multiplier_of_eval_steps=checkpoint_payload.get(
            "interval_multiplier_of_eval_steps"
        ),
    )

    logging = LoggingConfig(
        enabled=bool(logging_payload.get("enabled", False)),
        logging_steps=logging_payload.get("logging_steps"),
        report_to=logging_payload.get("report_to"),
        disable_tqdm=logging_payload.get("disable_tqdm"),
    )

    evaluation = EvaluationConfig(
        enabled=bool(evaluation_payload.get("enabled", False)),
        strategy=evaluation_payload.get("strategy"),
        eval_steps=evaluation_payload.get("eval_steps"),
    )

    return FeaturesConfig(
        augmentation=augmentation,
        teacher_augmentation=teacher_augmentation,
        teacher_pairing=teacher_pairing,
        phase=phase,
        checkpoint=checkpoint,
        logging=logging,
        evaluation=evaluation,
    )


def _build_training_config(config: Dict[str, Any]) -> TrainingConfig:
    model = ModelConfig(**config["model"])
    data = DataConfig(**config["data"])
    training = TrainingConfigSection(**config["training"])

    loss_dict = config["loss"]
    grouped = loss_dict["grouped"]
    loss = LossConfig(
        teacher_loss_weight=float(loss_dict["teacher_loss_weight"]),
        student_loss_weight=float(loss_dict["student_loss_weight"]),
        grouped=GroupLossWeights(
            caption=float(grouped["caption"]),
            grounding=float(grouped["grounding"]),
            formatting=float(grouped["formatting"]),
        ),
    )

    features = _build_features(config)

    output_dict = config["output"].copy()
    if "tb_dir" not in output_dict or not output_dict["tb_dir"]:
        output_dict["tb_dir"] = output_dict.get("output_dir")
    output_cfg = OutputConfig(**output_dict)

    advanced_payload = config.get("advanced", {})
    advanced = AdvancedConfig(
        span_include_im_end_in_labels=bool(
            advanced_payload.get("span_include_im_end_in_labels", True)
        ),
        debug_alignment=bool(advanced_payload.get("debug_alignment", False)),
        augmentation_smart_resize_defaults=_build_smart_resize(
            advanced_payload.get("augmentation_smart_resize_defaults"),
            "advanced.augmentation_smart_resize_defaults",
        ),
        trainable_token_strings=tuple(
            advanced_payload.get("trainable_token_strings", []) or []
        )
        or None,
    )

    runtime_payload = config.get("runtime", {})
    runtime = RuntimeConfig(seed=runtime_payload.get("seed"))

    return TrainingConfig(
        model=model,
        data=data,
        training=training,
        loss=loss,
        features=features,
        output=output_cfg,
        runtime=runtime,
        advanced=advanced,
    )


def load_layered_config(config_name: str) -> TrainingConfig:
    root = _project_root()
    configs_dir = _configs_dir(root)

    # Remove the requirement for global base.yaml since we use extends mechanism
    # base_path = configs_dir / "base.yaml"
    # if not base_path.exists():
    #     raise FileNotFoundError(f"Global base configuration missing: {base_path}")
    # base_data = _load_yaml(base_path)

    config_key = config_name
    if config_key.startswith("configs/"):
        config_key = config_key[len("configs/") :]
    alias_map = {
        "phase_1_debug": "debug",
        "phase_2_debug": "debug",
        "phase_3_debug": "debug",
    }
    try:
        config_path = _resolve_config_path(config_key, configs_dir)
    except FileNotFoundError as err:
        alt_key = config_key.replace("/", "_")
        candidates = []
        if alt_key != config_key:
            candidates.append(alt_key)
        if config_key in alias_map:
            candidates.append(alias_map[config_key])
        if alt_key in alias_map:
            candidates.append(alias_map[alt_key])
        for candidate in candidates:
            try:
                config_path = _resolve_config_path(candidate, configs_dir)
                break
            except FileNotFoundError:
                continue
        else:
            raise err
    layers = _gather_layers(config_path, configs_dir)

    # Start with empty dict instead of base_data since extends mechanism handles inheritance
    merged: Dict[str, Any] = {}
    for _, data in layers:
        merged = _deep_merge(merged, data)

    phase_name = _derive_phase_name(merged)
    if phase_name:
        phase_dir = (
            phase_name
            if str(phase_name).startswith("phase_")
            else f"phase_{phase_name}"
        )
        phase_base = configs_dir / phase_dir / "base.yaml"
        if phase_base.exists():
            layer_paths = [layer_path for layer_path, _ in layers]
            if phase_base not in layer_paths:
                phase_data = _load_yaml(phase_base)
                # Merge phase base first, then apply layers
                merged = _deep_merge(phase_data, merged)

    _validate_raw_config(merged)
    _normalize_paths(merged)

    training_config = _build_training_config(merged)
    return training_config


__all__ = ["load_layered_config", "LayerResolutionError"]
