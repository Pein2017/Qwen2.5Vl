from __future__ import annotations

from typing import Dict, Optional

from src_new.config.augmentation_config import AugmentationConfig, TypePolicyConfig

from .object_registry import resolve_object_type


SUPPORTED_TYPES = {"bbu", "bbu_shield", "connect_point", "label", "fiber", "wire"}


def get_policy_for_type(
    tp: Dict[str, TypePolicyConfig], typ: str
) -> Optional[TypePolicyConfig]:
    return tp[typ] if typ in tp else None


def resolve_policy_for_desc(
    cfg: AugmentationConfig, desc: str
) -> Optional[TypePolicyConfig]:
    if cfg.type_policies is None:
        return None
    typ = resolve_object_type(desc)
    if typ is None:
        return None
    return get_policy_for_type(cfg.type_policies, typ)


def is_move_allowed(cfg: AugmentationConfig, desc: str) -> bool:
    pol = resolve_policy_for_desc(cfg, desc)
    return bool(pol and pol.allow_move)


def is_copy_paste_allowed(cfg: AugmentationConfig, desc: str) -> bool:
    pol = resolve_policy_for_desc(cfg, desc)
    return bool(pol and pol.allow_copy_paste)


def is_blur_allowed(cfg: AugmentationConfig, desc: str) -> bool:
    pol = resolve_policy_for_desc(cfg, desc)
    return bool(pol and pol.allow_blur)
