from typing import Any, Dict

from .registry import register_augmentation


@register_augmentation("identity")
def augmentation_identity(sample: Dict[str, Any]) -> Dict[str, Any]:
    return sample
