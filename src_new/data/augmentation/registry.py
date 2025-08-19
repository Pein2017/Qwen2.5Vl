from typing import Callable, Dict


AUGMENTATION_REGISTRY: Dict[str, Callable] = {}


def register_augmentation(name: str):
    def _dec(fn: Callable) -> Callable:
        AUGMENTATION_REGISTRY[name] = fn
        return fn

    return _dec
