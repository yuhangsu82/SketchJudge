"""Core registry / factory / interface layer."""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable

MODEL_REGISTRY: Dict[str, "type[AbstractModel]"] = {}


def register_model(name: str) -> Callable:
    """Decorator used on wrapper classes:  @register_model("qwen")"""
    def _decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return _decorator


def make_model(name: str, **kwargs) -> "AbstractModel":
    """Factory – build a model by key."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {name}. "
                         f"Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


class AbstractModel:
    """Every wrapper must implement `generate_from_sample`."""

    def generate_from_sample(self, sample: Dict[str, Any]) -> str:     # noqa: D401
        raise NotImplementedError

    def batch_generate(
        self, samples: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Default serial loop; override for custom batching."""
        return [(s, self.generate_from_sample(s)) for s in samples]
