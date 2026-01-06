from typing import Any, Dict, Type

from image2biomass.modeling.models.base import BaseModel

_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register(name: str):
    def deco(cls: Type[BaseModel]):
        _REGISTRY[name] = cls
        return cls

    return deco


def create(name: str, **kwargs) -> Any:
    if name not in _REGISTRY:
        raise ValueError(f"Modèle inconnu: {name}. Disponibles: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def get_class(name: str) -> Type[BaseModel]:
    """Get model class without instantiating."""
    if name not in _REGISTRY:
        raise ValueError(f"Modèle inconnu: {name}. Disponibles: {list(_REGISTRY)}")
    return _REGISTRY[name]
