# modeling/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np


class BaseModel(ABC):
    """Interface commune à tous les modèles."""

    def __init__(self, **kwargs):
        self.cfg = kwargs

    @abstractmethod
    def config(self) -> dict[str, Any]: ...

    @abstractmethod
    def fit(self, X, y) -> "BaseModel": ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Retourne les labels prédits (ou logits)."""
        ...

    def predict_proba(self, X) -> Optional[Any]:
        """Optionnel : proba par classe si dispo."""
        return None

    def save(self, path: Path) -> None:
        """Sauvegarde le modèle sur le disque."""
        ...
