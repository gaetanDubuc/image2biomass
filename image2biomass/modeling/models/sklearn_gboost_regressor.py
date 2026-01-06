from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle

from loguru import logger
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from image2biomass.modeling.models.base import BaseModel
from image2biomass.modeling.models.registry import register


@dataclass(frozen=True)
class Config:
    pass


@register("sklearn_gboost")
class SkLearnGradientBoostingRegressor(BaseModel):
    """Scikit-learn GradientBoosting regressor wrapper with pickle serialization."""

    def __init__(self, config: Config = Config()):
        self._model = GradientBoostingRegressor(**asdict(config))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SkLearnGradientBoostingRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        """Save model using pickle for sklearn compatibility."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model with pickle
        with open(path, "wb") as f:
            pickle.dump(self._model, f, protocol=5)

        logger.info("Saved sklearn GradientBoosting model to {}", path)

    @classmethod
    def load(cls, path: Path) -> "SkLearnGradientBoostingRegressor":
        """Load model from pickle."""
        with open(path, "rb") as f:
            model = pickle.load(f)

        obj = cls()
        obj._model = model
        logger.info("Loaded sklearn GradientBoosting model from {}", path)
        return obj
