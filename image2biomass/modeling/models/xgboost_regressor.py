from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
import xgboost as xgb

from image2biomass.modeling.models.base import BaseModel
from image2biomass.modeling.models.registry import register


@dataclass(frozen=True)
class Config:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    min_child_weight: int = 1
    gamma: float = 0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    objective: str = "reg:squarederror"
    n_jobs: int = -1
    random_state: int = 42


@register("xgboost")
class XGBoostRegressor(BaseModel):
    """Native XGBoost regressor using xgboost.train + DMatrix.

    This avoids the sklearn wrapper which can cause platform/runtime issues
    (OpenMP, libomp) on some macOS setups.
    """

    def __init__(self, config: Config = Config()):
        self.config = config
        self._booster: xgb.Booster | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        dtrain = xgb.DMatrix(X, label=y)

        params = {
            "objective": self.config.objective,
            "eta": self.config.learning_rate,
            "max_depth": int(self.config.max_depth),
            "min_child_weight": int(self.config.min_child_weight),
            "gamma": float(self.config.gamma),
            "subsample": float(self.config.subsample),
            "colsample_bytree": float(self.config.colsample_bytree),
            "nthread": int(self.config.n_jobs) if self.config.n_jobs > 0 else -1,
            "seed": int(self.config.random_state),
        }

        num_boost_round = int(self.config.n_estimators)

        self._booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model is not trained. Call fit(...) before predict(...).")
        dtest = xgb.DMatrix(X)
        return self._booster.predict(dtest)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._booster is None:
            raise RuntimeError("No trained model to save.")

        # xgboost expects a string path
        self._booster.save_model(str(path))
        logger.info("Saved XGBoost native model to {}", path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostRegressor":
        obj = cls()
        booster = xgb.Booster()
        booster.load_model(str(path))
        obj._booster = booster
        logger.info("Loaded native XGBoost model from {}", path)
        return obj
