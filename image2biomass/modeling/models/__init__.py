"""Model implementations.

Train/predict CLIs should depend on these abstractions, not the other way around.
"""

from .registry import create
from .sklearn_gboost_regressor import SkLearnGradientBoostingRegressor

__all__ = ["SkLearnGradientBoostingRegressor", "create"]
