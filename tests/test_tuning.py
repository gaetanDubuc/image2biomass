"""Quick test script for hyperparameter tuning (small sample for fast validation)."""

from pathlib import Path

from loguru import logger
import numpy as np
import optuna

from image2biomass.modeling.models.xgboost_regressor import Config, XGBoostRegressor
from image2biomass.modeling.train import get_Xy_from_csv


def test_tuning_quick():
    """Run a quick 5-trial tuning test to validate the setup."""

    # Use fold 0 from month splits (or create dummy data if not available)
    features_path = Path("data/processed/features_dinov2_small.csv")
    train_labels = Path("data/processed/splits/month/0/train_split.csv")
    val_labels = Path("data/processed/splits/month/0/val_split.csv")

    if not all([features_path.exists(), train_labels.exists(), val_labels.exists()]):
        logger.warning("Test data not found, creating synthetic data...")
        # Create synthetic test
        X_train = np.random.rand(100, 50)
        y_train = np.random.rand(100)
        X_val = np.random.rand(30, 50)
        y_val = np.random.rand(30)
        sample_weights_train = None
        sample_weights_val = None
    else:
        logger.info("Loading real data...")
        X_train, y_train, sample_weights_train = get_Xy_from_csv(
            features_csv=features_path,
            labels_csv=train_labels,
        )
        X_val, y_val, sample_weights_val = get_Xy_from_csv(
            features_csv=features_path,
            labels_csv=val_labels,
        )

    logger.info("Train: {} samples, Val: {} samples", len(y_train), len(y_val))

    # Quick 5-trial test
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eta": trial.suggest_float("eta", 0.05, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "seed": 42,
        }

        config = Config(
            n_estimators=params["n_estimators"],
            learning_rate=params["eta"],
            max_depth=params["max_depth"],
        )
        model = XGBoostRegressor(config=config)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        from sklearn import metrics

        r2 = metrics.r2_score(y_val, y_pred, sample_weight=sample_weights_val)
        return -r2  # minimize negative R²

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5, show_progress_bar=True)

    logger.success("Test complete!")
    logger.success("Best R²: {:.4f}", -study.best_value)
    logger.success("Best params: {}", study.best_params)


if __name__ == "__main__":
    test_tuning_quick()
