"""Hyperparameter tuning for XGBoost using Optuna."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import optuna
from sklearn import metrics
import typer
import xgboost as xgb

from image2biomass.config import MODELS_DIR, PROCESSED_DATA_DIR
from image2biomass.modeling.models.xgboost_regressor import Config, XGBoostRegressor
from image2biomass.modeling.train import get_Xy_from_csv

app = typer.Typer()


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights_train: np.ndarray | None = None,
    sample_weights_val: np.ndarray | None = None,
) -> float:
    """Optuna objective function for XGBoost hyperparameter tuning.

    Returns:
        Negative R² score on validation set (Optuna minimizes by default)
    """

    # Suggest hyperparameters
    params = {
        "objective": "reg:squarederror",
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.0, 10.0),
        "seed": 42,
    }

    n_estimators = trial.suggest_int("n_estimators", 50, 500)

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=sample_weights_val)

    # Train model with early stopping
    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    # Predict and compute R² score
    y_pred = booster.predict(dval)
    r2 = metrics.r2_score(y_val, y_pred, sample_weight=sample_weights_val)

    # Return negative R² (Optuna minimizes)
    return -r2


@app.command()
def main(
    train_labels: Path,
    val_labels: Path,
    features_path: Path = PROCESSED_DATA_DIR / "features_dinov2_small.csv",
    n_trials: int = 100,
    output_name: str = "xgboost_tuned",
    study_name: str = "xgboost_optimization",
):
    """Tune XGBoost hyperparameters using Optuna.

    Args:
        train_labels: Path to training labels CSV
        val_labels: Path to validation labels CSV
        features_path: Path to features CSV
        n_trials: Number of Optuna trials
        output_name: Name for the output model file
        study_name: Name for the Optuna study
    """

    logger.info("Loading training data from {}", train_labels)
    X_train, y_train, sample_weights_train = get_Xy_from_csv(
        features_csv=features_path,
        labels_csv=train_labels,
    )

    logger.info("Loading validation data from {}", val_labels)
    X_val, y_val, sample_weights_val = get_Xy_from_csv(
        features_csv=features_path,
        labels_csv=val_labels,
    )

    logger.info(
        "Starting hyperparameter optimization with {} trials...",
        n_trials,
    )
    logger.info("Train: {} samples, Val: {} samples", len(y_train), len(y_val))

    # Create study
    study = optuna.create_study(
        direction="minimize",  # minimize negative R²
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Optimize
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_val,
            y_val,
            sample_weights_train,
            sample_weights_val,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Log best results
    logger.success("Optimization complete!")
    logger.success("Best trial: {}", study.best_trial.number)
    logger.success("Best validation R²: {:.4f}", -study.best_value)
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info("  {}: {}", key, value)

    # Train final model with best parameters
    logger.info("Training final model with best hyperparameters...")

    best_params = study.best_params
    config = Config(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["eta"],
        max_depth=best_params["max_depth"],
        min_child_weight=best_params["min_child_weight"],
        gamma=best_params["gamma"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
    )

    # Create and train model
    model = XGBoostRegressor(config=config)
    model.fit(X_train, y_train)

    # Evaluate on validation
    y_pred = model.predict(X_val)
    final_r2 = metrics.r2_score(y_val, y_pred, sample_weight=sample_weights_val)
    logger.success("Final model validation R²: {:.4f}", final_r2)

    # Save model
    model_path = MODELS_DIR / f"{output_name}.pkl"
    model.save(model_path)
    logger.success("Model saved to {}", model_path)

    # Save study results
    results_path = MODELS_DIR / f"{output_name}_optuna_study.csv"
    study_df = study.trials_dataframe()
    study_df.to_csv(results_path, index=False)
    logger.info("Study results saved to {}", results_path)


if __name__ == "__main__":
    app()
