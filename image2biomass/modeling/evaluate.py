from pathlib import Path

from loguru import logger
from sklearn import metrics
import typer

from image2biomass.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from image2biomass.modeling.models.registry import get_class
from image2biomass.modeling.train import get_Xy_from_csv

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features_dinov2_small.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "train.csv",
    model_name: str = "sklearn_gboost",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    """Evaluate a trained model on validation data.

    Loads a saved model, runs predictions, computes metrics.
    Optionally saves predictions to CSV.
    """

    logger.info("Evaluating model from {}", model_path)

    # Get model class from registry and load
    model_cls = get_class(model_name)
    model = model_cls.load(model_path)

    # Load validation data
    X, y, sample_weights = get_Xy_from_csv(
        features_csv=features_path,
        labels_csv=labels_path,
    )

    # Predict
    logger.info("Running predictions on {} samples...", len(y))
    y_pred = model.predict(X)

    # Compute metrics
    score = metrics.r2_score(y, y_pred, sample_weight=sample_weights)
    logger.success("Weighted R² score: {:.4f}", score)


if __name__ == "__main__":
    app()
