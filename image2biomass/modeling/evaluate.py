from pathlib import Path

from loguru import logger
import polars as pl
from sklearn import metrics
import typer

from image2biomass.config import (
    IMAGE_PATH_COL,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COL,
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
    output_path: Path | None = None,
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

    # Optionally save predictions
    if output_path:
        logger.info("Loading features to get image_path for predictions output...")
        features_df = pl.read_csv(features_path)
        labels_df = pl.read_csv(labels_path)

        df = labels_df.join(features_df.select([IMAGE_PATH_COL]), on=IMAGE_PATH_COL, how="inner")

        preds_df = df.select([IMAGE_PATH_COL, TARGET_COL]).with_columns(
            pl.Series("prediction", y_pred)
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df.write_csv(output_path)
        logger.success("Saved predictions to {}", output_path)


if __name__ == "__main__":
    app()
