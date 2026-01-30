from pathlib import Path

from loguru import logger
import numpy as np
import polars as pl
from sklearn import metrics
import typer

from image2biomass.config import (
    IMAGE_PATH_COL,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_WEIGHTS,
    TARGET_COL,
)
from image2biomass.modeling.models.registry import get_class

app = typer.Typer()


def get_Xy_from_csv(
    features_csv: Path,
    labels_csv: Path,
):
    """Create and train an XGBoost model from CSV files.

    This keeps all XGBoost-specific ingestion details in the model module.
    """

    logger.info("Loading features from {}", features_csv)
    features_df = pl.read_csv(features_csv)

    logger.info("Loading labels from {}", labels_csv)
    labels_df = pl.read_csv(labels_csv)

    if IMAGE_PATH_COL not in features_df.columns:
        raise ValueError(
            f"Missing '{IMAGE_PATH_COL}' in features csv. Got columns={features_df.columns}"
        )
    if IMAGE_PATH_COL not in labels_df.columns or TARGET_COL not in labels_df.columns:
        raise ValueError(
            f"Missing '{IMAGE_PATH_COL}' or '{TARGET_COL}' in labels csv. Got columns={labels_df.columns}"
        )

    df = labels_df.join(features_df, on=IMAGE_PATH_COL, how="inner")
    if df.height == 0:
        raise ValueError("Join produced 0 rows. Check that image_path keys match.")

    feature_cols = features_df.select(pl.exclude(IMAGE_PATH_COL)).columns
    if not feature_cols:
        raise ValueError("No feature columns found (expected f_0000...).")

    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df[TARGET_COL].to_numpy().astype(np.float32)

    print(
        X.shape,
        y.shape,
    )

    sample_weights = (
        df.select(pl.col("target_name").replace(SAMPLE_WEIGHTS).alias("sample_weight"))
        .to_numpy()
        .squeeze()
        .astype(np.float32)
    )

    return (
        X,
        y,
        sample_weights,
    )


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features_dinov2_small.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "train.csv",
    model_name: str = "sklearn_gboost",
    model_out: Path = MODELS_DIR / "model.json",
):
    """Train a model and save it.

    The model module owns all ingestion + training logic.
    This CLI just orchestrates: load data → train → save.
    """

    logger.info("Training model '{}'...", model_name)

    # Get model class from registry
    model_cls = get_class(model_name)

    X, y, sample_weights = get_Xy_from_csv(
        features_csv=features_path,
        labels_csv=labels_path,
    )

    model = model_cls().fit(X, y)

    score = metrics.r2_score(y, model.predict(X), sample_weight=sample_weights)

    logger.info("Training complete. In-sample score: {:.4f}", score)

    model.save(model_out)

    logger.success("Saved trained model to {}", model_out)


if __name__ == "__main__":
    app()
