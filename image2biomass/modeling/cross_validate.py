from pathlib import Path

from loguru import logger
import numpy as np
from sklearn import metrics
import typer

from image2biomass.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from image2biomass.modeling.models.registry import get_class
from image2biomass.modeling.train import get_Xy_from_csv

app = typer.Typer()


def train_single_fold(
    model_cls,
    features_path: Path,
    train_labels: Path,
    model_out: Path,
) -> None:
    """Train a model on a single fold."""
    logger.info("Training on {}", train_labels)

    X, y, sample_weights = get_Xy_from_csv(
        features_csv=features_path,
        labels_csv=train_labels,
    )

    model = model_cls().fit(X, y)
    score = metrics.r2_score(y, model.predict(X), sample_weight=sample_weights)

    logger.info("In-sample R² score: {:.4f}", score)

    model.save(model_out)

    logger.info("Model saved to {}", model_out)


def evaluate_single_fold(
    model_cls,
    model_path: Path,
    features_path: Path,
    labels_path: Path,
) -> float:
    """Evaluate a single model on a single fold. Returns R² score."""

    logger.info("Loading model from {}", model_path)
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
    logger.info("Validation R² score: {:.4f}", score)

    return score


@app.command()
def main(
    splits_dir: Path,
    features_path: Path = PROCESSED_DATA_DIR / "features_dinov2_small.csv",
    model_name: str = "sklearn_gboost",
    train: bool = False,
):
    """Run cross-validation over a directory of train/val splits.

    Args:
        splits_dir: Directory containing fold subdirectories (0/, 1/, 2/...)
        features_path: Path to features CSV
        model_name: Model to use (from registry)
        train: If True, train models for each fold before evaluating
        save_predictions: If True, save predictions for each fold
    """

    logger.info("Cross-validation on splits in {}", splits_dir)

    model_cls = get_class(model_name)

    fold_dirs = sorted([d for d in splits_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not fold_dirs:
        raise ValueError(f"No numeric fold directories found in {splits_dir}")

    logger.info("Found {} folds", len(fold_dirs))

    scores = []
    for fold_dir in fold_dirs:
        fold_id = fold_dir.name
        logger.info("=" * 60)
        logger.info("Processing fold {}", fold_id)

        train_split = fold_dir / "train_split.csv"
        val_split = fold_dir / "val_split.csv"

        if not val_split.exists():
            logger.warning("Skipping fold {}: val_split.csv not found", fold_id)
            continue

        fold_model_path = MODELS_DIR / f"{model_name}_fold{fold_id}.pkl"

        # Train if requested
        if train:
            if not train_split.exists():
                logger.warning("Skipping fold {}: train_split.csv not found", fold_id)
                continue

            train_single_fold(
                model_cls=model_cls,
                features_path=features_path,
                train_labels=train_split,
                model_out=fold_model_path,
            )

        score = evaluate_single_fold(
            model_cls=model_cls,
            model_path=fold_model_path,
            features_path=features_path,
            labels_path=val_split,
        )
        scores.append(score)

    if not scores:
        logger.error("No folds were evaluated successfully.")
        return

    logger.info("=" * 60)
    logger.success("Cross-validation complete!")
    logger.success("Scores per fold: {}", [f"{s:.4f}" for s in scores])
    logger.success("Mean R²: {:.4f} ± {:.4f}", np.mean(scores), np.std(scores))


if __name__ == "__main__":
    app()
