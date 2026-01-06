from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger
import polars as pl

from image2biomass.config import RAW_DATA_DIR


class EmbedderFn(Protocol):
    def __call__(self, paths):  # noqa: D102
        ...


@dataclass(frozen=True)
class BuildFeaturesConfig:
    image_root: Path = RAW_DATA_DIR
    image_path_col: str = "image_path"


def resolve_image_path(image_path: str, *, root: Path) -> Path:
    """Resolve an image path from the CSV to a filesystem path.

    The competition csv seems to store paths like `train/IDxxxx.jpg`.
    We interpret them relative to `data/raw/`.
    """

    p = Path(image_path)
    if p.is_absolute():
        return p
    return root / p


def build_features_dataframe(
    dataset_csv: Path,
    *,
    config: BuildFeaturesConfig = BuildFeaturesConfig(),
    embedder_fn: EmbedderFn | None = None,
) -> pl.DataFrame:
    df = pl.read_csv(dataset_csv)

    if config.image_path_col not in df.columns:
        raise ValueError(
            f"Missing required column '{config.image_path_col}' in {dataset_csv}. "
            f"Available: {df.columns}"
        )

    image_paths_str = df[config.image_path_col].to_list()
    image_paths = [resolve_image_path(p, root=config.image_root) for p in image_paths_str]

    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Some images referenced by the dataset CSV don't exist. "
            f"First missing: {missing[0]} (total missing: {len(missing)})"
        )

    if embedder_fn is None:
        raise ValueError(
            "embedder_fn is required. Provide a callable that maps a list[Path] to a 2D array-like "
            "of embeddings (n_images, embed_dim). Model selection must live in the CLI/pipeline."
        )

    embeddings = embedder_fn(image_paths)
    # Accept numpy, torch tensors, lists... convert via Polars
    feat_df = pl.DataFrame(embeddings)
    if feat_df.height != len(image_paths):
        raise RuntimeError(f"Expected {len(image_paths)} embeddings, got {feat_df.height}.")

    # Rename feature columns deterministically
    feat_cols = [f"f_{i:04d}" for i in range(feat_df.width)]
    feat_df.columns = feat_cols

    out = pl.DataFrame({config.image_path_col: image_paths_str}).hstack(feat_df)
    return out


def build_features_csv(
    dataset_csv: Path,
    output_csv: Path,
    *,
    config: BuildFeaturesConfig = BuildFeaturesConfig(),
    embedder_fn: EmbedderFn | None = None,
    pipeline_name: str = "features",
) -> None:
    logger.info(
        "Building {} from '{}' -> '{}'...",
        pipeline_name,
        dataset_csv,
        output_csv,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = build_features_dataframe(dataset_csv, config=config, embedder_fn=embedder_fn)

    df.write_csv(output_csv)
    logger.success(
        "Wrote features CSV with {} rows and {} features to {}",
        df.height,
        df.width - 1,
        output_csv,
    )
