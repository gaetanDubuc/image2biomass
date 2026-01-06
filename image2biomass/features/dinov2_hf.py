from __future__ import annotations

from dataclasses import dataclass
from os import sep
from pathlib import Path
from typing import Iterable, Sequence

from loguru import logger
import numpy as np
from PIL import Image
import polars as pl
import torch
import tqdm
from transformers import AutoImageProcessor, AutoModel
import typer

from image2biomass.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


@dataclass(frozen=True)
class DinoV2HFConfig:
    model_name: str = "facebook/dinov2-small"
    batch_size: int = 32
    num_workers: int = 0  # kept for future (DataLoader)
    device: str | None = None  # "cuda" | "mps" | "cpu" | None(auto)


class DinoV2HFEmbedder:
    """Extract DINOv2 embeddings using HuggingFace Transformers.

    Contract:
      - input: iterable of image paths
      - output: np.ndarray of shape (n_images, embed_dim)

    Notes:
      - We use mean pooling on patch tokens (excluding CLS when present).
      - Keeps responsibilities small: model loading + embedding.
    """

    def __init__(self, config: DinoV2HFConfig = DinoV2HFConfig()):
        self.config = config
        self.device = self._resolve_device(config.device)

        logger.info(
            "Loading DINOv2 model '{}' on device '{}'...",
            config.model_name,
            self.device,
        )
        self.processor = AutoImageProcessor.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(requested: str | None) -> torch.device:
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # macOS acceleration
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def embed(self, image_paths: Sequence[Path]) -> np.ndarray:
        if len(image_paths) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []

        with torch.inference_mode():
            for batch in tqdm.tqdm(
                self._batched(image_paths, self.config.batch_size),
                total=len(image_paths) // self.config.batch_size,
            ):
                images = [self._load_image(p) for p in batch]
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                # Transformers models typically expose last_hidden_state: (B, T, C)
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    raise RuntimeError(
                        "Model output has no last_hidden_state; can't pool embeddings."
                    )

                # Heuristic: if token count includes CLS at position 0, exclude it.
                if hidden.shape[1] > 1:
                    tokens = hidden[:, 1:, :]
                else:
                    tokens = hidden

                emb = tokens.mean(dim=1)  # (B, C)
                all_embeddings.append(emb.detach().float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        img = Image.open(path)
        # Ensure 3-channel RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @staticmethod
    def _batched(items: Sequence[Path], batch_size: int) -> Iterable[list[Path]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        for i in range(0, len(items), batch_size):
            yield list(items[i : i + batch_size])


app = typer.Typer()


@app.command()
def create_features(
    input_path: Path = PROCESSED_DATA_DIR / "train.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features_dinov2_small.csv",
) -> np.ndarray:
    image_paths = pl.read_csv(input_path)["image_path"].unique()
    image_paths_with_prefix = (str(RAW_DATA_DIR) + sep + image_paths).to_list()

    embedder = DinoV2HFEmbedder()
    embeddings = embedder.embed(image_paths_with_prefix)

    df = pl.DataFrame(embeddings)

    df = df.with_columns(pl.lit(image_paths).alias("image_path"))

    df.write_csv(output_path)
    return embeddings


if __name__ == "__main__":
    app()
