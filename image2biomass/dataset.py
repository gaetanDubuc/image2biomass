import os
from pathlib import Path

import polars as pl
from sklearn.model_selection import GroupKFold
import tqdm
import typer

from image2biomass.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "train.csv",
):
    splits_path = PROCESSED_DATA_DIR / "splits"

    train_df = pl.read_csv(input_path)

    train_df = train_df.with_columns(
        pl.col("Sampling_Date").str.to_datetime(format="%Y/%m/%d").dt.month().alias("month"),
    )

    for col in tqdm.tqdm(["month", "State", "Species"]):
        groups = train_df[col].to_numpy()
        image_ids = train_df["image_path"].to_numpy()

        splitter = GroupKFold(n_splits=train_df[col].n_unique() // 2)

        for idx, (train_indx, test_indx) in enumerate(splitter.split(image_ids, groups=groups)):
            tmp_splits_path = splits_path / col / str(idx)
            os.makedirs(tmp_splits_path, exist_ok=True)

            image_ids_train, image_ids_val = image_ids[train_indx], image_ids[test_indx]

            train_df.filter(pl.col("image_path").is_in(image_ids_train)).write_csv(
                tmp_splits_path / "train_split.csv"
            )

            train_df.filter(pl.col("image_path").is_in(image_ids_val)).write_csv(
                tmp_splits_path / "val_split.csv"
            )

    train_df.write_csv(PROCESSED_DATA_DIR / "train.csv")


if __name__ == "__main__":
    app()
