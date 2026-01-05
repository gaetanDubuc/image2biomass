from pathlib import Path

import typer

from image2biomass.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "train.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features_train.csv",
    # -----------------------------------------
):
    pass


if __name__ == "__main__":
    app()
