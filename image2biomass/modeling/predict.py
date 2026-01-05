from pathlib import Path

import typer

from image2biomass.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features_test.csv",
    model_path: Path = MODELS_DIR / "xgb_dinov2_small.joblib",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions_test.csv",
    # -----------------------------------------
):
    pass


if __name__ == "__main__":
    app()
