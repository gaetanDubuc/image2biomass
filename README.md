# image2biomass

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         image2biomass and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── image2biomass   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes image2biomass a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models
    │   ├── train.py            <- Code to train models
    │   ├── evaluate.py         <- Code to evaluate trained models
    │   ├── cross_validate.py   <- Code to run cross-validation
    │   └── tune_hyperparameters.py <- Code to optimize XGBoost hyperparameters with Optuna
    │
    └── plots.py                <- Code to create visualizations
```

## Quick Start

### 1. Environment Setup

```bash
# Create environment and install dependencies
make create_environment
source .venv/bin/activate  # Unix/macOS
make requirements
```

### 2. Data Pipeline

```bash
# Download competition data
make ingest

# Process data and create train/test splits
make data

# Extract DINOv2 features
uv run image2biomass/features.py \
  data/raw/train \
  --output data/processed/features_dinov2_small.csv
```

### 3. Model Training

```bash
# Train a single model
make train \
  MODEL_NAME=xgboost \
  FEATURES_PATH=data/processed/features_dinov2_small.csv

# Cross-validation on multiple folds
make cross-validate \
  MODEL_NAME=xgboost \
  SPLITS_DIR=data/processed/splits/month

# Hyperparameter tuning with Optuna (see docs/hyperparameter-tuning.md)
make tune \
  TRAIN_SPLIT=data/processed/splits/month/0/train_split.csv \
  VAL_SPLIT=data/processed/splits/month/0/val_split.csv \
  N_TRIALS=100
```

### 4. Evaluation & Prediction

```bash
# Evaluate a trained model
make evaluate \
  MODEL_PATH=models/xgboost_tuned.pkl \
  LABELS_PATH=data/processed/splits/month/0/val_split.csv

# Generate predictions
uv run image2biomass/modeling/predict.py \
  --model-path models/xgboost_tuned.pkl \
  --features-path data/processed/features_dinov2_small.csv \
  --output data/processed/predictions.csv
```

## Documentation

- **[Hyperparameter Tuning Guide](docs/hyperparameter-tuning.md)** - Detailed guide for XGBoost optimization with Optuna

--------

