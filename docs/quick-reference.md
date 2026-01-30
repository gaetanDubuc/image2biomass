# Référence Rapide des Commandes

## Commandes Makefile

### Setup & Installation
```bash
make create_environment  # Créer l'environnement virtuel
make requirements        # Installer les dépendances
```

### Data Pipeline
```bash
make ingest             # Télécharger les données Kaggle
make data               # Créer les splits train/val
```

### Training
```bash
# Entraîner un modèle unique
make train \
  MODEL_NAME=xgboost \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  LABELS_PATH=data/processed/splits/month/0/train_split.csv \
  MODEL_OUT=models/my_model.pkl

# Cross-validation sur tous les folds
make cross-validate \
  MODEL_NAME=xgboost \
  SPLITS_DIR=data/processed/splits/month \
  FEATURES_PATH=data/processed/features_dinov2_base.csv
```

### Hyperparameter Tuning
```bash
# Optimisation avec Optuna (défaut: 100 trials)
make tune \
  TRAIN_SPLIT=data/processed/splits/month/0/train_split.csv \
  VAL_SPLIT=data/processed/splits/month/0/val_split.csv \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  N_TRIALS=150

# Résultats sauvegardés dans:
# - models/xgboost_tuned.pkl (modèle optimisé)
# - models/xgboost_tuned_optuna_study.csv (historique)
```

### Evaluation
```bash
make evaluate \
  MODEL_NAME=xgboost \
  MODEL_PATH=models/xgboost_tuned.pkl \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  LABELS_PATH=data/processed/splits/month/0/val_split.csv
```

### Linting & Testing
```bash
make lint               # Vérifier le code (ruff)
make format             # Formater le code
make test               # Lancer les tests
```

## Commandes Python Directes

### Feature Extraction
```bash
# DINOv2 small (rapide)
uv run image2biomass/features.py \
  data/raw/train \
  --output data/processed/features_dinov2_small.csv \
  --model facebook/dinov2-small \
  --batch-size 32

# DINOv2 base (meilleure qualité)
uv run image2biomass/features.py \
  data/raw/train \
  --output data/processed/features_dinov2_base.csv \
  --model facebook/dinov2-base \
  --batch-size 16
```

### Training (CLI direct)
```bash
uv run image2biomass/modeling/train.py \
  --features-path data/processed/features_dinov2_base.csv \
  --labels-path data/processed/splits/month/0/train_split.csv \
  --model-name xgboost \
  --model-out models/my_model.pkl
```

### Hyperparameter Tuning (CLI direct)
```bash
uv run image2biomass/modeling/tune_hyperparameters.py \
  data/processed/splits/month/0/train_split.csv \
  data/processed/splits/month/0/val_split.csv \
  --features-path data/processed/features_dinov2_base.csv \
  --n-trials 200 \
  --output-name xgboost_optimized \
  --study-name my_optimization_study
```

### Evaluation (CLI direct)
```bash
uv run image2biomass/modeling/evaluate.py \
  --features-path data/processed/features_dinov2_base.csv \
  --labels-path data/processed/splits/month/0/val_split.csv \
  --model-name xgboost \
  --model-path models/xgboost_tuned.pkl \
  --save-predictions data/processed/my_predictions.csv
```

### Cross-Validation (CLI direct)
```bash
uv run image2biomass/modeling/cross_validate.py \
  data/processed/splits/month \
  --features-path data/processed/features_dinov2_base.csv \
  --model-name xgboost \
  --train
```

### Prediction
```bash
uv run image2biomass/modeling/predict.py \
  --model-path models/xgboost_tuned.pkl \
  --features-path data/processed/features_dinov2_base.csv \
  --output data/processed/submission.csv
```

## Paramètres Variables Makefile

Tous les paramètres peuvent être surchargés :

```bash
FEATURES_PATH      # Chemin vers le CSV de features
LABELS_PATH        # Chemin vers le CSV de labels
MODEL_NAME         # Nom du modèle (xgboost, sklearn_gboost, etc.)
MODEL_OUT          # Chemin de sortie du modèle
MODEL_PATH         # Chemin du modèle à charger
PREDICTIONS_OUT    # Chemin de sortie des prédictions
SPLITS_DIR         # Répertoire des splits pour CV
TRAIN_SPLIT        # Chemin du split d'entraînement pour tuning
VAL_SPLIT          # Chemin du split de validation pour tuning
N_TRIALS           # Nombre d'essais Optuna (défaut: 100)
```

## Workflow Recommandé

```bash
# 1. Setup
make create_environment && source .venv/bin/activate
make requirements

# 2. Data
make ingest
make data

# 3. Features
uv run image2biomass/features.py data/raw/train \
  --output data/processed/features_dinov2_base.csv \
  --model facebook/dinov2-base

# 4. Baseline
make train MODEL_NAME=xgboost \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  LABELS_PATH=data/processed/splits/month/0/train_split.csv \
  MODEL_OUT=models/xgboost_baseline.pkl

make evaluate MODEL_PATH=models/xgboost_baseline.pkl \
  LABELS_PATH=data/processed/splits/month/0/val_split.csv

# 5. Optimization
make tune \
  TRAIN_SPLIT=data/processed/splits/month/0/train_split.csv \
  VAL_SPLIT=data/processed/splits/month/0/val_split.csv \
  N_TRIALS=150

# 6. Update Config dans xgboost_regressor.py avec les meilleurs params

# 7. Cross-validation
make cross-validate \
  SPLITS_DIR=data/processed/splits/month \
  MODEL_NAME=xgboost

# 8. Predictions & Submit
uv run image2biomass/modeling/predict.py \
  --model-path models/xgboost_fold2.pkl \
  --features-path data/processed/features_dinov2_base.csv \
  --output data/processed/submission.csv

kaggle competitions submit -c csiro-biomass \
  -f data/processed/submission.csv \
  -m "XGBoost optimized with Optuna"
```

## Aide

```bash
# Aide générale
make help

# Aide sur un CLI spécifique
uv run image2biomass/modeling/tune_hyperparameters.py --help
uv run image2biomass/features.py --help
uv run image2biomass/modeling/cross_validate.py --help
```

## Documentation Complète

- **[Workflow Complet](workflow-complete.md)** - Guide détaillé étape par étape
- **[Hyperparameter Tuning](hyperparameter-tuning.md)** - Guide d'optimisation avec Optuna
- **[README Principal](../README.md)** - Vue d'ensemble du projet
