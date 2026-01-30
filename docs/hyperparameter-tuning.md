# Hyperparameter Tuning

## Overview

Le module `tune_hyperparameters.py` permet d'optimiser les hyperparamètres du modèle XGBoost en utilisant **Optuna**, une bibliothèque d'optimisation bayésienne.

## Fonctionnalités

- **Optimisation automatique** : teste différentes combinaisons d'hyperparamètres
- **Early stopping** : arrête l'entraînement quand la performance n'augmente plus
- **Sauvegarde des résultats** : enregistre le modèle optimisé et l'historique des essais
- **Validation cross-fold** : utilise les splits train/val existants

## Hyperparamètres optimisés

Le script optimise les paramètres suivants :
- `eta` (learning_rate) : 0.01 à 0.3 (log scale)
- `max_depth` : 3 à 10
- `min_child_weight` : 1 à 10
- `gamma` : 0.0 à 5.0
- `subsample` : 0.6 à 1.0
- `colsample_bytree` : 0.6 à 1.0
- `lambda` (L2 regularization) : 0.1 à 10.0 (log scale)
- `alpha` (L1 regularization) : 0.0 à 10.0
- `n_estimators` : 50 à 500

## Usage

### Via Makefile (recommandé)

```bash
# Utiliser les valeurs par défaut (fold 0 du split month, 100 trials)
make tune

# Personnaliser les paramètres
make tune \
  TRAIN_SPLIT=data/processed/splits/Species/0/train_split.csv \
  VAL_SPLIT=data/processed/splits/Species/0/val_split.csv \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  N_TRIALS=200
```

### Directement avec Python

```bash
uv run image2biomass/modeling/tune_hyperparameters.py \
  data/processed/splits/month/0/train_split.csv \
  data/processed/splits/month/0/val_split.csv \
  --features-path data/processed/features_dinov2_small.csv \
  --n-trials 100 \
  --output-name xgboost_tuned \
  --study-name xgboost_optimization
```

## Paramètres CLI

- `train_labels` (requis) : chemin vers le CSV de labels d'entraînement
- `val_labels` (requis) : chemin vers le CSV de labels de validation
- `--features-path` : chemin vers le CSV de features (défaut: `features_dinov2_small.csv`)
- `--n-trials` : nombre d'essais Optuna (défaut: 100)
- `--output-name` : nom du modèle sauvegardé (défaut: `xgboost_tuned`)
- `--study-name` : nom de l'étude Optuna (défaut: `xgboost_optimization`)

## Sorties

Le script génère deux fichiers dans `models/` :
1. **`{output_name}.pkl`** : modèle XGBoost optimisé (format natif XGBoost)
2. **`{output_name}_optuna_study.csv`** : historique complet des essais avec tous les hyperparamètres testés et scores

## Exemple de workflow complet

```bash
# 1. Extraire les features DINOv2
make features

# 2. Créer les splits
uv run image2biomass/dataset.py

# 3. Optimiser les hyperparamètres sur le fold 0
make tune \
  TRAIN_SPLIT=data/processed/splits/month/0/train_split.csv \
  VAL_SPLIT=data/processed/splits/month/0/val_split.csv \
  N_TRIALS=200

# 4. Évaluer le modèle optimisé
make evaluate \
  MODEL_NAME=xgboost \
  MODEL_PATH=models/xgboost_tuned.pkl \
  LABELS_PATH=data/processed/splits/month/0/val_split.csv

# 5. Cross-valider avec les hyperparamètres optimaux (modifier Config dans xgboost_regressor.py)
make cross-validate \
  MODEL_NAME=xgboost \
  SPLITS_DIR=data/processed/splits/month
```

## Conseils d'utilisation

### Nombre de trials

- **Quick test** : 20-50 trials (~5-15 min)
- **Standard** : 100-200 trials (~30-60 min)
- **Recherche exhaustive** : 500+ trials (plusieurs heures)

### Stratégie d'optimisation

1. **Phase 1** : lancer 100 trials avec les bornes larges (défaut)
2. **Phase 2** : analyser les résultats dans le CSV
3. **Phase 3** : affiner les bornes et relancer avec plus de trials
4. **Phase 4** : valider avec cross-validation complète

### Analyse des résultats

```python
import pandas as pd

# Charger l'historique des essais
df = pd.read_csv("models/xgboost_tuned_optuna_study.csv")

# Top 10 meilleurs essais
top_trials = df.nsmallest(10, "value")  # value = negative R²
print(top_trials[["number", "value", "params_eta", "params_max_depth", "params_n_estimators"]])

# Distribution des hyperparamètres
import matplotlib.pyplot as plt
df.hist(column=["params_eta", "params_max_depth", "params_subsample"], bins=20)
plt.tight_layout()
plt.savefig("reports/figures/optuna_distributions.png")
```

## Intégration avec cross-validation

Une fois les meilleurs hyperparamètres identifiés, les intégrer dans `Config` de `xgboost_regressor.py` :

```python
@dataclass(frozen=True)
class Config:
    n_estimators: int = 300  # depuis le tuning
    learning_rate: float = 0.05  # depuis le tuning
    max_depth: int = 7  # depuis le tuning
    # ... autres paramètres optimaux
```

Puis lancer la cross-validation complète :

```bash
make cross-validate MODEL_NAME=xgboost SPLITS_DIR=data/processed/splits/month
```
