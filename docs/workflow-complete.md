# Guide de Workflow Complet : DINOv2 → XGBoost Optimisé

Ce guide présente le workflow complet pour entraîner et optimiser un modèle XGBoost sur les features DINOv2.

## Vue d'ensemble

```
Images → DINOv2 → Features CSV → XGBoost → Prédictions
                     ↓
                  Optuna tuning → Meilleurs hyperparamètres
```

## Étape 1 : Préparation des données

### 1.1 Télécharger les données

```bash
make ingest
```

Cela télécharge les images de la compétition Kaggle CSIRO Biomass dans `data/raw/`.

### 1.2 Créer les splits (train/val/test)

```bash
make data
```

Cela crée les splits dans `data/processed/splits/` par différentes stratégies :
- `month/` : splits par mois
- `Species/` : splits par espèce
- `State/` : splits par état

Chaque split contient des sous-dossiers numérotés (0/, 1/, 2/...) pour la cross-validation.

## Étape 2 : Extraction des features DINOv2

### 2.1 Extraire les features (modèle base recommandé)

```bash
# DINOv2 base (meilleure qualité, plus lent)
uv run image2biomass/features.py \
  data/raw/train \
  --output data/processed/features_dinov2_base.csv \
  --model facebook/dinov2-base \
  --batch-size 16

# OU DINOv2 small (plus rapide, qualité correcte)
uv run image2biomass/features.py \
  data/raw/train \
  --output data/processed/features_dinov2_small.csv \
  --model facebook/dinov2-small \
  --batch-size 32
```

**Temps estimé** : 
- DINOv2-small : ~30-60 min sur GPU (M1/M2 Mac ou CUDA)
- DINOv2-base : ~1-2h sur GPU

Le CSV résultant contient :
- Colonne `image_path` : chemin relatif de l'image
- Colonnes `f_0000` à `f_0767` (ou `f_0383` pour small) : embeddings

## Étape 3 : Entraînement baseline

### 3.1 Entraîner un modèle XGBoost avec les paramètres par défaut

```bash
make train \
  MODEL_NAME=xgboost \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  LABELS_PATH=data/processed/splits/month/0/train_split.csv \
  MODEL_OUT=models/xgboost_baseline.pkl
```

### 3.2 Évaluer le modèle baseline

```bash
make evaluate \
  MODEL_NAME=xgboost \
  MODEL_PATH=models/xgboost_baseline.pkl \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  LABELS_PATH=data/processed/splits/month/0/val_split.csv
```

Notez le score R² de référence (baseline).

## Étape 4 : Optimisation des hyperparamètres

### 4.1 Lancer l'optimisation Optuna (recommandé : 100-200 trials)

```bash
make tune \
  TRAIN_SPLIT=data/processed/splits/month/0/train_split.csv \
  VAL_SPLIT=data/processed/splits/month/0/val_split.csv \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  N_TRIALS=150
```

**Temps estimé** : 1-3h selon les données et le matériel

### 4.2 Analyser les résultats

```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger l'historique
df = pd.read_csv("models/xgboost_tuned_optuna_study.csv")

# Top 10 essais
print(df.nsmallest(10, "value")[["number", "value", 
                                   "params_eta", 
                                   "params_max_depth",
                                   "params_n_estimators"]])

# Visualiser l'évolution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df["number"], -df["value"])  # Positive R²
plt.xlabel("Trial")
plt.ylabel("R² score")
plt.title("Optimization Progress")

plt.subplot(1, 2, 2)
plt.scatter(df["params_eta"], -df["value"], alpha=0.5)
plt.xlabel("Learning Rate (eta)")
plt.ylabel("R² score")
plt.xscale("log")
plt.title("Learning Rate vs Performance")

plt.tight_layout()
plt.savefig("reports/figures/tuning_analysis.png")
```

### 4.3 Comparer avec le baseline

```bash
make evaluate \
  MODEL_NAME=xgboost \
  MODEL_PATH=models/xgboost_tuned.pkl \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  LABELS_PATH=data/processed/splits/month/0/val_split.csv
```

**Amélioration attendue** : +2% à +10% de R² selon les données

## Étape 5 : Intégrer les meilleurs hyperparamètres

### 5.1 Mettre à jour la Config dans `xgboost_regressor.py`

Ouvrir `image2biomass/modeling/models/xgboost_regressor.py` et modifier la classe `Config` :

```python
@dataclass(frozen=True)
class Config:
    # Depuis les résultats Optuna (exemple)
    n_estimators: int = 250      # était 100
    learning_rate: float = 0.05  # était 0.1
    max_depth: int = 7           # était 3
    min_child_weight: int = 3    # était 1
    gamma: float = 0.2           # était 0
    subsample: float = 0.85      # était 1.0
    colsample_bytree: float = 0.9  # était 1.0
    objective: str = "reg:squarederror"
    n_jobs: int = -1
    random_state: int = 42
```

## Étape 6 : Cross-validation complète

### 6.1 Entraîner sur tous les folds

```bash
make cross-validate \
  MODEL_NAME=xgboost \
  SPLITS_DIR=data/processed/splits/month \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  --train
```

Cela va :
1. Entraîner un modèle par fold (`xgboost_fold0.pkl`, `xgboost_fold1.pkl`, ...)
2. Évaluer chaque modèle sur son fold de validation
3. Afficher les scores moyens et écart-type

**Résultat attendu** :
```
Cross-validation complete!
Scores per fold: ['0.7234', '0.7189', '0.7301', '0.7156', '0.7267']
Mean R²: 0.7229 ± 0.0052
```

### 6.2 Analyser la stabilité

Si l'écart-type est élevé (>0.02), cela peut indiquer :
- **Overfitting** : réduire `max_depth` ou `n_estimators`
- **Underfitting** : augmenter `n_estimators` ou `learning_rate`
- **Distribution déséquilibrée** : vérifier les splits

## Étape 7 : Prédictions finales

### 7.1 Choisir le meilleur fold ou faire un ensemble

```bash
# Option 1 : utiliser le meilleur modèle d'un fold
make evaluate \
  MODEL_NAME=xgboost \
  MODEL_PATH=models/xgboost_fold2.pkl \
  FEATURES_PATH=data/processed/features_dinov2_base.csv \
  LABELS_PATH=data/raw/test.csv

# Option 2 : moyenner les prédictions de tous les folds (ensemble)
# (nécessite un script custom)
```

### 7.2 Soumettre à Kaggle

```bash
# Générer le fichier de soumission
uv run image2biomass/modeling/predict.py \
  --model-path models/xgboost_fold2.pkl \
  --features-path data/processed/features_dinov2_base.csv \
  --output data/processed/submission.csv

# Soumettre
kaggle competitions submit -c csiro-biomass \
  -f data/processed/submission.csv \
  -m "XGBoost with optimized hyperparameters on DINOv2-base features"
```

## Conseils avancés

### Améliorer les performances

1. **Essayer d'autres stratégies de split**
   ```bash
   make cross-validate SPLITS_DIR=data/processed/splits/Species
   ```

2. **Combiner plusieurs extractions de features**
   ```python
   # Fusionner dinov2_small et dinov2_base
   import polars as pl
   
   df_small = pl.read_csv("data/processed/features_dinov2_small.csv")
   df_base = pl.read_csv("data/processed/features_dinov2_base.csv")
   
   # Renommer les colonnes pour éviter les conflits
   df_small = df_small.rename({f"f_{i:04d}": f"small_{i:04d}" 
                                for i in range(384)})
   
   df_combined = df_small.join(df_base, on="image_path")
   df_combined.write_csv("data/processed/features_combined.csv")
   ```

3. **Tuner avec plusieurs folds**
   ```bash
   # Tuner sur fold 0
   make tune TRAIN_SPLIT=data/processed/splits/month/0/train_split.csv \
             VAL_SPLIT=data/processed/splits/month/0/val_split.csv \
             N_TRIALS=100
   
   # Valider sur fold 1
   make tune TRAIN_SPLIT=data/processed/splits/month/1/train_split.csv \
             VAL_SPLIT=data/processed/splits/month/1/val_split.csv \
             N_TRIALS=50
   ```

4. **Utiliser un ensemble de modèles**
   - Moyenne pondérée des prédictions de tous les folds
   - Stacking avec un meta-modèle (LightGBM, CatBoost)

### Déboguer les problèmes

**Problème : R² négatif ou très faible**
- Vérifier que les features et labels sont bien alignés (même `image_path`)
- Vérifier la distribution des targets (y) : doivent être continues, pas de valeurs manquantes
- Essayer avec un modèle plus simple (sklearn RandomForest)

**Problème : Temps d'entraînement trop long**
- Réduire `n_estimators` temporairement
- Utiliser `n_jobs=-1` pour paralléliser
- Commencer avec moins de trials Optuna (50 au lieu de 150)

**Problème : Optuna plante avec OOM**
- Réduire `batch_size` dans les features
- Limiter les données avec un subset pour le tuning
- Utiliser des machines avec plus de RAM

## Checklist complète

- [ ] Données téléchargées (`make ingest`)
- [ ] Splits créés (`make data`)
- [ ] Features DINOv2 extraites
- [ ] Modèle baseline entraîné et évalué
- [ ] Hyperparamètres optimisés avec Optuna (100+ trials)
- [ ] Config mise à jour avec les meilleurs paramètres
- [ ] Cross-validation complète exécutée
- [ ] Scores analysés et stables (std < 0.02)
- [ ] Prédictions générées et soumises

**Bon entraînement ! 🚀**
