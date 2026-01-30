#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = image2biomass
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = uv run

# Training parameters (override with make train FEATURES_PATH=... etc.)
FEATURES_PATH ?= data/processed/features_dinov2_small.csv
LABELS_PATH ?= data/processed/train.csv
MODEL_NAME ?= sklearn_gboost
MODEL_OUT ?= models/model.pkl
MODEL_PATH ?= models/model.pkl
PREDICTIONS_OUT ?= data/processed/predictions.csv
SPLITS_DIR ?= data/processed/splits/month
TRAIN_SPLIT ?= data/processed/splits/month/0/train_split.csv
VAL_SPLIT ?= data/processed/splits/month/0/val_split.csv
N_TRIALS ?= 100

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make ingest
.PHONY: ingest
ingest: requirements
	uv run kaggle competitions download -c csiro-biomass -p data/raw
	unzip -o "data/raw/csiro-biomass.zip" -d "data/raw"

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) image2biomass/dataset.py

## Train a model
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) image2biomass/modeling/train.py \
		--features-path $(FEATURES_PATH) \
		--labels-path $(LABELS_PATH) \
		--model-name $(MODEL_NAME) \
		--model-out $(MODEL_OUT)

## Evaluate a trained model
.PHONY: evaluate
evaluate: requirements
	$(PYTHON_INTERPRETER) image2biomass/modeling/evaluate.py \
		--features-path $(FEATURES_PATH) \
		--labels-path $(LABELS_PATH) \
		--model-name $(MODEL_NAME) \
		--model-path $(MODEL_PATH) \

## Cross-validate on multiple folds (train + eval)
.PHONY: cross-validate
cross-validate: requirements
	$(PYTHON_INTERPRETER) image2biomass/modeling/cross_validate.py \
		$(SPLITS_DIR) \
		--features-path $(FEATURES_PATH) \
		--model-name $(MODEL_NAME) \
		--train \

## Tune XGBoost hyperparameters with Optuna
.PHONY: tune
tune: requirements
	$(PYTHON_INTERPRETER) image2biomass/modeling/tune_hyperparameters.py \
		$(TRAIN_SPLIT) \
		$(VAL_SPLIT) \
		--features-path $(FEATURES_PATH) \
		--n-trials $(N_TRIALS) \
		--output-name xgboost_tuned

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
