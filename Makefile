# ProductLift developer tasks. `make help` lists everything.
# On Windows use Git Bash / WSL, or copy the underlying command.

.PHONY: help install data features train evaluate experiment serve test test-exercise lint format typecheck check eda clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install with dev dependencies
	pip install -e ".[dev]"

data:  ## Download the Olist dataset into data/raw
	python -m scripts.download_data

features:  ## Build the analytic base table + feature matrix
	python -m scripts.build_features

train:  ## Train the model and register it
	python -m scripts.train

evaluate:  ## Evaluate the current model on the holdout set
	python -m scripts.evaluate

serve:  ## Start the FastAPI scoring service
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

eda:  ## Launch JupyterLab for exploratory analysis
	jupyter lab

test:  ## Run the full test suite (fast; uses fixtures, no real data)
	pytest

test-exercise:  ## Run only the exercise tests — your daily to-do list
	pytest -m exercise

lint:  ## Lint with ruff
	ruff check .

format:  ## Auto-format with ruff
	ruff format .

typecheck:  ## Static type check with mypy
	mypy app evaluation experimentation causal observability

check: lint typecheck test  ## What CI runs

clean:  ## Remove caches and built artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage models artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
