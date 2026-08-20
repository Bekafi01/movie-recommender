.PHONY: setup install install-dev lint format typecheck test clean preprocess train evaluate api app

PYTHON ?= python

setup: install-dev

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest tests -v --cov=src/recsys --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache build dist *.egg-info

preprocess:
	$(PYTHON) main.py data preprocess

train:
	$(PYTHON) main.py train --all-models

evaluate:
	$(PYTHON) main.py evaluate --top-k 10

api:
	$(PYTHON) -m uvicorn src.recsys.serving.api:app --reload --port 8000

app:
	$(PYTHON) -m streamlit run app.py
