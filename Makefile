.PHONY: setup install install-dev lint format test preprocess train benchmark serve ui docker-up docker-down clean help

PYTHON ?= python

help:
	@echo "======================================================================="
	@echo "CineFlow AI - Multi-Paradigm Recommendation Engine Automation"
	@echo "======================================================================="
	@echo "make install      - Install production dependencies"
	@echo "make install-dev  - Install development & test dependencies"
	@echo "make lint         - Run ruff static code analysis"
	@echo "make format       - Auto-format code with ruff"
	@echo "make test         - Run full pytest test suite with coverage"
	@echo "make preprocess   - Run data cleaning, ID bridge & SQLite pipeline"
	@echo "make train        - Train all 6 multi-paradigm recommendation models"
	@echo "make benchmark    - Run scientific offline evaluation benchmark suite"
	@echo "make serve        - Launch FastAPI production REST microservice (port 8000)"
	@echo "make ui           - Launch Streamlit Cinema Experience UI (port 8501)"
	@echo "make docker-up    - Build and launch Docker Compose services"
	@echo "make docker-down  - Stop all running Docker containers"
	@echo "make clean        - Remove Python bytecode and cache directories"
	@echo "======================================================================="

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check src tests main.py

format:
	$(PYTHON) -m ruff check --fix src tests main.py
	$(PYTHON) -m ruff format src tests main.py

test:
	$(PYTHON) -m pytest tests -v --cov=src/recsys --cov-report=term-missing

preprocess:
	$(PYTHON) main.py data preprocess

train:
	$(PYTHON) main.py train all

benchmark:
	$(PYTHON) main.py eval benchmark --top-k 10

serve:
	$(PYTHON) main.py serve --host 127.0.0.1 --port 8000

ui:
	$(PYTHON) main.py ui --port 8501

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache build dist *.egg-info
