.PHONY: help install install-dev test lint format clean

help:
	@echo "Available commands:"
	@echo "  make install       Install package dependencies"
	@echo "  make install-dev   Install package + dev dependencies"
	@echo "  make test          Run unit tests"
	@echo "  make lint          Run linters (flake8, mypy)"
	@echo "  make format        Format code (black, isort)"
	@echo "  make clean         Clean up temporary files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics
	mypy . --ignore-missing-imports --no-strict-optional

format:
	black . --line-length 120
	isort . --profile black --line-length 120

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	@echo "Cleaned up temporary files."
