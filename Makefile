# Makefile for Story Factory development tasks

.PHONY: help install test lint format check clean run healthcheck

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

healthcheck:  ## Run system health check
	python healthcheck.py

install:  ## Install dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements-dev.txt

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=. --cov-report=term --cov-report=html

lint:  ## Run linters (ruff and black check)
	ruff check .
	black --check .

format:  ## Format code with black and fix ruff issues
	black .
	ruff check --fix .

check: lint test  ## Run linters and tests

clean:  ## Clean up build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

run:  ## Run the web UI
	python main.py

run-cli:  ## Run in CLI mode
	python main.py --cli
