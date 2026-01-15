# Makefile for Story Factory development tasks

.PHONY: help install test test-unit test-smoke test-integration test-e2e test-all lint format check clean run healthcheck

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

healthcheck:  ## Run system health check
	python scripts/healthcheck.py

install:  ## Install dependencies
	pip install -r requirements.txt

test:  ## Run unit tests (default)
	pytest tests/unit

test-unit:  ## Run unit tests only
	pytest tests/unit

test-smoke:  ## Run smoke tests (quick startup validation)
	pytest tests/smoke

test-integration:  ## Run integration tests
	pytest tests/integration

test-e2e:  ## Run E2E browser tests (requires: playwright install chromium)
	pytest tests/e2e

test-all:  ## Run all tests (unit + smoke + integration, excludes e2e)
	pytest tests/unit tests/smoke tests/integration

test-cov:  ## Run tests with coverage
	pytest --cov=. --cov-report=term --cov-report=html

lint:  ## Run linters (ruff check and format check)
	ruff check .
	ruff format --check .

format:  ## Format code with ruff
	ruff format .
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
