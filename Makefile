.PHONY: clean test lint test-unit test-integration test-coverage help

.DEFAULT_GOAL := help

## Variables
PYTHON := python
PYTEST := pytest
PIP := pip
COVERAGE := coverage

## Help
help:
	@echo "Available commands:"
	@echo "  clean            - Clean cache files and test artifacts"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-coverage    - Run tests with coverage report"
	@echo "  lint             - Run linting checks"
	@echo "  format           - Format code using black"
	@echo "  setup            - Install development dependencies"

## Clean cache files and test artifacts
clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

## Run all tests
test:
	$(PYTEST)

## Run unit tests only
test-unit:
	$(PYTEST) tests/unit -v

## Run integration tests only
test-integration:
	$(PYTEST) tests/integration -v

## Run tests with coverage report
test-coverage:
	$(PYTEST) --cov=scripts --cov-report=html --cov-report=term

## Run linting checks
lint:
	flake8 scripts tests
	mypy scripts tests

## Format code using black
format:
	black scripts tests

## Setup development environment
setup:
	$(PIP) install -e ".[dev,test]" 