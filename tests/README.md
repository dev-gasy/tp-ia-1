# Tests for Insurance Claim Duration Prediction

This directory contains tests for the Insurance Claim Duration Prediction project.

## Test Structure

The tests are organized in a hierarchical structure:

```
tests/
├── conftest.py            # Common pytest fixtures
├── test_data/             # Test data files
├── unit/                  # Unit tests
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   ├── test_utils.py
│   └── test_kpi_calculation.py
└── integration/           # Integration tests
    ├── test_pipeline_integration.py
    └── test_api.py
```

## Running Tests

You can run tests using the following commands:

### Run all tests

```bash
python -m pytest
```

### Run unit tests only

```bash
python -m pytest tests/unit
```

### Run integration tests only

```bash
python -m pytest tests/integration
```

### Run a specific test file

```bash
python -m pytest tests/unit/test_data_processing.py
```

### Run with coverage

```bash
python -m pytest --cov=scripts --cov-report=html
```

## Using the Makefile

The project includes a Makefile with shortcuts for common test commands:

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run tests with coverage report
make test-coverage

# Clean cache files and test artifacts
make clean
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

- `test_data_dir`: Path to the test data directory
- `sample_claims_data`: Sample DataFrame with claim data
- `sample_statcan_data`: Sample DataFrame with StatCanada data
- `preprocessor`: Sample preprocessing pipeline
- `processed_sample_data`: Sample DataFrame with processed data
- `train_test_data`: Train/test split of sample data

## Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test how components work together
- **API Tests**: Test the FastAPI endpoints

## Skip Slow Tests

Some tests are marked as slow. You can skip them with:

```bash
python -m pytest -m "not slow"
```
