# Makefile for Anaerobic Mixing OpenFOAM

.PHONY: help setup install test lint format clean run analyze docker

# Default target
help:
	@echo "Available targets:"
	@echo "  setup      - Set up development environment"
	@echo "  install    - Install package and dependencies"
	@echo "  test       - Run unit tests"
	@echo "  lint       - Run linting checks"
	@echo "  format     - Format code"
	@echo "  clean      - Clean build artifacts"
	@echo "  run        - Run production case"
	@echo "  analyze    - Analyze results"
	@echo "  docker     - Build Docker image"

# Setup development environment
setup:
	pip install --upgrade pip
	pip install uv
	uv pip install -e .[dev]
	pre-commit install

# Install package
install:
	pip install -e .

# Run tests
test:
	pytest -v

test-cov:
	pytest --cov=amx --cov-report=html --cov-report=term

# Linting
lint:
	ruff check .
	black --check .
	isort --check-only .

# Format code
format:
	ruff check --fix .
	black .
	isort .

# Clean artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run production case
run:
	amx run-case --config configs/case_prod.yaml --out runs/prod

# Analyze results
analyze:
	amx analyze-mix --in runs/prod --out data/processed/prod_metrics

# PIV validation
piv:
	amx analyze-piv --config configs/piv_lab.yaml --cfd runs/prod --piv data/raw/piv

# Generate report
report:
	amx build-report \
		--config configs/case_prod.yaml \
		--metrics data/processed/prod_metrics/metrics.json \
		--out docs/report.md

# Docker operations
docker-build:
	docker build -t amx-openfoam docker/

docker-run:
	docker run -it -v $(PWD):/app amx-openfoam

docker-exec:
	docker run -v $(PWD):/app amx-openfoam \
		amx run-case --config configs/case_prod.yaml --out runs/prod

# HPC Singularity
singularity-build:
	singularity build of11.sif docker/apptainer.def

# Documentation
docs:
	mkdocs serve

docs-build:
	mkdocs build

# CI/CD checks
ci: lint test
	@echo "CI checks passed!"

# Quick check before commit
pre-commit: format lint test
	@echo "Ready to commit!"

# Full pipeline
pipeline: clean install test run analyze report
	@echo "Full pipeline completed!"

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
PYTEST := pytest