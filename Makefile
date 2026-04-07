.PHONY: help install install-dev test test-cov lint format clean build publish

help:
	@echo "TurboQuant - Makefile Commands"
	@echo ""
	@echo "Installation:"
	@echo "  install        Install the package"
	@echo "  install-dev    Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-cov       Run tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run all linters"
	@echo "  format         Format code with black and isort"
	@echo "  typecheck      Run mypy type checking"
	@echo ""
	@echo "Building:"
	@echo "  build          Build the package"
	@echo "  publish        Publish to PyPI"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Remove build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=turboquant --cov-report=html --cov-report=term

lint:
	ruff check turboquant/ tests/
	black --check turboquant/ tests/
	isort --check-only turboquant/ tests/

format:
	black turboquant/ tests/
	isort turboquant/ tests/

typecheck:
	mypy turboquant/

build: clean
	python -m build

publish: build
	twine upload dist/*

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
