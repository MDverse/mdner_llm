# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Configuration
VENV_NAME := .venv
TEST_PATH := $(shell [ -d tests ] && echo tests/ || echo "")
SRC_PATH := src/

# Command shortcuts
UV := uv
RUFF := uv run ruff
TY := uv run ty
PYTEST := uv run pytest


.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
.PHONY: venv
venv: ## Create a virtual environment using uv
	@echo "$(BLUE)Creating virtual environment...$(RESET)"
	$(UV) sync
	@echo "$(GREEN)Virtual environment created. Activate with: source $(VENV_NAME)/bin/activate$(RESET)"

.PHONY: install
install: ## Install package dependencies
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	$(UV) pip install -e .
	@echo "$(GREEN)Dependencies installed$(RESET)"

.PHONY: install-dev
install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(UV) pip install -e ".[test,dev]"
	@echo "$(GREEN)Development dependencies installed$(RESET)"

.PHONY: deps
deps: ## Update and sync dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(UV) lock
	$(UV) sync
	@echo "$(GREEN)Dependencies updated$(RESET)"

# Package management
.PHONY: update
update: ## Update all dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(UV) lock --upgrade
	$(UV) sync
	@echo "$(GREEN)Dependencies updated$(RESET)"

.PHONY: build
build: clean ## Build the package
	@echo "$(BLUE)Building package...$(RESET)"
	$(UV) build
	@echo "$(GREEN)Package built successfully$(RESET)"

# Cleaning tasks
.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)Build artifacts cleaned$(RESET)"

# Testing tasks
.PHONY: test
test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(RESET)"
	$(PYTEST) $(TEST_PATH)
	@echo "$(GREEN)Tests completed$(RESET)"

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)Running tests with verbose output...$(RESET)"
	$(PYTEST) $(TEST_PATH) -v -s --tb=long
	@echo "$(GREEN)Tests completed$(RESET)"

# Code quality tasks
.PHONY: lint-check
lint-check: ## Check if code is properly formatted
	@echo "$(BLUE)Running linter...$(RESET)"
	$(RUFF) check $(SRC_PATH) $(TEST_PATH)
	$(TY) check $(SRC_PATH)
	@echo "$(GREEN)Linting completed$(RESET)"

.PHONY: lint-format
lint-format: ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(RUFF) format $(SRC_PATH) $(TEST_PATH)
	$(RUFF) check $(SRC_PATH) $(TEST_PATH) --fix
	@echo "$(GREEN)Code formatted$(RESET)"
