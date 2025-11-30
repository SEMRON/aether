SHELL := /bin/bash
export PATH := $(HOME)/.local/bin:$(PATH)

.PHONY: all install install-cpu install-cuda install-rocm install-no-uv clean docs setup

# Helper to run commands in venv
IN_VENV = . .venv/bin/activate &&

# Setup: Install uv (if missing) and create venv (if missing)
setup:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv --python=3.10; \
	fi

# Default target
all: setup
	$(IN_VENV) uv pip install "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0"
	$(IN_VENV) uv pip install -e .

install: setup
	$(IN_VENV) uv pip install "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0"
	$(IN_VENV) uv pip install -e .

# Install CPU-only version
install-cpu: setup
	$(IN_VENV) uv pip install "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0" --index-url https://download.pytorch.org/whl/cpu
	$(IN_VENV) uv pip install -e .

# Install CUDA version
install-cuda: setup
	$(IN_VENV) uv pip install "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0" --index-url https://download.pytorch.org/whl/cu128
	$(IN_VENV) uv pip install -e .

# Install ROCM version
install-rocm: setup
	$(IN_VENV) uv pip install "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0" --index-url https://download.pytorch.org/whl/rocm6.4
	$(IN_VENV) uv pip install -e .

# Install without uv (uses system pip or active environment)
install-no-uv:
	pip install "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0"
	pip install -e .

# Build documentation
docs: setup
	$(IN_VENV) uv pip install -e .[docs]
	$(IN_VENV) cd docs && make html

# Clean up
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
