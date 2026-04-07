# Contributing to TurboQuant

First off, thank you for considering contributing to TurboQuant! It's people like you that make TurboQuant such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to team@anomaly.dev.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed and what you expected**
* **Include screenshots or animated GIFs if helpful**
* **Include your environment details** (OS, Python version, PyTorch version, GPU model)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and explain the expected behavior**
* **Explain why this enhancement would be useful**
* **List some other similar tools if applicable**

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the code style guidelines
* Include thoughtfully-worded, well-structured tests
* Document new code based on the Documentation Style Guide
* End all files with a newline

## Development Setup

### Prerequisites

* Python 3.8 or higher
* PyTorch 1.12.0 or higher
* Git

### Setup Steps

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/turboquant.git
   cd turboquant
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

6. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Style Guidelines

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Consider starting the commit message with an applicable emoji:
    * 🎨 `:art:` when improving the format/structure of the code
    * 🐎 `:racehorse:` when improving performance
    * 🚱 `:non-potable_water:` when plugging memory leaks
    * 📝 `:memo:` when writing docs
    * 🐛 `:bug:` when fixing a bug
    * 🔥 `:fire:` when removing code or files
    * 💚 `:green_heart:` when fixing the CI build
    * ✅ `:white_check_mark:` when adding tests
    * 🔒 `:lock:` when dealing with security
    * ⬆️ `:arrow_up:` when upgrading dependencies
    * ⬇️ `:arrow_down:` when downgrading dependencies

### Python Style Guide

* Follow PEP 8
* Use Black for formatting (line length: 100)
* Use isort for import sorting
* Use type hints for all public functions
* Write docstrings for all public functions and classes

Example:
```python
def compress_keys(keys: torch.Tensor, total_bits: int = 3) -> Dict[str, torch.Tensor]:
    """Compress key tensors using Lloyd-Max quantization.
    
    Args:
        keys: Key tensor of shape (seq_len, num_heads, head_dim)
        total_bits: Number of bits per coordinate (default: 3)
    
    Returns:
        Dictionary containing compressed keys with:
            - indices_packed: Packed quantized indices
            - residuals: Residual vectors for QJL correction
            - norms: L2 norms of keys
    
    Raises:
        ValueError: If total_bits is not in [1, 2, 3, 4]
    """
    pass
```

### Testing Guidelines

* Write unit tests for all new functionality
* Ensure all tests pass before submitting a PR
* Aim for at least 80% code coverage
* Use pytest fixtures for common test setup
* Mark slow tests with `@pytest.mark.slow`
* Mark tests requiring GPU with `@pytest.mark.gpu`

Example test:
```python
import pytest
import torch
from turboquant import TurboQuantEngine


@pytest.fixture
def engine():
    return TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")


def test_compress_keys_basic(engine):
    """Test basic key compression."""
    keys = torch.randn(100, 8, 128)
    compressed = engine.compress_keys(keys)
    
    assert "indices_packed" in compressed
    assert "residuals" in compressed
    assert compressed["indices_packed"].dtype == torch.uint8


def test_compress_keys_invalid_bits():
    """Test that invalid bits raises error."""
    with pytest.raises(ValueError, match="total_bits must be"):
        TurboQuantEngine(head_dim=128, total_bits=5)
```

## Project Structure

```
turboquant/
├── turboquant/
│   ├── core/           # Core compression engine
│   ├── backends/       # vLLM, SGLang integrations
│   ├── models/         # Model-specific implementations
│   └── kernels/        # CUDA kernels
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Jupyter notebooks and scripts
└── scripts/            # Utility scripts
```

## Release Process

1. Update version in `turboquant/__init__.py`
2. Update `CHANGELOG.md`
3. Create a new GitHub release
4. Build and publish to PyPI:
   ```bash
   python -m build
   twine upload dist/*
   ```

## Getting Help

* GitHub Issues: https://github.com/anomaly/turboquant/issues
* GitHub Discussions: https://github.com/anomaly/turboquant/discussions
* Email: team@anomaly.dev

## Recognition

Contributors will be recognized in our README.md file. Thank you for your contributions!
