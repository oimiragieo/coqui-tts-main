# Contributing to Coqui TTS

Thank you for your interest in contributing to Coqui TTS! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.9, 3.10, or 3.11
- Git
- Basic understanding of PyTorch and deep learning
- (Recommended) CUDA-capable GPU for training

### Quick Links

- **Main Repository**: https://github.com/coqui-ai/TTS
- **Documentation**: https://tts.readthedocs.io/
- **Discord Community**: https://discord.gg/5eXr5seRrv
- **Issue Tracker**: https://github.com/coqui-ai/TTS/issues

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/TTS.git
cd TTS

# Add upstream remote
git remote add upstream https://github.com/coqui-ai/TTS.git
```

### 2. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[all,dev,notebooks]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run a quick test
pytest tests/ -k "test_load_tts_model" -v

# Check code formatting
pre-commit run --all-files
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes** - Fix issues and bugs
2. **New Features** - Add new models, datasets, or functionality
3. **Documentation** - Improve docs, add examples, write tutorials
4. **Performance** - Optimize code, reduce memory usage, speed up inference
5. **Tests** - Add test coverage, improve test infrastructure
6. **Code Quality** - Refactoring, type hints, better error handling

### Finding Work

- Check [Good First Issues](https://github.com/coqui-ai/TTS/labels/good%20first%20issue) for beginner-friendly tasks
- Look at the [Modernization Roadmap](docs/development/MODERNIZATION_ROADMAP.md) for larger initiatives
- Browse [open issues](https://github.com/coqui-ai/TTS/issues) for bugs and feature requests
- Join [Discord](https://discord.gg/5eXr5seRrv) to discuss new ideas

### Before You Start

1. **Check existing issues** - Make sure someone isn't already working on it
2. **Open an issue** - For major changes, discuss your approach first
3. **Create a branch** - Use a descriptive name: `fix/issue-123` or `feature/new-vocoder`

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Good
def synthesize_speech(
    text: str,
    speaker_id: Optional[int] = None,
    language: str = "en",
) -> np.ndarray:
    """Synthesize speech from text.

    Args:
        text: Input text to synthesize
        speaker_id: Optional speaker identifier for multi-speaker models
        language: Language code (default: "en")

    Returns:
        Audio waveform as numpy array
    """
    # Implementation
    pass
```

### Automated Formatting

We use automated tools to enforce code style:

- **black** - Code formatting
- **isort** - Import sorting
- **ruff** - Fast linting
- **mypy** - Type checking

These run automatically via pre-commit hooks:

```bash
# Manual run
black TTS/
isort TTS/
ruff check TTS/ --fix
mypy TTS/api.py
```

### Type Hints

All public APIs should have comprehensive type hints:

```python
from typing import Optional, List, Dict, Any
import numpy.typing as npt

# Required for public methods
def load_model(
    model_path: str,
    config_path: Optional[str] = None,
    use_cuda: bool = False,
) -> Dict[str, Any]:
    ...

# Return types are mandatory
def tts(self, text: str) -> npt.NDArray[np.float32]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(config: TrainingConfig, output_dir: str) -> None:
    """Train a TTS model from scratch.

    This function handles the complete training pipeline including
    dataset preparation, model initialization, and checkpoint saving.

    Args:
        config: Training configuration object
        output_dir: Directory to save checkpoints and logs

    Raises:
        ValueError: If config is invalid
        RuntimeError: If CUDA is required but not available

    Example:
        >>> config = load_config("config.json")
        >>> train_model(config, "./output")
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/tts_tests/test_vits.py

# Run with coverage
pytest --cov=TTS --cov-report=html tests/

# Run in parallel (faster)
pytest -n auto tests/

# Run only fast tests (skip slow integration tests)
pytest -m "not slow" tests/
```

### Writing Tests

- Place tests in `tests/` with clear names: `test_<functionality>.py`
- Use descriptive test function names: `test_vits_inference_with_cuda`
- Test both success and failure cases
- Mock expensive operations (model loading, GPU operations)

Example test:

```python
import pytest
from TTS.api import TTS

def test_tts_inference_cpu():
    """Test TTS inference on CPU."""
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    wav = tts.tts("Hello world")

    assert wav is not None
    assert len(wav) > 0
    assert wav.dtype == np.float32

def test_tts_invalid_model():
    """Test TTS raises error for invalid model."""
    with pytest.raises(ValueError):
        TTS("invalid_model_name")
```

### Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.slow
def test_xtts_training():
    """Slow test for XTTS training."""
    pass

@pytest.mark.require_cuda
def test_gpu_inference():
    """Test requiring CUDA."""
    pass
```

## Documentation

### Types of Documentation

1. **Code Docstrings** - Document all public APIs
2. **README Updates** - Keep the main README current
3. **Sphinx Docs** - User-facing documentation in `docs/source/`
4. **Examples** - Jupyter notebooks in `notebooks/`
5. **Architecture Docs** - Technical documentation in `docs/architecture/`

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build Sphinx docs
cd docs
make html

# View docs
open build/html/index.html  # macOS
# or
xdg-open build/html/index.html  # Linux
```

### Documentation Guidelines

- Use clear, concise language
- Include code examples
- Add type hints to all code samples
- Link to related documentation
- Keep docs in sync with code changes

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Create a feature branch
git checkout -b feature/my-awesome-feature

# Make your changes
# ... edit files ...

# Run tests
pytest tests/

# Run code quality checks
pre-commit run --all-files

# Commit your changes
git add .
git commit -m "feat: add awesome new feature"
```

### 2. Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(models): add new VITS-2 model implementation
fix(api): resolve memory leak in streaming synthesis
docs(readme): update installation instructions for Python 3.11
perf(inference): optimize attention mechanism with torch.compile
```

### 3. Push and Create PR

```bash
# Push to your fork
git push origin feature/my-awesome-feature

# Open a Pull Request on GitHub
# - Use a clear, descriptive title
# - Reference related issues (Fixes #123)
# - Describe what changed and why
# - Include testing details
# - Add screenshots/examples if applicable
```

### 4. PR Review Process

1. **Automated Checks** - CI/CD must pass (tests, linting, type checking)
2. **Code Review** - Maintainers will review your code
3. **Feedback** - Address review comments
4. **Approval** - PR must be approved before merging
5. **Merge** - Maintainers will merge approved PRs

### PR Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123
Related to #456

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Added new tests
- [ ] All tests pass locally
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Community

### Communication Channels

- **GitHub Discussions**: General questions and discussions
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat and community support
- **Twitter**: Follow [@coqui_ai](https://twitter.com/coqui_ai) for updates

### Getting Help

1. **Check documentation** - Read [docs](https://tts.readthedocs.io/)
2. **Search issues** - Someone may have asked before
3. **Ask in Discord** - Community is friendly and helpful
4. **Open an issue** - For bugs or specific questions

### Being a Good Community Member

- Be respectful and welcoming
- Help others when you can
- Provide constructive feedback
- Credit others' contributions
- Follow the Code of Conduct

## Additional Resources

### Project Documentation

- [Quick Reference](docs/architecture/QUICK_REFERENCE.md) - Fast overview
- [Architecture Overview](docs/architecture/ARCHITECTURAL_OVERVIEW.md) - Deep dive
- [Modernization Roadmap](docs/development/MODERNIZATION_ROADMAP.md) - Future plans
- [Migration Guide](docs/development/MIGRATION_GUIDE.md) - Upgrade guide

### Learning Resources

- [TTS Papers](https://github.com/erogol/TTS-papers) - Research papers
- [Tutorial for Beginners](docs/source/tutorial_for_nervous_beginners.md) - Start here
- [Training Guide](docs/source/training_a_model.md) - How to train models
- [Model Implementations](docs/source/implementing_a_new_model.md) - Add new models

### Tools and Technologies

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NumPy](https://numpy.org/) - Numerical computing
- [pytest](https://docs.pytest.org/) - Testing framework
- [Sphinx](https://www.sphinx-doc.org/) - Documentation generator
- [black](https://black.readthedocs.io/) - Code formatter
- [mypy](https://mypy.readthedocs.io/) - Type checker

## Questions?

If you have questions about contributing, please:

1. Check this document thoroughly
2. Read the [FAQ](docs/source/faq.md)
3. Search [GitHub Discussions](https://github.com/coqui-ai/TTS/discussions)
4. Ask in [Discord](https://discord.gg/5eXr5seRrv)
5. Open a [GitHub Issue](https://github.com/coqui-ai/TTS/issues) with the "question" label

---

Thank you for contributing to Coqui TTS! Your efforts help make high-quality text-to-speech accessible to everyone.

**Last Updated**: November 19, 2025
