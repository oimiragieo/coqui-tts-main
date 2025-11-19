# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation structure with organized directories:
  - `docs/architecture/` for architectural documentation
  - `docs/development/` for development guides and roadmaps
  - `docs/audit/` for security audit reports
- Modern CONTRIBUTING.md guide for contributors
- CHANGELOG.md for tracking project changes
- Documentation index (docs/DOCUMENTATION_INDEX.md) for easy navigation
- PyTorch 2.0+ optimization support with `torch.compile()` utilities
- Performance benchmarking infrastructure
- Comprehensive type hints for public APIs

### Changed
- Reorganized documentation files from root to organized subdirectories
- Updated README.md to reflect modern project state
- Migrated testing framework from nose2 to pytest
- Modernized development tooling (ruff, mypy, updated pre-commit hooks)
- Updated dependencies for security and compatibility

### Fixed
- Security vulnerabilities in numpy and numba dependencies
- Path traversal vulnerability in server endpoint
- Improved error handling throughout codebase

### Documentation
- Created Quick Reference guide for developers
- Created Architectural Overview with detailed codebase analysis
- Created Modernization Roadmap for future improvements
- Created Migration Guide for upgrading
- Added comprehensive docstrings coverage (97.8%)

## [0.22.0] - 2023-XX-XX

### Added
- XTTS v2 with 16 languages and improved performance
- XTTS fine-tuning code and example recipes
- XTTS streaming support with <200ms latency
- Bark model support for unconstrained voice cloning
- Fairseq integration for 1100+ language support
- Tortoise model support with faster inference

### Notable Models
- **XTTS**: Production TTS with 13+ languages, voice cloning
- **Bark**: Emotional speech synthesis with non-speech sounds
- **VITS**: Fast end-to-end synthesis
- **YourTTS**: Multilingual voice cloning
- **Tortoise**: High-quality synthesis

### Infrastructure
- 14 CI/CD workflows for comprehensive testing
- Docker support for containerized deployment
- Flask-based REST API server
- 243 test functions across 14 categories
- Support for 30+ dataset formats

## Previous Versions

For changes before version 0.22.0, please refer to the [GitHub Releases](https://github.com/coqui-ai/TTS/releases) page.

---

## Version Notes

### Versioning Strategy

We use [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backward compatible manner
- **PATCH** version for backward compatible bug fixes

### Release Process

1. Update CHANGELOG.md with changes
2. Update version in `pyproject.toml` and `setup.py`
3. Create git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
4. Push tag: `git push origin vX.Y.Z`
5. Create GitHub release with notes
6. Publish to PyPI: `python -m build && python -m twine upload dist/*`

### Types of Changes

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

---

**Note**: This CHANGELOG tracks significant changes. For detailed commit history, see the [GitHub commits](https://github.com/coqui-ai/TTS/commits/).

**Last Updated**: November 19, 2025
