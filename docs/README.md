# Coqui TTS Documentation

Welcome to the Coqui TTS documentation! This directory contains comprehensive guides, references, and resources for using and developing with Coqui TTS.

---

## =€ Quick Start

**New to Coqui TTS?** Start here:

1. **[Getting Started Guide](GETTING_STARTED.md)** P - Complete guide from installation to your first TTS
2. **[Quick Reference](architecture/QUICK_REFERENCE.md)** - Fast overview for developers
3. **[Documentation Index](DOCUMENTATION_INDEX.md)** - Navigate all documentation

---

## =Ú Documentation Structure

### For Users

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Step-by-step installation, usage, and troubleshooting
- **[source/](source/)** - User-facing Sphinx documentation
  - [Installation](source/installation.md)
  - [Inference Guide](source/inference.md)
  - [Training Guide](source/training_a_model.md)
  - [FAQ](source/faq.md)
  - [Model-specific docs](source/models/)

### For Developers

- **[architecture/](architecture/)** - Architectural documentation
  - [QUICK_REFERENCE.md](architecture/QUICK_REFERENCE.md) - Fast developer overview
  - [ARCHITECTURAL_OVERVIEW.md](architecture/ARCHITECTURAL_OVERVIEW.md) - Deep dive into codebase

- **[development/](development/)** - Development guides
  - [TTS_PACKAGE_GUIDE.md](development/TTS_PACKAGE_GUIDE.md) - API reference and workflows
  - [MIGRATION_GUIDE.md](development/MIGRATION_GUIDE.md) - Upgrading versions
  - [MODERNIZATION_ROADMAP.md](development/MODERNIZATION_ROADMAP.md) - Future improvements

### For Contributors

- **[GOVERNANCE.md](GOVERNANCE.md)** - Project governance and code ownership
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - How to contribute to the project

### Model Documentation

- **[models/](models/)** - Model-specific documentation
  - [voicecraft_x.md](models/voicecraft_x.md) - VoiceCraft-X guide
  - [voicecraft_x_implementation.md](models/voicecraft_x_implementation.md) - Technical details

### Security & Audits

- **[audit/](audit/)** - Security audit reports
  - [EXECUTIVE_SUMMARY.md](audit/EXECUTIVE_SUMMARY.md)
  - [AUDIT_COMPLETE.md](audit/AUDIT_COMPLETE.md)

---

## <¯ Find What You Need

### By Task

| What You Want to Do | Where to Look |
|---------------------|---------------|
| **Install Coqui TTS** | [Getting Started ’ Installation](GETTING_STARTED.md#installation) |
| **Generate your first TTS** | [Getting Started ’ Quick Start](GETTING_STARTED.md#quick-start) |
| **Clone a voice** | [Getting Started ’ Use Case 2](GETTING_STARTED.md#use-case-2-multilingual-text-to-speech-with-voice-cloning) |
| **Use multiple languages** | [Getting Started ’ Multilingual TTS](GETTING_STARTED.md#use-case-2-multilingual-text-to-speech-with-voice-cloning) |
| **Choose the right model** | [Getting Started ’ Model Selection](GETTING_STARTED.md#model-selection-guide) |
| **Run TTS as a server** | [Getting Started ’ Running as Server](GETTING_STARTED.md#use-case-6-running-as-a-server) |
| **Train a custom model** | [source/training_a_model.md](source/training_a_model.md) |
| **Understand the architecture** | [architecture/ARCHITECTURAL_OVERVIEW.md](architecture/ARCHITECTURAL_OVERVIEW.md) |
| **Contribute to the project** | [GOVERNANCE.md](GOVERNANCE.md) + [../CONTRIBUTING.md](../CONTRIBUTING.md) |
| **Use the Python API** | [development/TTS_PACKAGE_GUIDE.md](development/TTS_PACKAGE_GUIDE.md) |
| **Troubleshoot issues** | [Getting Started ’ Troubleshooting](GETTING_STARTED.md#troubleshooting) |

### By Role

| Your Role | Recommended Reading Order |
|-----------|---------------------------|
| **New User** | 1. [GETTING_STARTED.md](GETTING_STARTED.md)<br>2. [source/faq.md](source/faq.md) |
| **Developer** | 1. [GETTING_STARTED.md](GETTING_STARTED.md)<br>2. [architecture/QUICK_REFERENCE.md](architecture/QUICK_REFERENCE.md)<br>3. [development/TTS_PACKAGE_GUIDE.md](development/TTS_PACKAGE_GUIDE.md) |
| **ML Engineer** | 1. [source/training_a_model.md](source/training_a_model.md)<br>2. [architecture/ARCHITECTURAL_OVERVIEW.md](architecture/ARCHITECTURAL_OVERVIEW.md)<br>3. [source/formatting_your_dataset.md](source/formatting_your_dataset.md) |
| **Contributor** | 1. [../CONTRIBUTING.md](../CONTRIBUTING.md)<br>2. [GOVERNANCE.md](GOVERNANCE.md)<br>3. [development/MODERNIZATION_ROADMAP.md](development/MODERNIZATION_ROADMAP.md) |
| **DevOps** | 1. [GETTING_STARTED.md ’ Docker](GETTING_STARTED.md#use-case-7-docker-deployment)<br>2. [source/docker_images.md](source/docker_images.md) |

---

## =Ö Complete Documentation Map

For a comprehensive navigation guide with detailed descriptions of each document, see **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)**.

---

## = Key Topics

### Models

- **All Models Overview**: [../README.md ’ Model Implementations](../README.md#model-implementations)
- **XTTS v2** (Multilingual): [source/models/xtts.md](source/models/xtts.md)
- **VoiceCraft-X** (Speech Editing): [models/voicecraft_x.md](models/voicecraft_x.md)
- **Bark** (Emotional Speech): [source/models/bark.md](source/models/bark.md)
- **Tortoise** (High Quality): [source/models/tortoise.md](source/models/tortoise.md)
- **VITS** (Fast & Quality): [source/models/vits.md](source/models/vits.md)

### Training

- **Training Overview**: [source/training_a_model.md](source/training_a_model.md)
- **Dataset Formatting**: [source/formatting_your_dataset.md](source/formatting_your_dataset.md)
- **What Makes a Good Dataset**: [source/what_makes_a_good_dataset.md](source/what_makes_a_good_dataset.md)
- **Tutorial for Beginners**: [source/tutorial_for_nervous_beginners.md](source/tutorial_for_nervous_beginners.md)

### API & Integration

- **Python API**: [development/TTS_PACKAGE_GUIDE.md](development/TTS_PACKAGE_GUIDE.md)
- **Command Line**: [../README.md ’ Command-line](../README.md#command-line-tts)
- **Server/API**: [source/inference.md](source/inference.md)
- **Docker**: [source/docker_images.md](source/docker_images.md)

### Development

- **Architecture**: [architecture/ARCHITECTURAL_OVERVIEW.md](architecture/ARCHITECTURAL_OVERVIEW.md)
- **Quick Reference**: [architecture/QUICK_REFERENCE.md](architecture/QUICK_REFERENCE.md)
- **Contributing**: [../CONTRIBUTING.md](../CONTRIBUTING.md)
- **Governance**: [GOVERNANCE.md](GOVERNANCE.md)
- **Migration**: [development/MIGRATION_GUIDE.md](development/MIGRATION_GUIDE.md)
- **Roadmap**: [development/MODERNIZATION_ROADMAP.md](development/MODERNIZATION_ROADMAP.md)

---

## <˜ Getting Help

### Documentation Not Enough?

- **Bug Reports**: [GitHub Issues](https://github.com/coqui-ai/TTS/issues)
- **Questions**: [GitHub Discussions](https://github.com/coqui-ai/TTS/discussions)
- **Chat**: [Discord](https://discord.gg/5eXr5seRrv)
- **FAQ**: [source/faq.md](source/faq.md)

### Before Asking

1. Check the [Getting Started Guide](GETTING_STARTED.md)
2. Search [existing issues](https://github.com/coqui-ai/TTS/issues)
3. Read the [FAQ](source/faq.md)
4. Review [Troubleshooting](GETTING_STARTED.md#troubleshooting)

---

## =Ê Documentation Stats

- **Total Documentation Pages**: 40+ markdown files
- **Lines of Documentation**: 4500+ lines
- **Languages Covered**: Code in 3.9-3.11, Docs in English
- **Last Major Update**: November 20, 2025
- **Documentation Version**: 2.0

---

## > Contributing to Documentation

We welcome documentation improvements! To contribute:

1. **Identify gaps** - What's missing or unclear?
2. **Read the style guide** - See existing docs for format
3. **Make changes** - Edit markdown files
4. **Test links** - Ensure all cross-references work
5. **Submit PR** - Follow [CONTRIBUTING.md](../CONTRIBUTING.md)

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add cross-references to related docs
- Keep consistent formatting
- Update [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for new files

---

## =Ü Recent Documentation Updates

**November 20, 2025**:
- ( Added comprehensive [Getting Started Guide](GETTING_STARTED.md)
- =Ú Created [TTS Package Guide](development/TTS_PACKAGE_GUIDE.md) (moved from TTS/claude.md)
- <Û Added [Governance documentation](GOVERNANCE.md) (converted from CODE_OWNERS.rst)
- =ú Enhanced [Documentation Index](DOCUMENTATION_INDEX.md) with new files
- = Updated all cross-references across documentation

**Previous Updates**:
- VoiceCraft-X implementation documentation
- Modernization roadmap and migration guides
- Comprehensive architecture documentation
- Security audit reports

See [../CHANGELOG.md](../CHANGELOG.md) for complete history.

---

## < Popular Documentation

**Most Viewed**:
1. [Getting Started Guide](GETTING_STARTED.md)
2. [Quick Reference](architecture/QUICK_REFERENCE.md)
3. [Training Guide](source/training_a_model.md)
4. [XTTS Documentation](source/models/xtts.md)
5. [FAQ](source/faq.md)

**Most Useful for Beginners**:
1. [Getting Started](GETTING_STARTED.md)
2. [Tutorial for Nervous Beginners](source/tutorial_for_nervous_beginners.md)
3. [FAQ](source/faq.md)

**Most Useful for Developers**:
1. [Architectural Overview](architecture/ARCHITECTURAL_OVERVIEW.md)
2. [TTS Package Guide](development/TTS_PACKAGE_GUIDE.md)
3. [Contributing Guide](../CONTRIBUTING.md)

---

**Happy Learning!** =Ú

For the complete navigation experience, see [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md).

---

**Version**: 2.0
**Last Updated**: November 20, 2025
**Maintained by**: Coqui TTS Community
