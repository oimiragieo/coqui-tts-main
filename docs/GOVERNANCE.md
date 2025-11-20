# TTS Code Owners & Governance System

**Last Updated**: November 20, 2025

---

## Overview

TTS is run under a governance system inspired by the [Mozilla module ownership system](https://www.mozilla.org/about/governance/policies/module-ownership/). The project is roughly divided into modules, and each module has its owners, which are responsible for reviewing pull requests and deciding on technical direction for their modules.

Module ownership authority is given to people who have worked extensively on areas of the project.

---

## Module Ownership Philosophy

Module owners also have the authority of naming other module owners or appointing module peers, which are people with authority to review pull requests in that module. They can also sub-divide their module into sub-modules with their owners.

**Important**: Module owners are not tyrants. They are chartered to make decisions with input from the community and in the best interest of the community. Module owners are not required to make code changes or additions solely because the community wants them to do so.

### Responsibilities

Module owners need to pay attention to patches submitted to their module. However, "pay attention" does not mean agreeing to every patch. Some patches may not make sense for the project; some may be poorly implemented. Module owners have the authority to:

- **Decline a patch** - This is a necessary part of the role
- **Request changes** - Owners should describe their reasons for wanting changes
- **Postpone review** - For example, if a patch is not needed for the next milestone

We expect these decisions to be described in the relevant GitHub issue. Module owners should not be expected to rewrite patches to make them acceptable.

---

## How to Use This File

This file describes module owners who are active on the project and which parts of the code they have expertise on (and interest in). If you're making changes to the code and are wondering who's an appropriate person to talk to, this list will tell you who to ping.

**Note**: There's overlap in the areas of expertise of each owner, and in particular when looking at which files are covered by each area, there is a lot of overlap. Don't worry about getting it exactly right when requesting review - any code owner will be happy to redirect the request to a more appropriate person.

---

## Global Owners

These are people who have worked on the project extensively and are familiar with all or most parts of it. Their expertise and review guidance is trusted by other code owners to cover their own areas of expertise. In case of conflicting opinions from other owners, global owners will make a final decision.

- **Eren Gölge** ([@erogol](https://github.com/erogol))
- **Reuben Morais** ([@reuben](https://github.com/reuben))

---

## Module Owners

### Training & Data Pipeline

**Owners**:
- Eren Gölge ([@erogol](https://github.com/erogol))

**Areas**:
- Training infrastructure and scripts
- Data loading and preprocessing
- Dataset formatters and utilities
- Training configuration

---

### Model Exporting

**Owners**:
- Eren Gölge ([@erogol](https://github.com/erogol))

**Areas**:
- Model export functionality
- ONNX conversion
- Model optimization

---

### Multi-Speaker TTS

**Owners**:
- Eren Gölge ([@erogol](https://github.com/erogol))
- Edresson Casanova ([@edresson](https://github.com/edresson))

**Areas**:
- Multi-speaker model implementations
- Speaker encoder models
- Speaker embedding systems
- Voice cloning features

---

### TTS Models

**Owners**:
- Eren Gölge ([@erogol](https://github.com/erogol))

**Areas**:
- All TTS model implementations
- Model architectures and layers
- TTS configurations
- Text processing pipelines

---

### Vocoders

**Owners**:
- Eren Gölge ([@erogol](https://github.com/erogol))

**Areas**:
- Vocoder model implementations
- Vocoder training and inference
- Vocoder configurations
- Audio synthesis

---

### Speaker Encoder

**Owners**:
- Eren Gölge ([@erogol](https://github.com/erogol))

**Areas**:
- Speaker encoder models (LSTM, ResNet)
- Speaker verification
- Speaker embedding extraction

---

### Testing & CI/CD

**Owners**:
- Eren Gölge ([@erogol](https://github.com/erogol))
- Reuben Morais ([@reuben](https://github.com/reuben))

**Areas**:
- Test infrastructure
- GitHub Actions workflows
- CI/CD pipelines
- Quality assurance

---

### Python Bindings & API

**Owners**:
- Eren Gölge ([@erogol](https://github.com/erogol))
- Reuben Morais ([@reuben](https://github.com/reuben))

**Areas**:
- Public Python API (`TTS/api.py`)
- CLI tools
- Server implementation
- Library interfaces

---

### Documentation

**Owners**:
- Eren Gölge ([@erogol](https://github.com/erogol))

**Areas**:
- All documentation files
- Tutorials and guides
- API documentation
- README and contribution guides

---

### Third Party Bindings

**Ownership**: Owned by the respective authors

**Note**: Third-party bindings and integrations are maintained by their respective authors and communities.

---

## Escalation Process

If there are conflicting opinions between module owners or if you need a final decision:

1. Discuss in the relevant GitHub issue
2. Tag the appropriate module owner(s)
3. If needed, escalate to Global Owners
4. Global Owners will make the final decision

---

## How to Become a Module Owner

Module owners are selected based on:

- Extensive contributions to the module
- Deep understanding of the code
- Active participation in the community
- Demonstrated good judgment in reviews

If you're interested in becoming a module owner or peer, contribute actively to the project and discuss with existing module owners.

---

## Community Guidelines

All module owners, peers, and contributors are expected to follow:

- The [Code of Conduct](CODE_OF_CONDUCT.md)
- The [Contributing Guidelines](../CONTRIBUTING.md)
- Professional and respectful communication
- Best interests of the project and community

---

## Related Documentation

- [Contributing Guide](../CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Development Guides](development/)
- [Architecture Overview](architecture/ARCHITECTURAL_OVERVIEW.md)

---

**Version**: 2.0
**Format**: Converted from RST to Markdown (November 20, 2025)
**Original**: CODE_OWNERS.rst
**Maintained by**: Global Owners
