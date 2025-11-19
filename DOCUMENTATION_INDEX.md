# Coqui TTS - Complete Documentation Index

## Overview
This directory now contains comprehensive documentation for understanding and modernizing the Coqui TTS codebase. This index helps you navigate all the documentation.

---

## Documents Created

### 1. QUICK_REFERENCE.md (11 KB, 270 lines)
**Best for**: Getting started quickly, refreshing architecture knowledge
- What is Coqui TTS
- Core statistics and metrics
- Architecture layers diagram
- Inference and training pipelines
- File organization guide
- Critical files to know
- Quick usage patterns
- Architecture patterns used
- Known issues and workarounds
- Development workflow
- Testing commands
- Performance tips

**Start here if**: You need a quick overview or reminder

---

### 2. ARCHITECTURAL_OVERVIEW.md (34 KB, 1027 lines)
**Best for**: In-depth understanding of the entire system
- Executive summary
- Complete directory structure with explanations
- Core architecture (pipeline, class hierarchy, config system)
- Model implementations (15+ TTS models, 8+ vocoders)
- API structure (public API, CLI, server)
- Training infrastructure
- Complete data pipeline
- Configuration system deep dive
- Text processing pipeline
- Testing infrastructure and CI/CD
- Integration points
- Documentation structure
- Dependency stack
- Architectural patterns
- Known architectural issues
- Metrics and statistics

**Start here if**: You want to understand the codebase deeply

---

### 3. MODERNIZATION_ROADMAP.md (9 KB, 320 lines)
**Best for**: Planning modernization initiatives
- Current state analysis
- 11 modernization opportunities (prioritized)
  - 4 high priority (unify config, PyTorch 2.0+, type hints, consolidate layers)
  - 4 medium priority (testing, async API, decouple trainer, docs)
  - 3 low priority (dependencies, performance, DX)
- 4-phase implementation plan (16 weeks total)
- Breaking changes to plan for
- Recommended reading
- Success metrics
- Quick wins that can be done immediately
- Risk mitigation strategies

**Start here if**: You're planning modernization work

---

## Quick Navigation Guide

### By Role

#### If you're a **New Developer**:
1. Start with QUICK_REFERENCE.md (overview)
2. Read ARCHITECTURAL_OVERVIEW.md sections 1-4 (directory structure + core architecture)
3. Follow "Adding a New Model" guide in QUICK_REFERENCE.md
4. Look at recipes/ljspeech/glow_tts/train_glowtts.py as example

#### If you're an **ML Engineer** (training models):
1. QUICK_REFERENCE.md → Training Pipeline section
2. ARCHITECTURAL_OVERVIEW.md → Sections 5, 6, 7 (training, data, config)
3. Check recipes/ljspeech/ for your model type
4. QUICK_REFERENCE.md → Performance Tips

#### If you're a **DevOps/Infrastructure Engineer**:
1. QUICK_REFERENCE.md → Dependencies section
2. ARCHITECTURAL_OVERVIEW.md → Section 9 (testing), Section 12 (dependencies)
3. Review .github/workflows/ for CI/CD setup
4. Check setup.py and requirements.txt

#### If you're **Planning Modernization**:
1. ARCHITECTURAL_OVERVIEW.md → Sections 14-15 (known issues, modernization)
2. MODERNIZATION_ROADMAP.md → All sections
3. QUICK_REFERENCE.md → Architecture Patterns section
4. Review critical files list for impact analysis

#### If you're **Adding Integration** (using Coqui in your app):
1. QUICK_REFERENCE.md → "Quick Usage Patterns" section
2. ARCHITECTURAL_OVERVIEW.md → Section 10 (integration points)
3. ARCHITECTURAL_OVERVIEW.md → Section 4 (API structure)
4. Check examples in notebooks/

---

### By Topic

#### **Architecture & Design**
- ARCHITECTURAL_OVERVIEW.md sections 1-4
- QUICK_REFERENCE.md "Architecture Layers" diagram
- QUICK_REFERENCE.md "Architecture Patterns Used"

#### **Models & Algorithms**
- ARCHITECTURAL_OVERVIEW.md section 3
- Model files: TTS/tts/models/*.py
- Config files: TTS/tts/configs/*_config.py

#### **Data & Training**
- ARCHITECTURAL_OVERVIEW.md sections 5-7
- QUICK_REFERENCE.md "Training Pipeline"
- recipes/ljspeech/ (22 example scripts)

#### **API & Integration**
- QUICK_REFERENCE.md "Quick Usage Patterns"
- ARCHITECTURAL_OVERVIEW.md section 4 + 10
- TTS/api.py (main public API)

#### **Configuration**
- ARCHITECTURAL_OVERVIEW.md section 7
- TTS/config/__init__.py
- TTS/tts/configs/*.py (17 config classes)

#### **Testing & CI/CD**
- ARCHITECTURAL_OVERVIEW.md section 9
- .github/workflows/ (14 workflow files)
- tests/ (14 test categories)

#### **Text Processing**
- ARCHITECTURAL_OVERVIEW.md section 8
- TTS/tts/utils/text/ (phonemizers, tokenizers)

#### **Audio Processing**
- QUICK_REFERENCE.md "Core Dependencies"
- TTS/utils/audio/ (processor.py, transforms)
- ARCHITECTURAL_OVERVIEW.md "Audio Processing"

#### **Modernization**
- MODERNIZATION_ROADMAP.md (all)
- ARCHITECTURAL_OVERVIEW.md sections 14-15

---

### By Codebase Area

#### TTS/tts/ (Text-to-Speech models)
- Overview: QUICK_REFERENCE.md "File Organization"
- Models: ARCHITECTURAL_OVERVIEW.md section 3
- Configs: ARCHITECTURAL_OVERVIEW.md section 7
- Datasets: ARCHITECTURAL_OVERVIEW.md section 6
- Utils: ARCHITECTURAL_OVERVIEW.md sections 7-8

#### TTS/vocoder/ (Audio synthesis)
- Overview: ARCHITECTURAL_OVERVIEW.md section 3.3
- Implementation: See vocoder files in TTS/vocoder/models/

#### TTS/utils/ (Core utilities)
- Audio: QUICK_REFERENCE.md "Core Dependencies"
- API: ARCHITECTURAL_OVERVIEW.md section 4.1-4.2
- Management: ARCHITECTURAL_OVERVIEW.md section 10.3

#### tests/ (Testing)
- Overview: QUICK_REFERENCE.md "Testing Commands"
- Details: ARCHITECTURAL_OVERVIEW.md section 9

#### recipes/ (Training examples)
- Overview: QUICK_REFERENCE.md "Adding a New Model"
- Details: ARCHITECTURAL_OVERVIEW.md section 5.3

#### docs/ (Documentation)
- Overview: QUICK_REFERENCE.md "Useful Links"
- Structure: ARCHITECTURAL_OVERVIEW.md section 11

---

## Key Files Reference

### Must Read (in order)
1. TTS/api.py - Main public interface
2. TTS/tts/models/base_tts.py - Base class for all models
3. TTS/utils/synthesizer.py - Inference logic
4. TTS/config/__init__.py - Config system
5. TTS/tts/datasets/dataset.py - Data loading

### For Training
1. recipes/ljspeech/glow_tts/train_glowtts.py - Perfect example
2. TTS/tts/configs/shared_configs.py - Config patterns
3. TTS/tts/datasets/formatters.py - Dataset support

### For New Models
1. TTS/tts/models/base_tts.py - Inherit from this
2. TTS/tts/configs/shared_configs.py - Config pattern
3. TTS/tts/layers/ - Layer implementations
4. Look at TTS/tts/models/glow_tts.py as example

---

## Statistics Summary

| Metric | Count | File |
|--------|-------|------|
| Total Python Files | 293 | ARCHITECTURAL_OVERVIEW.md |
| Code Size | 2.9 MB | ARCHITECTURAL_OVERVIEW.md |
| TTS Models | 15+ | ARCHITECTURAL_OVERVIEW.md section 3 |
| Vocoders | 8+ | ARCHITECTURAL_OVERVIEW.md section 3.3 |
| Config Classes | 17 | QUICK_REFERENCE.md |
| Test Categories | 14 | QUICK_REFERENCE.md |
| CI/CD Workflows | 14 | QUICK_REFERENCE.md |
| Dataset Formatters | 30+ | ARCHITECTURAL_OVERVIEW.md section 6.2 |
| Modernization Opportunities | 11 | MODERNIZATION_ROADMAP.md |

---

## Common Questions & Answers

### "How do I get started with Coqui TTS?"
See QUICK_REFERENCE.md "Quick Usage Patterns"

### "Where should I add a new model?"
See QUICK_REFERENCE.md "Adding a New Model" + ARCHITECTURAL_OVERVIEW.md section 3

### "How is training set up?"
See ARCHITECTURAL_OVERVIEW.md section 5 + QUICK_REFERENCE.md "Training Pipeline"

### "What are the main architectural issues?"
See MODERNIZATION_ROADMAP.md + ARCHITECTURAL_OVERVIEW.md section 14

### "How do I understand the config system?"
See ARCHITECTURAL_OVERVIEW.md section 7 + QUICK_REFERENCE.md

### "Where's the inference code?"
See ARCHITECTURAL_OVERVIEW.md section 4 + TTS/api.py + TTS/utils/synthesizer.py

### "What datasets are supported?"
See ARCHITECTURAL_OVERVIEW.md section 6.2 (30+ formatters)

### "How do I improve performance?"
See QUICK_REFERENCE.md "Performance Tips" + MODERNIZATION_ROADMAP.md section "PyTorch 2.0+"

---

## Documentation Standards

### Terminology
- **TTS Model**: Text-to-spectrogram models (Tacotron2, VITS, GlowTTS, etc.)
- **Vocoder**: Spectrogram-to-waveform models (HiFiGAN, MelGAN, etc.)
- **End-to-End Model**: Text-to-waveform directly (VITS, XTTS, Bark, Tortoise)
- **Spectrogram**: Mel-spectrogram (time-frequency representation)
- **Config**: Coqpit configuration object for model/training settings
- **Recipe**: Training script + config for a specific model/dataset combo
- **Formatter**: Dataset-specific metadata reader (30+ available)

### Code References
- File paths use **absolute paths** from repo root: `/TTS/tts/models/vits.py`
- Class names use **backticks**: `BaseTTS`, `AudioProcessor`
- Functions use **backticks**: `setup_model()`, `load_config()`
- Config keys use **backticks**: `batch_size`, `num_speakers`

---

## Contributing to This Documentation

When adding/updating documentation:
1. Keep QUICK_REFERENCE.md as the "one-pager"
2. Put detailed info in ARCHITECTURAL_OVERVIEW.md
3. Add roadmap items to MODERNIZATION_ROADMAP.md if applicable
4. Update this INDEX when adding new sections

---

## Version Information

- **Documentation Created**: November 19, 2025
- **Coqui TTS Version**: Latest (2 years old, actively maintained)
- **Python Support**: 3.9, 3.10, 3.11
- **PyTorch**: 2.1+
- **Total Documentation Pages**: 3 comprehensive guides + this index
- **Total Lines**: 2131 lines of documentation

---

## Next Steps

1. **Immediate**: Read QUICK_REFERENCE.md for overview
2. **Short-term**: Review ARCHITECTURAL_OVERVIEW.md for your role
3. **Medium-term**: Begin work on MODERNIZATION_ROADMAP.md items
4. **Long-term**: Implement 16-week modernization plan

---

**Last Updated**: November 19, 2025
**Maintained by**: Architecture Documentation Task
