# Coqui TTS - Root Directory Guide for Claude AI

**Last Updated**: November 19, 2025
**Version**: 0.22.0
**Purpose**: Guide AI assistants working with this Text-to-Speech library

---

## 📁 PROJECT OVERVIEW

**Coqui TTS** is a comprehensive, production-ready Text-to-Speech library featuring:
- 15+ TTS models (XTTS, VITS, Bark, Tacotron2, GlowTTS, etc.)
- 8+ vocoder models (HiFiGAN, MelGAN, WaveRNN, UnivNet, etc.)
- Multi-speaker and multi-language support (1100+ languages via Fairseq)
- Voice cloning capabilities
- Professional training and inference infrastructure

**Repository**: https://github.com/coqui-ai/TTS
**Documentation**: https://tts.readthedocs.io/
**License**: MPL-2.0

---

## 🎯 QUICK START FOR AI ASSISTANTS

### Most Important Files
1. **TTS/api.py** - Public Python API (start here!)
2. **TTS/utils/synthesizer.py** - Low-level inference engine
3. **TTS/tts/models/xtts.py** - XTTS model (voice cloning, streaming)
4. **TTS/server/server.py** - Flask REST API server
5. **setup.py** - Package configuration and dependencies

### Common Tasks

**Task: Add a new TTS model**
```
1. Create config: TTS/tts/configs/mymodel_config.py
2. Create model: TTS/tts/models/mymodel.py (inherit from BaseTTS)
3. Create layers: TTS/tts/layers/mymodel/*.py
4. Add to registry: TTS/tts/models/__init__.py
5. Create recipe: recipes/ljspeech/mymodel/train.py
6. Add tests: tests/tts_tests2/test_mymodel.py
```

**Task: Fix a bug in inference**
```
1. Start with TTS/api.py (high-level)
2. Trace to TTS/utils/synthesizer.py (inference logic)
3. Check model: TTS/tts/models/{model_name}.py
4. Verify audio: TTS/utils/audio/processor.py
```

**Task: Add a new language**
```
1. Add phonemizer: TTS/tts/utils/text/phonemizers/{lang}.py
2. Update language list: TTS/tts/configs/{model}_config.py
3. Add character mapping: TTS/tts/utils/text/characters.py
4. Test: tests/text_tests/test_{lang}_phonemizer.py
```

---

## 📂 DIRECTORY STRUCTURE

```
coqui-tts-main/
├── TTS/                    # Main package (core library)
│   ├── api.py             # 🔥 Public Python API
│   ├── bin/               # CLI tools (synthesize, train_tts, etc.)
│   ├── config/            # Configuration system
│   ├── encoder/           # Speaker encoding models
│   ├── server/            # 🔥 Flask REST API server
│   ├── tts/               # 🔥 TTS models and utilities
│   │   ├── configs/       # Model-specific configs (17 files)
│   │   ├── datasets/      # Dataset loaders (30+ formatters)
│   │   ├── layers/        # Model layers (12 subdirectories)
│   │   ├── models/        # 🔥 TTS model implementations
│   │   └── utils/         # Text processing, audio, helpers
│   ├── utils/             # Shared utilities (audio, synthesis, etc.)
│   ├── vc/                # Voice conversion models
│   └── vocoder/           # Vocoder models
├── docs/                  # Sphinx documentation
├── notebooks/             # Jupyter notebooks (examples, analysis)
├── recipes/               # Training recipes for datasets
├── scripts/               # Utility scripts
├── tests/                 # 243 test functions (14 categories)
├── requirements.txt       # Core dependencies
├── setup.py              # Package setup
└── README.md             # Main documentation

🔥 = Most frequently accessed
```

### Key Subdirectories Explained

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| **TTS/tts/models/** | TTS model implementations | xtts.py, vits.py, bark.py, base_tts.py |
| **TTS/tts/configs/** | Model configurations | glow_tts_config.py, xtts_config.py |
| **TTS/tts/layers/** | Model-specific layers | xtts/, vits/, glow_tts/ |
| **TTS/tts/datasets/** | Data loading | dataset.py, formatters.py |
| **TTS/vocoder/models/** | Vocoder implementations | hifigan_generator.py, melgan_generator.py |
| **TTS/utils/** | Shared utilities | synthesizer.py, audio/processor.py |
| **TTS/bin/** | CLI entry points | synthesize.py, train_tts.py |
| **TTS/server/** | REST API | server.py (Flask), templates/ |
| **recipes/** | Training scripts | ljspeech/, vctk/, etc. |
| **tests/** | Test suite | tts_tests/, vocoder_tests/, etc. |

---

## 🏗️ ARCHITECTURE OVERVIEW

### Inference Pipeline

```
User Text
    ↓
[TTSTokenizer] (language-specific)
    ↓
[TTS Model] (XTTS, VITS, Tacotron2, etc.)
    ↓ (mel-spectrogram or waveform)
[Vocoder] (optional: HiFiGAN, MelGAN, etc.)
    ↓
[AudioProcessor] (normalization, formatting)
    ↓
Output WAV File
```

### Class Hierarchy

```
BaseTTS (TTS/tts/models/base_tts.py)
├── Tacotron2
├── GlowTTS
├── Vits
├── XTTS
├── Bark
└── ... (15+ models)

BaseVocoder (TTS/vocoder/models/base_vocoder.py)
├── HiFiGAN
├── MelGAN
├── WaveRNN
└── ... (8+ vocoders)
```

### Design Patterns

1. **Factory Pattern**: Dynamic model loading via config
2. **Manager Pattern**: ModelManager, SpeakerManager, AudioProcessor
3. **Template Method**: BaseTTS defines algorithm, subclasses implement
4. **Strategy Pattern**: Pluggable phonemizers for different languages
5. **Registry Pattern**: Models registered by name for dynamic loading

---

## 🔧 CONFIGURATION SYSTEM

**Library**: Coqpit (Coqui's configuration library)
**Pattern**: Python dataclasses with type hints

```python
from coqpit import Coqpit

class MyModelConfig(Coqpit):
    model: str = "my_model"
    num_speakers: int = 1
    # ... model-specific config
```

### Config Files

| File | Purpose |
|------|---------|
| **TTS/config/shared_configs.py** | BaseAudioConfig, BaseTrainingConfig |
| **TTS/tts/configs/glow_tts_config.py** | GlowTTS-specific config |
| **TTS/tts/configs/vits_config.py** | VITS-specific config |
| **TTS/tts/configs/xtts_config.py** | XTTS-specific config |

### Known Config Issues ⚠️

**Problem**: Some models use `config.model_args` (nested), others use flat config
**Workaround**: Use helper functions like `get_from_config_or_model_args()`
**Status**: Should be unified in modernization (see MODERNIZATION_ROADMAP.md)

---

## 🧪 TESTING

### Test Structure

```
tests/
├── aux_tests/              # Auxiliary tests
├── data_tests/             # Dataset tests
├── inference_tests/        # Inference tests
├── text_tests/             # Text processing tests
├── tts_tests/              # TTS model tests
├── tts_tests2/             # Additional TTS tests
├── vocoder_tests/          # Vocoder tests
├── xtts_tests/             # XTTS-specific tests
└── zoo_tests/              # Pre-trained model tests
```

### Running Tests

```bash
# All tests
make test

# Specific category
make test_tts          # TTS models
make test_vocoder      # Vocoders
make test_xtts         # XTTS
make test_data         # Datasets

# Single test file
python -m pytest tests/tts_tests/test_vits.py
```

### Test Framework: nose2 (⚠️ Should migrate to pytest)

---

## 🚀 DEVELOPMENT WORKFLOW

### Setup Development Environment

```bash
git clone https://github.com/coqui-ai/TTS.git
cd TTS
pip install -e .[all,dev,notebooks]
```

### Code Style

```bash
make style    # Format code (black, isort)
make lint     # Check style (pylint, black, isort)
```

### Pre-commit Hooks

```bash
pre-commit install
# Runs automatically on commit:
# - black (code formatting)
# - isort (import sorting)
# - trailing whitespace removal
```

### Adding a New Feature

1. **Create feature branch**: `git checkout -b feature/my-feature`
2. **Write code**: Follow existing patterns
3. **Add tests**: Required for new features
4. **Run tests**: `make test`
5. **Format code**: `make style`
6. **Commit**: Clear, descriptive message
7. **Push and PR**: Submit pull request

---

## ⚠️ KNOWN ISSUES & WORKAROUNDS

### 1. Config Inconsistency

**Problem**: Mixed `config.model_args` vs flat config
**Workaround**: Use `get_from_config_or_model_args(config, key)`
**Fix**: See MODERNIZATION_ROADMAP.md (Phase 1)

### 2. Global Lock in Flask Server

**Problem**: `TTS/server/server.py` uses global lock → serializes all requests
**Impact**: Can't handle concurrent requests
**Workaround**: Use Python API directly for better performance
**Fix**: Migrate to FastAPI (see MODERNIZATION_ROADMAP.md, Phase 3)

### 3. Missing Type Hints

**Problem**: 0% return type hint coverage → poor IDE support
**Impact**: Developer experience, type safety
**Workaround**: Read docstrings for return types
**Fix**: See MODERNIZATION_ROADMAP.md (Phase 2)

### 4. No Model Quantization

**Problem**: Models are large (XTTS=550MB), slow on CPU
**Workaround**: Use GPU, smaller models (FastSpeech2, Tacotron2)
**Fix**: Implement INT8/FP16 quantization (see MODERNIZATION_ROADMAP.md, Phase 4)

---

## 🔐 SECURITY CONSIDERATIONS

### Critical Security Issues (Fix Immediately!)

1. **Outdated numpy** (CVE-2021-33430, CVE-2021-41495)
   - File: `requirements.txt:2`
   - Fix: Update to numpy>=1.24.3

2. **Path Traversal Risk**
   - File: `TTS/server/server.py:142`
   - Risk: Arbitrary file reading via `../../` paths
   - Fix: Add Path.resolve() validation

3. **Command Injection**
   - Files: `prepare_voxceleb.py:81`, `xtts_demo.py:189`
   - Risk: shell=True in subprocess calls
   - Fix: Use subprocess.run() with shell=False

See **EXECUTIVE_SUMMARY.md** for complete security audit.

---

## 📊 PERFORMANCE TIPS

### Inference Optimization

**Fast Models (CPU-friendly)**:
- FastSpeech2 (~100ms on CPU)
- Tacotron2 with Griffin-Lim (~300ms on CPU)
- VITS (~1s on GPU, ~3s on CPU)

**High-Quality Models (GPU recommended)**:
- XTTS (~2-3s on GPU, streaming <100ms/chunk)
- Bark (~5-10s on GPU)
- YourTTS (~2s on GPU)

**Tips**:
- Use `gpu=True` for large models
- Enable `split_sentences=True` for long text
- Cache speaker embeddings for repeated voices
- Use streaming for real-time (XTTS only)

### Training Optimization

**Config Settings**:
```python
mixed_precision = True          # Enable FP16 training
batch_group_size = 4           # Group similar-length sequences
num_loader_workers = 4         # Parallel data loading
gradient_accumulation_steps = 2 # Larger effective batch size
```

**Data Pipeline**:
- Precompute phonemes and cache
- Precompute F0 (pitch) if model uses it
- Use fast audio loading (soundfile, not librosa for real-time)

---

## 🎓 LEARNING RESOURCES

### For New Contributors

1. **Start Here**:
   - README.md - High-level overview
   - docs/source/tutorial_for_nervous_beginners.md - Step-by-step guide

2. **Understanding Models**:
   - docs/source/models/*.md - Model-specific guides
   - TTS/tts/models/base_tts.py - Base class, understand this first

3. **Adding Your First Model**:
   - docs/source/implementing_a_new_model.md - Detailed guide
   - recipes/ljspeech/ - Example training recipes

### For ML Engineers

1. **Architecture**:
   - ARCHITECTURAL_OVERVIEW.md - Complete codebase structure
   - TTS/tts/models/vits.py - Modern, well-written model

2. **Training**:
   - docs/source/training_a_model.md - Training guide
   - recipes/ - Real-world training examples

3. **Fine-tuning**:
   - docs/source/finetuning.md - Fine-tuning guide
   - TTS/tts/layers/xtts/trainer/ - XTTS fine-tuning trainer

### For DevOps/SRE

1. **Deployment**:
   - Dockerfile - Basic container
   - TTS/server/server.py - Flask API (⚠️ needs modernization)
   - MODERNIZATION_ROADMAP.md - Enterprise deployment plan

2. **Monitoring**:
   - Currently minimal (⚠️ needs Prometheus, see roadmap)
   - TTS/server/server.py - Add metrics here

3. **Scaling**:
   - EXECUTIVE_SUMMARY.md - Cost-benefit analysis
   - Recommendation: FastAPI + Celery + Redis (see roadmap)

---

## 🔗 RELATED DOCUMENTATION

| File | Purpose |
|------|---------|
| **EXECUTIVE_SUMMARY.md** | High-level audit, security findings, ROI |
| **QUICK_REFERENCE.md** | Developer quick start, commands, patterns |
| **ARCHITECTURAL_OVERVIEW.md** | Deep dive, all models, configs, detailed structure |
| **MODERNIZATION_ROADMAP.md** | 11 improvements, 5-phase plan, 16 weeks |
| **DOCUMENTATION_INDEX.md** | Navigation guide for all docs |
| **README.md** | Original project README |
| **docs/** | Sphinx documentation (ReadTheDocs) |

---

## 🚨 CURRENT STATUS & PRIORITIES

### Codebase Health: **B+ (Good, with Critical Security Issues)**

| Area | Status | Priority |
|------|--------|----------|
| **Security** | ❌ Critical | 🔴 Fix now (4 hours) |
| **AI Capabilities** | ✅ Excellent | ✅ Maintain |
| **Architecture** | ⚠️ Good, needs update | 🟠 Modernize (8 weeks) |
| **Documentation** | ✅ Very good | ✅ Maintain |
| **Testing** | ✅ Good | 🟡 Improve coverage |
| **Type Safety** | ❌ Poor | 🟠 Add type hints |
| **Performance** | ⚠️ Okay | 🟡 Optimize |

### Immediate Actions Required

1. **This Week**: Fix security vulnerabilities (numpy, path traversal)
2. **This Month**: Add type hints, improve error handling
3. **This Quarter**: Migrate to FastAPI, add monitoring
4. **Next 6 Months**: Full modernization (see MODERNIZATION_ROADMAP.md)

---

## 🤖 AI ASSISTANT GUIDELINES

### When Working with This Codebase

**DO**:
✅ Start with TTS/api.py for public API changes
✅ Read EXECUTIVE_SUMMARY.md for security concerns
✅ Check existing tests before making changes
✅ Use black and isort for formatting
✅ Add docstrings (project has 97.8% coverage)
✅ Follow existing patterns (factory, manager, base classes)

**DON'T**:
❌ Modify models without understanding base classes
❌ Add dependencies without updating requirements.txt
❌ Skip tests (make test before committing)
❌ Use global state (see Flask server anti-pattern)
❌ Ignore security warnings (see EXECUTIVE_SUMMARY.md)

### Common Pitfalls

1. **Config Access**: Use helper functions, not direct attribute access
2. **Device Management**: Always check `use_cuda` flag
3. **Audio Normalization**: Different models expect different ranges
4. **Sentence Splitting**: Some models handle long text, others don't
5. **Speaker Embeddings**: Cache when possible (expensive computation)

### Debugging Tips

**Inference Issues**:
```
1. Check model config (print config)
2. Verify audio input (sample rate, format)
3. Check tokenizer (print tokens)
4. Verify device (CPU vs GPU)
5. Check audio output (normalization)
```

**Training Issues**:
```
1. Check dataset formatter (print batch)
2. Verify data paths (absolute vs relative)
3. Check config (all required fields)
4. Monitor GPU memory (nvidia-smi)
5. Check batch size (reduce if OOM)
```

---

## 📞 GETTING HELP

### Resources

- **GitHub Issues**: https://github.com/coqui-ai/TTS/issues
- **Discussions**: https://github.com/coqui-ai/TTS/discussions
- **Discord**: https://discord.gg/5eXr5seRrv
- **Docs**: https://tts.readthedocs.io/

### Common Questions

**Q: Which model should I use?**
A: XTTS for voice cloning, VITS for fast high-quality, Bark for emotions

**Q: How do I add a new language?**
A: Add phonemizer in TTS/tts/utils/text/phonemizers/, update config

**Q: How do I train a model?**
A: See recipes/ folder, copy and modify for your dataset

**Q: How do I deploy to production?**
A: See MODERNIZATION_ROADMAP.md for enterprise deployment guide

---

## 📝 CHANGE LOG

**Last Updated**: November 19, 2025
- Created comprehensive claude.md for root directory
- Completed security audit (see EXECUTIVE_SUMMARY.md)
- Identified critical security vulnerabilities
- Created modernization roadmap (MODERNIZATION_ROADMAP.md)
- Generated 2000+ lines of documentation

---

**End of Root claude.md**

For detailed subdirectory guidance, see:
- TTS/claude.md - Main package structure
- TTS/tts/claude.md - TTS models and utilities
- TTS/tts/models/claude.md - Model implementations
- TTS/server/claude.md - REST API server
