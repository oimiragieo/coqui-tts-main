# Coqui TTS - Quick Architecture Reference

## Key Takeaways

### What is Coqui TTS?
A comprehensive, production-ready Text-to-Speech (TTS) library with:
- 15+ TTS models (XTTS, VITS, Bark, Tacotron, GlowTTS, etc.)
- 8+ vocoders (HiFiGAN, MelGAN, WaveRNN, etc.)
- Multi-speaker and multi-language support (1100+ languages via fairseq)
- Professional training and inference infrastructure
- 100+ pre-trained models ready to use

### Core Statistics
| Metric | Value |
|--------|-------|
| Python Files | 293 |
| Code Size | 2.9 MB |
| Estimated LOC | 30,000+ |
| Config Classes | 17 |
| Test Suites | 14 categories |
| CI/CD Workflows | 14 |
| Supported Frameworks | PyTorch 2.1+ |
| Python Versions | 3.9, 3.10, 3.11 |

---

## Architecture Layers

```
┌─────────────────────────────────────┐
│     Public API (TTS.api.TTS)        │
│  - List models, load, synthesize    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Synthesizer (high-level inf.)     │
│  - Handles TTS + Vocoder + Utils    │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┬─────────────┐
       │                │             │
   ┌───▼────┐    ┌─────▼────┐   ┌───▼──────┐
   │  TTS   │    │  Vocoder │   │ Encoder  │
   │ Models │    │  Models  │   │ Models   │
   └───┬────┘    └─────┬────┘   └───┬──────┘
       │                │             │
   ┌───▼──────────────────────────────▼────┐
   │ Layers + Utilities (audio, text, etc) │
   └────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────┐
   │  PyTorch + External Dependencies │
   └────────────────────────────────────┘
```

---

## Inference Pipeline

```
User Input (Text)
    ↓
[TTSTokenizer]
    ↓ (sequence IDs)
[TTS Model] (Tacotron2, VITS, XTTS, etc.)
    ↓ (mel-spectrogram or waveform)
[Vocoder] (if spectrogram output)
    ↓
[AudioProcessor] (normalization, formatting)
    ↓
Output (WAV file)
```

---

## Training Pipeline

```
Config Creation
    ↓
Dataset Loading (30+ formatters supported)
    ↓
AudioProcessor Initialization
    ↓
Tokenizer Initialization
    ↓
Model Creation (from config)
    ↓
Trainer Initialization (uses trainer library)
    ↓
trainer.fit()
    ↓
Checkpoints saved to output_path
```

---

## Data Flow Diagram

```
                    ┌─────────────┐
                    │  .models.json│ (100+ pre-trained models)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ModelManager │ (download/cache)
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
     │Config   │     │Model    │     │Tokenizer│
     │Files    │     │Files    │     │+Audio   │
     └────┬────┘     └────┬────┘     └────┬────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Synthesizer │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ WAV Output  │
                    └─────────────┘
```

---

## File Organization

### Core Directories
| Dir | Purpose | Key Files |
|-----|---------|-----------|
| `TTS/tts/models/` | TTS models (15+) | vits.py, xtts.py, bark.py |
| `TTS/tts/configs/` | Model configs (17) | glow_tts_config.py, vits_config.py |
| `TTS/tts/layers/` | Model layers (12 dirs) | Model-specific components |
| `TTS/tts/datasets/` | Data loading | dataset.py, formatters.py |
| `TTS/vocoder/models/` | Vocoders (8+) | hifigan_generator.py, melgan_generator.py |
| `TTS/utils/audio/` | Audio processing | processor.py, numpy_transforms.py |
| `TTS/bin/` | CLI tools | synthesize.py, train_tts.py |
| `TTS/server/` | Flask server | server.py |

---

## Critical Files to Know

### For Inference
- **TTS/api.py** - Main public API (use this!)
- **TTS/utils/synthesizer.py** - Low-level inference
- **TTS/config/__init__.py** - Config loading

### For Training
- **recipes/ljspeech/glow_tts/train_glowtts.py** - Training example
- **TTS/tts/models/base_tts.py** - Base class for all TTS models
- **TTS/tts/datasets/dataset.py** - Data loading logic

### For Adding New Models
- **TTS/tts/models/base_tts.py** - Base class to inherit
- **TTS/tts/configs/shared_configs.py** - Config patterns
- **TTS/tts/layers/{model}/ ** - Layer implementations

### For Understanding Config
- **TTS/config/shared_configs.py** - BaseAudioConfig, BaseTrainingConfig
- **TTS/tts/configs/*.py** - Model-specific configs (17 files)

---

## Quick Usage Patterns

### Basic Inference
```python
from TTS.api import TTS

# Load model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)

# Synthesize
tts.tts_to_file("Hello world", file_path="output.wav")
```

### Multilingual
```python
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
tts.tts_to_file("Hello", speaker=tts.speakers[0], 
                language=tts.languages[0], file_path="output.wav")
```

### Voice Cloning
```python
tts.tts_to_file("Custom text", speaker_wav="reference.wav",
                language="en", file_path="cloned.wav")
```

### List Available Models
```python
print(TTS.list_models())  # All 100+ pre-trained models
```

---

## Architecture Patterns Used

1. **Factory Pattern**: Dynamic model loading via config name
   - `setup_model(config)` → Returns appropriate model class

2. **Manager Pattern**: Centralized resource management
   - `ModelManager` - Download/cache models
   - `SpeakerManager` - Manage speakers
   - `AudioProcessor` - Audio features

3. **Template Method**: Base classes define algorithm
   - `BaseTTS` - TTS template
   - `BaseVocoder` - Vocoder template

4. **Strategy Pattern**: Pluggable phonemizers
   - Routes text processing by language

5. **Registry Pattern**: Dynamic config/model lookup
   - `register_config()`, `find_module()`

---

## Known Issues & Workarounds

### Config Inconsistency
**Problem**: Some models use `config.model_args`, others use flat config
**Workaround**: Use `get_from_config_or_model_args(config, key)` helpers
**Status**: Should unify in v1.0 modernization

### Audio Normalization
**Problem**: Different models expect different audio ranges
**Workaround**: AudioProcessor handles most cases, check model-specific docs
**Status**: Document more thoroughly

### Model-Specific Layers
**Problem**: Each model has duplicate layer implementations
**Workaround**: Extract common layers when adding new models
**Status**: Consolidate in modernization phase

### Trainer Coupling
**Problem**: Hard to use models outside of trainer framework
**Workaround**: Use Synthesizer for inference directly
**Status**: Decouple in future major version

---

## Dependencies You Should Know

### Core
- **torch/torchaudio** - PyTorch ecosystem
- **coqpit** - Configuration management
- **trainer** - Coqui's training framework
- **gruut** - Grapheme-to-phoneme conversion
- **librosa** - Audio processing

### Optional
- **transformers** - For Bark (BERT tokenizer)
- **einops** - For Tortoise/Bark
- **encodec** - For Bark (codec)

---

## Testing Commands

```bash
# All tests
make test

# Specific categories
make test_tts          # TTS model tests
make test_vocoder      # Vocoder tests
make test_tts2         # Training tests
make test_xtts         # XTTS tests
make test_zoo          # Pre-trained models
make test_data         # Dataset tests
make test_text         # Text processing

# Single test file
python -m pytest tests/tts_tests/test_vits.py
```

---

## Development Workflow

### Setup
```bash
# Clone and install
git clone https://github.com/coqui-ai/TTS.git
cd TTS
pip install -e .[all]
```

### Code Style
```bash
make style    # Format code (black, isort)
make lint     # Check style (pylint, black, isort)
```

### Adding a New Model
1. Create `TTS/tts/configs/mymodel_config.py`
2. Create `TTS/tts/models/mymodel.py` (inherit from BaseTTS)
3. Create `TTS/tts/layers/mymodel/` (layer implementations)
4. Add to `TTS/tts/models/__init__.py`
5. Create training recipe in `recipes/ljspeech/mymodel/`
6. Add tests in `tests/tts_tests2/`

---

## Performance Tips

### Inference
- Use GPU for larger models (XTTS, Tortoise, VITS)
- CPU-friendly options: FastSpeech2, Tacotron2 with CPU vocoder
- Batch processing not yet optimized (one-at-a-time faster)

### Training
- Use mixed precision: `mixed_precision=True` in config
- Use bucket batch sampling: `batch_group_size > 0`
- Cache phonemes and F0 computations
- Distributed training: Multi-GPU via trainer

### Data
- Use `batch_group_size > 0` to gather similar-length sequences
- Precompute expensive features (F0, energy, phonemes)
- Use multiple loader workers: `num_loader_workers > 0`

---

## Next Steps for Modernization

See `MODERNIZATION_ROADMAP.md` for detailed 16-week plan with 11 improvements.

**Top priorities**:
1. Unify config system (reduce tech debt)
2. Add type hints (improve DX)
3. Leverage PyTorch 2.0+ (improve performance)
4. Consolidate layers (reduce duplication)
5. Modernize tests (migrate from nose2 → pytest)

---

## Useful Links

- **GitHub**: https://github.com/coqui-ai/TTS
- **Docs**: https://tts.readthedocs.io/
- **Discord**: https://discord.gg/5eXr5seRrv
- **Model Zoo**: https://huggingface.co/coqui
- **Papers**: https://github.com/erogol/TTS-papers

