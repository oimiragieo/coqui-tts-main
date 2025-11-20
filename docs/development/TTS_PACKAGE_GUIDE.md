# TTS Package - Claude AI Guide

**Directory**: `/TTS` (Main Package)
**Purpose**: Core TTS library - models, utilities, APIs
**Last Updated**: November 19, 2025

---

## 📂 DIRECTORY OVERVIEW

```
TTS/
├── api.py                 # 🔥 PUBLIC PYTHON API (start here!)
├── bin/                   # CLI tools
│   ├── synthesize.py     # CLI for TTS
│   └── train_tts.py      # Training CLI
├── config/                # Configuration system
├── encoder/               # Speaker encoding
├── server/                # Flask REST API
├── tts/                   # 🔥 TTS models, datasets, layers
├── utils/                 # Shared utilities
├── vc/                    # Voice conversion
└── vocoder/               # Vocoder models
```

---

## 🔥 MOST IMPORTANT FILES

### 1. **api.py** - Public Python API
**Purpose**: High-level interface for users
**Key Classes**: `TTS`
**Example**:
```python
from TTS.api import TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
wav = tts.tts("Hello world")
tts.tts_to_file("Hello", file_path="output.wav")
```

**Key Methods**:
- `tts()` → Generate speech (returns numpy array)
- `tts_to_file()` → Generate and save to file
- `voice_conversion()` → Convert voice (FreeVC)
- `tts_with_vc()` → TTS + voice conversion

**Properties**:
- `models` → List available models
- `speakers` → List speakers (multi-speaker models)
- `languages` → List languages (multi-lingual models)
- `is_multi_speaker` → Check if multi-speaker
- `is_multi_lingual` → Check if multi-lingual

---

### 2. **utils/synthesizer.py** - Low-Level Inference
**Purpose**: Core inference engine
**Key Class**: `Synthesizer`
**When to Use**: Advanced control, custom pipelines

**Responsibilities**:
- Load models (TTS + Vocoder)
- Handle speaker/language selection
- Manage audio processing
- Sentence splitting and batching

---

### 3. **server/server.py** - Flask REST API
**Purpose**: HTTP API for TTS
**Endpoints**:
- `POST /api/tts` → Synthesize speech
- `GET /details` → Model info
- `GET /` → Web UI

**⚠️ Known Issues**:
- Uses global lock (serializes requests)
- No async support
- Path traversal vulnerability (needs fix)
- Should migrate to FastAPI (see MODERNIZATION_ROADMAP.md)

---

## 📁 SUBDIRECTORY GUIDE

### TTS/bin/ - CLI Tools

| File | Purpose | Usage |
|------|---------|-------|
| **synthesize.py** | CLI for TTS | `tts --text "Hello" --model_name ...` |
| **train_tts.py** | Train TTS model | `python train_tts.py --config config.json` |
| **train_vocoder.py** | Train vocoder | `python train_vocoder.py --config config.json` |
| **find_unique_chars.py** | Analyze dataset | Find unique characters in dataset |
| **compute_statistics.py** | Dataset stats | Compute audio statistics |

---

### TTS/config/ - Configuration System

**Key Files**:
- `__init__.py` - Config loading utilities
- `shared_configs.py` - BaseAudioConfig, BaseTrainingConfig

**Pattern**: Uses Coqpit (dataclass-based configs)

```python
from TTS.config import load_config
config = load_config("config.json")
```

---

### TTS/encoder/ - Speaker Encoding

**Purpose**: Generate speaker embeddings (d-vectors)
**Models**: LSTM, ResNet-based speaker encoders
**Use Case**: Voice cloning, multi-speaker TTS

**Key Files**:
- `models/lstm.py` - LSTM speaker encoder
- `models/resnet.py` - ResNet speaker encoder
- `utils/generic_utils.py` - Helper functions

---

### TTS/server/ - REST API Server

**See**: `TTS/server/claude.md` for detailed guide

**Quick Reference**:
```bash
# Start server
tts-server --model_name "tts_models/en/ljspeech/tacotron2-DDC"

# Use API
curl -X POST http://localhost:5002/api/tts \
  -d "text=Hello world" \
  --output output.wav
```

---

### TTS/tts/ - TTS Models & Utilities

**See**: `TTS/tts/claude.md` for detailed guide

**Structure**:
```
tts/
├── configs/        # Model configs (17 files)
├── datasets/       # Data loading (30+ formatters)
├── layers/         # Model layers (12 subdirs)
├── models/         # 🔥 Model implementations
└── utils/          # Text, audio, helpers
```

**Most Important**: `TTS/tts/models/` - All TTS models

---

### TTS/utils/ - Shared Utilities

**Key Files**:
- **synthesizer.py** - 🔥 Core inference engine
- **manage.py** - Model download/management
- **audio/processor.py** - Audio processing (STFTs, mel-specs)
- **audio/numpy_transforms.py** - Audio transformations

**Audio Pipeline**:
```
Raw Audio → AudioProcessor → Mel-Spectrogram → Model → Vocoder → Audio
```

---

### TTS/vc/ - Voice Conversion

**Models**: FreeVC (zero-shot voice conversion)
**Purpose**: Convert source audio to target speaker

**Structure**:
```
vc/
├── configs/
│   └── freevc_config.py
├── models/
│   └── freevc.py
└── modules/
    └── freevc/
```

**Usage**:
```python
from TTS.api import TTS
tts = TTS("voice_conversion_models/multilingual/vctk/freevc24")
wav = tts.voice_conversion(source_wav="a.wav", target_wav="b.wav")
```

---

### TTS/vocoder/ - Vocoder Models

**Purpose**: Convert mel-spectrograms → waveforms
**Models**: HiFiGAN, MelGAN, WaveRNN, UnivNet, etc.

**Structure**:
```
vocoder/
├── configs/         # Vocoder configs
├── datasets/        # Vocoder datasets
├── layers/          # Vocoder layers
├── models/          # 🔥 Vocoder implementations
└── utils/           # Vocoder utilities
```

**Key Models**:
- **HiFiGAN** - Fast, high-quality (most popular)
- **MelGAN** - Fast, lightweight
- **WaveRNN** - Autoregressive (slower, high-quality)
- **UnivNet** - Universal vocoder

---

## 🔍 COMMON WORKFLOWS

### Workflow 1: Simple Synthesis

```python
from TTS.api import TTS

# Load model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)

# Synthesize
wav = tts.tts("Hello world")

# Save
tts.tts_to_file("Hello world", file_path="output.wav")
```

---

### Workflow 2: Voice Cloning (XTTS)

```python
from TTS.api import TTS

# Load XTTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Clone voice
tts.tts_to_file(
    text="Hello, this is a cloned voice!",
    speaker_wav="reference_speaker.wav",
    language="en",
    file_path="cloned.wav"
)
```

---

### Workflow 3: Multi-Speaker

```python
from TTS.api import TTS

# Load multi-speaker model
tts = TTS("tts_models/en/vctk/vits")

# List speakers
print(tts.speakers)  # ['p225', 'p226', ...]

# Synthesize with specific speaker
tts.tts_to_file("Hello", speaker="p225", file_path="output.wav")
```

---

### Workflow 4: Advanced (Custom Pipeline)

```python
from TTS.utils.synthesizer import Synthesizer

# Load synthesizer directly
synthesizer = Synthesizer(
    tts_checkpoint="path/to/model.pth",
    tts_config_path="path/to/config.json",
    vocoder_checkpoint="path/to/vocoder.pth",
    vocoder_config="path/to/vocoder_config.json",
    use_cuda=True
)

# Synthesize with custom settings
wav = synthesizer.tts(
    text="Hello world",
    speaker_name="p225",
    style_wav=None,  # Optional GST style
    language_name="en"
)
```

---

## 🐛 DEBUGGING GUIDE

### Issue: Import Error

**Problem**: `ModuleNotFoundError: No module named 'TTS'`
**Solution**: Install package `pip install -e .` from root

---

### Issue: Model Not Found

**Problem**: `Model not found: tts_models/...`
**Solution**:
```python
from TTS.api import TTS
print(TTS().list_models())  # List all available models
```

---

### Issue: CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`
**Solutions**:
1. Use CPU: `gpu=False`
2. Use smaller model (VITS instead of XTTS)
3. Split sentences: `split_sentences=True`
4. Reduce batch size in training

---

### Issue: Poor Audio Quality

**Check**:
1. Sample rate: Model expects 22050 or 16000 Hz
2. Audio normalization: Check `AudioProcessor` settings
3. Vocoder: Try different vocoder (HiFiGAN recommended)
4. Text preprocessing: Check phonemizer output

---

## ⚠️ KNOWN ISSUES

### 1. Config Inconsistency

**Files Affected**: Multiple models
**Problem**: Some use `config.model_args`, others use flat config
**Workaround**: Use `get_from_config_or_model_args(config, key)` helper
**Status**: Needs unification (see MODERNIZATION_ROADMAP.md)

---

### 2. Type Hints Missing

**Files Affected**: All files (0% return type coverage)
**Impact**: Poor IDE support, type safety
**Workaround**: Read docstrings
**Status**: High priority fix (see MODERNIZATION_ROADMAP.md, Phase 2)

---

### 3. Flask Server Limitations

**File**: `TTS/server/server.py`
**Issues**:
- Global lock serializes requests
- No async support
- Path traversal vulnerability
**Status**: Needs FastAPI migration (see MODERNIZATION_ROADMAP.md, Phase 3)

---

## 🚀 PERFORMANCE TIPS

### Faster Inference

1. **Use GPU**: `gpu=True` (2-10x speedup)
2. **Smaller Models**: FastSpeech2, Tacotron2 (vs XTTS, Bark)
3. **Streaming**: Use XTTS streaming for real-time
4. **Caching**: Cache speaker embeddings for repeated voices

### Faster Training

1. **Mixed Precision**: `mixed_precision=True` in config
2. **Batch Size**: Increase `batch_size` (if GPU memory allows)
3. **Workers**: Increase `num_loader_workers` (4-8 recommended)
4. **Bucket Batching**: `batch_group_size > 0` (group similar lengths)

---

## 📚 RELATED DOCUMENTATION

- **api.py docstrings** - Complete API reference
- **TTS/tts/claude.md** - TTS models guide
- **TTS/server/claude.md** - REST API guide
- **ARCHITECTURAL_OVERVIEW.md** - Deep dive
- **QUICK_REFERENCE.md** - Quick start guide

---

## 🎯 NEXT STEPS FOR AI ASSISTANTS

**If working on**:
- **Public API**: Start with `api.py`
- **Models**: Go to `TTS/tts/models/`
- **Training**: Check `TTS/bin/train_tts.py` and `recipes/`
- **Server**: See `TTS/server/server.py`
- **Audio Processing**: Check `TTS/utils/audio/`
- **Text Processing**: See `TTS/tts/utils/text/`

**Always**:
1. Read existing code first
2. Check tests for examples
3. Run `make test` before committing
4. Use `make style` for formatting
5. Update docstrings (97.8% coverage standard)

---

**End of TTS/ claude.md**
