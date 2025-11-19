# TTS/tts/ - Models, Datasets, and Utilities - Claude AI Guide

**Directory**: `/TTS/tts`
**Purpose**: Core TTS models, layers, datasets, and text processing
**Last Updated**: November 19, 2025

---

## 📂 STRUCTURE

```
tts/
├── configs/              # Model-specific configurations (17 files)
├── datasets/             # Dataset loaders and formatters (30+)
├── layers/               # Model-specific layers (12 subdirs)
├── models/               # 🔥 TTS MODEL IMPLEMENTATIONS
└── utils/                # Text processing, audio, helpers
```

---

## 🔥 MODELS (TTS/tts/models/)

### Available Models

| Model | File | Type | Speed | Quality | Voice Clone |
|-------|------|------|-------|---------|-------------|
| **XTTS** | xtts.py | Autoregressive GPT | Slow | ⭐⭐⭐⭐⭐ | ✅ Excellent |
| **VITS** | vits.py | End-to-end | Fast | ⭐⭐⭐⭐⭐ | ⚠️ Limited |
| **Bark** | bark.py | Hierarchical | Very Slow | ⭐⭐⭐⭐ | ✅ Good |
| **Tacotron2** | tacotron2.py | Seq2Seq | Medium | ⭐⭐⭐⭐ | ❌ No |
| **GlowTTS** | glow_tts.py | Flow-based | Fast | ⭐⭐⭐⭐ | ❌ No |
| **FastSpeech2** | fast_speech2.py | Non-AR | Very Fast | ⭐⭐⭐ | ❌ No |
| **YourTTS** | your_tts.py | End-to-end | Medium | ⭐⭐⭐⭐ | ✅ Good |

### Model Selection Guide

**Use XTTS when**:
- Need voice cloning with few samples
- Multilingual support (17 languages)
- Streaming is required
- Quality is priority over speed

**Use VITS when**:
- Need fast, high-quality synthesis
- Single/multi-speaker (pre-defined)
- End-to-end is preferred
- GPU available

**Use Bark when**:
- Need emotional prosody
- Non-speech sounds (laughs, sighs)
- Quality over speed
- Zero-shot voice generation

**Use FastSpeech2/Tacotron2 when**:
- CPU-only environment
- Speed is critical
- Pre-trained voices are sufficient

---

## 📄 KEY MODEL FILES

### 1. **base_tts.py** - Base Class for All Models

**Purpose**: Abstract base class defining TTS interface
**Inherit From**: All TTS models inherit from `BaseTTS`

**Key Methods to Implement**:
```python
class MyTTS(BaseTTS):
    def forward(self, x, x_lengths, ...):
        # Model forward pass
        pass

    def inference(self, x, ...):
        # Inference mode
        pass

    def load_checkpoint(self, config, checkpoint_path, ...):
        # Load pre-trained weights
        pass
```

---

### 2. **xtts.py** - XTTS Model (Voice Cloning)

**Lines**: 790
**Architecture**: GPT (30 layers) + HiFiGAN
**Capabilities**:
- Zero-shot voice cloning (3-30 second reference)
- 17 languages
- Streaming support (`inference_stream()`)
- Temperature-controlled generation

**Key Methods**:
```python
# Get conditioning latents from reference audio
get_conditioning_latents(audio_path, ...)

# Inference (full synthesis)
inference(text, language, gpt_cond_latent, speaker_embedding, ...)

# Streaming inference
inference_stream(text, language, ..., stream_chunk_size=20)
```

**Example**:
```python
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path)

# Get voice embedding
gpt_cond, speaker_emb = model.get_conditioning_latents("reference.wav")

# Stream synthesis
for chunk in model.inference_stream(
    text="Hello world",
    language="en",
    gpt_cond_latent=gpt_cond,
    speaker_embedding=speaker_emb
):
    # Process audio chunk
    pass
```

---

### 3. **vits.py** - VITS Model (Fast End-to-End)

**Lines**: 750+
**Architecture**: Variational Inference + Flow-based
**Capabilities**:
- End-to-end waveform generation (no vocoder needed)
- Multi-speaker support
- Fast inference (~1s on GPU)

**Key Methods**:
```python
forward(x, x_lengths, wav, g=None)  # Training
inference(x, x_lengths, g=None)      # Inference
```

---

### 4. **bark.py** - Bark Model (Emotional Speech)

**Lines**: 285
**Architecture**: Hierarchical transformers
**Capabilities**:
- Emotional prosody via text markers: `[happy]`, `[sad]`
- Non-speech sounds: `[laughs]`, `[sighs]`
- Zero-shot voice cloning
- Multilingual (via BERT)

**Special Syntax**:
```python
text = "Hello [happy]! Oh no [scared]!"
model.generate_audio(text, history_prompt="voice_template")
```

---

## 🧱 LAYERS (TTS/tts/layers/)

**Problem**: 12 layer subdirectories with duplication
**Recommendation**: See MODERNIZATION_ROADMAP.md for consolidation plan

### Layer Directories

| Directory | Purpose | Models Using It |
|-----------|---------|----------------|
| **xtts/** | XTTS layers | XTTS |
| **vits/** | VITS layers | VITS, YourTTS |
| **glow_tts/** | GlowTTS layers | GlowTTS |
| **tacotron/** | Tacotron layers | Tacotron2 |
| **bark/** | Bark layers | Bark |
| **delightful_tts/** | DelightfulTTS | DelightfulTTS |
| **overflow/** | Overflow | Overflow |
| **tortoise/** | Tortoise | Tortoise |
| **feed_forward/** | FFN | Multiple |
| **generic/** | Shared layers | Multiple |
| **align_tts/** | AlignTTS | AlignTTS |

### Common Layer Patterns

**Attention Mechanisms** (8+ implementations):
- Tacotron attention (location-sensitive)
- Transformer attention (multi-head)
- Glow attention (flow-based)
- Alignment attention

**⚠️ Duplication**: Many attention variants can be unified

**Normalization Layers**:
- LayerNorm (5+ implementations)
- InstanceNorm
- GroupNorm

---

## 📊 DATASETS (TTS/tts/datasets/)

### Key Files

| File | Purpose |
|------|---------|
| **dataset.py** | Main dataset class |
| **formatters.py** | 30+ dataset format parsers |

### Dataset Formatters (formatters.py)

**Popular Datasets**:
- `ljspeech()` - LJSpeech format
- `vctk()` - VCTK format
- `libritts()` - LibriTTS format
- `common_voice()` - Mozilla Common Voice
- `mailabs()` - M-AILABS format

**Custom Format**:
```python
def my_formatter(root_path, meta_file, **kwargs):
    """
    Returns: List[List[str, str, str]]
        [audio_file, text, speaker_name]
    """
    items = []
    # Parse your dataset
    items.append([wav_path, text, speaker_id])
    return items
```

### Dataset Class (dataset.py)

**Purpose**: Load and preprocess data for training

**Key Methods**:
```python
__init__(
    outputs_per_step,    # Frames per decoder step
    compute_linear_spec, # Compute linear spectrogram
    ap,                  # AudioProcessor
    ...
)

__getitem__(idx)  # Load single sample
load_data(idx)    # Load and preprocess audio+text
```

---

## 📝 TEXT PROCESSING (TTS/tts/utils/text/)

### Structure

```
text/
├── characters.py           # Character sets for languages
├── phonemizers/            # G2P (grapheme-to-phoneme)
│   ├── base.py
│   ├── espeak_wrapper.py
│   ├── gruut_wrapper.py
│   └── ... (8+ phonemizers)
├── bangla/                 # Bangla text processing
├── chinese_mandarin/       # Chinese processing
├── english/                # English processing
├── french/                 # French processing
├── japanese/               # Japanese processing
├── korean/                 # Korean processing
└── belarusian/             # Belarusian processing
```

### Phonemizers

**Available Phonemizers**:
1. **ESpeak** (espeak_wrapper.py) - 100+ languages
2. **Gruut** (gruut_wrapper.py) - 13 languages, better quality
3. **JA** (ja_jp_phonemizer.py) - Japanese (MeCab)
4. **ZH** (zh_cn_phonemizer.py) - Chinese (Pinyin)
5. **KO** (ko_kr_phonemizer.py) - Korean (Hangul romanization)
6. **BN** (bn_phonemizer.py) - Bangla
7. **BE** (belarusian_phonemizer.py) - Belarusian

**Example**:
```python
from TTS.tts.utils.text.phonemizers import Gruut

phonemizer = Gruut(language="en-us")
phonemes = phonemizer.phonemize("Hello world")
# Output: "həloʊ wɜrld"
```

---

## ⚙️ CONFIGS (TTS/tts/configs/)

### Config Files (17 Total)

| File | Model |
|------|-------|
| **xtts_config.py** | XTTS |
| **vits_config.py** | VITS |
| **glow_tts_config.py** | GlowTTS |
| **tacotron2_config.py** | Tacotron2 |
| **bark_config.py** | Bark |
| **fast_speech_config.py** | FastSpeech |
| **fast_pitch_config.py** | FastPitch |
| **align_tts_config.py** | AlignTTS |
| **overflow_config.py** | Overflow |
| **delightful_tts_config.py** | DelightfulTTS |
| **shared_configs.py** | Shared base configs |

### Config Pattern

```python
from dataclasses import dataclass, field
from TTS.tts.configs.shared_configs import BaseTTSConfig

@dataclass
class MyModelConfig(BaseTTSConfig):
    model: str = "my_model"

    # Model-specific config
    num_layers: int = 6
    hidden_dim: int = 512

    # Audio config
    sample_rate: int = 22050
    hop_length: int = 256
```

---

## 🔨 COMMON WORKFLOWS

### Workflow 1: Add New Model

1. **Create config**: `TTS/tts/configs/mymodel_config.py`
```python
@dataclass
class MyModelConfig(BaseTTSConfig):
    model: str = "my_model"
    # ... model-specific config
```

2. **Create model**: `TTS/tts/models/mymodel.py`
```python
from TTS.tts.models.base_tts import BaseTTS

class MyModel(BaseTTS):
    def __init__(self, config):
        super().__init__()
        # Initialize model

    def forward(self, x, x_lengths):
        # Forward pass
        pass

    def inference(self, x, x_lengths):
        # Inference
        pass
```

3. **Add layers** (if needed): `TTS/tts/layers/mymodel/*.py`

4. **Register model**: `TTS/tts/models/__init__.py`
```python
from TTS.tts.models.mymodel import MyModel
```

5. **Create training recipe**: `recipes/ljspeech/mymodel/train.py`

6. **Add tests**: `tests/tts_tests2/test_mymodel.py`

---

### Workflow 2: Add New Language

1. **Add phonemizer** (if needed): `TTS/tts/utils/text/phonemizers/my_lang.py`
```python
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

class MyLangPhonemizer(BasePhonemizer):
    def __init__(self, language):
        super().__init__(language)

    def phonemize(self, text, separator="|"):
        # Convert text to phonemes
        pass
```

2. **Add character set**: `TTS/tts/utils/text/characters.py`
```python
MyLangCharacters = "abcdefg..."
```

3. **Update model config**: Add language to supported list
```python
languages = ["en", "fr", "my_lang"]
```

4. **Test**: `tests/text_tests/test_my_lang_phonemizer.py`

---

### Workflow 3: Add New Dataset

1. **Create formatter**: `TTS/tts/datasets/formatters.py`
```python
def my_dataset(root_path, meta_file, **kwargs):
    """Parse my dataset format"""
    items = []
    # Read dataset
    with open(meta_file) as f:
        for line in f:
            wav_path, text, speaker = line.strip().split("|")
            items.append([wav_path, text, speaker])
    return items
```

2. **Use in config**:
```python
dataset_config = BaseDatasetConfig(
    formatter="my_dataset",
    meta_file_train="train.txt",
    meta_file_val="val.txt",
    path="/path/to/dataset"
)
```

---

## 🐛 DEBUGGING

### Issue: Model Not Loading

**Check**:
1. Config file path correct?
2. Checkpoint file exists?
3. Model name in config matches model class?

**Debug**:
```python
import torch
checkpoint = torch.load("model.pth")
print(checkpoint.keys())  # See what's in checkpoint
```

---

### Issue: Poor Quality Output

**Check**:
1. Audio sample rate (config vs actual)
2. Text preprocessing (phonemes correct?)
3. Speaker embedding (if multi-speaker)
4. Vocoder quality

**Debug**:
```python
# Check phonemes
from TTS.tts.utils.text.phonemizers import Gruut
phonemizer = Gruut("en-us")
print(phonemizer.phonemize("Test text"))

# Check audio
import torchaudio
wav, sr = torchaudio.load("test.wav")
print(f"Sample rate: {sr}, Shape: {wav.shape}")
```

---

### Issue: CUDA Out of Memory

**Solutions**:
1. Reduce batch size
2. Reduce sequence length
3. Use gradient checkpointing
4. Enable mixed precision training

---

## ⚠️ KNOWN ISSUES

### 1. Layer Duplication

**Problem**: 12 layer directories, many duplicates
**Impact**: Maintenance burden, code bloat
**Fix**: Consolidate (see MODERNIZATION_ROADMAP.md, Phase 1)

### 2. Config Inconsistency

**Problem**: Mixed `config.model_args` vs flat
**Files**: Multiple models
**Fix**: Unify (see MODERNIZATION_ROADMAP.md, Phase 1)

### 3. No Type Hints

**Impact**: Poor IDE support
**Fix**: Add type hints (see MODERNIZATION_ROADMAP.md, Phase 2)

---

## 📚 RELATED DOCS

- **ARCHITECTURAL_OVERVIEW.md** - Full structure
- **QUICK_REFERENCE.md** - Quick patterns
- **TTS/models/claude.md** - Model-specific guide
- **docs/source/implementing_a_new_model.md** - Official guide

---

**End of TTS/tts/ claude.md**
