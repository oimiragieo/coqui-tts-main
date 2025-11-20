# Getting Started with Coqui TTS

**Welcome to Coqui TTS!** This guide will help you get up and running quickly with text-to-speech synthesis.

**Last Updated**: November 20, 2025

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Common Use Cases](#common-use-cases)
5. [Model Selection Guide](#model-selection-guide)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 18.04+, macOS, or Windows 10+
- **Python**: 3.9, 3.10, or 3.11
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: 2-10GB for models (varies by model)
- **GPU** (optional but recommended): CUDA-compatible GPU with 4GB+ VRAM

### Check Your Python Version

```bash
python --version
# Should show Python 3.9.x, 3.10.x, or 3.11.x
```

If you need to install Python, visit [python.org](https://www.python.org/downloads/).

---

## Installation

### Option 1: Quick Install (Recommended for Most Users)

For inference and using pre-trained models:

```bash
pip install TTS
```

### Option 2: Development Install

For contributing, training models, or development:

```bash
# Clone the repository
git clone https://github.com/coqui-ai/TTS
cd TTS

# Install in development mode with all extras
pip install -e ".[all,dev,notebooks]"

# Install pre-commit hooks (for contributors)
pre-commit install
```

### Option 3: Platform-Specific Instructions

**Ubuntu/Debian**:
```bash
# Install system dependencies
sudo apt-get install -y espeak-ng libsndfile1-dev

# Install TTS
pip install TTS
```

**macOS**:
```bash
# Install espeak (required for phonemization)
brew install espeak-ng

# Install TTS
pip install TTS
```

**Windows**:
See [@GuyPaddock's detailed instructions](https://stackoverflow.com/questions/66726331/how-can-i-run-mozilla-tts-coqui-tts-training-with-cuda-on-a-windows-system)

### Verify Installation

```bash
tts --help
```

You should see the TTS command-line interface help message.

---

## Quick Start

### 1. Your First Text-to-Speech

**Command Line** (easiest):

```bash
tts --text "Hello world! This is my first text to speech." --out_path output.wav
```

The model will download automatically on first run (~100-500MB depending on model).

**Python API**:

```python
from TTS.api import TTS

# Initialize TTS (downloads model on first run)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Generate speech
tts.tts_to_file(text="Hello world!", file_path="output.wav")
```

### 2. List Available Models

```bash
tts --list_models
```

Or in Python:

```python
from TTS.api import TTS

# List all available models
print(TTS().list_models())
```

### 3. Using a Different Model

```bash
tts --text "Hello world" \
    --model_name "tts_models/en/ljspeech/vits" \
    --out_path output.wav
```

---

## Common Use Cases

### Use Case 1: Simple English Text-to-Speech

**Best Model**: `tts_models/en/ljspeech/tacotron2-DDC` or `tts_models/en/ljspeech/vits`

**Command Line**:
```bash
tts --text "Your text here" --out_path output.wav
```

**Python**:
```python
from TTS.api import TTS

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file("Your text here", file_path="output.wav")
```

---

### Use Case 2: Multilingual Text-to-Speech with Voice Cloning

**Best Model**: `tts_models/multilingual/multi-dataset/xtts_v2` (16 languages)

**Supported Languages**: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko)

**Python Example**:
```python
from TTS.api import TTS

# Initialize XTTS v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Clone a voice and speak in English
tts.tts_to_file(
    text="Hello, this is voice cloning!",
    speaker_wav="path/to/your/voice_sample.wav",  # 6+ seconds recommended
    language="en",
    file_path="output_en.wav"
)

# Same voice, different language
tts.tts_to_file(
    text="Hola, esto es clonación de voz!",
    speaker_wav="path/to/your/voice_sample.wav",
    language="es",
    file_path="output_es.wav"
)
```

**Tips for Voice Cloning**:
- Use clear audio with minimal background noise
- 6-10 seconds of speech is ideal
- WAV format at 22050 Hz or 16000 Hz recommended
- Single speaker only in the reference audio

---

### Use Case 3: Multi-Speaker TTS (Pre-defined Voices)

**Best Model**: `tts_models/en/vctk/vits`

**Python Example**:
```python
from TTS.api import TTS

# Initialize multi-speaker model
tts = TTS("tts_models/en/vctk/vits")

# List available speakers
print(f"Available speakers: {tts.speakers}")

# Generate speech with specific speaker
tts.tts_to_file(
    text="Hello from speaker p225!",
    speaker="p225",
    file_path="output_p225.wav"
)

# Try a different speaker
tts.tts_to_file(
    text="Hello from speaker p226!",
    speaker="p226",
    file_path="output_p226.wav"
)
```

---

### Use Case 4: Voice Conversion

**Best Model**: `voice_conversion_models/multilingual/vctk/freevc24`

**Python Example**:
```python
from TTS.api import TTS

# Initialize voice conversion model
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24")

# Convert source voice to target voice
tts.voice_conversion_to_file(
    source_wav="my_voice.wav",      # Voice to convert
    target_wav="target_voice.wav",  # Voice to sound like
    file_path="converted_output.wav"
)
```

---

### Use Case 5: High-Quality Emotional Speech (Bark)

**Best Model**: `tts_models/multilingual/multi-dataset/bark`

**Features**: Emotional speech, laughter, sighs, music

**Python Example**:
```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/bark").to("cuda")

# Generate emotional speech
tts.tts_to_file(
    text="[LAUGHS] This is so funny! [SIGHS] But also quite impressive.",
    file_path="output_bark.wav"
)
```

**Special Tokens**:
- `[LAUGHS]` - Laughter
- `[SIGHS]` - Sighing
- `[MUSIC]` - Music generation
- `[GASPS]` - Gasping

---

### Use Case 6: Running as a Server

**Start the server**:
```bash
tts-server --model_name tts_models/en/ljspeech/tacotron2-DDC
```

**Access the web UI**: Open http://localhost:5002 in your browser

**Use the API**:
```bash
curl -X POST http://localhost:5002/api/tts \
    -d "text=Hello from the API" \
    --output output.wav
```

**Python requests**:
```python
import requests

response = requests.post(
    "http://localhost:5002/api/tts",
    data={"text": "Hello from Python"}
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

---

### Use Case 7: Docker Deployment

**CPU Version**:
```bash
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
python3 TTS/server/server.py --model_name tts_models/en/ljspeech/vits
```

Access at http://localhost:5002

**GPU Version** (with NVIDIA GPU):
```bash
docker run --rm -it --gpus all -p 5002:5002 ghcr.io/coqui-ai/tts-gpu
```

---

## Model Selection Guide

### By Speed (Fastest to Slowest)

1. **FastPitch** - Fastest, good quality
2. **Tacotron2** - Fast, very good quality
3. **VITS** - Medium speed, excellent quality
4. **XTTS** - Slower, multilingual with voice cloning
5. **Tortoise** - Slowest, highest quality

### By Quality (Highest to Good)

1. **Tortoise** - Highest quality, very slow
2. **XTTS** - Excellent quality, multilingual
3. **VITS** - Excellent quality, fast
4. **Bark** - Great quality, emotional
5. **Tacotron2** - Very good quality, fast

### By Features

| Feature | Recommended Models |
|---------|-------------------|
| **Multilingual** | XTTS v2, YourTTS, Bark |
| **Voice Cloning** | XTTS v2, YourTTS, FreeVC |
| **Multi-Speaker** | VITS (VCTK), YourTTS |
| **Emotional Speech** | Bark |
| **Fastest Inference** | FastPitch, FastSpeech2 |
| **Best Quality** | Tortoise, VITS |
| **Production/Real-time** | VITS, XTTS streaming |
| **Speech Editing** | VoiceCraft-X |

### By Language Support

- **English only**: Tacotron2, FastPitch, Tortoise
- **16 languages**: XTTS v2
- **Multilingual (many)**: Bark, YourTTS
- **1100+ languages**: Fairseq models (via `tts_models/<lang-code>/fairseq/vits`)

---

## Troubleshooting

### Issue: Model Download Fails

**Symptoms**: Download timeout or connection error

**Solutions**:
1. Check internet connection
2. Try again (downloads resume automatically)
3. Manually download from [releases](https://github.com/coqui-ai/TTS/releases)
4. Set environment variable: `export TTS_HOME=/path/to/models`

---

### Issue: "espeak not found" Error

**Symptoms**: `OSError: espeak not found`

**Solutions**:

**Ubuntu/Debian**:
```bash
sudo apt-get install espeak-ng
```

**macOS**:
```bash
brew install espeak-ng
```

**Windows**: Download from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)

---

### Issue: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Use CPU instead of GPU:
   ```python
   tts = TTS("model_name").to("cpu")
   ```
2. Use a smaller model (e.g., Tacotron2 instead of XTTS)
3. Close other GPU-using applications
4. Reduce batch size during training

---

### Issue: Poor Audio Quality

**Possible Causes & Solutions**:

1. **Wrong sample rate**: Ensure input audio matches model expectations (usually 22050 Hz)
   ```python
   # Resample audio to 22050 Hz before using
   import librosa
   audio, sr = librosa.load("input.wav", sr=22050)
   ```

2. **Background noise**: Use clean audio for voice cloning

3. **Text issues**: Check for special characters or formatting
   ```python
   # Clean text
   text = text.strip().replace("\n", " ")
   ```

4. **Wrong model**: Try a different vocoder (e.g., HiFiGAN)

---

### Issue: Slow Inference

**Solutions**:

1. **Use GPU**:
   ```python
   tts = TTS("model_name").to("cuda")
   ```

2. **Enable PyTorch 2.0+ optimizations**:
   ```python
   from TTS.api import TTS
   from TTS.utils.torch_compile import maybe_compile

   tts = TTS("model_name").to("cuda")
   tts.synthesizer.tts_model = maybe_compile(tts.synthesizer.tts_model)
   ```

3. **Use faster models**: FastPitch, Tacotron2, or VITS

4. **Enable streaming** (XTTS only):
   ```python
   # See examples in documentation
   ```

---

### Issue: Import Errors

**Symptoms**: `ModuleNotFoundError: No module named 'TTS'`

**Solutions**:
1. Ensure TTS is installed: `pip install TTS`
2. Check Python version: `python --version` (must be 3.9-3.11)
3. Verify installation: `pip show TTS`
4. Reinstall: `pip uninstall TTS && pip install TTS`

---

## Next Steps

### For Users

1. **Explore Examples**: Check the [examples/](../examples/) directory
2. **Try Different Models**: Experiment with various models for your use case
3. **Read Model Docs**: See model-specific documentation in [docs/source/models/](source/models/)
4. **Join Community**:
   - [Discord](https://discord.gg/5eXr5seRrv)
   - [GitHub Discussions](https://github.com/coqui-ai/TTS/discussions)

### For Developers

1. **Contributing Guide**: Read [CONTRIBUTING.md](../CONTRIBUTING.md)
2. **Architecture Overview**: See [Architecture Overview](architecture/ARCHITECTURAL_OVERVIEW.md)
3. **Development Setup**:
   ```bash
   git clone https://github.com/coqui-ai/TTS
   cd TTS
   pip install -e ".[all,dev,notebooks]"
   pre-commit install
   ```
4. **Run Tests**: `pytest tests/`

### For Training Custom Models

1. **Dataset Preparation**: See [Formatting Your Dataset](source/formatting_your_dataset.md)
2. **Training Guide**: Read [Training a Model](source/training_a_model.md)
3. **Training Recipes**: Check [recipes/](../recipes/) directory
4. **Beginner Tutorial**: See [Tutorial for Nervous Beginners](source/tutorial_for_nervous_beginners.md)

---

## Additional Resources

### Documentation

- **[Quick Reference](architecture/QUICK_REFERENCE.md)** - Fast overview
- **[Architecture Overview](architecture/ARCHITECTURAL_OVERVIEW.md)** - Deep dive
- **[TTS Package Guide](development/TTS_PACKAGE_GUIDE.md)** - API reference
- **[Migration Guide](development/MIGRATION_GUIDE.md)** - Upgrading versions
- **[Documentation Index](DOCUMENTATION_INDEX.md)** - Full navigation

### Model Documentation

- **[XTTS](source/models/xtts.md)** - Multilingual voice cloning
- **[Bark](source/models/bark.md)** - Emotional speech
- **[Tortoise](source/models/tortoise.md)** - High-quality synthesis
- **[VITS](source/models/vits.md)** - Fast end-to-end TTS
- **[VoiceCraft-X](models/voicecraft_x.md)** - Speech editing

### External Links

- **[Official Documentation](https://tts.readthedocs.io/)**
- **[GitHub Repository](https://github.com/coqui-ai/TTS)**
- **[TTS Papers](https://github.com/erogol/TTS-papers)** - Research papers
- **[Model Zoo](https://github.com/coqui-ai/TTS/wiki/Experimental-Released-Models)** - Released models

---

## Getting Help

### Where to Ask Questions

| Type | Platform |
|------|----------|
| **Bug Reports** | [GitHub Issues](https://github.com/coqui-ai/TTS/issues) |
| **Feature Requests** | [GitHub Issues](https://github.com/coqui-ai/TTS/issues) |
| **Usage Questions** | [GitHub Discussions](https://github.com/coqui-ai/TTS/discussions) |
| **General Discussion** | [Discord](https://discord.gg/5eXr5seRrv) |

### Before Asking

1. Check this Getting Started guide
2. Search [existing issues](https://github.com/coqui-ai/TTS/issues)
3. Read the [FAQ](source/faq.md)
4. Check [Troubleshooting](#troubleshooting) section above

---

## Quick Tips

- **First run downloads models** - Be patient! Models are 100MB-1GB
- **GPU is recommended** - Especially for large models (XTTS, Bark, Tortoise)
- **Use `.to("cuda")` or `.to("cpu")`** - Control device placement
- **PyTorch 2.0+** - Get 20-40% speedup with `torch.compile()`
- **Clean audio for cloning** - 6-10 seconds, minimal background noise
- **Start simple** - Begin with Tacotron2 or VITS, then explore

---

**Happy Synthesizing!** 🎙️

If you find Coqui TTS useful, please consider:
- ⭐ Starring the [repository](https://github.com/coqui-ai/TTS)
- 📢 Sharing with others
- 🤝 Contributing improvements
- 📖 Improving documentation

---

**Version**: 1.0
**Last Updated**: November 20, 2025
**Compatible with**: Coqui TTS 0.22.0+
