
## 🐸 Coqui.ai News

### 🆕 Latest Updates (November 2025)

#### **NEW: VoiceCraft-X - Next Generation Multilingual TTS** 🚀
- ✨ **Unified TTS + Speech Editing**: Single model for both text-to-speech AND seamless audio editing
- 🌍 **True Multilingual**: 11+ languages without phoneme conversion (powered by Qwen3 LLM)
- 🎯 **State-of-the-Art Quality**: EnCodec-style 4-codebook architecture with 50Hz temporal resolution
- ⚡ **Zero-Shot Voice Cloning**: Advanced speaker embeddings with CAM++ voiceprint model
- 📝 **See the [VoiceCraft-X Guide](docs/models/voicecraft_x.md)** for details and implementation

#### Modernization & Performance
- ✅ **PyTorch 2.0+**: `torch.compile()` support for 20-40% faster inference
- ✅ **Type Safety**: Comprehensive type hints throughout the codebase
- ✅ **Security Hardened**: Vulnerabilities addressed, dependencies updated
- ✅ **Modern Testing**: Migrated to pytest with better coverage and parallel execution
- ✅ **Improved Documentation**: Reorganized with comprehensive guides and examples

### 🎙️ Available TTS Models
- 🆕 **VoiceCraft-X**: Multilingual TTS + speech editing (11+ languages) [Docs](docs/models/voicecraft_x.md)
- 📣 **ⓍTTSv2**: 16 languages, voice cloning, streaming <200ms [Docs](https://tts.readthedocs.io/en/dev/models/xtts.html)
- 📣 **🐶 Bark**: Emotional speech, unconstrained voice cloning [Docs](https://tts.readthedocs.io/en/dev/models/bark.html)
- 📣 **🐢 Tortoise**: High-quality, slower inference [Docs](https://tts.readthedocs.io/en/dev/models/tortoise.html)
- 📣 **VITS**: Fast end-to-end synthesis with multi-speaker support
- 📣 **~1100 languages**: Via [Fairseq models](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)

<div align="center">
<img src="https://static.scarf.sh/a.png?x-pxid=cf317fe7-2188-4721-bc01-124bb5d5dbb2" />

## <img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/coqui-log-green-TTS.png" height="56"/>


**🐸TTS is a library for advanced Text-to-Speech generation.**

🚀 Pretrained models in +1100 languages.

🛠️ Tools for training new models and fine-tuning existing models in any language.

📚 Utilities for dataset analysis and curation.

______________________________________________________________________

## 📋 Table of Contents

- [🚀 Getting Started](#-getting-started---step-by-step) - **Start here if you're new!**
- [Installation](#installation)
- [Model Implementations](#model-implementations)
- [Usage Examples](#synthesizing-speech-by-tts)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation--resources)
- [Contributing](#-contributing)

______________________________________________________________________

[![Discord](https://img.shields.io/discord/1037326658807533628?color=%239B59B6&label=chat%20on%20discord)](https://discord.gg/5eXr5seRrv)
[![License](<https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg>)](https://opensource.org/licenses/MPL-2.0)
[![PyPI version](https://badge.fury.io/py/TTS.svg)](https://badge.fury.io/py/TTS)
[![Covenant](https://camo.githubusercontent.com/7d620efaa3eac1c5b060ece5d6aacfcc8b81a74a04d05cd0398689c01c4463bb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f6e7472696275746f72253230436f76656e616e742d76322e3025323061646f707465642d6666363962342e737667)](https://github.com/coqui-ai/TTS/blob/master/CODE_OF_CONDUCT.md)
[![Downloads](https://pepy.tech/badge/tts)](https://pepy.tech/project/tts)
[![DOI](https://zenodo.org/badge/265612440.svg)](https://zenodo.org/badge/latestdoi/265612440)

![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/aux_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/data_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/docker.yaml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/inference_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/style_check.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/text_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/tts_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/vocoder_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/zoo_tests0.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/zoo_tests1.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/zoo_tests2.yml/badge.svg)
[![Docs](<https://readthedocs.org/projects/tts/badge/?version=latest&style=plastic>)](https://tts.readthedocs.io/en/latest/)

</div>

______________________________________________________________________

## 💬 Where to ask questions
Please use our dedicated channels for questions and discussion. Help is much more valuable if it's shared publicly so that more people can benefit from it.

| Type                            | Platforms                               |
| ------------------------------- | --------------------------------------- |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker]                  |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker]                  |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]                    |
| 🗯 **General Discussion**       | [GitHub Discussions] or [Discord]   |

[github issue tracker]: https://github.com/coqui-ai/tts/issues
[github discussions]: https://github.com/coqui-ai/TTS/discussions
[discord]: https://discord.gg/5eXr5seRrv
[Tutorials and Examples]: https://github.com/coqui-ai/TTS/wiki/TTS-Notebooks-and-Tutorials


## 🔗 Links and Resources

| Type | Links |
| ---- | ----- |
| 💼 **Documentation** | [ReadTheDocs](https://tts.readthedocs.io/en/latest/) • [Quick Reference](docs/architecture/QUICK_REFERENCE.md) • [Architecture Overview](docs/architecture/ARCHITECTURAL_OVERVIEW.md) |
| 📚 **Developer Guides** | [Contributing](CONTRIBUTING.md) • [Migration Guide](docs/development/MIGRATION_GUIDE.md) • [Modernization Roadmap](docs/development/MODERNIZATION_ROADMAP.md) |
| 🚀 **Released Models** | [TTS Releases](https://github.com/coqui-ai/TTS/releases) • [Model Zoo](https://github.com/coqui-ai/TTS/wiki/Experimental-Released-Models) |
| 📖 **Learning** | [Tutorial for Beginners](docs/source/tutorial_for_nervous_beginners.md) • [TTS Papers](https://github.com/erogol/TTS-papers) |
| 🗺️ **Project** | [Roadmap](docs/development/MODERNIZATION_ROADMAP.md) • [Changelog](CHANGELOG.md) • [Documentation Index](docs/DOCUMENTATION_INDEX.md) |


## 🥇 TTS Performance
<p align="center"><img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/TTS-performance.png" width="800" /></p>

Underlined "TTS*" and "Judy*" are **internal** 🐸TTS models that are not released open-source. They are here to show the potential. Models prefixed with a dot (.Jofish .Abe and .Janice) are real human voices.

## Features
- High-performance Deep Learning models for Text2Speech tasks.
    - Text2Spec models (Tacotron, Tacotron2, Glow-TTS, SpeedySpeech).
    - Speaker Encoder to compute speaker embeddings efficiently.
    - Vocoder models (MelGAN, Multiband-MelGAN, GAN-TTS, ParallelWaveGAN, WaveGrad, WaveRNN)
- Fast and efficient model training.
- Detailed training logs on the terminal and Tensorboard.
- Support for Multi-speaker TTS.
- Efficient, flexible, lightweight but feature complete `Trainer API`.
- Released and ready-to-use models.
- Tools to curate Text2Speech datasets under```dataset_analysis```.
- Utilities to use and test your models.
- Modular (but not too much) code base enabling easy implementation of new ideas.

## Model Implementations
### Spectrogram models
- Tacotron: [paper](https://arxiv.org/abs/1703.10135)
- Tacotron2: [paper](https://arxiv.org/abs/1712.05884)
- Glow-TTS: [paper](https://arxiv.org/abs/2005.11129)
- Speedy-Speech: [paper](https://arxiv.org/abs/2008.03802)
- Align-TTS: [paper](https://arxiv.org/abs/2003.01950)
- FastPitch: [paper](https://arxiv.org/pdf/2006.06873.pdf)
- FastSpeech: [paper](https://arxiv.org/abs/1905.09263)
- FastSpeech2: [paper](https://arxiv.org/abs/2006.04558)
- SC-GlowTTS: [paper](https://arxiv.org/abs/2104.05557)
- Capacitron: [paper](https://arxiv.org/abs/1906.03402)
- OverFlow: [paper](https://arxiv.org/abs/2211.06892)
- Neural HMM TTS: [paper](https://arxiv.org/abs/2108.13320)
- Delightful TTS: [paper](https://arxiv.org/abs/2110.12612)

### End-to-End Models
- **VoiceCraft-X**: [paper](https://arxiv.org/abs/2511.12347) [docs](docs/models/voicecraft_x.md) ⭐ NEW
- ⓍTTS: [blog](https://coqui.ai/blog/tts/open_xtts)
- VITS: [paper](https://arxiv.org/pdf/2106.06103)
- 🐸 YourTTS: [paper](https://arxiv.org/abs/2112.02418)
- 🐢 Tortoise: [orig. repo](https://github.com/neonbjb/tortoise-tts)
- 🐶 Bark: [orig. repo](https://github.com/suno-ai/bark)

### Attention Methods
- Guided Attention: [paper](https://arxiv.org/abs/1710.08969)
- Forward Backward Decoding: [paper](https://arxiv.org/abs/1907.09006)
- Graves Attention: [paper](https://arxiv.org/abs/1910.10288)
- Double Decoder Consistency: [blog](https://erogol.com/solving-attention-problems-of-tts-models-with-double-decoder-consistency/)
- Dynamic Convolutional Attention: [paper](https://arxiv.org/pdf/1910.10288.pdf)
- Alignment Network: [paper](https://arxiv.org/abs/2108.10447)

### Speaker Encoder
- GE2E: [paper](https://arxiv.org/abs/1710.10467)
- Angular Loss: [paper](https://arxiv.org/pdf/2003.11982.pdf)

### Vocoders
- MelGAN: [paper](https://arxiv.org/abs/1910.06711)
- MultiBandMelGAN: [paper](https://arxiv.org/abs/2005.05106)
- ParallelWaveGAN: [paper](https://arxiv.org/abs/1910.11480)
- GAN-TTS discriminators: [paper](https://arxiv.org/abs/1909.11646)
- WaveRNN: [origin](https://github.com/fatchord/WaveRNN/)
- WaveGrad: [paper](https://arxiv.org/abs/2009.00713)
- HiFiGAN: [paper](https://arxiv.org/abs/2010.05646)
- UnivNet: [paper](https://arxiv.org/abs/2106.07889)

### Voice Conversion
- FreeVC: [paper](https://arxiv.org/abs/2210.15418)

You can also help us implement more models.

## Installation

🐸TTS is tested on Ubuntu 18.04+ with **Python 3.9, 3.10, and 3.11**.

### Quick Install (PyPI)

For inference only (recommended for most users):

```bash
pip install TTS
```

### Development Install

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

### Platform-Specific Instructions

**Ubuntu/Debian:**
```bash
make system-deps  # Install system dependencies
make install      # Install TTS
```

**Windows:**
See [@GuyPaddock's instructions](https://stackoverflow.com/questions/66726331/how-can-i-run-mozilla-tts-coqui-tts-training-with-cuda-on-a-windows-system)

**macOS:**
```bash
brew install espeak  # Required for phonemization
pip install TTS
```

### Optional: Enable Performance Optimizations

For 20-40% faster inference with PyTorch 2.0+:

```python
from TTS.api import TTS
from TTS.utils.torch_compile import maybe_compile

tts = TTS("tts_models/en/ljspeech/vits").to("cuda")
tts.synthesizer.tts_model = maybe_compile(tts.synthesizer.tts_model)
```

See the [Migration Guide](docs/development/MIGRATION_GUIDE.md) for more details.

---

## 🚀 Getting Started - Step by Step

New to Coqui TTS? Follow these simple steps to get up and running!

### Step 1: Install Coqui TTS

**For most users (inference only):**
```bash
pip install TTS
```

**For advanced users (development/training):**
```bash
git clone https://github.com/coqui-ai/TTS
cd TTS
pip install -e ".[all,dev,notebooks]"
```

### Step 2: Choose Your Use Case

#### 🎤 Use Case A: Simple Text-to-Speech (Easiest)

Generate speech from text using a pre-trained English model:

```bash
# Command-line (simplest)
tts --text "Hello world! This is Coqui TTS." --out_path output.wav

# The model will download automatically on first run
```

**Python API:**
```python
from TTS.api import TTS

# Initialize TTS (downloads model on first run)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Generate speech
tts.tts_to_file(text="Hello world!", file_path="output.wav")
```

#### 🌍 Use Case B: Multilingual Text-to-Speech

Speak in multiple languages with voice cloning:

```python
from TTS.api import TTS

# Initialize XTTS v2 (supports 16 languages)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Clone a voice and speak in English
tts.tts_to_file(
    text="Hello, this is voice cloning!",
    speaker_wav="path/to/your/voice_sample.wav",
    language="en",
    file_path="output.wav"
)

# Same voice, different language
tts.tts_to_file(
    text="Hola, esto es clonación de voz!",
    speaker_wav="path/to/your/voice_sample.wav",
    language="es",
    file_path="output_spanish.wav"
)
```

#### ✨ Use Case C: VoiceCraft-X - Speech Editing (NEW!)

Edit existing audio or generate speech in 11+ languages:

```python
from TTS.tts.models.voicecraft_x import VoiceCraftX, VoiceCraftXConfig
import torch

# Initialize VoiceCraft-X
config = VoiceCraftXConfig(num_codebooks=4, codebook_size=2048)
model = VoiceCraftX(config)

# Generate speech with voice cloning
prompt_audio = torch.randn(1, 16000 * 3)  # Load your audio here
output = model.inference_tts(
    text="Hello from VoiceCraft-X!",
    prompt_audio=prompt_audio,
    temperature=1.0,
    top_k=20
)

# See examples/voicecraft_x_example.py for more details
```

#### 🎨 Use Case D: Voice Conversion

Convert one voice to sound like another:

```python
from TTS.api import TTS

# Initialize voice conversion model
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24")

# Convert source voice to target voice
tts.voice_conversion_to_file(
    source_wav="my_voice.wav",
    target_wav="target_voice.wav",
    file_path="converted_output.wav"
)
```

### Step 3: Explore Available Models

List all available pre-trained models:

```bash
tts --list_models
```

**Popular Models:**
- `tts_models/en/ljspeech/tacotron2-DDC` - Fast English TTS
- `tts_models/en/ljspeech/vits` - High-quality English TTS
- `tts_models/multilingual/multi-dataset/xtts_v2` - 16 languages + voice cloning
- `tts_models/multilingual/multi-dataset/your_tts` - Multilingual voice cloning
- `tts_models/en/ljspeech/fast_pitch` - Fast synthesis

### Step 4: Use the Server (Optional)

Run TTS as a web service:

```bash
# Start the server
tts-server --model_name tts_models/en/ljspeech/tacotron2-DDC

# Access at http://localhost:5002
```

Or with Docker:

```bash
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits
```

### Step 5: Next Steps

**For Users:**
- 📖 Read the [Quick Reference Guide](docs/architecture/QUICK_REFERENCE.md)
- 🎯 Check [examples/](examples/) for more code samples
- 💬 Join [Discord](https://discord.gg/5eXr5seRrv) for help

**For Developers:**
- 🔧 Read [CONTRIBUTING.md](CONTRIBUTING.md)
- 🏗️ See [Architecture Overview](docs/architecture/ARCHITECTURAL_OVERVIEW.md)
- 🧪 Run tests: `pytest tests/`

**For Training Custom Models:**
- 📚 See [training recipes](recipes/)
- 📖 Read [training guide](docs/source/training_a_model.md)
- 🎓 Check [tutorial for beginners](docs/source/tutorial_for_nervous_beginners.md)

---

### 💡 Quick Tips

**Troubleshooting:**
- First run downloads models (~100MB-1GB) - be patient!
- GPU recommended for large models (XTTS, VoiceCraft-X)
- Use `--use_cuda` flag or `.to("cuda")` for GPU acceleration

**Performance:**
- Use PyTorch 2.0+ for 20-40% speedup with `torch.compile()`
- Smaller models (Tacotron2, FastPitch) work fine on CPU
- For production: cache model downloads, use GPU, enable compile mode

**Common Issues:**
- `espeak` not found: Install espeak (`apt-get install espeak` or `brew install espeak`)
- CUDA out of memory: Reduce batch size or use CPU
- Model download fails: Check internet connection or download manually

---

## Docker Image
You can also try TTS without install with the docker image.
Simply run the following command and you will be able to run TTS without installing it.

```bash
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
python3 TTS/server/server.py --list_models #To get the list of available models
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits # To start a server
```

You can then enjoy the TTS server [here](http://[::1]:5002/)
More details about the docker images (like GPU support) can be found [here](https://tts.readthedocs.io/en/latest/docker_images.html)


## Synthesizing speech by 🐸TTS

### 🐍 Python API

#### Running a multi-speaker and multi-lingual model

```python
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
```

#### Running a single speaker model

```python
# Init TTS with the target model name
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(device)

# Run TTS
tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path=OUTPUT_PATH)

# Example voice cloning with YourTTS in English, French and Portuguese
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")
```

#### Example voice conversion

Converting the voice in `source_wav` to the voice of `target_wav`

```python
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")
tts.voice_conversion_to_file(source_wav="my/source.wav", target_wav="my/target.wav", file_path="output.wav")
```

#### Example voice cloning together with the voice conversion model.
This way, you can clone voices by using any model in 🐸TTS.

```python

tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
tts.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)
```

#### Example text to speech using **Fairseq models in ~1100 languages** 🤯.
For Fairseq models, use the following name format: `tts_models/<lang-iso_code>/fairseq/vits`.
You can find the language ISO codes [here](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html)
and learn about the Fairseq models [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms).

```python
# TTS with on the fly voice conversion
api = TTS("tts_models/deu/fairseq/vits")
api.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)
```

### Command-line `tts`

<!-- begin-tts-readme -->

Synthesize speech on command line.

You can either use your trained model or choose a model from the provided list.

If you don't specify any models, then it uses LJSpeech based English model.

#### Single Speaker Models

- List provided models:

  ```
  $ tts --list_models
  ```

- Get model info (for both tts_models and vocoder_models):

  - Query by type/name:
    The model_info_by_name uses the name as it from the --list_models.
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```
    For example:
    ```
    $ tts --model_info_by_name tts_models/tr/common-voice/glow-tts
    $ tts --model_info_by_name vocoder_models/en/ljspeech/hifigan_v2
    ```
  - Query by type/idx:
    The model_query_idx uses the corresponding idx from --list_models.

    ```
    $ tts --model_info_by_idx "<model_type>/<model_query_idx>"
    ```

    For example:

    ```
    $ tts --model_info_by_idx tts_models/3
    ```

  - Query info for model info by full name:
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```

- Run TTS with default models:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav
  ```

- Run TTS and pipe out the generated TTS wav file data:

  ```
  $ tts --text "Text for TTS" --pipe_out --out_path output/path/speech.wav | aplay
  ```

- Run a TTS model with its default vocoder model:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --out_path output/path/speech.wav
  ```

- Run with specific TTS and vocoder models from the list:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --vocoder_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --vocoder_name "vocoder_models/en/ljspeech/univnet" --out_path output/path/speech.wav
  ```

- Run your own TTS model (Using Griffin-Lim Vocoder):

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
  ```

- Run your own TTS and Vocoder models:

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
      --vocoder_path path/to/vocoder.pth --vocoder_config_path path/to/vocoder_config.json
  ```

#### Multi-speaker Models

- List the available speakers and choose a <speaker_id> among them:

  ```
  $ tts --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
  ```

- Run the multi-speaker TTS model with the target speaker ID:

  ```
  $ tts --text "Text for TTS." --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
  ```

- Run your own multi-speaker TTS model:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav --model_path path/to/model.pth --config_path path/to/config.json --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
  ```

### Voice Conversion Models

```
$ tts --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>" --source_wav <path/to/speaker/wav> --target_wav <path/to/reference/wav>
```

<!-- end-tts-readme -->

## 📁 Project Structure

```
coqui-tts/
├── TTS/                    # Main package
│   ├── api.py             # 🔥 High-level Python API
│   ├── bin/               # CLI tools and training scripts
│   ├── tts/               # TTS models, configs, and utilities
│   ├── vocoder/           # Vocoder models
│   ├── encoder/           # Speaker encoder
│   └── vc/                # Voice conversion
├── docs/                   # Documentation
│   ├── architecture/      # Architecture docs and quick references
│   ├── development/       # Development guides and roadmaps
│   ├── audit/             # Security audit reports
│   └── source/            # Sphinx documentation
├── tests/                  # Comprehensive test suite
├── recipes/                # Training recipes for various datasets
├── notebooks/              # Jupyter notebooks and examples
├── CONTRIBUTING.md         # 📖 Contribution guidelines
├── CHANGELOG.md            # 📝 Project changelog
└── README.md               # This file
```

🔥 = Most frequently used

For detailed architecture information, see the [Architecture Overview](docs/architecture/ARCHITECTURAL_OVERVIEW.md).

---

## 📚 Documentation & Resources

### For Users
- **[Quick Start Guide](docs/architecture/QUICK_REFERENCE.md)** - Get started quickly
- **[Installation Guide](https://tts.readthedocs.io/en/latest/installation.html)** - Detailed installation instructions
- **[Model Documentation](https://tts.readthedocs.io/en/latest/)** - Model-specific guides
- **[FAQ](docs/source/faq.md)** - Frequently asked questions

### For Developers
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Architecture Overview](docs/architecture/ARCHITECTURAL_OVERVIEW.md)** - Deep dive into codebase
- **[Development Setup](CONTRIBUTING.md#development-setup)** - Set up your dev environment
- **[Migration Guide](docs/development/MIGRATION_GUIDE.md)** - Upgrade to latest version

### For Maintainers
- **[Modernization Roadmap](docs/development/MODERNIZATION_ROADMAP.md)** - Future improvements
- **[Security Audit](docs/audit/EXECUTIVE_SUMMARY.md)** - Security assessment
- **[Changelog](CHANGELOG.md)** - Version history

### Additional Resources
- **[Documentation Index](docs/DOCUMENTATION_INDEX.md)** - Complete documentation navigation
- **[Tutorial for Beginners](docs/source/tutorial_for_nervous_beginners.md)** - Step-by-step tutorial
- **[TTS Papers](https://github.com/erogol/TTS-papers)** - Research papers collection

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs** - Open an issue with a clear description
2. **Suggest Features** - Discuss new ideas in GitHub Discussions
3. **Submit PRs** - Fix bugs, add features, improve docs
4. **Improve Documentation** - Help others understand the project
5. **Share Examples** - Contribute notebooks and tutorials

See our [Contributing Guide](CONTRIBUTING.md) for detailed instructions.

### Quick Start for Contributors

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/TTS.git
cd TTS

# Install in development mode
pip install -e ".[all,dev,notebooks]"

# Install pre-commit hooks
pre-commit install

# Create a branch and make your changes
git checkout -b feature/my-awesome-feature

# Run tests
pytest tests/

# Submit a pull request!
```

---

## 📜 Citation

If you use Coqui TTS in your research, please cite:

```bibtex
@misc{coqui-tts,
  author = {Coqui},
  title = {Coqui TTS},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/coqui-ai/TTS}}
}
```

See [CITATION.cff](CITATION.cff) for more citation formats.

---

## 📄 License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0). See [LICENSE.txt](LICENSE.txt) for details.

---

## 🙏 Acknowledgments

- The Coqui team and community for their contributions
- All the researchers whose models are implemented in this library
- The open-source community for their support and feedback

---

**Last Updated**: November 20, 2025 | **Version**: 0.22.0+ | **Status**: Actively Maintained
