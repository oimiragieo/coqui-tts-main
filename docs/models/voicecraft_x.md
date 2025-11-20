# VoiceCraft-X Implementation for Coqui TTS

## Overview

This implementation integrates the key innovations from the VoiceCraft-X paper (arXiv:2511.12347v1) into Coqui TTS, enabling:

- **Unified multilingual TTS and speech editing** across 11+ languages
- **Phoneme-free multilingual text processing** using Qwen3 LLM
- **High-quality speech synthesis** with EnCodec-style multi-codebook tokenization
- **Zero-shot voice cloning** with improved speaker embeddings
- **Speech editing capabilities** for seamless audio modifications

## Paper Reference

**VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech Synthesis and Speech Editing**
Zhisheng Zheng, Puyuan Peng, Anuj Diwan, Cong Phuoc Huynh, Xiaohang Sun, Zhu Liu, Vimal Bhat, David Harwath
arXiv:2511.12347v1 [eess.AS] 15 Nov 2025
Demo: https://zhishengzheng.com/voicecraft-x/

## Key Innovations Implemented

### 1. EnCodec-Style Speech Tokenizer (`encodec_tokenizer.py`)

**What it does:**
- Replaces the single-codebook DVAE with a multi-codebook Residual Vector Quantizer (RVQ)
- Provides better temporal resolution and compression efficiency

**Technical details:**
- **4 RVQ codebooks** with 2048 entries each (vs 1×8192 in XTTS)
- **50Hz framerate** (320 sample stride @ 16kHz = 20ms resolution)
- **Residual quantization** for hierarchical representation
- 2.3x better temporal resolution compared to original XTTS

**Benefits:**
- Finer-grained audio representation
- Better reconstruction quality
- More efficient multi-token prediction

### 2. Delay Pattern Mechanism (`delay_pattern.py`)

**What it does:**
- Implements MusicGen-style delay pattern for autoregressive multi-codebook generation
- Enables conditioning on previous codebook levels for the same timestep

**Technical details:**
```
Pattern visualization (4 codebooks, 4 timesteps):
Position: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
CB1:      A0 A1 A2 A3 -  -  -  -  -  -  -  -  -  -  -  -
CB2:      -  A0 A1 A2 A3 -  -  -  -  -  -  -  -  -  -  -
CB3:      -  -  A0 A1 A2 A3 -  -  -  -  -  -  -  -  -  -
CB4:      -  -  -  A0 A1 A2 A3 -  -  -  -  -  -  -  -  -
```

**Benefits:**
- Codebook 2 at time t can be conditioned on Codebook 1 at time t
- More coherent multi-codebook predictions
- Improved audio quality

### 3. Token Reordering (`token_reordering.py`)

**What it does:**
- Implements prefix-suffix-middle reordering strategy
- Time-aligns text and speech tokens
- Unifies TTS and speech editing under single framework

**Technical details:**
- **Training:** Randomly segment into prefix, middle, suffix → reorder to prefix + suffix + middle
- **Middle segment** is the prediction target
- Enables infilling/editing conditioned on surrounding context

**Benefits:**
- Single model for both TTS and editing
- Eliminates repetition loops (major issue in original VoiceCraft)
- Stable inference without multi-sample filtering
- Seamless audio editing capabilities

### 4. Qwen3 Backbone Integration (`qwen3_backbone.py`)

**What it does:**
- Replaces phoneme-based text processing with Qwen3 LLM
- Provides native multilingual support for 119 languages

**Technical details:**
- **Qwen3-0.6B-Base** model (613M total parameters, 457M non-embedding)
- 28 Transformer layers, 1024 hidden dim
- Grouped-Query Attention (16 heads, 8 KV heads)
- No G2P (grapheme-to-phoneme) conversion needed

**Benefits:**
- Eliminate language-specific pronunciation lexicons
- Unified tokenizer across all languages
- Better multilingual representations
- Simplified pipeline

### 5. Enhanced Speaker Embedding (`speaker_embedding.py`)

**What it does:**
- Implements CosyVoice CAM++ voiceprint model
- Provides better speaker disentanglement

**Technical details:**
- **CAM++** (Conformer-based architecture)
- Supports both ONNX and PyTorch implementations
- Fallback to WavLM if CAM++ unavailable
- 512-dim L2-normalized embeddings

**Benefits:**
- Better speaker similarity
- Robust to different languages
- Improved zero-shot voice cloning

### 6. Weighted Loss Function (`voicecraft_x_loss.py`)

**What it does:**
- Implements weighted cross-entropy with codebook and segment weighting
- Focuses training on important aspects

**Technical details:**
```python
Loss = sum(w_seg(z_i) * α_k * CE(pred_i, target_i))

where:
  α_k = [1.0, 0.8, 0.6, 0.4]  # Codebook weights for CB1-CB4
  w_seg = {
    "prefix": 1.0,
    "suffix": 1.0,
    "middle": 3.0  # Higher weight for target
  }
```

**Benefits:**
- Balanced training across codebooks
- Focus on target generation (middle segment)
- Better convergence

### 7. Unified VoiceCraft-X Model (`voicecraft_x.py`)

**What it does:**
- Integrates all components into a single unified model
- Provides both TTS and editing inference modes

**Features:**
- `inference_tts()`: Zero-shot text-to-speech
- `inference_edit()`: Seamless speech editing
- `encode_audio()` / `decode_audio()`: Audio tokenization
- `extract_speaker_embedding()`: Speaker embedding extraction

## Architecture Comparison

| Component | XTTS (Current) | VoiceCraft-X (New) | Improvement |
|-----------|---------------|-------------------|-------------|
| **Text Processing** | BPE (language-specific) | Qwen3 LLM (119 languages) | No phoneme conversion needed |
| **Speech Tokenizer** | DVAE (1×8192) | EnCodec RVQ (4×2048) | 2.3x better temporal resolution |
| **Framerate** | ~22Hz (1024 stride) | 50Hz (320 stride) | 2.3x finer granularity |
| **Token Prediction** | Sequential | Delay pattern | Better multi-codebook coherence |
| **Capabilities** | TTS only | TTS + Editing | Unified framework |
| **Training Stability** | Repetition issues | Reordering eliminates loops | Stable single-pass generation |

### 7. Alignment Utilities (`align_utils.py`) ✨ **NEW**

**What it does:**
- Provides sophisticated text alignment for speech editing
- Detects common prefix/suffix between prompt and target text
- Maps text positions to audio frame boundaries

**Key functions:**
- `get_diff_time_frame_and_segment()`: Find editing boundaries
- `build_mapping()`: Map cleaned text to original positions
- `remove_punctuation()`: Unicode-aware punctuation handling
- Language-aware processing (word-based vs character-based)

**Benefits:**
- Precise speech editing with exact boundary detection
- Handles both word-based (English, etc.) and character-based (Chinese, Japanese) languages
- Essential for high-quality speech editing

### 8. Text Preprocessing Pipeline (`text_processor.py`) ✨ **NEW**

**What it does:**
- Normalizes text before tokenization
- Converts numbers to words
- Splits long texts into manageable segments

**Features:**
- `spell_out_number()`: Digit to word conversion
- `split_paragraph()`: Intelligent text segmentation
- `TextPreprocessor`: All-in-one preprocessing class
- Multi-language support

**Benefits:**
- Consistent text formatting across languages
- Improved number pronunciation
- Handles long documents efficiently

### 9. Enhanced Sampling with Repetition Penalty ✨ **NEW**

**What it does:**
- Adds repetition_penalty parameter to reduce repetition loops
- Implements min-p filtering as alternative to top-p
- Critical for VoiceCraft stability

**Parameters:**
- `repetition_penalty`: Penalty for repeating tokens (>1.0 to discourage, default: 1.0)
- `min_p`: Minimum probability threshold relative to max prob (default: 0.0)
- Compatible with existing top-k and top-p sampling

**Benefits:**
- Dramatically reduces repetition issues
- More natural prosody
- Better quality control

### 10. Special Speech Tokens ✨ **NEW**

**What it does:**
- Adds prosody control tokens to tokenizer
- Enables natural non-verbal sounds

**Tokens:**
- `[breath]`: Breathing sound
- `[noise]`: Background noise
- `[laughter]`: Laughter
- `[cough]`: Coughing
- `[sigh]`: Sighing
- `[pause]`: Pause marker

**Benefits:**
- More natural and expressive speech
- Control over prosodic elements
- Better emotional expression

## File Structure

```
TTS/tts/
├── layers/xtts/
│   ├── encodec_tokenizer.py       # EnCodec-style RVQ tokenizer
│   ├── delay_pattern.py           # Delay pattern mechanism
│   ├── token_reordering.py        # Token reordering strategy
│   ├── qwen3_backbone.py          # Qwen3 integration + special tokens
│   ├── speaker_embedding.py       # Enhanced speaker encoder
│   ├── voicecraft_x_loss.py       # Weighted loss function
│   ├── align_utils.py             # Alignment utilities (NEW)
│   └── text_processor.py          # Text preprocessing pipeline (NEW)
│
└── models/
    └── voicecraft_x.py            # Unified VoiceCraft-X model
```

## Usage Examples

### Basic TTS Inference

```python
from TTS.tts.models.voicecraft_x import VoiceCraftX, VoiceCraftXConfig
import torch

# Create config
config = VoiceCraftXConfig(
    num_codebooks=4,
    codebook_size=2048,
    sample_rate=16000,
    qwen_model_name="Qwen/Qwen2.5-0.5B",
)

# Initialize model
model = VoiceCraftX(config)

# Load prompt audio for voice cloning
prompt_audio = torch.randn(16000 * 3)  # 3 seconds

# Generate speech with repetition penalty
output = model.inference_tts(
    text="Hello, this is a test of VoiceCraft-X!",
    prompt_audio=prompt_audio,
    temperature=1.0,
    top_k=20,
    repetition_penalty=1.1,  # NEW: Reduce repetition
)
```

### Speech Editing

```python
# Load audio to edit
prefix_audio = torch.randn(16000 * 2)  # 2 seconds before edit
suffix_audio = torch.randn(16000 * 2)  # 2 seconds after edit

# Edit speech
edited_audio = model.inference_edit(
    prefix_audio=prefix_audio,
    suffix_audio=suffix_audio,
    new_middle_text="This is the new text to insert",
    temperature=1.0,
    top_k=20,
)
```

### Training

```python
# Prepare training data
text_tokens = tokenizer(text)
audio = load_audio("sample.wav")
audio_tokens, quant_loss = model.encode_audio(audio)
speaker_emb = model.extract_speaker_embedding(audio)

# Forward pass
loss, loss_dict = model(
    text_tokens=text_tokens,
    audio_tokens=audio_tokens,
    speaker_embedding=speaker_emb,
    segment_lengths={"prefix": 20, "suffix": 30, "middle": 50},
    return_loss=True,
)

# Backward
loss.backward()
```

## Expected Improvements

Based on VoiceCraft-X paper results:

### Quantitative Metrics

- **WER/CER:** Competitive with SOTA models despite using less training data
- **Speaker Similarity (SIM-o):** 0.54-0.68 depending on language
- **CMOS (Naturalness):** Up to 0.63 on English (highest among compared models)

### Qualitative Improvements

1. **Training Stability:** Token reordering eliminates repetition loops
2. **Data Efficiency:** Transfer learning enables low-resource language support
3. **Multilingual:** Single model supports 11+ languages
4. **Speech Editing:** First multilingual speech editing model
5. **No Phonemes:** Simplified pipeline without G2P

### Performance Examples from Paper

| Language | Training Hours | WER/CER | SIM-o |
|----------|---------------|---------|-------|
| English | 14.5K | 4.20 | 0.54 |
| Chinese | 5K | 3.29 | 0.68 |
| Spanish | 1.2K | 4.67 | 0.63 |
| German | 3.4K | 8.19 | 0.60 |
| Korean | 832 | 31.11 | 0.56 |

## Dependencies

### Required
- PyTorch >= 2.0
- transformers >= 4.30 (for Qwen3)
- einops
- numpy

### Optional
- onnxruntime-gpu (for CAM++ ONNX inference)
- peft (for LoRA fine-tuning)
- torchaudio (for mel-spectrogram extraction)

## Installation

```bash
# Install base requirements
pip install torch transformers einops

# Optional dependencies
pip install onnxruntime-gpu peft torchaudio
```

## Training Data

The paper uses ~32K hours across 11 languages:

- **English:** LibriTTS-R (516h), GigaSpeech (5.8K h), MLS (8.2K h)
- **Chinese:** WenetSpeech4TTS (3.3K h), AISHELL-2 (997h), MAGICDATA (707h)
- **Korean:** KsponSpeech (832h)
- **Japanese:** ReazonSpeech (3.5K h)
- **European languages:** MLS + CML-TTS (200-3.4K h per language)

## Limitations

1. **Data Scale:** 32K hours is less than some SOTA models (50K-100K hours)
2. **Language Coverage:** Currently 11 languages (paper explored ~20-30 internally)
3. **Model Size:** Current implementation uses Qwen3-0.6B; larger variants untested

## Future Work

1. **Scale to larger Qwen3 models** (1.5B, 3B, 7B)
2. **Expand language coverage** to 50+ languages
3. **Joint codec training** (currently codec is frozen)
4. **Streaming inference** optimization
5. **Audio watermarking** for synthetic speech detection

## Ethical Considerations

As noted in the paper:

⚠️ **This technology can be misused for:**
- Unauthorized voice cloning
- Deepfake creation
- Misinformation/propaganda

**Responsible use guidelines:**
- Only clone voices with explicit consent
- Include audio watermarking where possible
- Follow model license restrictions
- Consider detection tool integration

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{zheng2025voicecraftx,
  title={VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech Synthesis and Speech Editing},
  author={Zheng, Zhisheng and Peng, Puyuan and Diwan, Anuj and Huynh, Cong Phuoc and Sun, Xiaohang and Liu, Zhu and Bhat, Vimal and Harwath, David},
  journal={arXiv preprint arXiv:2511.12347},
  year={2025}
}
```

## License

This implementation follows the Coqui TTS license. The VoiceCraft-X paper is published under CC BY-NC-SA 4.0.

## Acknowledgments

- Original VoiceCraft-X paper authors
- Qwen3 team at Alibaba
- CosyVoice team
- EnCodec/MusicGen authors at Meta
- Coqui TTS community

## Contact

For issues or questions about this implementation, please open an issue on the Coqui TTS GitHub repository.

---

**Implementation Date:** November 2025
**Paper:** arXiv:2511.12347v1
**Implementation by:** Claude (Anthropic)
