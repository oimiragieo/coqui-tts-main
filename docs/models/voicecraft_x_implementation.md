# VoiceCraft-X Implementation Plan

## Overview
This document outlines the implementation plan for integrating VoiceCraft-X innovations into the Coqui TTS codebase.

## Key Innovations from VoiceCraft-X Paper

### 1. EnCodec-Style Multi-Codebook Speech Tokenizer
**Current State:** DVAE with single codebook (8192 entries), 1024 sample stride (~46ms @ 22kHz)

**Target State:**
- **4 Residual Vector Quantization (RVQ) codebooks**
- **2048 entries per codebook** (vs 8192 single)
- **50Hz framerate** (320 sample stride @ 16kHz = 20ms resolution)
- **Residual quantization** for better compression

**Implementation:**
- File: `TTS/tts/layers/xtts/encodec_tokenizer.py` (new)
- Architecture:
  - Encoder: Strided convolutions (stride 320 for 50Hz @ 16kHz)
  - RVQ: 4 layers, each 2048 tokens
  - Decoder: Transposed convolutions
- Training: On multilingual data (32K hours target)

### 2. Token Reordering Mechanism
**Current State:** Sequential left-to-right autoregressive generation only

**Target State:**
- **Time-aligned text and speech token interleaving**
- **Prefix-Suffix-Middle reordering** for unified TTS/editing
- **Monotonic alignment** between text and speech

**Implementation:**
- File: `TTS/tts/layers/xtts/token_reordering.py` (new)
- Components:
  - `reorder_tokens()`: Implements prefix-suffix-middle logic
  - `create_alignment_mask()`: Time-alignment masks
  - Support for MFA (Montreal Forced Aligner) timestamps
- Input: Original sequence + alignment info
- Output: Reordered sequence with mask tokens

**Algorithm:**
```python
def reorder_tokens(text, speech, alignment):
    # Randomly segment into prefix, middle, suffix
    prefix_text, middle_text, suffix_text = segment_text(text)

    # Reorder: prefix + suffix + middle
    reordered_text = [prefix_text, suffix_text, middle_text]

    # Reorder speech tokens based on alignment
    reordered_speech = align_and_reorder(speech, alignment, reordered_text)

    # Insert mask tokens
    return insert_masks(reordered_text, reordered_speech)
```

### 3. Delay Pattern for Multi-Codebook Prediction
**Current State:** None

**Target State:**
- **MusicGen-style delay pattern** from Encodec paper
- Cumulative 1-step delay per RVQ layer
- Enable conditioning on previous codebooks for same timestep

**Implementation:**
- File: `TTS/tts/layers/xtts/delay_pattern.py` (new)
- Pattern visualization (4 codebooks, 3 timesteps):
  ```
  t:  0  1  2  3  4  5
  C1: A0 A1 A2 -  -  -
  C2: -  A0 A1 A2 -  -
  C3: -  -  A0 A1 A2 -
  C4: -  -  -  A0 A1 A2
  ```
- Flattened sequence: [C1_0, C1_1, C2_0, C1_2, C2_1, C3_0, ...]

### 4. Qwen3 Integration for Phoneme-Free Text Processing
**Current State:** BPE tokenizer (language-specific)

**Target State:**
- **Qwen3-0.6B-Base** as text tokenizer AND backbone
- **Native support for 119 languages**
- **No G2P (grapheme-to-phoneme) conversion needed**

**Implementation:**
- File: `TTS/tts/layers/xtts/qwen3_backbone.py` (new)
- Model: Qwen/Qwen3-0.6B-Base from HuggingFace
- Architecture:
  - 28 Transformer layers
  - Hidden dim: 1024
  - FFN dim: 3072
  - 16 attention heads (GQA: 8 KV heads)
  - Context length: 32,768 tokens
- Replace GPT-2 backbone with Qwen3

### 5. Speaker Embedding Enhancement
**Current State:** Basic HiFiGAN speaker encoder

**Target State:**
- **CosyVoice CAM++ voiceprint model**
- **Linear projection** to match model dimension
- Better speaker disentanglement

**Implementation:**
- File: `TTS/tts/layers/xtts/speaker_embedding.py` (update)
- Download: `iic/CosyVoice-300M/campplus.onnx`
- Add ONNX runtime support
- Project to Qwen3's 1024-dim input

### 6. Weighted Loss Function
**Current State:** Simple weighted CE (0.01 text + 1.0 mel)

**Target State:**
- **Codebook weighting:** α = (1.0, 0.8, 0.6, 0.4) for C1-C4
- **Segment weighting:** prefix/suffix = 1.0, middle = 3.0
- **Combined weighted cross-entropy**

**Implementation:**
- File: `TTS/tts/layers/xtts/trainer/gpt_trainer.py` (update)
- Loss formula:
  ```python
  loss = sum(w_seg(z_i) * α_k * CE(pred_i, target_i))
  ```

### 7. Speech Editing Support
**Current State:** TTS only

**Target State:**
- **Unified inference** for TTS and editing
- **Seamless audio splicing**
- **In-place editing** of existing recordings

**Implementation:**
- File: `TTS/tts/models/xtts.py` (update inference methods)
- Add `inference_edit()` method
- Input: prefix_audio, suffix_audio, new_middle_text
- Output: Seamless edited audio

## Implementation Priority

### Phase 1: Core Architecture (High Priority)
1. ✅ Create implementation plan
2. EnCodec-style tokenizer with RVQ
3. Delay pattern mechanism
4. Token reordering mechanism

### Phase 2: Model Integration (High Priority)
5. Qwen3 backbone integration
6. Speaker embedding enhancement
7. Weighted loss implementation

### Phase 3: Training & Inference (Medium Priority)
8. Update training pipeline
9. Speech editing inference
10. Multi-codebook prediction heads

### Phase 4: Testing & Documentation (Medium Priority)
11. Unit tests for all components
12. Integration tests
13. Documentation and examples
14. Benchmarking against baselines

## File Structure

```
TTS/tts/layers/xtts/
├── encodec_tokenizer.py      # NEW: EnCodec-style RVQ tokenizer
├── delay_pattern.py           # NEW: Delay pattern for multi-codebook
├── token_reordering.py        # NEW: Prefix-suffix-middle reordering
├── qwen3_backbone.py          # NEW: Qwen3 integration
├── speaker_embedding.py       # UPDATE: CAM++ voiceprint
├── gpt.py                     # UPDATE: Multi-head for codebooks
└── trainer/
    ├── gpt_trainer.py         # UPDATE: Weighted loss, reordering
    └── dataset.py             # UPDATE: Alignment data loading

TTS/tts/models/
└── xtts.py                    # UPDATE: Add editing inference

TTS/tts/configs/
└── xtts_config.py             # UPDATE: New config parameters
```

## Configuration Parameters

```python
# New config additions
config.num_codebooks = 4
config.codebook_size = 2048
config.codec_framerate = 50  # Hz
config.codec_stride = 320     # samples @ 16kHz
config.use_delay_pattern = True
config.use_token_reordering = True
config.codebook_weights = [1.0, 0.8, 0.6, 0.4]
config.segment_weights = {"prefix": 1.0, "suffix": 1.0, "middle": 3.0}
config.backbone = "qwen3"  # or "gpt2"
config.speaker_encoder = "campplus"  # or "default"
```

## Expected Improvements

Based on VoiceCraft-X paper results:

1. **Better temporal resolution:** 50Hz vs ~22Hz (2.3x improvement)
2. **More efficient compression:** 4x2048 RVQ vs 1x8192 single
3. **Multilingual without phonemes:** Eliminate G2P pipeline
4. **Unified TTS + Editing:** Single model for both tasks
5. **Better speaker similarity:** CAM++ embeddings
6. **Training stability:** Token reordering reduces repetition loops
7. **Data efficiency:** Transfer learning across languages

## Validation Metrics

- **WER/CER:** Word/Character Error Rate on test sets
- **SIM-o:** Objective speaker similarity (WavLM embeddings)
- **CMOS:** Comparative Mean Opinion Score (naturalness)
- **SMOS:** Similarity Mean Opinion Score
- **NMOS/IMOS:** Naturalness/Intelligibility for editing

## References

- VoiceCraft-X: arXiv:2511.12347v1
- Qwen3: Qwen/Qwen3-0.6B-Base
- EnCodec: defossez2022high
- MusicGen: copet2023simple (delay pattern)
- CosyVoice: du2024cosyvoice1/2
