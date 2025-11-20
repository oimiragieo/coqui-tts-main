"""
VoiceCraft-X Example: Multilingual TTS and Speech Editing

This example demonstrates how to use VoiceCraft-X for:
1. Zero-shot text-to-speech with voice cloning
2. Seamless speech editing (insert/replace audio segments)

VoiceCraft-X supports 11+ languages without phoneme conversion, using a
Qwen3 LLM backbone and EnCodec-style multi-codebook speech tokenizer.

Requirements:
    pip install torch transformers einops torchaudio

For ONNX speaker encoder (optional, faster):
    pip install onnxruntime-gpu
"""

import torch
import torchaudio
from pathlib import Path

from TTS.tts.models.voicecraft_x import VoiceCraftX, VoiceCraftXConfig


def example_tts():
    """Example: Zero-shot text-to-speech with voice cloning."""
    print("\n" + "=" * 60)
    print("Example 1: Zero-Shot Text-to-Speech")
    print("=" * 60)

    # Create config
    config = VoiceCraftXConfig(
        num_codebooks=4,
        codebook_size=2048,
        sample_rate=16000,
        qwen_model_name="Qwen/Qwen2.5-0.5B",
        use_delay_pattern=True,
    )

    # Initialize model
    print("\n[1/3] Initializing VoiceCraft-X model...")
    model = VoiceCraftX(config)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✓ Model initialized on {device}")

    # Load prompt audio for voice cloning
    # Replace with your own audio file (3-10 seconds recommended)
    print("\n[2/3] Loading prompt audio...")
    prompt_audio_path = "path/to/your/speaker_sample.wav"

    # For demo, we'll create synthetic audio
    # In practice, load real audio: prompt_audio, sr = torchaudio.load(prompt_audio_path)
    prompt_audio = torch.randn(1, 16000 * 3)  # 3 seconds of demo audio
    prompt_audio = prompt_audio.to(device)
    print(f"✓ Prompt audio loaded: {prompt_audio.shape}")

    # Generate speech
    print("\n[3/3] Generating speech...")
    text = "Hello! This is a demonstration of VoiceCraft-X, a multilingual text-to-speech system with speech editing capabilities."

    with torch.no_grad():
        output = model.inference_tts(
            text=text,
            prompt_audio=prompt_audio,
            temperature=1.0,
            top_k=20,
            repetition_penalty=1.1,
        )

    print(f"✓ Generated audio: {output.shape}")

    # Save output
    output_path = "voicecraft_x_output.wav"
    torchaudio.save(output_path, output.cpu(), sample_rate=16000)
    print(f"✓ Saved to {output_path}")

    return output


def example_speech_editing():
    """Example: Seamless speech editing (insert/replace audio segments)."""
    print("\n" + "=" * 60)
    print("Example 2: Speech Editing")
    print("=" * 60)

    # Create config
    config = VoiceCraftXConfig(
        num_codebooks=4,
        codebook_size=2048,
        sample_rate=16000,
        qwen_model_name="Qwen/Qwen2.5-0.5B",
        use_delay_pattern=True,
    )

    # Initialize model
    print("\n[1/4] Initializing VoiceCraft-X model...")
    model = VoiceCraftX(config)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✓ Model initialized on {device}")

    # Load audio segments to edit
    # In practice: prefix_audio, _ = torchaudio.load("before.wav")
    #              suffix_audio, _ = torchaudio.load("after.wav")
    print("\n[2/4] Loading audio segments...")
    prefix_audio = torch.randn(1, 16000 * 2)  # 2 seconds before edit point
    suffix_audio = torch.randn(1, 16000 * 2)  # 2 seconds after edit point

    prefix_audio = prefix_audio.to(device)
    suffix_audio = suffix_audio.to(device)
    print(f"✓ Prefix audio: {prefix_audio.shape}")
    print(f"✓ Suffix audio: {suffix_audio.shape}")

    # Text to insert/replace in the middle
    new_middle_text = "This is the new text that will be seamlessly inserted."
    print(f"\n[3/4] Inserting text: '{new_middle_text}'")

    # Perform speech editing
    with torch.no_grad():
        edited_audio = model.inference_edit(
            prefix_audio=prefix_audio,
            suffix_audio=suffix_audio,
            new_middle_text=new_middle_text,
            temperature=1.0,
            top_k=20,
            repetition_penalty=1.1,
        )

    print(f"✓ Edited audio: {edited_audio.shape}")

    # Save output
    output_path = "voicecraft_x_edited.wav"
    torchaudio.save(output_path, edited_audio.cpu(), sample_rate=16000)
    print(f"\n[4/4] ✓ Saved edited audio to {output_path}")

    return edited_audio


def example_multilingual():
    """Example: Multilingual TTS across different languages."""
    print("\n" + "=" * 60)
    print("Example 3: Multilingual TTS")
    print("=" * 60)

    config = VoiceCraftXConfig(
        num_codebooks=4,
        codebook_size=2048,
        sample_rate=16000,
        qwen_model_name="Qwen/Qwen2.5-0.5B",
    )

    model = VoiceCraftX(config)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Prompt audio (same speaker for all languages)
    prompt_audio = torch.randn(1, 16000 * 3).to(device)

    # Generate speech in multiple languages
    texts = {
        "en": "Hello, this is English.",
        "es": "Hola, esto es español.",
        "fr": "Bonjour, c'est du français.",
        "de": "Hallo, das ist Deutsch.",
        "zh": "你好，这是中文。",
        "ja": "こんにちは、これは日本語です。",
        "ko": "안녕하세요, 이것은 한국어입니다.",
    }

    print("\nGenerating speech in multiple languages...")
    outputs = {}

    for lang, text in texts.items():
        print(f"\n[{lang}] '{text}'")
        with torch.no_grad():
            output = model.inference_tts(
                text=text,
                prompt_audio=prompt_audio,
                temperature=1.0,
                top_k=20,
            )
        outputs[lang] = output

        # Save each language
        output_path = f"voicecraft_x_{lang}.wav"
        torchaudio.save(output_path, output.cpu(), sample_rate=16000)
        print(f"  ✓ Saved to {output_path}")

    print("\n✓ All languages generated successfully!")
    return outputs


def example_with_compile():
    """Example: Using torch.compile() for 20-40% faster inference."""
    print("\n" + "=" * 60)
    print("Example 4: Performance Optimization with torch.compile()")
    print("=" * 60)

    config = VoiceCraftXConfig(
        num_codebooks=4,
        codebook_size=2048,
        sample_rate=16000,
        qwen_model_name="Qwen/Qwen2.5-0.5B",
    )

    model = VoiceCraftX(config)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Compile the model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("\n[1/2] Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
        print("✓ Model compiled (expect 20-40% speedup)")
    else:
        print("\n⚠ torch.compile() not available (need PyTorch 2.0+)")

    # Generate speech
    print("\n[2/2] Running inference...")
    prompt_audio = torch.randn(1, 16000 * 3).to(device)
    text = "This model has been optimized with torch.compile for faster inference."

    import time
    start = time.time()

    with torch.no_grad():
        output = model.inference_tts(
            text=text,
            prompt_audio=prompt_audio,
            temperature=1.0,
            top_k=20,
        )

    elapsed = time.time() - start
    print(f"✓ Generated {output.shape[-1] / 16000:.2f}s audio in {elapsed:.2f}s")
    print(f"  Real-time factor: {elapsed / (output.shape[-1] / 16000):.2f}x")

    return output


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VoiceCraft-X Examples")
    print("=" * 60)
    print("\nThese examples demonstrate VoiceCraft-X capabilities:")
    print("1. Zero-shot TTS with voice cloning")
    print("2. Seamless speech editing")
    print("3. Multilingual synthesis (11+ languages)")
    print("4. Performance optimization with torch.compile()")

    # Run examples
    try:
        # Example 1: Basic TTS
        output_tts = example_tts()

        # Example 2: Speech editing
        output_edited = example_speech_editing()

        # Example 3: Multilingual
        # outputs_multilingual = example_multilingual()

        # Example 4: Performance optimization
        # output_compiled = example_with_compile()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Some examples may fail if:")
        print("- Qwen3 model not downloaded yet (will download on first run)")
        print("- GPU memory insufficient (try reducing audio length)")
        print("- Dependencies not installed (see requirements at top)")
