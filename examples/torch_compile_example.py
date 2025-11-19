#!/usr/bin/env python3
"""Example of using torch.compile() with Coqui TTS for faster inference.

This example demonstrates how to use PyTorch 2.0+ torch.compile() feature
to achieve 20-40% faster inference speeds with TTS models.

Requirements:
    - PyTorch >= 2.0
    - CUDA-capable GPU (optional, but recommended for best performance)
"""

import time
from pathlib import Path

import numpy as np
import torch

from TTS.api import TTS
from TTS.utils.torch_compile import (
    HAS_TORCH_COMPILE,
    compile_for_inference,
    maybe_compile,
)


def benchmark_model(tts: TTS, text: str, num_runs: int = 10) -> float:
    """Run inference multiple times and return average time.

    Args:
        tts: TTS model instance
        text: Text to synthesize
        num_runs: Number of runs to average

    Returns:
        Average inference time in seconds
    """
    times = []

    # Warmup
    for _ in range(3):
        _ = tts.tts(text, split_sentences=False)

    # Benchmark
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = tts.tts(text, split_sentences=False)
        end = time.perf_counter()
        times.append(end - start)

    return float(np.mean(times))


def main():
    """Main example function."""
    # Configuration
    model_name = "tts_models/en/ljspeech/vits"  # Fast model for demonstration
    test_text = "The quick brown fox jumps over the lazy dog."
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_runs = 10

    print("=" * 80)
    print("Torch.compile() Performance Example")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  torch.compile() available: {HAS_TORCH_COMPILE}")
    print(f"  Test text: '{test_text}'")
    print(f"  Benchmark runs: {num_runs}")

    if not HAS_TORCH_COMPILE:
        print("\n⚠ Warning: torch.compile() requires PyTorch >= 2.0")
        print("Install with: pip install torch>=2.1")
        return

    # Load model (baseline)
    print("\n" + "-" * 80)
    print("Loading model (baseline)...")
    tts_baseline = TTS(model_name=model_name, progress_bar=False)
    tts_baseline = tts_baseline.to(device)
    print("✓ Model loaded")

    # Benchmark baseline
    print("\nBenchmarking baseline performance...")
    baseline_time = benchmark_model(tts_baseline, test_text, num_runs)
    print(f"✓ Baseline average time: {baseline_time*1000:.2f} ms")

    # Load model again (for fair comparison)
    print("\n" + "-" * 80)
    print("Loading model (with torch.compile)...")
    tts_compiled = TTS(model_name=model_name, progress_bar=False)
    tts_compiled = tts_compiled.to(device)

    # Apply torch.compile()
    print("Applying torch.compile()...")
    if hasattr(tts_compiled.synthesizer, "tts_model"):
        # Compile with "reduce-overhead" mode for best inference performance
        tts_compiled.synthesizer.tts_model = maybe_compile(
            tts_compiled.synthesizer.tts_model, mode="reduce-overhead"
        )
        print("✓ torch.compile() applied with mode='reduce-overhead'")
    else:
        print("⚠ Warning: Model doesn't have tts_model attribute")
        return

    # Benchmark compiled model
    print("\nBenchmarking compiled performance...")
    print("(First run may be slower due to compilation...)")
    compiled_time = benchmark_model(tts_compiled, test_text, num_runs)
    print(f"✓ Compiled average time: {compiled_time*1000:.2f} ms")

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nBaseline time:     {baseline_time*1000:8.2f} ms")
    print(f"Compiled time:     {compiled_time*1000:8.2f} ms")
    print(f"Speedup:           {baseline_time/compiled_time:8.2f}x")
    print(f"Time saved:        {(baseline_time - compiled_time)*1000:8.2f} ms ({(1 - compiled_time/baseline_time)*100:.1f}%)")
    print("=" * 80)

    # Generate sample output
    print("\nGenerating sample output...")
    output_path = Path("output_torch_compile_example.wav")
    tts_compiled.tts_to_file(
        text="This audio was generated using PyTorch 2.0 torch compile for faster inference.",
        file_path=str(output_path),
    )
    print(f"✓ Sample saved to: {output_path}")

    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)
    if baseline_time / compiled_time > 1.2:
        print("✓ torch.compile() provides significant speedup for this model!")
        print("  Consider using it for production inference.")
    else:
        print("• torch.compile() speedup is modest for this model/device combination.")
        print("  Speedup may be greater with:")
        print("  - Longer text inputs")
        print("  - Batch inference")
        print("  - Different GPU models")

    print("\nTips for best performance:")
    print("  • Use GPU (CUDA) for maximum benefit")
    print("  • First inference is slower (compilation overhead)")
    print("  • Speedup increases with longer sequences")
    print("  • Try different modes: 'default', 'reduce-overhead', 'max-autotune'")
    print("=" * 80 + "\n")


def advanced_example():
    """Advanced example with custom compilation settings."""
    print("\n" + "=" * 80)
    print("ADVANCED: Custom Compilation Settings")
    print("=" * 80)

    model_name = "tts_models/en/ljspeech/vits"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tts = TTS(model_name=model_name, progress_bar=False).to(device)

    if hasattr(tts.synthesizer, "tts_model") and HAS_TORCH_COMPILE:
        # Try different compilation modes
        modes = ["default", "reduce-overhead", "max-autotune"]

        print("\nComparing different compilation modes:")
        print("-" * 80)

        test_text = "Testing different compilation modes for performance."

        for mode in modes:
            # Reload model for each mode
            tts = TTS(model_name=model_name, progress_bar=False).to(device)
            tts.synthesizer.tts_model = maybe_compile(tts.synthesizer.tts_model, mode=mode)

            # Benchmark
            avg_time = benchmark_model(tts, test_text, num_runs=5)
            print(f"  {mode:20s}: {avg_time*1000:8.2f} ms")

        print("-" * 80)
        print("\nMode descriptions:")
        print("  • default:         Balanced compilation (good starting point)")
        print("  • reduce-overhead: Optimized for latency (best for single inference)")
        print("  • max-autotune:    Maximum optimization (slower compile, best runtime)")
    else:
        print("\n⚠ torch.compile() not available or model incompatible")


if __name__ == "__main__":
    # Run basic example
    main()

    # Uncomment to run advanced example
    # advanced_example()
