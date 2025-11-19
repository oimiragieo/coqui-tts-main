#!/usr/bin/env python3
"""Performance benchmarking script for TTS models.

This script benchmarks TTS models with and without PyTorch 2.0+ optimizations
like torch.compile() to measure performance improvements.

Usage:
    python TTS/bin/benchmark_performance.py --model_name tts_models/en/ljspeech/vits
    python TTS/bin/benchmark_performance.py --model_path /path/to/model.pth --config_path /path/to/config.json
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.cuda import is_available as cuda_available

try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer


def benchmark_inference(
    tts_instance: TTS,
    test_texts: List[str],
    num_runs: int = 10,
    warmup_runs: int = 3,
    use_compile: bool = False,
) -> Dict[str, float]:
    """Benchmark TTS inference performance.

    Args:
        tts_instance: TTS model instance
        test_texts: List of test texts to synthesize
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (excluded from timing)
        use_compile: Whether to use torch.compile()

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "mean_time": 0.0,
        "std_time": 0.0,
        "min_time": float("inf"),
        "max_time": 0.0,
        "total_time": 0.0,
        "throughput": 0.0,  # characters per second
    }

    # Apply torch.compile if requested and available
    if use_compile and hasattr(torch, "compile"):
        print("Applying torch.compile() optimization...")
        try:
            if hasattr(tts_instance.synthesizer, "tts_model"):
                tts_instance.synthesizer.tts_model = torch.compile(
                    tts_instance.synthesizer.tts_model, mode="reduce-overhead"
                )
                print("✓ torch.compile() applied successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not apply torch.compile(): {e}")

    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    for i in range(warmup_runs):
        for text in test_texts[:1]:  # Use only first text for warmup
            try:
                _ = tts_instance.tts(text, split_sentences=False)
            except Exception as e:
                print(f"Warning: Warmup run {i} failed: {e}")

    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    times = []
    total_chars = 0

    for i in range(num_runs):
        run_times = []
        for text in test_texts:
            total_chars += len(text)
            start_time = time.perf_counter()
            try:
                _ = tts_instance.tts(text, split_sentences=False)
            except Exception as e:
                print(f"Warning: Benchmark run {i} failed: {e}")
                continue
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            run_times.append(elapsed)

        if run_times:
            times.extend(run_times)

    if times:
        results["mean_time"] = float(np.mean(times))
        results["std_time"] = float(np.std(times))
        results["min_time"] = float(np.min(times))
        results["max_time"] = float(np.max(times))
        results["total_time"] = float(np.sum(times))
        results["throughput"] = total_chars / results["total_time"]

    return results


def print_results(baseline_results: Dict, optimized_results: Optional[Dict] = None):
    """Print benchmark results in a formatted table.

    Args:
        baseline_results: Results from baseline (non-optimized) run
        optimized_results: Results from optimized run (optional)
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)

    print("\nBaseline Performance:")
    print(f"  Mean time:       {baseline_results['mean_time']*1000:.2f} ms")
    print(f"  Std deviation:   {baseline_results['std_time']*1000:.2f} ms")
    print(f"  Min time:        {baseline_results['min_time']*1000:.2f} ms")
    print(f"  Max time:        {baseline_results['max_time']*1000:.2f} ms")
    print(f"  Throughput:      {baseline_results['throughput']:.2f} chars/sec")

    if optimized_results:
        print("\nOptimized Performance (with torch.compile):")
        print(f"  Mean time:       {optimized_results['mean_time']*1000:.2f} ms")
        print(f"  Std deviation:   {optimized_results['std_time']*1000:.2f} ms")
        print(f"  Min time:        {optimized_results['min_time']*1000:.2f} ms")
        print(f"  Max time:        {optimized_results['max_time']*1000:.2f} ms")
        print(f"  Throughput:      {optimized_results['throughput']:.2f} chars/sec")

        print("\nPerformance Improvement:")
        speedup = baseline_results["mean_time"] / optimized_results["mean_time"]
        print(f"  Speedup:         {speedup:.2f}x")
        print(f"  Time reduction:  {(1 - 1/speedup) * 100:.1f}%")
        print(f"  Throughput gain: {(optimized_results['throughput'] / baseline_results['throughput'] - 1) * 100:.1f}%")

    print("=" * 80 + "\n")


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Benchmark TTS model performance")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the TTS model to benchmark (e.g., tts_models/en/ljspeech/vits)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to model config")
    parser.add_argument("--vocoder_path", type=str, default=None, help="Path to vocoder checkpoint")
    parser.add_argument("--vocoder_config", type=str, default=None, help="Path to vocoder config")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup_runs", type=int, default=3, help="Number of warmup runs")
    parser.add_argument(
        "--test_compile",
        action="store_true",
        help="Test torch.compile() optimization (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        default=["The quick brown fox jumps over the lazy dog."],
        help="Test text(s) to synthesize",
    )

    args = parser.parse_args()

    # Check GPU availability
    if args.gpu and not cuda_available():
        print("Warning: GPU requested but CUDA not available. Using CPU.")
        args.gpu = False

    print("=" * 80)
    print("TTS PERFORMANCE BENCHMARKING TOOL")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Device:          {'GPU (CUDA)' if args.gpu else 'CPU'}")
    print(f"  Num runs:        {args.num_runs}")
    print(f"  Warmup runs:     {args.warmup_runs}")
    print(f"  Test texts:      {len(args.text)}")
    print(f"  torch.compile:   {args.test_compile and hasattr(torch, 'compile')}")

    # Initialize TTS model
    print("\nInitializing TTS model...")
    try:
        if args.model_name:
            print(f"  Loading model: {args.model_name}")
            tts = TTS(model_name=args.model_name, gpu=args.gpu, progress_bar=True)
        elif args.model_path and args.config_path:
            print(f"  Loading model from: {args.model_path}")
            tts = TTS(
                model_path=args.model_path,
                config_path=args.config_path,
                vocoder_path=args.vocoder_path,
                vocoder_config_path=args.vocoder_config,
                gpu=args.gpu,
            )
        else:
            print("Error: Must specify either --model_name or both --model_path and --config_path")
            return

        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run baseline benchmark
    print("Running baseline benchmark...")
    baseline_results = benchmark_inference(
        tts_instance=tts,
        test_texts=args.text,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        use_compile=False,
    )

    optimized_results = None
    if args.test_compile and hasattr(torch, "compile"):
        print("\nReloading model for torch.compile() benchmark...")
        # Reload model for fair comparison
        try:
            if args.model_name:
                tts = TTS(model_name=args.model_name, gpu=args.gpu, progress_bar=False)
            elif args.model_path and args.config_path:
                tts = TTS(
                    model_path=args.model_path,
                    config_path=args.config_path,
                    vocoder_path=args.vocoder_path,
                    vocoder_config_path=args.vocoder_config,
                    gpu=args.gpu,
                )

            print("Running optimized benchmark with torch.compile()...")
            optimized_results = benchmark_inference(
                tts_instance=tts,
                test_texts=args.text,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs,
                use_compile=True,
            )
        except Exception as e:
            print(f"Warning: torch.compile() benchmark failed: {e}")

    # Print results
    print_results(baseline_results, optimized_results)

    # Save results to file
    output_file = "benchmark_results.txt"
    with open(output_file, "w") as f:
        f.write("TTS Performance Benchmark Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model_name or args.model_path}\n")
        f.write(f"Device: {'GPU (CUDA)' if args.gpu else 'CPU'}\n")
        f.write(f"PyTorch version: {torch.__version__}\n\n")
        f.write("Baseline Results:\n")
        for key, value in baseline_results.items():
            f.write(f"  {key}: {value}\n")
        if optimized_results:
            f.write("\nOptimized Results:\n")
            for key, value in optimized_results.items():
                f.write(f"  {key}: {value}\n")
            speedup = baseline_results["mean_time"] / optimized_results["mean_time"]
            f.write(f"\nSpeedup: {speedup:.2f}x\n")

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
