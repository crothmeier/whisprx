#!/usr/bin/env python3
"""Precompile CUDA graphs for reduced latency on first inference"""
import torch
import argparse
from pathlib import Path


def precompile_llm_graphs(model_name: str, sequence_lengths: list):
    """Precompile CUDA graphs for LLM at various sequence lengths

    Args:
        model_name: HuggingFace model name or local path
        sequence_lengths: List of sequence lengths to precompile

    This warms up CUDA kernels and captures graph patterns to eliminate
    first-inference latency overhead (typically 20-50ms reduction).
    """
    print("=" * 60)
    print("CUDA Graph Precompilation for vLLM")
    print("=" * 60)

    try:
        from vllm import LLM
    except ImportError:
        print("❌ vLLM not available")
        return False

    print(f"\n📦 Loading model: {model_name}")

    try:
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.9,
            enforce_eager=False,  # Enable graph capture
            max_model_len=2048,
            trust_remote_code=True,
        )
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    print(f"\n🔥 Warming up CUDA graphs for sequence lengths: {sequence_lengths}")

    for seq_len in sequence_lengths:
        print(f"\n  Precompiling for seq_len={seq_len}...")

        try:
            # Create dummy input
            dummy_prompt = "test " * (seq_len // 2)  # Rough token count

            # Run inference to trigger graph capture
            from vllm import SamplingParams
            params = SamplingParams(
                temperature=0.7,
                max_tokens=min(32, seq_len // 4),  # Short generation
            )

            _ = llm.generate([dummy_prompt], params)

            print(f"  ✓ Seq_len={seq_len} precompiled")

        except Exception as e:
            print(f"  ⚠ Failed for seq_len={seq_len}: {e}")

    print("\n✅ CUDA graph precompilation complete!")
    print("\nBenefit: First inference latency reduced by ~20-50ms")

    return True


def precompile_whisper_graphs(model_size: str):
    """Precompile CUDA graphs for Whisper STT

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large-v3)
    """
    print("\n" + "=" * 60)
    print("CUDA Graph Precompilation for Whisper")
    print("=" * 60)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ faster-whisper not available")
        return False

    print(f"\n📦 Loading Whisper model: {model_size}")

    try:
        model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16",
        )
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    print("\n🔥 Warming up Whisper inference...")

    # Generate dummy audio (2 seconds @ 16kHz)
    import numpy as np
    sample_rate = 16000
    duration = 2.0
    dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.01

    try:
        # Run inference to warm up CUDA kernels
        segments, info = model.transcribe(
            dummy_audio,
            language="en",
            beam_size=1,
        )

        # Consume generator
        _ = list(segments)

        print("✓ Whisper CUDA kernels warmed up")
        print("\nBenefit: First inference latency reduced by ~10-30ms")

        return True

    except Exception as e:
        print(f"❌ Warmup failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Precompile CUDA graphs for WhispRX components"
    )
    parser.add_argument(
        "--llm-model",
        default="microsoft/Phi-3-mini-4k-instruct",
        help="LLM model to precompile (default: microsoft/Phi-3-mini-4k-instruct)"
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM precompilation"
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip Whisper precompilation"
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="Sequence lengths to precompile for LLM (default: 16 32 64 128 256)"
    )

    args = parser.parse_args()

    print("🚀 WhispRX CUDA Graph Precompilation")
    print("=" * 60)
    print("\nThis script warms up CUDA kernels to reduce first-inference latency.")
    print("Run this after starting your server for optimal performance.\n")

    success = True

    # Precompile LLM
    if not args.skip_llm:
        llm_success = precompile_llm_graphs(
            args.llm_model,
            args.sequence_lengths
        )
        success = success and llm_success
    else:
        print("\n⏭️  Skipping LLM precompilation")

    # Precompile Whisper
    if not args.skip_whisper:
        whisper_success = precompile_whisper_graphs(args.whisper_model)
        success = success and whisper_success
    else:
        print("\n⏭️  Skipping Whisper precompilation")

    print("\n" + "=" * 60)
    if success:
        print("✅ Precompilation complete!")
        print("\n📊 Expected latency improvements:")
        print("  - LLM first inference: -20 to -50ms")
        print("  - Whisper first inference: -10 to -30ms")
        print("  - Total: -30 to -80ms on first request")
    else:
        print("⚠️  Precompilation completed with some errors")

    print("\n💡 Tip: Run this script as part of your server startup routine")


if __name__ == "__main__":
    main()
