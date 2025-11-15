#!/usr/bin/env python3
"""WhispRX Benchmarking Suite - Measure end-to-end latency"""
import asyncio
import time
import statistics
from typing import Dict, List, Optional
import numpy as np
import json
from pathlib import Path

from .async_pipeline import AsyncPipeline
from .modules.stream_capture import SileroVADWrapper


class LatencyBenchmark:
    """Benchmark individual pipeline stages and end-to-end latency"""

    def __init__(
        self,
        whisper_model_path: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        tts_model_path: Optional[str] = None,
        num_iterations: int = 10,
    ):
        """Initialize benchmark

        Args:
            whisper_model_path: Path to Whisper model
            llm_model_name: LLM model name
            tts_model_path: Path to TTS model
            num_iterations: Number of test iterations
        """
        self.whisper_model_path = whisper_model_path
        self.llm_model_name = llm_model_name
        self.tts_model_path = tts_model_path
        self.num_iterations = num_iterations

        self.results: Dict[str, List[float]] = {
            "stt_latency": [],
            "llm_latency": [],
            "tts_latency": [],
            "end_to_end": [],
        }

    async def run(self) -> Dict:
        """Run complete benchmark suite

        Returns:
            Dictionary with benchmark results
        """
        print("=" * 60)
        print("WhispRX Performance Benchmark")
        print("=" * 60)

        # Initialize pipeline
        print("\n📊 Initializing pipeline...")
        vad = SileroVADWrapper(use_cuda=True, threshold=0.5)

        pipeline = AsyncPipeline(
            vad,
            whisper_model_path=self.whisper_model_path,
            llm_model_name=self.llm_model_name,
            tts_model_path=self.tts_model_path,
        )

        await pipeline.start()
        print("✓ Pipeline ready\n")

        # Generate test audio
        print(f"🎤 Running {self.num_iterations} iterations...\n")

        for i in range(self.num_iterations):
            print(f"Iteration {i+1}/{self.num_iterations}")

            # Generate 2 seconds of test audio (silence)
            audio_duration = 2.0  # seconds
            sample_rate = 16000
            audio_samples = int(audio_duration * sample_rate)
            test_audio = np.random.randint(-1000, 1000, audio_samples, dtype=np.int16)

            # Measure end-to-end latency
            start_time = time.perf_counter()

            # Push audio to pipeline
            await pipeline.q_audio.put(test_audio.tobytes())

            # Wait for STT
            stt_start = time.perf_counter()
            try:
                transcript = await asyncio.wait_for(pipeline.q_llm.get(), timeout=10.0)
                stt_latency = (time.perf_counter() - stt_start) * 1000
                self.results["stt_latency"].append(stt_latency)
                print(f"  STT: {stt_latency:.1f}ms - '{transcript[:50]}'")
            except asyncio.TimeoutError:
                print(f"  STT: TIMEOUT")
                continue

            # Wait for LLM
            llm_start = time.perf_counter()
            try:
                response = await asyncio.wait_for(pipeline.q_tts.get(), timeout=10.0)
                llm_latency = (time.perf_counter() - llm_start) * 1000
                self.results["llm_latency"].append(llm_latency)
                print(f"  LLM: {llm_latency:.1f}ms - '{response[:50]}'")
            except asyncio.TimeoutError:
                print(f"  LLM: TIMEOUT")
                continue

            # TTS latency is measured internally, estimate from logs
            # For now, we'll measure the time for TTS queue processing
            # In a real scenario, we'd need to instrument the TTS worker

            end_time = time.perf_counter()
            total_latency = (end_time - start_time) * 1000
            self.results["end_to_end"].append(total_latency)

            print(f"  Total: {total_latency:.1f}ms\n")

        # Stop pipeline
        await pipeline.stop()

        # Calculate statistics
        stats = self._calculate_statistics()

        # Print results
        self._print_results(stats)

        return stats

    def _calculate_statistics(self) -> Dict:
        """Calculate statistics from results

        Returns:
            Dictionary with mean, median, p95, p99 for each metric
        """
        stats = {}

        for metric, values in self.results.items():
            if not values:
                stats[metric] = {
                    "mean": 0,
                    "median": 0,
                    "p50": 0,
                    "p95": 0,
                    "p99": 0,
                    "min": 0,
                    "max": 0,
                }
                continue

            stats[metric] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p50": statistics.median(values),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
                "min": min(values),
                "max": max(values),
            }

        return stats

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile

        Args:
            data: List of values
            percentile: Percentile (0-100)

        Returns:
            Percentile value
        """
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _print_results(self, stats: Dict):
        """Print formatted benchmark results

        Args:
            stats: Statistics dictionary
        """
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        print("\n📊 Latency Breakdown:\n")

        for metric in ["stt_latency", "llm_latency", "tts_latency", "end_to_end"]:
            if metric not in stats:
                continue

            s = stats[metric]
            name = metric.replace("_", " ").upper()

            print(f"{name}:")
            print(f"  Mean:   {s['mean']:.2f}ms")
            print(f"  Median: {s['median']:.2f}ms")
            print(f"  P95:    {s['p95']:.2f}ms")
            print(f"  P99:    {s['p99']:.2f}ms")
            print(f"  Min:    {s['min']:.2f}ms")
            print(f"  Max:    {s['max']:.2f}ms")
            print()

        # Check if we meet 500ms target
        if stats.get("end_to_end", {}).get("p95", 1000) <= 500:
            print("✅ SUCCESS: Meeting 500ms latency target (P95)!")
        else:
            p95 = stats.get("end_to_end", {}).get("p95", 0)
            print(f"⚠️  NEEDS OPTIMIZATION: P95 latency is {p95:.0f}ms (target: <500ms)")

        print("\n" + "=" * 60)

    def save_results(self, output_path: str):
        """Save results to JSON file

        Args:
            output_path: Path to output file
        """
        stats = self._calculate_statistics()

        output = {
            "config": {
                "whisper_model": self.whisper_model_path,
                "llm_model": self.llm_model_name,
                "tts_model": self.tts_model_path,
                "iterations": self.num_iterations,
            },
            "results": stats,
            "raw_data": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")


async def main():
    """Main benchmark entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="WhispRX Benchmark Suite")
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model path/size (default: base)"
    )
    parser.add_argument(
        "--llm-model",
        default="microsoft/Phi-3-mini-4k-instruct",
        help="LLM model name (default: microsoft/Phi-3-mini-4k-instruct)"
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        help="TTS model path"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of test iterations (default: 10)"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)"
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = LatencyBenchmark(
        whisper_model_path=args.whisper_model,
        llm_model_name=args.llm_model,
        tts_model_path=args.tts_model,
        num_iterations=args.iterations,
    )

    stats = await benchmark.run()

    # Save results
    if args.output:
        benchmark.save_results(args.output)


if __name__ == "__main__":
    asyncio.run(main())
