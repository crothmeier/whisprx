#!/usr/bin/env python3
"""Performance profiling utilities for WhispRX"""
import time
import functools
from typing import Dict, List
from collections import defaultdict
import asyncio


class PerformanceProfiler:
    """Simple performance profiler for measuring function execution times"""

    def __init__(self):
        self.measurements: Dict[str, List[float]] = defaultdict(list)
        self._enabled = True

    def enable(self):
        """Enable profiling"""
        self._enabled = True

    def disable(self):
        """Disable profiling"""
        self._enabled = False

    def measure(self, name: str):
        """Decorator to measure function execution time

        Args:
            name: Measurement name

        Example:
            @profiler.measure("stt_inference")
            def transcribe(audio):
                ...
        """
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    if not self._enabled:
                        return await func(*args, **kwargs)

                    start = time.perf_counter()
                    result = await func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000  # ms

                    self.measurements[name].append(elapsed)
                    return result

                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    if not self._enabled:
                        return func(*args, **kwargs)

                    start = time.perf_counter()
                    result = func(*args, **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000  # ms

                    self.measurements[name].append(elapsed)
                    return result

                return sync_wrapper

        return decorator

    def record(self, name: str, duration_ms: float):
        """Manually record a measurement

        Args:
            name: Measurement name
            duration_ms: Duration in milliseconds
        """
        if self._enabled:
            self.measurements[name].append(duration_ms)

    def get_stats(self, name: str) -> Dict:
        """Get statistics for a measurement

        Args:
            name: Measurement name

        Returns:
            Dictionary with mean, min, max, count
        """
        if name not in self.measurements or not self.measurements[name]:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }

        values = self.measurements[name]

        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def print_report(self):
        """Print profiling report"""
        print("\n" + "=" * 60)
        print("PERFORMANCE PROFILING REPORT")
        print("=" * 60)

        if not self.measurements:
            print("\nNo measurements recorded")
            return

        print(f"\n{'Function':<30} {'Count':>8} {'Mean':>10} {'Min':>10} {'Max':>10}")
        print("-" * 70)

        for name in sorted(self.measurements.keys()):
            stats = self.get_stats(name)

            print(f"{name:<30} {stats['count']:>8} "
                  f"{stats['mean']:>9.2f}ms {stats['min']:>9.2f}ms {stats['max']:>9.2f}ms")

        print("=" * 60)

    def reset(self):
        """Clear all measurements"""
        self.measurements.clear()


# Global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance"""
    return _profiler


# Context manager for timing code blocks
class Timer:
    """Context manager for timing code blocks

    Example:
        with Timer("my_operation") as t:
            # do something
            pass
        print(f"Took {t.elapsed_ms:.2f}ms")
    """

    def __init__(self, name: str = "", record_to_profiler: bool = False):
        """Initialize timer

        Args:
            name: Timer name (for display)
            record_to_profiler: Whether to record to global profiler
        """
        self.name = name
        self.record_to_profiler = record_to_profiler
        self.start_time = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        self.elapsed_ms = elapsed * 1000

        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed_ms:.2f}ms")

        if self.record_to_profiler and self.name:
            _profiler.record(self.name, self.elapsed_ms)


# Example usage
if __name__ == "__main__":
    import asyncio

    profiler = get_profiler()

    @profiler.measure("fast_function")
    def fast_func():
        time.sleep(0.01)
        return "fast"

    @profiler.measure("slow_function")
    def slow_func():
        time.sleep(0.05)
        return "slow"

    @profiler.measure("async_function")
    async def async_func():
        await asyncio.sleep(0.02)
        return "async"

    # Run some functions
    for _ in range(10):
        fast_func()
        slow_func()

    asyncio.run(async_func())

    # Use context manager
    with Timer("manual_timing", record_to_profiler=True):
        time.sleep(0.03)

    # Print report
    profiler.print_report()
