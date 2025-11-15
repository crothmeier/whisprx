#!/usr/bin/env python3
"""Monitoring and observability with Prometheus metrics"""
from typing import Optional
import time
import asyncio
from collections import defaultdict
from threading import Lock

# Try to import prometheus_client, but make it optional
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        generate_latest,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠ prometheus_client not available. Install with: pip install prometheus-client")


class Metrics:
    """WhispRX metrics collector"""

    def __init__(self, enabled: bool = True):
        """Initialize metrics

        Args:
            enabled: Whether to collect metrics
        """
        self.enabled = enabled and PROMETHEUS_AVAILABLE

        if not self.enabled:
            return

        # Request metrics
        self.requests_total = Counter(
            'whisprx_requests_total',
            'Total number of requests',
            ['endpoint', 'status']
        )

        self.requests_in_progress = Gauge(
            'whisprx_requests_in_progress',
            'Number of requests currently being processed'
        )

        # Latency metrics
        self.latency_seconds = Histogram(
            'whisprx_latency_seconds',
            'Request latency in seconds',
            ['stage'],  # stt, llm, tts, end_to_end
            buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
        )

        self.latency_summary = Summary(
            'whisprx_latency_summary',
            'Request latency summary',
            ['stage']
        )

        # Model metrics
        self.model_load_time = Histogram(
            'whisprx_model_load_seconds',
            'Model loading time',
            ['model_type'],  # whisper, llm, tts
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
        )

        self.model_inference_time = Histogram(
            'whisprx_model_inference_seconds',
            'Model inference time',
            ['model_type'],
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
        )

        # Queue metrics
        self.queue_size = Gauge(
            'whisprx_queue_size',
            'Current queue size',
            ['queue_name']  # audio, stt, llm, tts
        )

        self.queue_wait_time = Histogram(
            'whisprx_queue_wait_seconds',
            'Time spent waiting in queue',
            ['queue_name'],
            buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
        )

        # Error metrics
        self.errors_total = Counter(
            'whisprx_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )

        # WebSocket metrics
        self.websocket_connections = Gauge(
            'whisprx_websocket_connections',
            'Number of active WebSocket connections'
        )

        self.websocket_messages_total = Counter(
            'whisprx_websocket_messages_total',
            'Total WebSocket messages',
            ['direction']  # sent, received
        )

        # Audio metrics
        self.audio_bytes_processed = Counter(
            'whisprx_audio_bytes_total',
            'Total audio bytes processed'
        )

        self.audio_chunks_total = Counter(
            'whisprx_audio_chunks_total',
            'Total audio chunks processed',
            ['stage']  # input, output
        )

        # System info
        self.build_info = Info(
            'whisprx_build',
            'Build information'
        )
        self.build_info.info({
            'version': '0.1.0',
            'phase': '3',
        })

        # Custom metrics storage (for non-Prometheus use)
        self._custom_metrics = defaultdict(list)
        self._lock = Lock()

    def record_request(self, endpoint: str, status: str):
        """Record a request

        Args:
            endpoint: Endpoint name
            status: Status (success, error, etc.)
        """
        if self.enabled:
            self.requests_total.labels(endpoint=endpoint, status=status).inc()

    def record_latency(self, stage: str, duration_seconds: float):
        """Record latency for a stage

        Args:
            stage: Stage name (stt, llm, tts, end_to_end)
            duration_seconds: Duration in seconds
        """
        if self.enabled:
            self.latency_seconds.labels(stage=stage).observe(duration_seconds)
            self.latency_summary.labels(stage=stage).observe(duration_seconds)
        else:
            # Store in custom metrics
            with self._lock:
                self._custom_metrics[f"latency_{stage}"].append(duration_seconds)

    def record_model_load(self, model_type: str, duration_seconds: float):
        """Record model loading time

        Args:
            model_type: Model type (whisper, llm, tts)
            duration_seconds: Duration in seconds
        """
        if self.enabled:
            self.model_load_time.labels(model_type=model_type).observe(duration_seconds)

    def record_error(self, error_type: str, component: str):
        """Record an error

        Args:
            error_type: Error type
            component: Component name
        """
        if self.enabled:
            self.errors_total.labels(error_type=error_type, component=component).inc()

    def set_queue_size(self, queue_name: str, size: int):
        """Update queue size

        Args:
            queue_name: Queue name
            size: Current size
        """
        if self.enabled:
            self.queue_size.labels(queue_name=queue_name).set(size)

    def inc_websocket_connections(self):
        """Increment WebSocket connection count"""
        if self.enabled:
            self.websocket_connections.inc()

    def dec_websocket_connections(self):
        """Decrement WebSocket connection count"""
        if self.enabled:
            self.websocket_connections.dec()

    def get_metrics_text(self) -> str:
        """Get Prometheus metrics in text format

        Returns:
            Metrics in Prometheus text format
        """
        if not self.enabled:
            return "# Prometheus not available\n"

        return generate_latest(REGISTRY).decode('utf-8')

    def get_custom_stats(self) -> dict:
        """Get statistics from custom metrics (non-Prometheus)

        Returns:
            Dictionary with statistics
        """
        stats = {}

        with self._lock:
            for key, values in self._custom_metrics.items():
                if not values:
                    continue

                import statistics
                stats[key] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }

        return stats


# Global metrics instance
_metrics = Metrics()


def get_metrics() -> Metrics:
    """Get global metrics instance"""
    return _metrics


# Context manager for timing operations
class MetricsTimer:
    """Context manager for timing operations and recording to metrics

    Example:
        with MetricsTimer("stt"):
            # do STT inference
            pass
    """

    def __init__(self, stage: str):
        """Initialize timer

        Args:
            stage: Stage name for metrics
        """
        self.stage = stage
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self.start_time
        get_metrics().record_latency(self.stage, duration)

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, *args):
        duration = time.perf_counter() - self.start_time
        get_metrics().record_latency(self.stage, duration)


# Decorator for measuring function execution time
def measure_latency(stage: str):
    """Decorator to measure and record function latency

    Args:
        stage: Stage name for metrics

    Example:
        @measure_latency("stt")
        async def transcribe(audio):
            ...
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start
                    get_metrics().record_latency(stage, duration)

            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start
                    get_metrics().record_latency(stage, duration)

            return sync_wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    metrics = get_metrics()

    # Record some metrics
    metrics.record_request("/ws", "success")
    metrics.record_latency("stt", 0.145)
    metrics.record_latency("llm", 0.098)
    metrics.record_latency("tts", 0.180)
    metrics.record_latency("end_to_end", 0.480)

    metrics.record_error("timeout", "stt")
    metrics.set_queue_size("audio", 3)

    # Get metrics
    if PROMETHEUS_AVAILABLE:
        print(metrics.get_metrics_text())
    else:
        print(metrics.get_custom_stats())
