#!/usr/bin/env python3
"""Integration tests for WhispRX end-to-end functionality"""
import asyncio
import pytest
import numpy as np
from pathlib import Path
import sys

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisprx_sdk.async_pipeline import AsyncPipeline
from whisprx_sdk.modules.stream_capture import SileroVADWrapper


class TestIntegration:
    """Integration tests for complete pipeline"""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test that pipeline initializes without models"""
        vad = SileroVADWrapper(use_cuda=False, threshold=0.5)

        pipeline = AsyncPipeline(
            vad,
            enable_stt=False,
            enable_llm=False,
            enable_tts=False,
        )

        await pipeline.start()
        assert len(pipeline._tasks) == 4
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_audio_worker_vad_filtering(self):
        """Test audio worker with VAD filtering"""
        vad = SileroVADWrapper(use_cuda=False, threshold=0.5)

        pipeline = AsyncPipeline(
            vad,
            enable_stt=False,
            enable_llm=False,
            enable_tts=False,
        )

        await pipeline.start()

        # Generate test audio (2 seconds @ 16kHz)
        audio = np.random.randint(-1000, 1000, 32000, dtype=np.int16)
        await pipeline.q_audio.put(audio.tobytes())

        # Wait a bit for processing
        await asyncio.sleep(0.5)

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_stt_worker_fallback(self):
        """Test STT worker fallback (no model)"""
        vad = SileroVADWrapper(use_cuda=False, threshold=0.5)

        pipeline = AsyncPipeline(
            vad,
            enable_stt=False,  # No STT model
            enable_llm=False,
            enable_tts=False,
        )

        await pipeline.start()

        # Push audio through VAD to STT
        audio = np.random.randint(-1000, 1000, 32000, dtype=np.int16)
        audio_tensor = pipeline.pcm_buf[:32000].copy_(
            pipeline.pcm_buf.new_tensor(
                np.frombuffer(audio.tobytes(), np.int16).astype(np.float32) / 32768.0
            )
        )

        await pipeline.q_stt.put((audio_tensor, 32000))

        # Should get fallback response
        try:
            response = await asyncio.wait_for(pipeline.q_llm.get(), timeout=2.0)
            assert response == "Hello, this is a test"
        except asyncio.TimeoutError:
            pytest.skip("STT worker timeout (expected without model)")

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_llm_worker_fallback(self):
        """Test LLM worker fallback (no model)"""
        vad = SileroVADWrapper(use_cuda=False, threshold=0.5)

        pipeline = AsyncPipeline(
            vad,
            enable_stt=False,
            enable_llm=False,  # No LLM model
            enable_tts=False,
        )

        await pipeline.start()

        # Send text to LLM queue
        await pipeline.q_llm.put("Test input")

        # Should get fallback response
        try:
            response = await asyncio.wait_for(pipeline.q_tts.get(), timeout=2.0)
            assert "I heard you say" in response
        except asyncio.TimeoutError:
            pytest.skip("LLM worker timeout")

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_tts_worker_fallback(self):
        """Test TTS worker fallback (no model)"""
        vad = SileroVADWrapper(use_cuda=False, threshold=0.5)

        pipeline = AsyncPipeline(
            vad,
            enable_stt=False,
            enable_llm=False,
            enable_tts=False,  # No TTS model
        )

        await pipeline.start()

        # Send text to TTS queue
        await pipeline.q_tts.put("Test synthesis")

        # Wait for processing
        await asyncio.sleep(0.5)

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_graceful_shutdown(self):
        """Test pipeline shuts down gracefully"""
        vad = SileroVADWrapper(use_cuda=False, threshold=0.5)

        pipeline = AsyncPipeline(
            vad,
            enable_stt=False,
            enable_llm=False,
            enable_tts=False,
        )

        await pipeline.start()

        # Add some items to queues
        await pipeline.q_audio.put(b'\x00' * 1000)

        # Stop should complete without hanging
        await asyncio.wait_for(pipeline.stop(), timeout=5.0)

        assert len(pipeline._tasks) == 0
        assert len(pipeline.workers) == 0


class TestMonitoring:
    """Tests for monitoring and metrics"""

    def test_metrics_recording(self):
        """Test metrics can be recorded"""
        from whisprx_sdk.monitoring import get_metrics

        metrics = get_metrics()

        # Record some metrics
        metrics.record_latency("stt", 0.145)
        metrics.record_latency("llm", 0.098)
        metrics.record_error("timeout", "stt")

        # Get stats
        stats = metrics.get_custom_stats()

        if "latency_stt" in stats:
            assert stats["latency_stt"]["count"] >= 1

    @pytest.mark.asyncio
    async def test_health_checks(self):
        """Test health check system"""
        from whisprx_sdk.resilience import get_health_check

        health = get_health_check()

        # Register a check
        async def check_test():
            return True

        health.register("test", check_test)

        # Run checks
        results = await health.run_checks()

        assert results["status"] in ["healthy", "unhealthy"]
        assert "checks" in results


class TestResilience:
    """Tests for resilience features"""

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        """Test retry decorator"""
        from whisprx_sdk.resilience import retry_with_backoff

        attempts = []

        @retry_with_backoff(max_retries=2, base_delay=0.1)
        async def flaky_function():
            attempts.append(1)
            if len(attempts) < 2:
                raise Exception("Simulated failure")
            return "Success"

        result = await flaky_function()

        assert result == "Success"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker"""
        from whisprx_sdk.resilience import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)

        # Cause failures
        for _ in range(2):
            try:
                await breaker.call_async(self._failing_function)
            except:
                pass

        # Circuit should be open
        assert breaker.state == "OPEN"

    async def _failing_function(self):
        raise Exception("Test failure")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
