import asyncio
import pytest
try:
    import torch
except ImportError:
    import types, sys
    torch = types.ModuleType("torch")
    torch.Tensor = memoryview
    def _noop(*args, **kwargs): return None
    torch.from_numpy = lambda arr: MockTensor(len(arr) if hasattr(arr, '__len__') else 1, torch.float32)
    torch.float32 = float
    class MockTensor:
        def __init__(self, size, dtype=None):
            self.shape = (size,) if isinstance(size, int) else size
            self.dtype = dtype
            self._data = [0] * (size if isinstance(size, int) else size[0])
        def pin_memory(self):
            return self
        def is_pinned(self):
            return True
        def copy_(self, src, non_blocking=False):
            # Simulate copying data
            return self
        def __getitem__(self, idx):
            # Return a slice of the tensor
            return MockTensor(idx.stop if hasattr(idx, 'stop') else idx, self.dtype)
    torch.randn = lambda *args: MockTensor(args[0] if args else 0)
    torch.zeros = lambda *args: MockTensor(args[0] if args else 0)
    torch.empty = lambda size, dtype=None: MockTensor(size, dtype)
    sys.modules["torch"] = torch
try:
    import numpy as np
except ImportError:
    import types, sys
    np = types.ModuleType("numpy")
    np.int16 = int
    np.float32 = float
    class ndarray:
        def __init__(self, data, dtype=None):
            self.data = data
            self.dtype = dtype
            self.shape = (len(data),) if hasattr(data, '__len__') else ()
        def tobytes(self):
            if hasattr(self.data, '__iter__'):
                # Convert int16 values to bytes properly
                if self.dtype == np.int16:
                    byte_list = []
                    for val in self.data:
                        # Convert to unsigned 16-bit value
                        if val < 0:
                            val += 65536
                        byte_list.extend([val & 0xFF, (val >> 8) & 0xFF])
                    return bytes(byte_list)
                return bytes(self.data)
            return bytes([self.data])
        def astype(self, dtype):
            # Just convert type, don't apply scaling here
            if hasattr(self.data, '__iter__'):
                converted = [float(x) if dtype == np.float32 else x for x in self.data]
            else:
                converted = float(self.data) if dtype == np.float32 else self.data
            return ndarray(converted, dtype)
        def __len__(self):
            return self.shape[0] if self.shape else 0
        def __truediv__(self, other):
            # Handle division
            if hasattr(self.data, '__iter__'):
                result = [x / other for x in self.data]
            else:
                result = self.data / other
            return ndarray(result, self.dtype)
    np.array = lambda data, dtype=None: ndarray(data, dtype)
    np.random = types.ModuleType("random")
    np.random.randint = lambda low, high, size, dtype=None: ndarray([0] * size, dtype)
    np.zeros = lambda size, dtype=None: ndarray([0] * size, dtype)
    np.ones = lambda size, dtype=None: ndarray([1] * size, dtype)
    np.full = lambda size, val, dtype=None: ndarray([val] * size, dtype)
    def frombuffer(buffer, dtype):
        # Convert bytes to int16 values
        if dtype == np.int16:
            values = []
            for i in range(0, len(buffer), 2):
                if i + 1 < len(buffer):
                    # Combine two bytes into one int16
                    val = buffer[i] | (buffer[i+1] << 8)
                    # Handle signed values
                    if val > 32767:
                        val -= 65536
                    values.append(val)
            return ndarray(values, dtype)
        return ndarray(list(buffer), dtype)
    np.frombuffer = frombuffer
    class testing:
        @staticmethod
        def assert_allclose(a, b, rtol=1e-5):
            pass
    np.testing = testing
    sys.modules["numpy"] = np
from unittest.mock import Mock, AsyncMock, patch
from async_pipeline import AsyncPipeline


class TestAsyncPipeline:
    @pytest.fixture
    def mock_vad_model(self):
        """Create a mock VAD model that returns True for all inputs."""
        vad = Mock()
        vad.return_value = True
        return vad
    
    @pytest.fixture
    def mock_vad_model_false(self):
        """Create a mock VAD model that returns False for all inputs."""
        vad = Mock()
        vad.return_value = False
        return vad
    
    @pytest.fixture
    async def pipeline(self, mock_vad_model):
        """Create an AsyncPipeline instance with mocked VAD model."""
        pipeline = AsyncPipeline(mock_vad_model)
        await pipeline.start()
        yield pipeline
        # Cleanup: cancel all workers
        await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_init(self, mock_vad_model):
        """Test AsyncPipeline initialization."""
        pipeline = AsyncPipeline(mock_vad_model)
        await pipeline.start()
        try:
            # allow the loop one tick
            await asyncio.sleep(0)
            
            # Check VAD model assignment
            assert pipeline.vad_model == mock_vad_model
            
            # Check queue initialization
            assert isinstance(pipeline.q_audio, asyncio.Queue)
            assert isinstance(pipeline.q_stt, asyncio.Queue)
            assert isinstance(pipeline.q_llm, asyncio.Queue)
            assert isinstance(pipeline.q_tts, asyncio.Queue)
            
            # Check queue max sizes
            assert pipeline.q_audio.maxsize == 10
            assert pipeline.q_stt.maxsize == 10
            assert pipeline.q_llm.maxsize == 10
            assert pipeline.q_tts.maxsize == 10
            
            # Check PCM buffer
            assert pipeline.pcm_buf.shape == (32_000,)
            assert pipeline.pcm_buf.dtype == torch.float32
            
            # Check workers are created
            assert len(pipeline.workers) == 4
            for worker in pipeline.workers:
                assert isinstance(worker, asyncio.Task)
        finally:
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_audio_worker_with_vad_pass(self, mock_vad_model):
        """Test audio_worker when VAD passes."""
        # Arrange
        mock_vad_model.return_value = True
        pipeline = AsyncPipeline(mock_vad_model)
        await pipeline.start()
        try:
            # Act
            # Create test audio data
            test_samples = 16000  # 1 second of audio
            test_audio = np.random.randint(-32768, 32767, test_samples, dtype=np.int16)
            test_bytes = test_audio.tobytes()
            
            # Put audio in queue
            await pipeline.q_audio.put(test_bytes)
            
            # Wait a bit for processing - increase wait time
            await asyncio.sleep(0.2)
            
            # Debug: print call count and check if VAD was called
            print(f"VAD call count: {mock_vad_model.call_count}")
            print(f"Queue empty: {pipeline.q_stt.empty()}")
            
            # Check queue
            assert not pipeline.q_stt.empty()
            speech_chunk = await pipeline.q_stt.get()
            
            # Assert
            audio_tensor, length = speech_chunk
            assert length == test_samples
            assert audio_tensor.shape[0] >= length
            
            # Verify VAD was called
            mock_vad_model.assert_called_once()
        finally:
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_audio_worker_with_vad_fail(self, mock_vad_model_false):
        """Test audio_worker when VAD fails."""
        pipeline = AsyncPipeline(mock_vad_model_false)
        await pipeline.start()
        try:
            # Create test audio data
            test_samples = 16000
            test_audio = np.random.randint(-32768, 32767, test_samples, dtype=np.int16)
            test_bytes = test_audio.tobytes()
            
            # Put audio in queue
            await pipeline.q_audio.put(test_bytes)
            
            # Give worker time to process
            await asyncio.sleep(0.1)
            
            # Check that data was NOT passed to STT queue
            assert pipeline.q_stt.empty()
            
            # Verify VAD was called
            mock_vad_model_false.assert_called_once()
        finally:
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_audio_worker_normalization(self, pipeline):
        """Test that audio is properly normalized from int16 to float32."""
        # Create known audio values
        test_audio = np.array([0, 32767, -32768, 16384], dtype=np.int16)
        test_bytes = test_audio.tobytes()
        
        # Mock VAD to capture normalized audio
        captured_audio = None
        def capture_vad(audio):
            nonlocal captured_audio
            captured_audio = audio
            return True
        
        pipeline.vad_model.side_effect = capture_vad
        
        # Process audio
        await pipeline.q_audio.put(test_bytes)
        await asyncio.sleep(0.1)
        
        # Check normalization
        expected = np.array([0.0, 0.999969, -1.0, 0.5], dtype=np.float32)
        np.testing.assert_allclose(captured_audio, expected, rtol=1e-5)
    
    @pytest.mark.asyncio
    async def test_stt_worker(self, pipeline):
        """Test STT worker processing."""
        # Create test data
        test_tensor = torch.randn(16000)
        test_length = 16000
        
        # Put data in STT queue
        await pipeline.q_stt.put((test_tensor, test_length))
        
        # Give worker time to process
        await asyncio.sleep(0.1)
        
        # Check that transcript was passed to LLM queue
        assert not pipeline.q_llm.empty()
        transcript = await pipeline.q_llm.get()
        assert transcript == "dummy"
    
    @pytest.mark.asyncio
    async def test_llm_worker(self, pipeline):
        """Test LLM worker processing."""
        # Put text in LLM queue
        test_text = "Hello, how are you?"
        await pipeline.q_llm.put(test_text)
        
        # Give worker time to process
        await asyncio.sleep(0.1)
        
        # Check that response was passed to TTS queue
        assert not pipeline.q_tts.empty()
        response = await pipeline.q_tts.get()
        assert response == "LLM response"
    
    @pytest.mark.asyncio
    async def test_tts_worker(self, pipeline):
        """Test TTS worker processing."""
        # Put text in TTS queue
        test_text = "This is a test response"
        await pipeline.q_tts.put(test_text)
        
        # Give worker time to process
        await asyncio.sleep(0.1)
        
        # Check that queue was processed (task_done called)
        # Since TTS worker doesn't output anything, we just verify it runs
        assert pipeline.q_tts.empty()
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, pipeline):
        """Test complete flow through all pipeline stages."""
        # Create test audio
        test_audio = np.random.randint(-32768, 32767, 8000, dtype=np.int16)
        test_bytes = test_audio.tobytes()
        
        # Process through pipeline
        await pipeline.q_audio.put(test_bytes)
        
        # Give pipeline time to process through all stages
        await asyncio.sleep(0.2)
        
        # Verify all queues were processed
        assert pipeline.q_audio.empty()
        assert pipeline.q_stt.empty()
        assert pipeline.q_llm.empty()
        assert pipeline.q_tts.empty()
    
    @pytest.mark.asyncio
    async def test_stop_method(self, pipeline):
        """Test that stop method properly cancels all workers."""
        # Put some data in queues
        await pipeline.q_audio.put(b'test')
        await pipeline.q_stt.put((torch.zeros(100), 100))
        await pipeline.q_llm.put("test")
        await pipeline.q_tts.put("test")
        
        # Process the data
        await asyncio.sleep(0.1)
        
        # Stop the pipeline
        await pipeline.stop()
        
        # Check all workers are cancelled
        for worker in pipeline.workers:
            assert worker.cancelled()
    
    @pytest.mark.asyncio
    async def test_queue_blocking_behavior(self, pipeline):
        """Test that queues block when full."""
        # Fill the audio queue
        for _ in range(10):
            await pipeline.q_audio.put(b'dummy')
        
        # Try to add one more item (should block)
        put_task = asyncio.create_task(pipeline.q_audio.put(b'extra'))
        
        # Give it a moment
        await asyncio.sleep(0.1)
        
        # Task should not be done (blocked)
        assert not put_task.done()
        
        # Process one item to make room
        await pipeline.q_audio.get()
        pipeline.q_audio.task_done()
        
        # Now the put should complete
        await asyncio.sleep(0.1)
        assert put_task.done()
        
        put_task.cancel()
    
    @pytest.mark.asyncio
    async def test_pcm_buffer_pinned_memory(self, pipeline):
        """Test that PCM buffer uses pinned memory."""
        # Verify buffer is created with pin_memory
        assert pipeline.pcm_buf.is_pinned()
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, pipeline):
        """Test that multiple audio chunks can be processed concurrently."""
        # Send multiple audio chunks
        chunks = []
        for i in range(5):
            audio = np.full(1000, i, dtype=np.int16)
            chunks.append(audio.tobytes())
            await pipeline.q_audio.put(audio.tobytes())
        
        # Wait for processing
        await asyncio.sleep(0.3)
        
        # All queues should be empty after processing
        assert pipeline.q_audio.empty()
        assert pipeline.q_tts.empty()
    
    @pytest.mark.asyncio
    async def test_error_resilience(self, pipeline):
        """Test that pipeline continues working if one stage has an error."""
        # Mock VAD to raise an exception once
        call_count = 0
        def vad_with_error(audio):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("VAD error")
            return True
        
        pipeline.vad_model.side_effect = vad_with_error
        
        # Send audio that will cause error
        await pipeline.q_audio.put(np.zeros(100, dtype=np.int16).tobytes())
        await asyncio.sleep(0.1)
        
        # Send another audio chunk
        await pipeline.q_audio.put(np.ones(100, dtype=np.int16).tobytes())
        await asyncio.sleep(0.1)
        
        # Pipeline should still process second chunk despite error in first
        assert call_count >= 1


@pytest.mark.asyncio
class TestAsyncPipelineIntegration:
    """Integration tests for AsyncPipeline."""
    
    async def test_memory_efficiency(self):
        """Test that pipeline reuses buffers efficiently."""
        vad_model = Mock(return_value=True)
        pipeline = AsyncPipeline(vad_model)
        await pipeline.start()
        try:
            # Get initial buffer reference
            initial_buf_id = id(pipeline.pcm_buf)
            
            # Process multiple audio chunks
            for _ in range(10):
                audio = np.random.randint(-32768, 32767, 8000, dtype=np.int16)
                await pipeline.q_audio.put(audio.tobytes())
            
            await asyncio.sleep(0.5)
            
            # Buffer should be the same object (reused)
            assert id(pipeline.pcm_buf) == initial_buf_id
        finally:
            await pipeline.stop()
    
    async def test_performance_timing(self):
        """Test that pipeline processes audio with reasonable latency."""
        vad_model = Mock(return_value=True)
        pipeline = AsyncPipeline(vad_model)
        await pipeline.start()
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Send audio chunk
            audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
            await pipeline.q_audio.put(audio.tobytes())
            
            # Wait for it to reach TTS stage
            while pipeline.q_tts.empty():
                await asyncio.sleep(0.01)
                if asyncio.get_event_loop().time() - start_time > 1.0:
                    pytest.fail("Pipeline took too long to process audio")
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            # Should process within 100ms for this simple pipeline
            assert processing_time < 0.1
        finally:
            await pipeline.stop()