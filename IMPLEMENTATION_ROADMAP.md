# WhispRX Implementation Roadmap

**Goal**: Transform the 35% complete codebase into a functional 500ms low-latency conversational AI system

**Phases**: MVP → Optimization → Production
**Timeline**: 4-6 weeks to production-ready

---

## Phase 1: Core AI Integration (MVP) - CRITICAL
**Timeline**: 2-3 weeks
**Goal**: Functional end-to-end speech-to-speech pipeline

### 1.1 Whisper STT Integration ⚡ HIGHEST PRIORITY
**Effort**: 2-3 days
**Files to Modify**:
- `packages/whisprx_sdk/async_pipeline.py`

**Tasks**:
- [ ] Load CTranslate2 Whisper model in `AsyncPipeline.__init__`
  ```python
  import ctranslate2
  self.whisper_model = ctranslate2.models.Whisper(
      model_path="models/whisper-large-v3-ct2",
      device="cuda",
      compute_type="float16"
  )
  ```
- [ ] Replace `stt_worker()` placeholder (lines 73-82)
  ```python
  async def stt_worker(self):
      while True:
          audio_tensor, length = await self.q_stt.get()
          try:
              # Convert to numpy array
              audio_np = audio_tensor.cpu().numpy()

              # Run CTranslate2 inference
              result = self.whisper_model.transcribe(audio_np)
              transcript = result[0]["text"]

              await self.q_llm.put(transcript)
          except Exception as e:
              print(f"STT error: {e}")
          finally:
              self.q_stt.task_done()
  ```
- [ ] Add model download script
- [ ] Test with real audio input
- [ ] Measure STT latency

**Success Criteria**: Real speech → accurate transcript in <200ms

---

### 1.2 LLM Worker Integration
**Effort**: 1-2 days
**Files to Modify**:
- `packages/whisprx_sdk/llm_service.py`
- `packages/whisprx_sdk/async_pipeline.py`

**Tasks**:
- [ ] Add streaming generation to `LLMService`
  ```python
  async def stream_generate(self, prompt: str):
      """Yield tokens as they're generated"""
      sampling_params = SamplingParams(
          temperature=0.7,
          max_tokens=256,
          stream=True
      )

      async for output in self.llm.generate_async(prompt, sampling_params):
          if output.outputs:
              yield output.outputs[0].text
  ```
- [ ] Integrate into `llm_worker()` (lines 84-92)
  ```python
  async def llm_worker(self):
      while True:
          transcript = await self.q_llm.get()
          try:
              full_response = ""
              async for token in self.llm.stream_generate(transcript):
                  full_response += token

              await self.q_tts.put(full_response)
          except Exception as e:
              print(f"LLM error: {e}")
          finally:
              self.q_llm.task_done()
  ```
- [ ] Add model path configuration
- [ ] Test with sample transcripts
- [ ] Measure LLM latency

**Success Criteria**: Transcript → conversational response in <150ms

---

### 1.3 TTS Integration (Piper)
**Effort**: 3-5 days
**Files to Create**:
- `packages/whisprx_sdk/tts_service.py`

**Files to Modify**:
- `packages/whisprx_sdk/async_pipeline.py`

**Tasks**:
- [ ] Choose TTS backend:
  - **Recommended**: Piper (easier, faster to integrate)
  - Alternative: FishSpeech (higher quality, more complex)
- [ ] Create `TTSService` class
  ```python
  import subprocess
  import numpy as np

  class TTSService:
      def __init__(self, model_path: str = "models/piper/en_US-lessac-medium"):
          self.model_path = model_path
          self.sample_rate = 16000

      def synthesize(self, text: str) -> bytes:
          """Convert text to audio waveform"""
          # Run Piper CLI
          result = subprocess.run(
              ["piper", "--model", self.model_path, "--output_raw"],
              input=text.encode(),
              capture_output=True
          )

          if result.returncode != 0:
              raise RuntimeError(f"TTS failed: {result.stderr}")

          return result.stdout
  ```
- [ ] Implement `tts_worker()` (lines 95-102)
  ```python
  async def tts_worker(self):
      while True:
          text = await self.q_tts.get()
          try:
              # Generate audio
              audio_bytes = self.tts.synthesize(text)

              # TODO: Send via WebSocket
              print(f"Generated {len(audio_bytes)} bytes of audio")
          except Exception as e:
              print(f"TTS error: {e}")
          finally:
              self.q_tts.task_done()
  ```
- [ ] Download Piper models
- [ ] Test synthesis quality
- [ ] Measure TTS latency

**Success Criteria**: Text → audio in <150ms (using ONNX, pre-TensorRT)

---

### 1.4 WebSocket Server with MessagePack
**Effort**: 2-3 days
**Files to Create**:
- `packages/whisprx_sdk/ws_server.py`

**Files to Modify**:
- `services/whisprx_api/main.py`

**Tasks**:
- [ ] Create `MessagePackConnection` class
  ```python
  import msgpack
  from fastapi import WebSocket
  from .async_pipeline import AsyncPipeline
  from .modules.batched_vad import BatchedSileroVAD

  class MessagePackConnection:
      def __init__(self, websocket: WebSocket):
          self.ws = websocket
          self.pipeline = None
          self.session_id = None

      async def handle(self):
          await self.ws.accept()

          # Initialize pipeline
          from .modules.stream_capture import SileroVADWrapper
          vad = SileroVADWrapper(use_cuda=True)
          self.pipeline = AsyncPipeline(vad)
          await self.pipeline.start()

          try:
              while True:
                  # Receive MessagePack binary
                  data = await self.ws.receive_bytes()
                  msg = msgpack.unpackb(data)

                  await self.handle_message(msg)
          except Exception as e:
              print(f"WebSocket error: {e}")
          finally:
              if self.pipeline:
                  await self.pipeline.stop()

      async def handle_message(self, msg: dict):
          msg_type = msg.get("type")

          if msg_type == "audio":
              # Push audio to pipeline
              audio_data = msg["data"]
              await self.pipeline.q_audio.put(audio_data)

          # TODO: Add response handling
  ```
- [ ] Integrate with FastAPI endpoint
- [ ] Add response streaming from pipeline
- [ ] Implement session management
- [ ] Add error handling and reconnection

**Success Criteria**: WebSocket accepts audio, returns transcript + TTS audio

---

### 1.5 Client Demo Application
**Effort**: 2-3 days
**Files to Create**:
- `packages/whisprx_sdk/client_demo.py`

**Tasks**:
- [ ] Create WebSocket client
  ```python
  import asyncio
  import websockets
  import msgpack
  import sounddevice as sd
  import numpy as np

  async def run_client():
      uri = "ws://localhost:8000/ws"

      async with websockets.connect(uri) as ws:
          print("Connected to WhispRX server")

          # Audio capture task
          async def capture_audio():
              stream = sd.InputStream(
                  samplerate=16000,
                  channels=1,
                  dtype=np.int16,
                  blocksize=3200  # 200ms chunks
              )
              stream.start()

              while True:
                  audio_chunk, _ = stream.read(3200)
                  msg = msgpack.packb({
                      "type": "audio",
                      "data": audio_chunk.tobytes()
                  }, use_bin_type=True)

                  await ws.send(msg)
                  await asyncio.sleep(0.2)

          # Response handler task
          async def handle_responses():
              while True:
                  response = await ws.recv()
                  data = msgpack.unpackb(response)

                  if data["type"] == "transcript":
                      print(f"[STT] {data['text']}")
                  elif data["type"] == "llm_response":
                      print(f"[LLM] {data['text']}")
                  elif data["type"] == "audio":
                      # Play TTS audio
                      audio = np.frombuffer(data["data"], dtype=np.int16)
                      sd.play(audio, samplerate=16000)

          # Run both tasks
          await asyncio.gather(
              capture_audio(),
              handle_responses()
          )

  if __name__ == "__main__":
      asyncio.run(run_client())
  ```
- [ ] Add CLI interface
- [ ] Implement audio playback
- [ ] Add status indicators
- [ ] Test end-to-end flow

**Success Criteria**: Speak → hear AI response in <1 second

---

### 1.6 End-to-End Testing
**Effort**: 1-2 days

**Tasks**:
- [ ] Create integration test suite
- [ ] Test complete pipeline flow
- [ ] Measure actual latencies:
  - Audio capture: ?ms
  - STT: ?ms
  - LLM: ?ms
  - TTS: ?ms
  - Total: ?ms
- [ ] Debug integration issues
- [ ] Document findings

**Success Criteria**: Consistent end-to-end operation, latencies measured

---

## Phase 2: Optimization (Performance)
**Timeline**: 1-2 weeks
**Goal**: Achieve 500ms target latency

### 2.1 TensorRT Vocoder Optimization
**Effort**: 3-5 days
**Impact**: 2-3x TTS speedup (180ms → 60ms)

**Tasks**:
- [ ] Export Piper vocoder to ONNX
- [ ] Convert ONNX to TensorRT engine
  ```bash
  trtexec \
      --onnx=piper_vocoder.onnx \
      --saveEngine=vocoder.trt \
      --fp16 \
      --workspace=4096 \
      --minShapes=mel:1x80x32 \
      --optShapes=mel:1x80x256 \
      --maxShapes=mel:1x80x1024
  ```
- [ ] Implement TensorRT inference (based on patches/tts_service_trt.diff)
- [ ] Add CUDA stream for async execution
- [ ] Implement chunked processing (256-hop size)
- [ ] Benchmark speedup

**Success Criteria**: TTS latency <80ms

---

### 2.2 Early TTS Triggering
**Effort**: 1-2 days
**Impact**: 100-200ms perceived latency reduction

**Tasks**:
- [ ] Integrate `PipelineManager` logic into workers
- [ ] Start TTS after first 4 LLM tokens
- [ ] Feed incremental 8-token deltas to TTS
- [ ] Handle streaming audio output
- [ ] Test perceived latency improvement

**Success Criteria**: Audio starts playing before LLM finishes

---

### 2.3 CUDA Graph Precompilation
**Effort**: 1-2 days
**Impact**: 20-50ms reduction in first inference

**Tasks**:
- [ ] Create precompilation script
  ```python
  import torch
  from whisprx_sdk.llm_service import LLMService

  llm = LLMService()

  for seq_len in [16, 32, 64, 128, 256]:
      dummy = torch.randint(0, 32000, (1, seq_len), device='cuda')
      _ = llm.generate(dummy)  # Trigger graph capture
      print(f"Precompiled seq_len={seq_len}")
  ```
- [ ] Add to startup sequence
- [ ] Measure latency improvement

**Success Criteria**: First inference as fast as subsequent ones

---

### 2.4 Benchmarking Suite
**Effort**: 2 days

**Tasks**:
- [ ] Create benchmark script
  ```python
  import time
  import statistics

  async def benchmark():
      latencies = {"stt": [], "llm": [], "tts": [], "total": []}

      for i in range(100):
          start = time.perf_counter()

          # Measure each stage
          t0 = time.perf_counter()
          transcript = await run_stt(audio)
          t1 = time.perf_counter()
          response = await run_llm(transcript)
          t2 = time.perf_counter()
          audio_out = await run_tts(response)
          t3 = time.perf_counter()

          latencies["stt"].append((t1-t0)*1000)
          latencies["llm"].append((t2-t1)*1000)
          latencies["tts"].append((t3-t2)*1000)
          latencies["total"].append((t3-t0)*1000)

      # Print statistics
      for stage, times in latencies.items():
          print(f"{stage}:")
          print(f"  Mean: {statistics.mean(times):.2f}ms")
          print(f"  P50:  {statistics.median(times):.2f}ms")
          print(f"  P95:  {statistics.quantiles(times, n=20)[18]:.2f}ms")
          print(f"  P99:  {statistics.quantiles(times, n=100)[98]:.2f}ms")
  ```
- [ ] Profile each pipeline stage
- [ ] Identify bottlenecks
- [ ] Create performance report

**Success Criteria**: Detailed latency breakdown, 500ms target achieved

---

## Phase 3: Production Hardening
**Timeline**: 1-2 weeks
**Goal**: Reliable, scalable deployment

### 3.1 Error Recovery & Resilience
**Effort**: 2-3 days

**Tasks**:
- [ ] Add model loading retries
- [ ] Implement pipeline restart on failure
- [ ] Add health check endpoints
- [ ] Graceful degradation (fallback models)
- [ ] Circuit breakers for external deps
- [ ] Comprehensive error logging

---

### 3.2 Multi-User Support
**Effort**: 2-3 days

**Tasks**:
- [ ] Session management (UUID per connection)
- [ ] Concurrent pipeline instances
- [ ] Resource pooling (GPU memory limits)
- [ ] Rate limiting per client
- [ ] Queue management under load

---

### 3.3 Monitoring & Observability
**Effort**: 2-3 days

**Tasks**:
- [ ] Add Prometheus metrics
  - Latency histograms (per stage)
  - Request rate
  - Error rate
  - GPU utilization
  - Queue depths
- [ ] Add OpenTelemetry tracing
- [ ] Create Grafana dashboard
- [ ] Set up alerting

---

### 3.4 Documentation
**Effort**: 2-3 days

**Tasks**:
- [ ] Write `docs/ARCHITECTURE.md`
  - System diagram
  - Component descriptions
  - Data flow
- [ ] Write `docs/INSTALLATION.md`
  - CUDA setup
  - Python environment
  - Model downloads
  - Configuration
- [ ] Write `docs/USAGE.md`
  - API reference
  - Client examples
  - Performance tuning
- [ ] Create `docs/DEPLOYMENT.md`
  - Kubernetes setup
  - GPU scheduling
  - Scaling strategies

---

### 3.5 Kubernetes Production Deployment
**Effort**: 3-4 days

**Tasks**:
- [ ] Update Dockerfile
  ```dockerfile
  FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

  # Install Python 3.11
  RUN apt-get update && apt-get install -y \
      python3.11 python3-pip \
      && rm -rf /var/lib/apt/lists/*

  # Copy requirements
  COPY requirements.txt .
  RUN pip install -r requirements.txt

  # Copy models (or mount as volume)
  COPY models/ /models/

  # Copy application
  COPY packages/ /app/packages/
  COPY services/ /app/services/

  WORKDIR /app/services/whisprx_api
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] Update Helm chart
  ```yaml
  # values.yaml
  resources:
    limits:
      nvidia.com/gpu: 1
      memory: 16Gi
    requests:
      nvidia.com/gpu: 1
      memory: 12Gi

  volumes:
    - name: models
      persistentVolumeClaim:
        claimName: whisprx-models

  env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
  ```
- [ ] Add GPU node affinity
- [ ] Configure persistent volume for models
- [ ] Add horizontal pod autoscaling (based on queue depth)
- [ ] Set up ingress with load balancing
- [ ] Test production deployment

---

## Phase 4: Advanced Features (Optional)
**Timeline**: Ongoing

### 4.1 Voice Cloning
- Add speaker embedding extraction
- Fine-tune TTS with custom voices
- API for voice selection

### 4.2 Multi-Language Support
- Add language detection
- Multi-lingual Whisper models
- Multi-lingual TTS models

### 4.3 Custom Model Fine-Tuning
- Domain-specific Whisper fine-tuning
- LoRA adapters for LLM personalities
- Custom TTS voice training pipeline

### 4.4 Advanced Streaming
- WebRTC for lower latency
- Opus codec for better compression
- Echo cancellation

---

## Success Metrics

### MVP (Phase 1)
- ✅ Functional end-to-end pipeline
- ✅ Real audio input → AI audio output
- ✅ <1 second total latency
- ✅ Stable for 100+ interactions

### Optimized (Phase 2)
- ✅ <500ms P95 latency
- ✅ Early TTS triggering working
- ✅ TensorRT vocoder integrated
- ✅ Detailed performance metrics

### Production (Phase 3)
- ✅ 99.9% uptime
- ✅ 10+ concurrent users
- ✅ Full observability
- ✅ Documentation complete
- ✅ Kubernetes deployment successful

---

## Risk Mitigation

### Technical Risks
1. **Whisper Latency Too High**
   - Mitigation: Use smaller Whisper model (medium vs large-v3)
   - Fallback: Switch to faster STT (Vosk, Wav2Vec2)

2. **TTS Quality Poor**
   - Mitigation: Test multiple backends (Piper, FishSpeech, Coqui)
   - Fallback: Accept higher latency for better quality

3. **GPU Memory Exhaustion**
   - Mitigation: Reduce batch sizes, model quantization
   - Fallback: Model offloading, CPU fallback

4. **500ms Target Unachievable**
   - Mitigation: Accept 800ms as "good enough"
   - Fallback: Focus on perceived latency (early TTS)

### Resource Risks
1. **Model Downloads Slow**
   - Mitigation: Pre-download and cache models
   - Fallback: Use smaller models initially

2. **GPU Availability**
   - Mitigation: Cloud GPU instances (Lambda Labs, RunPod)
   - Fallback: Develop with CPU, deploy to GPU

---

## Getting Started

### Immediate Next Steps (This Week)
1. **Set up development environment**
   ```bash
   # Install CUDA 12.1 if not present
   # Install PyTorch with CUDA
   pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

   # Install project dependencies
   cd /home/user/whisprx
   pip install -r requirements.txt
   pip install -e packages/whisprx_sdk
   ```

2. **Download models**
   ```bash
   # Whisper (CTranslate2 format)
   ct2-transformers-converter --model openai/whisper-large-v3 \
       --output_dir models/whisper-large-v3-ct2 \
       --quantization float16

   # Mistral-7B (vLLM will auto-download)

   # Piper TTS
   wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.tar.gz
   tar -xzf voice-en-us-lessac-medium.tar.gz -C models/piper/
   ```

3. **Start with STT integration (Task 1.1)**
   - This is the critical path blocker
   - Once STT works, everything else can follow

---

## Questions to Resolve

1. **TTS Backend Choice**:
   - Piper (recommended for MVP, easier)
   - FishSpeech (higher quality, more complex)
   - Other (Coqui, StyleTTS2)?

2. **LLM Model**:
   - Mistral-7B (current)
   - Llama-3-8B?
   - Smaller model for lower latency?

3. **Whisper Model Size**:
   - Large-v3 (best quality, ~150ms)
   - Medium (good quality, ~80ms)
   - Small (fast, ~40ms, lower quality)

4. **Deployment Target**:
   - Single GPU (A100, H100)?
   - Multi-GPU (model parallelism)?
   - CPU fallback needed?

5. **Multi-User Priority**:
   - Single user for MVP?
   - Multi-user required from start?

---

**Ready to execute?** Let's start with Phase 1.1: Whisper STT Integration!
