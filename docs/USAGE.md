# WhispRX Usage Guide

How to use the WhispRX conversational AI system.

---

## Starting the Server

```bash
cd services/whisprx_api
python main.py
```

Server runs on `http://localhost:8000` with WebSocket endpoint at `ws://localhost:8000/ws`.

---

## Using the Client Demo

```bash
# Basic usage
python packages/whisprx_sdk/client_demo.py

# Custom server
python packages/whisprx_sdk/client_demo.py --server ws://192.168.1.100:8000/ws

# Different sample rate
python packages/whisprx_sdk/client_demo.py --sample-rate 22050
```

Speak into your microphone. The AI will respond with both text and audio.

---

## WebSocket API

### Connection

Connect to `ws://localhost:8000/ws` using MessagePack binary protocol.

### Client → Server Messages

**Send Audio**:
```python
import msgpack

# Send audio chunk (raw PCM int16)
msg = msgpack.packb({
    "type": "audio",
    "data": audio_bytes  # 16-bit PCM data
}, use_bin_type=True)

await websocket.send(msg)
```

**Ping**:
```python
msg = msgpack.packb({"type": "ping"}, use_bin_type=True)
await websocket.send(msg)
```

**Stop**:
```python
msg = msgpack.packb({"type": "stop"}, use_bin_type=True)
await websocket.send(msg)
```

### Server → Client Messages

**Connection Established**:
```python
{"type": "connected", "session_id": "uuid", "message": "WhispRX ready"}
```

**Transcript (STT Result)**:
```python
{"type": "transcript", "text": "user speech text"}
```

**LLM Response**:
```python
{"type": "response", "text": "AI response text"}
```

**Audio Output**:
```python
{"type": "audio", "data": bytes}  # TTS audio (16-bit PCM)
```

**Error**:
```python
{"type": "error", "message": "error description"}
```

---

## Python API

### Direct Pipeline Usage

```python
import asyncio
from whisprx_sdk.async_pipeline import AsyncPipeline
from whisprx_sdk.modules.stream_capture import SileroVADWrapper

async def main():
    # Create VAD
    vad = SileroVADWrapper(use_cuda=True)

    # Create pipeline
    pipeline = AsyncPipeline(
        vad,
        whisper_model_path="base",
        llm_model_name="microsoft/Phi-3-mini-4k-instruct",
        tts_model_path="models/tts/piper/model.onnx",
    )

    # Start pipeline
    await pipeline.start()

    # Send audio
    await pipeline.q_audio.put(audio_bytes)

    # ... processing happens automatically ...

    # Stop pipeline
    await pipeline.stop()

asyncio.run(main())
```

### Individual Components

**Speech-to-Text**:
```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda")
segments, info = model.transcribe("audio.wav")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**LLM**:
```python
from whisprx_sdk.llm_service import LLMService

llm = LLMService(model_name="microsoft/Phi-3-mini-4k-instruct")
response = llm.generate("Hello, how are you?")
print(response)
```

**Text-to-Speech**:
```python
from whisprx_sdk.tts_service import TTSService

tts = TTSService(model_path="models/tts/piper/model.onnx")
audio_bytes = tts.synthesize("Hello, world!")

# Save to file
import wave
with wave.open("output.wav", "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(audio_bytes)
```

---

## Configuration

### Environment Variables

Set in `.env` file or export:

```bash
# Model paths
export WHISPER_MODEL_PATH=base
export LLM_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
export TTS_MODEL_PATH=models/tts/piper/model.onnx

# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Server settings
export HOST=0.0.0.0
export PORT=8000
```

### Performance Tuning

Edit `performance_presets.yaml`:

```yaml
presets:
  ultra_low_latency:
    buffer_size_ms: 80
    chunk_size: 1024
    overlap: 128

  balanced:
    buffer_size_ms: 100
    chunk_size: 2048
    overlap: 256

acceleration:
  use_torch_tensorrt: false
```

---

## Examples

### Simple Echo Bot

```python
import asyncio
from whisprx_sdk.ws_server import MessagePackConnection
from fastapi import WebSocket, FastAPI

app = FastAPI()

@app.websocket("/echo")
async def echo(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_bytes()
        msg = msgpack.unpackb(data)

        if msg["type"] == "audio":
            # Echo back the audio
            response = msgpack.packb({
                "type": "audio",
                "data": msg["data"]
            }, use_bin_type=True)
            await websocket.send_bytes(response)
```

### Custom STT Processing

```python
from whisprx_sdk.async_pipeline import AsyncPipeline

class CustomPipeline(AsyncPipeline):
    async def stt_worker(self):
        while True:
            audio_t, length = await self.q_stt.get()

            # Custom processing
            transcript = await my_custom_stt(audio_t)

            # Post-process
            transcript = transcript.strip().lower()

            await self.q_llm.put(transcript)
            self.q_stt.task_done()
```

---

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "healthy"}
```

### API Info

```bash
curl http://localhost:8000/
```

Response shows model configuration and endpoints.

---

## Best Practices

1. **Audio Quality**: Use 16kHz mono PCM for best results
2. **Chunk Size**: 200ms chunks with 100ms overlap recommended
3. **GPU Memory**: Monitor with `nvidia-smi` during operation
4. **Latency**: Test different model sizes to balance quality vs speed
5. **Error Handling**: Always handle WebSocket disconnections gracefully

---

## Troubleshooting

**No response from AI**:
- Check server logs for errors
- Verify models are loaded (check startup messages)
- Test each component individually

**High Latency**:
- Use smaller models
- Reduce GPU memory utilization
- Check for CPU/GPU throttling

**Poor Audio Quality**:
- Increase microphone volume
- Reduce background noise
- Use better TTS model (Piper over espeak)

---

## Next Steps

- See [INSTALLATION.md](INSTALLATION.md) for setup
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [IMPLEMENTATION_ROADMAP.md](../IMPLEMENTATION_ROADMAP.md) for planned features
