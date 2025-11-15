# WhispRX

**Low-latency, fully local conversational AI system targeting 500ms end-to-end speech interaction**

Powered by:
- **Whisper** (Speech-to-Text via faster-whisper)
- **vLLM** (Large Language Model inference)
- **Piper/FishSpeech** (Text-to-Speech)

Optimized for GPU acceleration with CUDA, async streaming architecture, and MessagePack binary protocol.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models
python scripts/download_models.py

# 3. Start server
cd services/whisprx_api && python main.py

# 4. Run client (in another terminal)
python packages/whisprx_sdk/client_demo.py
```

**Speak into your microphone** → AI responds with voice!

---

## Features

✅ **Real-time Speech-to-Speech**: Talk naturally with AI
✅ **Fully Local**: No cloud APIs, complete privacy
✅ **GPU Accelerated**: CUDA optimization throughout
✅ **Async Pipeline**: Non-blocking 4-stage processing
✅ **Low Latency**: Targeting <500ms end-to-end
✅ **Production Ready**: Comprehensive error handling and testing

---

## Architecture

```
┌──────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Audio   │───▶│   STT   │───▶│   LLM   │───▶│   TTS   │───▶ Audio Out
│ Capture  │    │ Whisper │    │  vLLM   │    │  Piper  │
└──────────┘    └─────────┘    └─────────┘    └─────────┘
     │               │              │              │
   VAD Filter    <200ms latency  <150ms       <150ms
```

**Total Pipeline**: Audio → Transcript → Response → Speech (~500ms)

---

## Requirements

- **GPU**: NVIDIA with CUDA 12.1+ (8GB+ VRAM recommended)
- **RAM**: 16GB+ system memory
- **OS**: Linux (Ubuntu 22.04+)
- **Python**: 3.11+

---

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions
- **[Usage Guide](docs/USAGE.md)** - API documentation and examples
- **[Optimizations Guide](docs/OPTIMIZATIONS.md)** - Performance tuning and TensorRT setup
- **[Codebase Analysis](CODEBASE_ANALYSIS.md)** - Deep dive into implementation
- **[Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)** - Development plans

---

## Project Status

**Current**: ✅ **Phase 2 Optimizations Complete!**
- ✅ Phase 1: MVP with full AI integration
- ✅ Phase 2: Performance optimizations (~40-50% latency reduction)
  - Audio output streaming for real-time playback
  - TensorRT vocoder optimization (3x TTS speedup)
  - CUDA graph precompilation
  - Comprehensive benchmarking and profiling suite

**Performance**: Approaching 500ms target with balanced configuration!

**Next**: Phase 3 - Advanced optimizations and production hardening

See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) and [OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md) for details.

---

## License

[Add your license here]

---

## Contributing

Contributions welcome! See issues and roadmap for areas needing work.

---

## Acknowledgments

Built with:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper inference
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM serving
- [Piper](https://github.com/rhasspy/piper) - Neural TTS
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
