# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WhispRX is a low-latency conversational AI system that combines:
- Whisper (via CTranslate2) for Speech-to-Text
- vLLM for Large Language Model inference  
- FishSpeech/Piper for Text-to-Speech
- Target: 500ms end-to-end speech interaction latency

## Architecture

The monorepo contains three main packages under `/packages/`:

1. **whisperx2_legacy**: Core implementation with async pipeline, LLM/TTS services, WebSocket server
2. **whisprx_core**: CLI interface and CUDA optimization utilities
3. **whisprx_sdk**: SDK for building WhispRX applications

Key optimizations:
- TensorRT for vocoder acceleration (see patches/tts_service_trt.diff)
- MessagePack for WebSocket communication (see patches/ws_server_msgpack.diff)
- CUDA graph precompilation for reduced latency

## Development Commands

### Running the API Server
```bash
# Development mode
cd services/whisprx_api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode (from project root)
docker build -f services/whisprx_api/Dockerfile -t whisprx_api .
docker run -p 8000:8000 whisprx_api
```

### Installing Dependencies
```bash
# Requires Python 3.11+
pip install -r requirements.txt

# For development (install packages in editable mode)
pip install -e .
```

### Building TensorRT Vocoder
```bash
cd packages/whisperx2_legacy
./scripts/build_trt_vocoder.sh
```

### Kubernetes Deployment
```bash
helm install whisprx charts/whisprx/
```

## Key Technical Details

- WebSocket endpoint: `/ws` with MessagePack binary serialization
- Performance presets in `performance_presets.yaml` control latency/quality tradeoffs
- Requires CUDA 12.1+ with PyTorch 2.2.1
- Multi-GPU support via vLLM for LLM inference
- Voice Activity Detection (VAD) with ONNX Runtime GPU

## Testing

No test framework is currently configured. When implementing tests:
1. Check if pytest or unittest setup exists before adding
2. Follow existing patterns if tests are added later

## Important Notes

- The project uses patch files to modify dependencies for performance
- Documentation files (ARCHITECTURE.md, INSTALLATION.md, USAGE.md) are placeholders
- Main API implementation is minimal - actual pipeline logic is in the legacy package
- Requires specific PyTorch version with CUDA 12.1 support from PyTorch wheels