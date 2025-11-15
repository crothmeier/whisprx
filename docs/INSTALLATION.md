# WhispRX Installation Guide

Complete setup instructions for WhispRX - a low-latency conversational AI system.

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA 12.1+ support
  - Minimum: 8GB VRAM (RTX 3070 or better)
  - Recommended: 16GB+ VRAM (RTX 4080, A5000, A100)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space for models

### Software
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Python**: 3.11 or 3.12
- **CUDA**: 12.1 or newer
- **cuDNN**: 8.9+ (usually bundled with CUDA)

---

## Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/whisprx.git
cd whisprx

# 2. Create environment and install dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Download models
python scripts/download_models.py

# 4. Start server
cd services/whisprx_api
python main.py

# 5. In another terminal, run client
python packages/whisprx_sdk/client_demo.py
```

---

## Detailed Installation

### Step 1: Install System Dependencies

#### Ubuntu/Debian
```bash
# Update package lists
sudo apt-get update

# Install Python and build tools
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential

# Install audio libraries
sudo apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    espeak-ng

# Optional: Install Piper TTS (recommended for better quality)
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
sudo mv piper/piper /usr/local/bin/
```

#### Verify CUDA Installation
```bash
nvidia-smi
nvcc --version  # Should show CUDA 12.1+
```

### Step 2: Install Python Dependencies

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e packages/whisprx_sdk
```

### Step 3: Download Models

```bash
# Default setup (base Whisper, Phi-3 LLM, Piper TTS)
python scripts/download_models.py

# Custom configuration
python scripts/download_models.py \
    --whisper-size small \
    --llm-model mistralai/Mistral-7B-Instruct-v0.2 \
    --tts-voice en_US-amy-medium
```

### Step 4: Start Server

```bash
cd services/whisprx_api
python main.py
# Server starts on http://localhost:8000
```

### Step 5: Test Client

```bash
python packages/whisprx_sdk/client_demo.py
# Speak into your microphone!
```

---

## Model Selection Guide

**For Low Latency** (<500ms target):
- Whisper: `tiny` or `base`
- LLM: `microsoft/Phi-3-mini-4k-instruct`
- TTS: Any Piper voice

**For Best Quality** (latency not critical):
- Whisper: `large-v3`
- LLM: `mistralai/Mistral-7B-Instruct-v0.2`
- TTS: `en_US-lessac-high` (if available)

---

## Troubleshooting

### CUDA Out of Memory
- Use smaller models (`tiny` Whisper, `Phi-3` LLM)
- Reduce `gpu_memory_utilization` in `llm_service.py`

### No Audio Output
- Install Piper or espeak-ng
- Check audio device: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### Models Not Downloading
- Check internet connection
- Download manually from HuggingFace
- Update paths in `.env` file

---

## Next Steps

- Read [USAGE.md](USAGE.md) for API documentation
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [IMPLEMENTATION_ROADMAP.md](../IMPLEMENTATION_ROADMAP.md) for development plans
