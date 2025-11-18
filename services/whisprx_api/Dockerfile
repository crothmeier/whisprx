# WhispRX Production Dockerfile with CUDA Support
# Multi-stage build for optimized image size

# Stage 1: Base image with CUDA runtime
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Stage 2: Build dependencies
FROM base AS builder

WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --user -r requirements.txt

# Stage 3: Runtime image
FROM base AS runtime

# Create non-root user
RUN useradd -m -u 1000 whisprx && \
    mkdir -p /app /models && \
    chown -R whisprx:whisprx /app /models

# Copy installed packages from builder
COPY --from=builder /root/.local /home/whisprx/.local

# Set PATH
ENV PATH=/home/whisprx/.local/bin:$PATH
ENV PYTHONPATH=/app/packages:$PYTHONPATH

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=whisprx:whisprx packages/ /app/packages/
COPY --chown=whisprx:whisprx services/ /app/services/
COPY --chown=whisprx:whisprx scripts/ /app/scripts/
COPY --chown=whisprx:whisprx performance_presets.yaml /app/

# Switch to non-root user
USER whisprx

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Run application
WORKDIR /app/services/whisprx_api
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
