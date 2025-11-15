from fastapi import FastAPI, WebSocket
import asyncio
import os
from typing import Optional

app = FastAPI(
    title="WhispRX API",
    version="0.1.0",
    description="Low-latency conversational AI with Whisper, vLLM, and TTS"
)

# Model paths from environment variables
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
TTS_MODEL_PATH = os.getenv("TTS_MODEL_PATH")


@app.get("/")
async def root():
    """API info endpoint"""
    return {
        "name": "WhispRX API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health"
        },
        "models": {
            "whisper": WHISPER_MODEL_PATH or "not configured",
            "llm": LLM_MODEL_NAME or "not configured",
            "tts": TTS_MODEL_PATH or "not configured",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time speech-to-speech interaction

    Protocol: MessagePack binary messages

    Client -> Server:
        {"type": "audio", "data": <bytes>}  # Send audio chunk
        {"type": "ping"}                     # Ping server

    Server -> Client:
        {"type": "connected", "session_id": "<id>"}  # Initial connection
        {"type": "transcript", "text": "<str>"}      # STT result
        {"type": "response", "text": "<str>"}        # LLM response
        {"type": "audio", "data": <bytes>}           # TTS audio
        {"type": "pong"}                             # Ping response
        {"type": "error", "message": "<str>"}        # Error message
    """
    from whisprx_sdk.ws_server import websocket_endpoint as handle_websocket

    await handle_websocket(
        websocket,
        whisper_model_path=WHISPER_MODEL_PATH,
        llm_model_name=LLM_MODEL_NAME,
        tts_model_path=TTS_MODEL_PATH,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
