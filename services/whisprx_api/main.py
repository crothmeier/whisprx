from fastapi import FastAPI, WebSocket, Response
from fastapi.responses import PlainTextResponse
import asyncio
import os
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhispRX API",
    version="0.2.0",
    description="Low-latency conversational AI with Whisper, vLLM, and TTS"
)

# Model paths from environment variables
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
TTS_MODEL_PATH = os.getenv("TTS_MODEL_PATH")

# Import resilience and monitoring
health_check = None
metrics = None

try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../packages'))
    from whisprx_sdk.resilience import get_health_check
    from whisprx_sdk.monitoring import get_metrics

    health_check = get_health_check()
    metrics = get_metrics()

    # Register health checks
    async def check_models():
        """Check if models are configured"""
        return bool(WHISPER_MODEL_PATH or LLM_MODEL_NAME or TTS_MODEL_PATH)

    health_check.register("models_configured", check_models)

    logger.info("✓ Health checks and metrics initialized")
except Exception as e:
    logger.warning(f"⚠ Could not initialize health checks/metrics: {e}")


@app.on_event("startup")
async def startup():
    """Startup event handler"""
    logger.info("🚀 WhispRX API starting up...")
    logger.info(f"  Whisper model: {WHISPER_MODEL_PATH or 'not configured'}")
    logger.info(f"  LLM model: {LLM_MODEL_NAME or 'not configured'}")
    logger.info(f"  TTS model: {TTS_MODEL_PATH or 'not configured'}")


@app.on_event("shutdown")
async def shutdown():
    """Shutdown event handler"""
    logger.info("👋 WhispRX API shutting down...")


@app.get("/")
async def root():
    """API info endpoint"""
    return {
        "name": "WhispRX API",
        "version": "0.2.0",
        "phase": "3",
        "status": "running",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "health_detailed": "/health/detailed",
            "metrics": "/metrics",
            "status": "/status"
        },
        "models": {
            "whisper": WHISPER_MODEL_PATH or "not configured",
            "llm": LLM_MODEL_NAME or "not configured",
            "tts": TTS_MODEL_PATH or "not configured",
        },
        "features": {
            "health_checks": health_check is not None,
            "metrics": metrics is not None,
        }
    }


@app.get("/health")
async def health_simple():
    """Simple health check endpoint (Kubernetes liveness probe)"""
    return {"status": "healthy"}


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check with component status (Kubernetes readiness probe)"""
    if health_check:
        try:
            results = await health_check.run_checks()
            status_code = 200 if results["status"] == "healthy" else 503
            return Response(
                content=str(results),
                status_code=status_code,
                media_type="application/json"
            )
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return Response(
                content=str({"status": "error", "error": str(e)}),
                status_code=503,
                media_type="application/json"
            )
    else:
        return {"status": "healthy", "checks": {}, "note": "Health checks not available"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    if metrics:
        try:
            return metrics.get_metrics_text()
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return f"# Error generating metrics: {e}\n"
    else:
        return "# Metrics not available\n# Install prometheus-client: pip install prometheus-client\n"


@app.get("/status")
async def status():
    """Detailed status information"""
    status_info = {
        "api_version": "0.2.0",
        "phase": "3",
        "uptime_seconds": 0,  # TODO: Track uptime
        "models": {
            "whisper": {
                "path": WHISPER_MODEL_PATH,
                "configured": bool(WHISPER_MODEL_PATH),
            },
            "llm": {
                "name": LLM_MODEL_NAME,
                "configured": bool(LLM_MODEL_NAME),
            },
            "tts": {
                "path": TTS_MODEL_PATH,
                "configured": bool(TTS_MODEL_PATH),
            },
        },
        "features": {
            "health_checks": health_check is not None,
            "metrics": metrics is not None,
        }
    }

    # Add custom metrics stats if available
    if metrics:
        try:
            custom_stats = metrics.get_custom_stats()
            if custom_stats:
                status_info["performance_stats"] = custom_stats
        except:
            pass

    return status_info


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
    if metrics:
        metrics.inc_websocket_connections()

    try:
        from whisprx_sdk.ws_server import websocket_endpoint as handle_websocket

        await handle_websocket(
            websocket,
            whisper_model_path=WHISPER_MODEL_PATH,
            llm_model_name=LLM_MODEL_NAME,
            tts_model_path=TTS_MODEL_PATH,
        )
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if metrics:
            metrics.record_error("websocket_error", "ws_endpoint")
        raise
    finally:
        if metrics:
            metrics.dec_websocket_connections()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
