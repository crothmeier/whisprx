#!/usr/bin/env python3
"""WebSocket server with MessagePack binary protocol for low-latency communication"""
import asyncio
import uuid
from typing import Dict, Optional
import msgpack
from fastapi import WebSocket, WebSocketDisconnect
import torch

from .async_pipeline import AsyncPipeline
from .modules.stream_capture import SileroVADWrapper


class SessionManager:
    """Manages multiple concurrent WebSocket sessions"""

    def __init__(self):
        self.sessions: Dict[str, 'MessagePackConnection'] = {}

    def create_session(self, websocket: WebSocket) -> str:
        """Create a new session

        Args:
            websocket: FastAPI WebSocket connection

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = MessagePackConnection(session_id, websocket)
        return session_id

    def get_session(self, session_id: str) -> Optional['MessagePackConnection']:
        """Get session by ID"""
        return self.sessions.get(session_id)

    async def remove_session(self, session_id: str):
        """Remove and cleanup session"""
        session = self.sessions.pop(session_id, None)
        if session:
            await session.cleanup()


# Global session manager
session_manager = SessionManager()


class MessagePackConnection:
    """MessagePack-based WebSocket connection handler"""

    def __init__(self, session_id: str, websocket: WebSocket):
        """Initialize connection

        Args:
            session_id: Unique session identifier
            websocket: FastAPI WebSocket connection
        """
        self.session_id = session_id
        self.ws = websocket
        self.pipeline: Optional[AsyncPipeline] = None
        self.audio_output_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def handle(
        self,
        whisper_model_path: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        tts_model_path: Optional[str] = None,
    ):
        """Main connection handler

        Args:
            whisper_model_path: Path to Whisper model
            llm_model_name: LLM model name
            tts_model_path: Path to TTS model
        """
        await self.ws.accept()
        print(f"[Session {self.session_id}] WebSocket connected")

        # Send welcome message
        await self.send_message({
            "type": "connected",
            "session_id": self.session_id,
            "message": "WhispRX ready"
        })

        # Initialize pipeline
        try:
            # Create VAD model
            vad = SileroVADWrapper(
                use_cuda=torch.cuda.is_available(),
                threshold=0.5
            )

            # Create async pipeline with optional model paths
            self.pipeline = AsyncPipeline(
                vad,
                whisper_model_path=whisper_model_path,
                llm_model_name=llm_model_name,
                tts_model_path=tts_model_path,
                enable_stt=whisper_model_path is not None,
                enable_llm=llm_model_name is not None,
                enable_tts=tts_model_path is not None,
            )

            # Start pipeline
            await self.pipeline.start()
            print(f"[Session {self.session_id}] Pipeline started")

            self._running = True

            # Run receiver and sender concurrently
            await asyncio.gather(
                self._receive_loop(),
                self._send_loop(),
                return_exceptions=True
            )

        except WebSocketDisconnect:
            print(f"[Session {self.session_id}] Client disconnected")
        except Exception as e:
            print(f"[Session {self.session_id}] Error: {e}")
            await self.send_message({
                "type": "error",
                "message": str(e)
            })
        finally:
            await self.cleanup()

    async def _receive_loop(self):
        """Receive and process incoming messages"""
        try:
            while self._running:
                # Receive binary MessagePack data
                data = await self.ws.receive_bytes()

                # Unpack message
                msg = msgpack.unpackb(data, raw=False)

                # Handle message
                await self.handle_message(msg)

        except WebSocketDisconnect:
            self._running = False
        except Exception as e:
            print(f"[Session {self.session_id}] Receive error: {e}")
            self._running = False

    async def _send_loop(self):
        """Send outgoing messages from pipeline"""
        try:
            while self._running:
                # Get audio output from pipeline output queue
                if self.pipeline and hasattr(self.pipeline, 'q_audio_out'):
                    try:
                        # Wait for audio output with timeout
                        output = await asyncio.wait_for(
                            self.pipeline.q_audio_out.get(),
                            timeout=0.1
                        )

                        # Send audio to client
                        await self.send_message({
                            "type": "audio",
                            "data": output["data"],
                            "text": output.get("text", "")
                        })

                        print(f"[Session {self.session_id}] Sent {len(output['data'])} bytes of audio")

                    except asyncio.TimeoutError:
                        # No audio ready, continue
                        pass
                else:
                    await asyncio.sleep(0.1)  # Prevent tight loop

        except Exception as e:
            print(f"[Session {self.session_id}] Send error: {e}")
            self._running = False

    async def handle_message(self, msg: dict):
        """Handle incoming message

        Args:
            msg: Unpacked MessagePack message
        """
        msg_type = msg.get("type")

        if msg_type == "audio":
            # Received audio chunk from client
            audio_data = msg.get("data")
            if audio_data and self.pipeline:
                # Push to pipeline
                await self.pipeline.q_audio.put(audio_data)

        elif msg_type == "ping":
            # Respond to ping
            await self.send_message({"type": "pong"})

        elif msg_type == "stop":
            # Stop processing
            self._running = False

        else:
            print(f"[Session {self.session_id}] Unknown message type: {msg_type}")

    async def send_message(self, msg: dict):
        """Send MessagePack message to client

        Args:
            msg: Message dictionary
        """
        try:
            # Pack with MessagePack
            data = msgpack.packb(msg, use_bin_type=True)

            # Send binary data
            await self.ws.send_bytes(data)

        except Exception as e:
            print(f"[Session {self.session_id}] Send error: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        self._running = False

        if self.pipeline:
            await self.pipeline.stop()
            print(f"[Session {self.session_id}] Pipeline stopped")

        try:
            await self.ws.close()
        except:
            pass

        print(f"[Session {self.session_id}] Cleaned up")


# FastAPI integration helper
async def websocket_endpoint(
    websocket: WebSocket,
    whisper_model_path: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    tts_model_path: Optional[str] = None,
):
    """FastAPI WebSocket endpoint handler

    Args:
        websocket: FastAPI WebSocket connection
        whisper_model_path: Optional path to Whisper model
        llm_model_name: Optional LLM model name
        tts_model_path: Optional path to TTS model
    """
    session_id = session_manager.create_session(websocket)

    try:
        session = session_manager.get_session(session_id)
        if session:
            await session.handle(
                whisper_model_path=whisper_model_path,
                llm_model_name=llm_model_name,
                tts_model_path=tts_model_path,
            )
    finally:
        await session_manager.remove_session(session_id)
