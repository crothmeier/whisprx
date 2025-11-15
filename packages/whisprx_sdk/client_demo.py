#!/usr/bin/env python3
"""WhispRX Client Demo - Real-time speech-to-speech interaction"""
import asyncio
import sys
import signal
from typing import Optional
import msgpack
import websockets
import numpy as np

# Try to import audio libraries
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    print("⚠ sounddevice not available. Audio I/O disabled.")
    AUDIO_AVAILABLE = False


class WhispRXClient:
    """Client for WhispRX WebSocket server"""

    def __init__(
        self,
        server_url: str = "ws://localhost:8000/ws",
        sample_rate: int = 16000,
        chunk_duration_ms: int = 200,
    ):
        """Initialize client

        Args:
            server_url: WebSocket server URL
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Audio chunk duration in milliseconds
        """
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._audio_queue = asyncio.Queue()

    async def connect(self):
        """Connect to WebSocket server"""
        print(f"Connecting to {self.server_url}...")

        try:
            self.ws = await websockets.connect(self.server_url)
            print("✓ Connected to WhispRX server")

            # Wait for welcome message
            data = await self.ws.recv()
            msg = msgpack.unpackb(data, raw=False)

            if msg.get("type") == "connected":
                print(f"✓ Session ID: {msg.get('session_id')}")
                print(f"✓ {msg.get('message')}")
                return True
            else:
                print(f"⚠ Unexpected message: {msg}")
                return False

        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from server"""
        self._running = False

        if self.ws:
            try:
                # Send stop message
                await self.send_message({"type": "stop"})
                await self.ws.close()
                print("✓ Disconnected")
            except:
                pass

    async def send_message(self, msg: dict):
        """Send MessagePack message

        Args:
            msg: Message dictionary
        """
        if not self.ws:
            return

        try:
            data = msgpack.packb(msg, use_bin_type=True)
            await self.ws.send(data)
        except Exception as e:
            print(f"✗ Send error: {e}")

    async def receive_loop(self):
        """Receive and handle messages from server"""
        try:
            while self._running and self.ws:
                data = await self.ws.recv()
                msg = msgpack.unpackb(data, raw=False)

                await self.handle_message(msg)

        except websockets.exceptions.ConnectionClosed:
            print("✗ Connection closed by server")
        except Exception as e:
            print(f"✗ Receive error: {e}")
        finally:
            self._running = False

    async def handle_message(self, msg: dict):
        """Handle incoming message

        Args:
            msg: Unpacked message dictionary
        """
        msg_type = msg.get("type")

        if msg_type == "transcript":
            # STT result
            text = msg.get("text", "")
            print(f"\n[YOU]: {text}")

        elif msg_type == "response":
            # LLM response
            text = msg.get("text", "")
            print(f"[AI]:  {text}")

        elif msg_type == "audio":
            # TTS audio output
            audio_data = msg.get("data")
            if audio_data and AUDIO_AVAILABLE:
                # Play audio
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                sd.play(audio_array, samplerate=self.sample_rate)

        elif msg_type == "error":
            # Error message
            error_msg = msg.get("message", "Unknown error")
            print(f"✗ Server error: {error_msg}")

        elif msg_type == "pong":
            # Pong response
            pass

        else:
            print(f"⚠ Unknown message type: {msg_type}")

    async def audio_capture_loop(self):
        """Capture and send audio to server"""
        if not AUDIO_AVAILABLE:
            print("⚠ Audio capture disabled (sounddevice not available)")
            return

        print("\n✓ Starting audio capture...")
        print("  Speak into your microphone. Press Ctrl+C to stop.\n")

        def audio_callback(indata, frames, time_info, status):
            """Sounddevice callback"""
            if status:
                print(f"Audio status: {status}")

            if self._running:
                # Convert to int16 and put in queue
                audio_chunk = (indata[:, 0] * 32767).astype(np.int16)
                try:
                    self._audio_queue.put_nowait(audio_chunk.tobytes())
                except asyncio.QueueFull:
                    pass  # Drop frame if queue is full

        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=audio_callback,
        )

        with stream:
            try:
                while self._running:
                    # Get audio from queue and send to server
                    audio_bytes = await self._audio_queue.get()

                    await self.send_message({
                        "type": "audio",
                        "data": audio_bytes
                    })

            except Exception as e:
                print(f"✗ Audio capture error: {e}")

    async def ping_loop(self):
        """Send periodic pings to keep connection alive"""
        try:
            while self._running:
                await asyncio.sleep(30)  # Ping every 30 seconds
                await self.send_message({"type": "ping"})
        except:
            pass

    async def run(self):
        """Main client loop"""
        if not await self.connect():
            return

        self._running = True

        # Run receive, audio capture, and ping loops concurrently
        tasks = [
            asyncio.create_task(self.receive_loop()),
            asyncio.create_task(self.ping_loop()),
        ]

        if AUDIO_AVAILABLE:
            tasks.append(asyncio.create_task(self.audio_capture_loop()))
        else:
            print("\n⚠ Running in text-only mode (no audio I/O)")
            print("  The server is running but you won't hear responses.\n")

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n\n✓ Stopping...")
        finally:
            await self.disconnect()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="WhispRX Client Demo")
    parser.add_argument(
        "--server",
        default="ws://localhost:8000/ws",
        help="WebSocket server URL (default: ws://localhost:8000/ws)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=200,
        help="Audio chunk duration in ms (default: 200)"
    )

    args = parser.parse_args()

    # Create client
    client = WhispRXClient(
        server_url=args.server,
        sample_rate=args.sample_rate,
        chunk_duration_ms=args.chunk_ms,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n✓ Interrupt received, stopping...")
        client._running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Run client
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Exiting...")
        sys.exit(0)
