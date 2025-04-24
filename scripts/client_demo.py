# scripts/client_demo.py
import asyncio, argparse, msgpack, numpy as np, sounddevice as sd, websockets
from modules.stream_capture import AudioStream          # already in the repo

async def main(uri):
    # open WebSocket
    async with websockets.connect(uri, max_size=2**20) as ws:
        stream = AudioStream(sample_rate=16000)
        await stream.start()

        # playback stream
        async def playback():
            while True:
                pcm_bytes = await ws.recv()
                audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)/32768
                sd.play(audio, 16000, blocking=False)

        # start playback task
        asyncio.create_task(playback())

        print("🎙️  Speak into the mic – Ctrl-C to quit")
        async for chunk in stream:
            packed = msgpack.packb(chunk.tobytes())
            await ws.send(packed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True,
                        help="ws://<phx-ai02-ip>:8765")
    args = parser.parse_args()
    asyncio.run(main(args.server))
