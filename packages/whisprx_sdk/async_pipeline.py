# Production-grade lifecycle: explicit start()/stop()
import asyncio
from typing import List
import torch
import numpy as np


class AsyncPipeline:
    """
    Asynchronous audio → STT → LLM → TTS pipeline.

    Life-cycle:
        pipeline = AsyncPipeline(vad)
        await pipeline.start()
        …
        await pipeline.stop()
    """

    def __init__(self, vad_model, *, loop: asyncio.AbstractEventLoop | None = None):
        self.vad_model = vad_model

        # bounded queues per stage
        self.q_audio: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.q_stt:   asyncio.Queue = asyncio.Queue(maxsize=10)
        self.q_llm:   asyncio.Queue = asyncio.Queue(maxsize=10)
        self.q_tts:   asyncio.Queue = asyncio.Queue(maxsize=10)

        # pinned host buffer for 2 s@16 kHz mono
        self.pcm_buf = torch.empty(32_000, dtype=torch.float32).pin_memory()

        self._loop: asyncio.AbstractEventLoop = loop or asyncio.get_event_loop()
        self._tasks: List[asyncio.Task] = []
        self.workers = []  # Keep compatibility with existing tests

    async def start(self) -> None:
        """Launch background workers."""
        if self._tasks:  # already started
            return
        self._tasks = [
            self._loop.create_task(self.audio_worker()),
            self._loop.create_task(self.stt_worker()),
            self._loop.create_task(self.llm_worker()),
            self._loop.create_task(self.tts_worker()),
        ]
        self.workers = self._tasks  # Keep compatibility

    async def audio_worker(self):
        """Continuously read q_audio, VAD-filter, push speech chunks to q_stt."""
        try:
            while True:
                pcm_bytes = await self.q_audio.get()
                try:
                    np_pcm = (np.frombuffer(pcm_bytes, np.int16)
                                .astype(np.float32) / 32768.0)
                    length = len(np_pcm)
                    self.pcm_buf[:length].copy_(torch.from_numpy(np_pcm),
                                                non_blocking=True)
                    # simple VAD pass
                    vad_result = self.vad_model(np_pcm)
                    # Debug print
                    # print(f"VAD result: {vad_result}")
                    if vad_result:
                        await self.q_stt.put((self.pcm_buf[:length], length))
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error in audio_worker: {e}")
                finally:
                    self.q_audio.task_done()
        except asyncio.CancelledError:
            # graceful shutdown
            return

    async def stt_worker(self):
        try:
            while True:
                audio_t, length = await self.q_stt.get()
                # placeholder – call Whisper here
                transcript = "dummy"
                await self.q_llm.put(transcript)
                self.q_stt.task_done()
        except asyncio.CancelledError:
            return

    async def llm_worker(self):
        try:
            while True:
                text = await self.q_llm.get()
                # placeholder – call vLLM here
                response = "LLM response"
                await self.q_tts.put(response)
                self.q_llm.task_done()
        except asyncio.CancelledError:
            return

    async def tts_worker(self):
        try:
            while True:
                text = await self.q_tts.get()
                # placeholder – call TTS
                self.q_tts.task_done()
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        """Cancel background workers and wait for them to finish."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.workers.clear()