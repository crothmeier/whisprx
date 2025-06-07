import asyncio, torch, numpy as np

class AsyncPipeline:
    def __init__(self, vad_model):
        self.vad_model = vad_model

        # bounded queues per stage
        self.q_audio = asyncio.Queue(maxsize=10)
        self.q_stt   = asyncio.Queue(maxsize=10)
        self.q_llm   = asyncio.Queue(maxsize=10)
        self.q_tts   = asyncio.Queue(maxsize=10)

        # pinned host buffer for 2 s@16 kHz mono
        self.pcm_buf = torch.empty(32_000, dtype=torch.float32).pin_memory()

        # launch workers
        self.workers = [
            asyncio.create_task(self.audio_worker()),
            asyncio.create_task(self.stt_worker()),
            asyncio.create_task(self.llm_worker()),
            asyncio.create_task(self.tts_worker()),
        ]

    async def audio_worker(self):
        while True:
            pcm_bytes = await self.q_audio.get()
            np_pcm = (np.frombuffer(pcm_bytes, np.int16)
                        .astype(np.float32) / 32768.0)
            length = len(np_pcm)
            self.pcm_buf[:length].copy_(torch.from_numpy(np_pcm),
                                        non_blocking=True)
            # simple VAD pass
            if self.vad_model(np_pcm):
                await self.q_stt.put((self.pcm_buf[:length], length))
            self.q_audio.task_done()

    async def stt_worker(self):
        while True:
            audio_t, length = await self.q_stt.get()
            # placeholder – call Whisper here
            transcript = "dummy"
            await self.q_llm.put(transcript)
            self.q_stt.task_done()

    async def llm_worker(self):
        while True:
            text = await self.q_llm.get()
            # placeholder – call vLLM here
            response = "LLM response"
            await self.q_tts.put(response)
            self.q_llm.task_done()

    async def tts_worker(self):
        while True:
            text = await self.q_tts.get()
            # placeholder – call TTS
            self.q_tts.task_done()

    async def stop(self):
        for q in (self.q_audio, self.q_stt, self.q_llm, self.q_tts):
            await q.join()
        for w in self.workers:
            w.cancel()
