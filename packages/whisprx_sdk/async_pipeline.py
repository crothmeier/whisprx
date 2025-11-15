# Production-grade lifecycle: explicit start()/stop()
import asyncio
from typing import List, Optional
import torch
import numpy as np
from pathlib import Path


class AsyncPipeline:
    """
    Asynchronous audio → STT → LLM → TTS pipeline.

    Life-cycle:
        pipeline = AsyncPipeline(vad)
        await pipeline.start()
        …
        await pipeline.stop()
    """

    def __init__(
        self,
        vad_model,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        whisper_model_path: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        tts_model_path: Optional[str] = None,
        enable_stt: bool = True,
        enable_llm: bool = True,
        enable_tts: bool = True,
    ):
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

        # Model configuration
        self.enable_stt = enable_stt
        self.enable_llm = enable_llm
        self.enable_tts = enable_tts

        # Initialize models
        self.whisper_model = None
        self.llm_service = None
        self.tts_service = None

        # Load STT model (Whisper via faster-whisper)
        if enable_stt and whisper_model_path:
            try:
                from faster_whisper import WhisperModel
                self.whisper_model = WhisperModel(
                    whisper_model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    compute_type="float16" if torch.cuda.is_available() else "int8",
                )
                print(f"✓ Loaded Whisper model: {whisper_model_path}")
            except Exception as e:
                print(f"⚠ Failed to load Whisper model: {e}")
                self.enable_stt = False

        # Load LLM service
        if enable_llm and llm_model_name:
            try:
                from .llm_service import LLMService
                self.llm_service = LLMService(model_name=llm_model_name)
                print(f"✓ Loaded LLM service: {llm_model_name}")
            except Exception as e:
                print(f"⚠ Failed to load LLM service: {e}")
                self.enable_llm = False

        # Load TTS service
        if enable_tts and tts_model_path:
            try:
                from .tts_service import TTSService
                self.tts_service = TTSService(model_path=tts_model_path)
                print(f"✓ Loaded TTS service: {tts_model_path}")
            except Exception as e:
                print(f"⚠ Failed to load TTS service: {e}")
                self.enable_tts = False

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
        """STT worker: converts audio to text using Whisper"""
        try:
            while True:
                audio_t, length = await self.q_stt.get()
                try:
                    if self.enable_stt and self.whisper_model:
                        # Convert tensor to numpy array
                        audio_np = audio_t.cpu().numpy() if isinstance(audio_t, torch.Tensor) else audio_t

                        # Run Whisper inference (blocking, so use executor)
                        loop = asyncio.get_event_loop()
                        segments, info = await loop.run_in_executor(
                            None,
                            lambda: self.whisper_model.transcribe(
                                audio_np,
                                language="en",
                                vad_filter=False,  # We already did VAD
                                beam_size=1,  # Faster, slightly lower quality
                            )
                        )

                        # Collect all segments
                        transcript = " ".join(segment.text for segment in segments).strip()

                        if transcript:
                            print(f"[STT] {transcript}")
                            await self.q_llm.put(transcript)
                        else:
                            print("[STT] No speech detected")
                    else:
                        # Fallback to dummy for testing
                        transcript = "Hello, this is a test"
                        await self.q_llm.put(transcript)
                except Exception as e:
                    print(f"Error in stt_worker: {e}")
                finally:
                    self.q_stt.task_done()
        except asyncio.CancelledError:
            return

    async def llm_worker(self):
        """LLM worker: generates conversational responses"""
        try:
            while True:
                text = await self.q_llm.get()
                try:
                    if self.enable_llm and self.llm_service:
                        # Generate streaming response
                        full_response = ""
                        async for token in self.llm_service.stream_generate(text):
                            full_response += token

                        if full_response:
                            print(f"[LLM] {full_response}")
                            await self.q_tts.put(full_response)
                        else:
                            print("[LLM] No response generated")
                    else:
                        # Fallback to simple response for testing
                        response = f"I heard you say: {text}"
                        print(f"[LLM] {response}")
                        await self.q_tts.put(response)
                except Exception as e:
                    print(f"Error in llm_worker: {e}")
                    # Send error message to TTS for feedback
                    await self.q_tts.put("Sorry, I encountered an error processing that.")
                finally:
                    self.q_llm.task_done()
        except asyncio.CancelledError:
            return

    async def tts_worker(self):
        """TTS worker: converts text to speech audio"""
        try:
            while True:
                text = await self.q_tts.get()
                try:
                    if self.enable_tts and self.tts_service:
                        # Generate audio (blocking, so use executor)
                        loop = asyncio.get_event_loop()
                        audio_bytes = await loop.run_in_executor(
                            None,
                            self.tts_service.synthesize,
                            text
                        )

                        if audio_bytes:
                            # For now, just print that we generated audio
                            # In the WebSocket server, this will be sent to the client
                            print(f"[TTS] Generated {len(audio_bytes)} bytes of audio")
                            # TODO: Send to WebSocket client or audio output queue
                        else:
                            print("[TTS] No audio generated")
                    else:
                        # Fallback - just acknowledge
                        print(f"[TTS] Would synthesize: {text[:50]}...")
                except Exception as e:
                    print(f"Error in tts_worker: {e}")
                finally:
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