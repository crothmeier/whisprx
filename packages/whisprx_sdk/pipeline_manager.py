import asyncio

class PipelineManager:
    def __init__(self, llm_client, tts_client):
        self.llm_client = llm_client
        self.tts_client = tts_client

    async def process_llm_response(self, sess_id: str) -> None:
        """Stream LLM tokens; launch TTS after first ~4 tokens."""
        buf = []
        tts_started = False

        async for tok in self.llm_client.stream_response(sess_id):
            buf.append(tok)

            # kick TTS once we have a pronounceable chunk
            if len(buf) == 4 and not tts_started:
                first = "".join(buf)
                asyncio.create_task(
                    self.tts_client.start_synth(sess_id, first)
                )
                tts_started = True

            # feed incremental 8‑token deltas
            elif tts_started and len(buf) % 8 == 0:
                delta = "".join(buf[-8:])
                await self.tts_client.continue_synth(sess_id, delta)

        # flush tail if mis‑aligned
        if tts_started and len(buf) % 8:
            await self.tts_client.continue_synth(
                sess_id,
                "".join(buf[-(len(buf) % 8):])
            )

        # safety: wait max 100 ms for vocoder to finish
        try:
            await asyncio.wait_for(
                self.tts_client.wait_done(sess_id), 0.10
            )
        except asyncio.TimeoutError:
            print("TTS timeout – continuing")
