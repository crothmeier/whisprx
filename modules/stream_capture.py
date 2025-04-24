#!/usr/bin/env python3
# stream_capture.py  •  real‑time 200 ms / 50 %‑overlap audio chunks + Silero VAD
# ---------------------------------------------------------------------------
import asyncio, warnings
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple
import numpy as np, sounddevice as sd, torch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# --------------------------- Silero VAD wrapper --------------------------- #
class SileroVADWrapper:
    _URL = ('https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.jit')
    def __init__(self, sample_rate:int=16_000, threshold:float=0.5,
                 min_speech_ms:int=250, min_silence_ms:int=500, use_cuda:bool=True):
        self.sr=sample_rate; self.th=threshold
        self.min_speech=int(min_speech_ms*self.sr/1000)
        self.min_silence=int(min_silence_ms*self.sr/1000)
        self.dev=torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.in_speech=False; self.cum_samples=0; self.tmp_start=-1
    def reset(self): self.in_speech=False; self.cum_samples=0; self.tmp_start=-1
    def __call__(self, pcm_i16:np.ndarray)->Tuple[bool,bool]:
        x=torch.from_numpy(pcm_i16.astype(np.float32)/32768.0).to(self.dev).unsqueeze(0)
        with torch.no_grad(): p=float(self.model(x,self.sr))
        self.cum_samples+=len(pcm_i16); st=en=False
        if p>=self.th:
            if not self.in_speech:
                if self.tmp_start<0: self.tmp_start=self.cum_samples
                elif self.cum_samples-self.tmp_start>=self.min_speech:
                    self.in_speech,st=True,True
        else:
            if self.in_speech and self.cum_samples-self.tmp_start>=self.min_silence:
                self.in_speech,en=False,True
            elif not self.in_speech: self.tmp_start=-1
        return st,en
    def _load_model(self):
        cache=Path.home()/'.cache'/'silero_vad'; cache.mkdir(exist_ok=True)
        fp=cache/'silero_vad.jit'
        if not fp.exists():
            import urllib.request,shutil
            with urllib.request.urlopen(self._URL) as r, fp.open('wb') as f:
                shutil.copyfileobj(r,f)
        self.model=torch.jit.load(str(fp)).to(self.dev).eval()

# --------------------------- audio stream --------------------------------- #
class AudioStream(AsyncIterator[np.ndarray]):
    def __init__(self,sample_rate:int=16_000,chunk_ms:int=200,hop_ms:int=100,
                 device:Optional[int]=None,max_q:int=16,vad:Optional[SileroVADWrapper]=None):
        self.sr=sample_rate; self.chunk=int(chunk_ms*self.sr/1000); self.hop=int(hop_ms*self.sr/1000)
        if self.hop>self.chunk: raise ValueError('hop_ms must be ≤ chunk_ms')
        self.dev_idx=device; self.q=asyncio.Queue(maxsize=max_q); self.vad=vad
        self._buf=np.zeros(self.chunk,dtype=np.int16); self._fill=0; self._running=False
    def _cb(self, indata, frames, _t, status):
        if status: print('Audio status',status,flush=True)
        if not self._running: return
        pcm=indata[:,0].copy()
        if self.vad:
            st,en=self.vad(pcm)
            if st or en: self._buf.fill(0); self._fill=0
        ptr=0
        while ptr<len(pcm):
            need=self.chunk-self._fill
            take=min(len(pcm)-ptr,need)
            self._buf[self._fill:self._fill+take]=pcm[ptr:ptr+take]
            self._fill+=take; ptr+=take
            if self._fill==self.chunk:
                try: self.q.put_nowait(self._buf.copy())
                except asyncio.QueueFull: pass
                self._buf[:-self.hop]=self._buf[self.hop:]
                self._fill=self.chunk-self.hop
    # async iterator protocol
    def __aiter__(self): return self
    async def __anext__(self):
        if not self._running: raise StopAsyncIteration
        try: return await asyncio.wait_for(self.q.get(),1.0)
        except asyncio.TimeoutError: raise StopAsyncIteration
    # control
    async def start(self):
        if self._running: return
        self._running=True
        self.stream=sd.InputStream(samplerate=self.sr,channels=1,dtype=np.int16,
                                   blocksize=self.hop,callback=self._cb,device=self.dev_idx)
        self.stream.start()
    async def stop(self):
        if not self._running: return
        self._running=False; self.stream.stop(); self.stream.close()

if __name__=='__main__':
    async def demo():
        vad=SileroVADWrapper(use_cuda=torch.cuda.is_available())
        async with AudioStream(vad=vad) as mic:
            async for frame in mic: print('frame',frame.shape)
    import sys, asyncio; 
    try: asyncio.run(demo())
    except KeyboardInterrupt: sys.exit(0)
