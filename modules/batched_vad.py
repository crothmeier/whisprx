
#!/usr/bin/env python3
"""batched_vad.py – GPU‑batched Silero VAD"""
import numpy as np, onnxruntime as ort, logging
from typing import List, Generator
class BatchedSileroVAD:
    def __init__(self, model_path:str, threshold:float=0.5, sample_rate:int=16000,
                 frame_ms:int=32, max_batch_size:int=8):
        self.th=threshold; self.sr=sample_rate; self.spf=int(sample_rate*frame_ms/1000)
        self.max=max_batch_size
        self.session=ort.InferenceSession(model_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name
        self.h=np.zeros((2,1,64),np.float32); self.c=np.zeros((2,1,64),np.float32)
        self.buf=[]
        self.gen=None
    def reset(self):
        self.h[:]=0; self.c[:]=0; self.buf.clear(); self.gen=None
    def _run(self, batch:np.ndarray)->List[bool]:
        b=batch.shape[0]
        h=np.repeat(self.h,b,axis=1); c=np.repeat(self.c,b,axis=1)
        outs=self.session.run(None,{self.input_name:batch,'h':h,'c':c,'sr':np.array(self.sr,dtype=np.int64)})
        self.h, self.c = outs[1][:,-1:,:], outs[2][:,-1:,:]
        return (outs[0].squeeze()>self.th).tolist()
    def add_chunk(self, pcm_bytes:bytes)->Generator[bool,None,None]:
        if self.gen:
            try: yield next(self.gen); return
            except StopIteration: self.gen=None
        frame=np.frombuffer(pcm_bytes,np.int16).astype(np.float32)/32768.0
        if len(frame)!=self.spf:
            if len(frame)<self.spf: frame=np.pad(frame,(0,self.spf-len(frame)))
            else: frame=frame[:self.spf]
        self.buf.append(frame)
        if len(self.buf)>=self.max:
            batch=np.stack(self.buf,axis=0); self.buf=[]
            res=self._run(batch)
            self.gen=(r for r in res)
            yield next(self.gen)
        else:
            res=self._run(np.stack(self.buf,axis=0))
            yield res[-1]
