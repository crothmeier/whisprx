# Real‑Time Speech Pipeline (patched set)

## 1· Unpack

```bash
mkdir -p /data/nvme/rt_pipeline
cd /data/nvme/rt_pipeline
unzip ~/rt_pipeline_patched.zip
```

## 2· Python env

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3· Model weights (run on each node)

```bash
# STT – small‑en CT2 int8
ct2-transformers-converter --model openai/whisper-small     --output_dir whisper-small-ct2 --quantization int8

# LLM (A4000 node)
python -m vllm.entrypoints.openai.api_server --model mistral-7b
# or download weights ahead of time
```

## 4· Run WS pipeline (T4 node)

```bash
python rt_pipeline.py --ws --stt-device cuda:1 --tts-device cuda:1
```

## 5· Benchmark from laptop / controller

```bash
python benchmark.py --host 192.168.66.2 --runs 30 --realtime
```

### Storage layout (Gen8)

* OS: `/` on `sda2` (SAS, 1.1 TB)
* Fast scratch + models: `/data/nvme` on `nvme0n1p1` (931 GB ZFS)

Mount already active (per `lsblk`).  
Place model dirs and virtual‑env in `/data/nvme` for SSD speed.

### Graceful stop

Inside WS process:

```python
await pipeline.stop()
```

### Next steps

* Integrate real Whisper / TTS calls in `async_pipeline.py`
* Run benchmark, upload `results.csv`
* Proceed with MessagePack and 200 ms capture optimisation
