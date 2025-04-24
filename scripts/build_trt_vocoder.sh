#!/bin/bash
# build_trt_vocoder.sh - Convert and benchmark TTS vocoder with TensorRT
set -e
echo "Installing TensorRT CLI tools..."
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install --upgrade pip
pip3 install nvidia-pyindex
pip3 install nvidia-tensorrt

if [ ! -f "vocoder.onnx" ]; then
  echo "Error: vocoder.onnx not found in current directory"
  exit 1
fi

echo "Converting vocoder.onnx to vocoder_fp16.trt ..."
trtexec --onnx=vocoder.onnx --saveEngine=vocoder_fp16.trt \
        --fp16 --workspace=4096 --verbose \
        --tacticSources=+CUDNN,+CUBLAS

echo "Benchmarking TensorRT engine..."
trtexec --loadEngine=vocoder_fp16.trt --warmUp=50 --duration=60 \
        --avgRuns=100 --verbose --useCudaGraph

echo "Done. Engine stored at vocoder_fp16.trt"
