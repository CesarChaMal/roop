#!/bin/bash
set -e
export PYTHONUTF8=1

# ✅ Isolate from Anaconda interference
unset PYTHONPATH
unset CONDA_PREFIX
unset LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# ✅ Remove conflicting CUDA packages
#sudo apt remove --purge -y nvidia-cuda-toolkit || true

# ✅ Setup CUDA 11.8 explicitly
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 🔧 Auto install CUDA 11.8 if missing
if [[ ! -d "/usr/local/cuda-11.8" ]]; then
    echo "[INFO] CUDA 11.8 not found. Installing..."
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O cuda_11.8.run
    chmod +x cuda_11.8.run
    sudo ./cuda_11.8.run --toolkit --silent --override
    rm -f cuda_11.8.run
fi

# ✅ Install cuDNN v8.9.7.29 for CUDA 11.8 (.tar.xz method)
if [[ ! -f "/usr/local/cuda-11.8/lib64/libcudnn.so.8" ]]; then
    echo "[INFO] cuDNN not found. Installing cuDNN v8.9.7.29 for CUDA 11.8..."
    CUDNN_TAR="cudnn-linux-x86_64-8.9.5.29_cuda11-archive.tar.xz"
    CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.5.29_cuda11-archive.tar.xz"
    wget "$CUDNN_URL" -O "$CUDNN_TAR"
    tar -xf "$CUDNN_TAR"
    CUDNN_DIR=$(find . -type d -name "cudnn-linux-x86_64*" | head -n 1)
    sudo cp -P "$CUDNN_DIR/include/"* /usr/local/cuda-11.8/include/
    sudo cp -P "$CUDNN_DIR/lib/"* /usr/local/cuda-11.8/lib64/
    sudo ldconfig
    rm -rf "$CUDNN_TAR" "$CUDNN_DIR"
    echo "[✅] cuDNN v8.9.7.29 installed for CUDA 11.8"
fi

# ✅ Create and activate virtual env
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# ✅ Install correct onnxruntime-gpu manually
pip install onnxruntime-gpu==1.16.3

# ✅ Install project requirements
pip install -r requirements.txt

# ✅ Check ONNXRuntime GPU availability now (after venv, after install)
echo "[DEBUG] Checking ONNXRuntime GPU availability..."
python3 -c '
import onnxruntime as ort
providers = ort.get_available_providers()
device = ort.get_device()
print(f"[DEBUG] Providers available: {providers}")
print(f"[DEBUG] Execution device selected: {device}")
if "CUDAExecutionProvider" in providers:
    print("[✅] GPU (CUDA) is available and will be used.")
else:
    print("[⚠️] CUDAExecutionProvider not found. Running on CPU.")
' || true

# ✅ Install system dependencies
if [[ "$(uname -s)" == "Linux" ]]; then
    sudo apt update
    sudo apt install -y ffmpeg
fi

# ✅ Download required models
mkdir -p models
[ ! -f "models/inswapper_128.onnx" ] && \
curl -L https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -o models/inswapper_128.onnx

[ ! -f "models/GFPGANv1.4.pth" ] && \
curl -L https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth -o models/GFPGANv1.4.pth

# ✅ Default to CUDA
EXECUTION_PROVIDER="cuda"

echo "[INFO] Execution provider selected: $EXECUTION_PROVIDER"

# ✅ Run face swap (Image)
python3 run.py \
  --target content/target_image.png \
  --source content/source_image.png \
  -o content/swapped_image.png \
  --execution-provider $EXECUTION_PROVIDER \
  --frame-processor face_swapper face_enhancer

# ✅ Run face swap (Video)
python3 run.py \
  --target content/target_video.mp4 \
  --source content/source_image.png \
  -o content/swapped_video.mp4 \
  --execution-provider $EXECUTION_PROVIDER \
  --frame-processor face_swapper face_enhancer
