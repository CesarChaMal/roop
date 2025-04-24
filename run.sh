#!/bin/bash
set -e
export PYTHONUTF8=1

# ✅ Isolate from Anaconda interference
unset PYTHONPATH
unset CONDA_PREFIX
unset LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# ✅ Remove conflicting CUDA packages
sudo apt remove --purge -y nvidia-cuda-toolkit || true

# ✅ Setup CUDA 11.7 explicitly
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 🔧 Auto install CUDA 11.7 if missing
if [[ ! -d "/usr/local/cuda-11.7" ]]; then
    echo "[INFO] CUDA 11.7 not found. Installing..."
    wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run -O cuda_11.7.run
    chmod +x cuda_11.7.run
    sudo ./cuda_11.7.run --toolkit --silent --override
    rm -f cuda_11.7.run
fi

# ✅ Install cuDNN if missing
if [[ ! -f "/usr/local/cuda-11.7/lib64/libcudnn.so.8" ]]; then
    echo "[INFO] cuDNN not found. Installing cuDNN v8.6.0 for CUDA 11.7..."
    CUDNN_URL="https://developer.download.nvidia.com/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz"
    wget "$CUDNN_URL" -O cudnn.tar.xz
    if [[ -f cudnn.tar.xz ]]; then
        tar -xf cudnn.tar.xz
        CUDNN_DIR=$(find . -type d -name "cudnn-linux-x86_64*" | head -n 1)
        sudo cp -P "$CUDNN_DIR/include/"* /usr/local/cuda-11.7/include/
        sudo cp -P "$CUDNN_DIR/lib/"* /usr/local/cuda-11.7/lib64/
        sudo ldconfig
        rm -rf cudnn.tar.xz "$CUDNN_DIR"
        echo "[OK] cuDNN installed for CUDA 11.7"
    else
        echo "[ERROR] cuDNN download failed. Please check the URL or your network."
    fi
fi

# ✅ Default to CUDA
EXECUTION_PROVIDER="cuda"

# ✅ Check libcufft.so.10
REQUIRED_LIB="/usr/local/cuda-11.7/lib64/libcufft.so.10"
if [[ ! -f "$REQUIRED_LIB" ]]; then
    echo "[WARN] $REQUIRED_LIB not found. Trying to fix..."
    FOUND_CUFFT=$(find /usr/local -name "libcufft.so.10" 2>/dev/null | head -n 1)
    if [[ -n "$FOUND_CUFFT" ]]; then
        sudo ln -s "$FOUND_CUFFT" "$REQUIRED_LIB"
        echo "[OK] Linked: $REQUIRED_LIB → $FOUND_CUFFT"
    else
        echo "[FATAL] cuFFT lib not found. Falling back to CPU."
        EXECUTION_PROVIDER="cpu"
    fi
fi

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

# ✅ Log final selection
echo "[INFO] Execution provider selected: $EXECUTION_PROVIDER"

# ✅ Create and activate virtual env
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

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

# 🔁 Optional: enable for many-faces mode (swap multiple faces in video)
# python3 run.py \
#   --target content/target_video.mp4 \
#   --source content/source_image.png \
#   -o content/swapped_video_multi.mp4 \
#   --execution-provider $EXECUTION_PROVIDER \
#   --frame-processor face_swapper face_enhancer \
#   --many-faces
