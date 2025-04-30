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
#pip install --upgrade pip

# ✅ Install correct onnxruntime-gpu manually
#pip install onnxruntime-gpu==1.16.3

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
#    sudo apt update
    sudo apt install -y ffmpeg
fi

# ✅ Download required models
mkdir -p models
[ ! -f "models/inswapper_128.onnx" ] && \
curl -L https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -o models/inswapper_128.onnx

[ ! -f "models/GFPGANv1.4.pth" ] && \
curl -L https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth -o models/GFPGANv1.4.pth

# ✅ Determine execution provider (CUDA or CPU)
if [[ "$1" == "cuda" || "$1" == "cpu" ]]; then
  EXECUTION_PROVIDER="$1"
  echo "[INFO] Execution provider manually set to: $EXECUTION_PROVIDER"
else
  EXECUTION_PROVIDER=$(python3 -c 'import onnxruntime as ort; print("cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu")')
  if [[ "$EXECUTION_PROVIDER" == "cpu" ]]; then
    echo "[⚠️] CUDA not detected. Defaulting to CPU execution."
  else
    echo "[✅] CUDA detected. Using GPU execution."
  fi
fi

# ✅ Menu-based execution
mkdir -p logs
while true; do
  echo ""
  echo "🟢 Choose a face swap mode to run:"
  echo "1) Face Swap - Image (HQ with Enhancer)"
  echo "2) Face Swap - Image (Fast, no Enhancer)"
  echo "3) Face Swap - Video (HQ)"
  echo "4) Face Swap - Multi-face Video (custom reference)"
  echo "5) Face Swap - Compressed Video (HEVC, lower quality)"
  echo "6) Face Swap - NVENC Video (fast encoding)"
  echo "7) Face Swap - Debug (keep temp frames, no audio)"
  echo "8) Face Swap - Minimal Example (manual testing)"
  echo "9) Face Swap - Image (2 source faces)"
  echo "10) Face Swap - Video (2 source faces)"
  echo "11) Face Swap - Video (2 source faces onto 3 target faces, ref by position 0)"
  echo "12) Face Swap - Video (2 source faces onto 3 target faces, ref by position 1)"
  echo "13) Face Swap - Video (3 source faces onto 3 target faces, ref by position 0)"
  echo "14) Face Swap - Video (3 source faces onto 3 target faces, ref by position 1)"
  echo "0) ❌ Exit"

  read -p "👉 Enter number [0-8]: " choice
  LOG_FILE="logs/$(date +%F_%T)_choice${choice}.log"

  case $choice in
    1)
      echo "[🔁] Running: Image (HQ with Enhancer)..."
      python3 run.py \
        --target content/target_image.png \
        --source content/source_image.png \
        --output content/output_image.png \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer | tee "$LOG_FILE"
      ;;
    2)
      echo "[🔁] Running: Image (Fast, no Enhancer)..."
      python3 run.py \
        --target content/target_image.png \
        --source content/source_image.png \
        --output content/output_image_fast.png \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper | tee "$LOG_FILE"
      ;;
    3)
      echo "[🔁] Running: Video (HQ)..."
      python3 run.py \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise | tee "$LOG_FILE"
      ;;
    4)
      echo "[🔁] Running: Multi-face Video (with reference face selection)..."
      python3 run.py \
        --target content/target_multiface_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_multiface.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --reference-face-position 0 \
        --reference-frame-number 10 | tee "$LOG_FILE"
      ;;
    5)
      echo "[🔁] Running: Compressed Video (HEVC)..."
      python3 run.py \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_compressed.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --output-video-encoder libx265 \
        --output-video-quality 40 \
        --keep-fps \
        --framewise | tee "$LOG_FILE"
      ;;
    6)
      echo "[🔁] Running: NVENC Video..."
      python3 run.py \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_nvenc.mp4 \
        --execution-provider cuda \
        --frame-processor face_swapper face_enhancer \
        --output-video-encoder h264_nvenc \
        --output-video-quality 30 \
        --keep-fps \
        --framewise | tee "$LOG_FILE"
      ;;
    7)
      echo "[🔁] Running: Debug Video (keep frames, skip audio)..."
      python3 run.py \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_debug.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --keep-fps \
        --framewise \
        --keep-frames \
        --skip-audio | tee "$LOG_FILE"
      ;;
    8)
      echo "[🔁] Running: Minimal Example (manual testing)..."
      python3 run.py \
        --target content/target_video.mp4 \
        --source content/source_image.png \
        --output content/output_video_test.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper \
        --keep-fps \
        --framewise | tee "$LOG_FILE"
      ;;
    9)
      echo "[🔁] Running: Image (with 2 source faces)..."
      python3 run.py \
        --target content/target_multiface_image.png \
        --source "content/source_image1.png;content/source_image1.png" \
        --output content/output_image_multiface.png \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --many-faces \
        --multi-source | tee "$LOG_FILE"
      ;;
    10)
      echo "[🔁] Running: Video (with 2 source faces)..."
      python3 run.py \
        --target content/target_multiface_video.mp4 \
        --source "content/source_image1.png;content/source_image1.png" \
        --output content/output_video_multifaces.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source | tee "$LOG_FILE"
      ;;
    11)
      echo "[🔁] Running: Video (2 source faces onto 3 target faces, ref by position 0)..."
      python3 run.py \
        --target content/target_3faces_video.mp4 \
        --source "content/source_face1.png;content/source_face2.png" \
        --output content/output_video_multifaces_2sources_3targets_byPosition0.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source \
        --reference-face-position 0 \
        --reference-frame-number 0 | tee "$LOG_FILE"
      ;;
    12)
      echo "[🔁] Running: Video (2 source faces onto 3 target faces, ref by position 1)..."
      python3 run.py \
        --target content/target_3faces_video.mp4 \
        --source "content/source_face1.png;content/source_face2.png" \
        --output content/output_video_multifaces_2sources_3targets_byPosition1.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source \
        --reference-face-position 1 \
        --reference-frame-number 0 | tee "$LOG_FILE"
      ;;
    13)
      echo "[🔁] Running: Video (3 source faces onto 3 target faces, ref by position 0)..."
      python3 run.py \
        --target content/target_3faces_video.mp4 \
        --source "content/source_face1.png;content/source_face2.png;content/source_face3.png" \
        --output content/output_video_multifaces_3sources_3targets_byPosition0.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source \
        --reference-face-position 0 \
        --reference-frame-number 0 | tee "$LOG_FILE"
      ;;
    14)
      echo "[🔁] Running: Video (3 source faces onto 3 target faces, ref by position 1)..."
      python3 run.py \
        --target content/target_3faces_video.mp4 \
        --source "content/source_face1.png;content/source_face2.png;content/source_face3.png" \
        --output content/output_video_multifaces_3sources_3targets_byPosition1.mp4 \
        --execution-provider $EXECUTION_PROVIDER \
        --frame-processor face_swapper face_enhancer \
        --execution-threads 8 \
        --keep-fps \
        --framewise \
        --many-faces \
        --multi-source \
        --reference-face-position 0 \
        --reference-frame-number 0 | tee "$LOG_FILE"
      ;;
    0)
      echo "[👋] Exiting. Have a nice day!"
      break
      ;;
    *)
      echo "[❌] Invalid option. Please enter a number between 0 and 8."
      ;;
  esac

  echo ""
  read -p "🔁 Press [Enter] to continue or [Ctrl+C] to exit..."
  clear
done
