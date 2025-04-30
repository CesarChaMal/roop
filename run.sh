#!/bin/bash
set -e
export PYTHONUTF8=1

# ✅ Global Variables
VENV=".venv"
PYTHON_VERSION="3.10.13"
CUDA_PATH="/usr/local/cuda-11.8"

# ✅ Color Codes
BOLD="\033[1m"
RESET="\033[0m"
BLUE="\033[1;34m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"

# ✅ Logging Utilities with Colors
log()     { echo -e "${BLUE}[INFO]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[⚠️ WARNING]${RESET} $*"; }
error()   { echo -e "${RED}[❌ ERROR]${RESET}   $*" >&2; }
success() { echo -e "${GREEN}[✅ SUCCESS]${RESET} $*"; }
debug()   { echo -e "${CYAN}[DEBUG]${RESET}   $*"; }

# ✅ 1. Remove conflicting CUDA packages
remove_conflicting_CUDA_packages() {
    log "Removing potentially conflicting CUDA packages..."
    sudo apt remove --purge -y nvidia-cuda-toolkit || warn "nvidia-cuda-toolkit not found or already removed."
    #sudo apt remove --purge -y nvidia-cuda-toolkit || true
    success "Conflicting CUDA packages removed (if present)."
}

# ✅ 2. Environment Cleanup
cleanup_env() {
    log "Cleaning up environment variables..."
    unset PYTHONPATH CONDA_PREFIX LD_LIBRARY_PATH
    debug "Unset PYTHONPATH, CONDA_PREFIX, LD_LIBRARY_PATH"
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    success "Environment cleaned."
}

# ✅ 3. CUDA Setup
setup_cuda() {
  log "Setting up CUDA environment..."
  sudo apt remove --purge -y nvidia-cuda-toolkit || true
  export PATH="$CUDA_PATH/bin${PATH:+:${PATH}}"
  export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  debug "Exporting CUDA PATH and LD_LIBRARY_PATH"
  debug "[DEBUG] Test: ! -d "$CUDA_PATH" → $([[ ! -d "$CUDA_PATH" ]] && echo true || echo false)"

  if [[ ! -d "$CUDA_PATH" ]]; then
    warn "CUDA directory not found at $CUDA_PATH"
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O cuda_11.8.run
    log "Downloading and installing CUDA 11.8..."
    chmod +x cuda_11.8.run
    sudo ./cuda_11.8.run --toolkit --silent --override
    rm -f cuda_11.8.run
    success "CUDA 11.8 installed successfully."
  else
    success "CUDA already installed."
  fi
  success "Setup CUDA completed."
}

# ✅ 4. cuDNN Setup
install_cudnn() {
  log "Checking for cuDNN installation..."
  debug "[DEBUG] Test: ! -f $CUDA_PATH/lib64/libcudnn.so.8 → $([[ ! -f "$CUDA_PATH/lib64/libcudnn.so.8" ]] && echo true || echo false)"

  if [[ ! -f "$CUDA_PATH/lib64/libcudnn.so.8" ]]; then
    warn "cuDNN not found. Installing now..."
    log "Installing cuDNN v8.9.7.29 for CUDA 11.8..."
    local TARBALL="cudnn-linux-x86_64-8.9.5.29_cuda11-archive.tar.xz"
    local URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/$TARBALL"
    wget "$URL" -O "$TARBALL"
    tar -xf "$TARBALL"
    local DIR=$(find . -type d -name "cudnn-linux-x86_64*" | head -n 1)
    sudo cp -P "$DIR/include/"* "$CUDA_PATH/include/"
    sudo cp -P "$DIR/lib/"* "$CUDA_PATH/lib64/"
    sudo ldconfig
    rm -rf "$TARBALL" "$DIR"
    log "cuDNN installed."
    success "cuDNN installed successfully."
  else
    success "cuDNN already present."
  fi
  success "install_cudnn completed"
}

# ✅ 5. pyenv + venv setup
setup_python_env() {
  log "Setting up Python environment using pyenv and venv..."
  debug "Checking for pyenv at $HOME/.pyenv"
  debug "[DEBUG] Test: ! -d "$HOME/.pyenv" → $([[ ! -d "$HOME/.pyenv" ]] && echo true || echo false)"

  if [[ ! -d "$HOME/.pyenv" ]]; then
    log "Installing pyenv..."
    sudo apt update && sudo apt install -y make build-essential libssl-dev zlib1g-dev \
      libbz2-dev libreadline-dev libsqlite3-dev curl llvm libncursesw5-dev xz-utils tk-dev \
      libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    curl https://pyenv.run | bash
  fi

  export PATH="$HOME/.pyenv/bin:$PATH"
  eval "$(pyenv init --path)"
  eval "$(pyenv virtualenv-init -)"

  if ! pyenv versions --bare | grep -q "^$PYTHON_VERSION$"; then
    log "Installing Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
  fi
  pyenv local "$PYTHON_VERSION"

  if [[ ! -d "$VENV" ]]; then
    log "Creating virtual environment..."
    python -m venv "$VENV"
  fi

  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
  success "Python environment set up successfully."
}

# ✅ 6. ONNX GPU Check
check_onnx_gpu() {
  log "[DEBUG] Beginning of check_onnx_gpu()"
  python -c '
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"[DEBUG] Providers available: {providers}")
print(f"[DEBUG] Execution device selected: {ort.get_device()}")
if "CUDAExecutionProvider" in providers:
    print("[✅] GPU (CUDA) is available and will be used.")
else:
    print("[⚠️] CUDAExecutionProvider not found. Running on CPU.")
' || true
  success "Finish of check_onnx_gpu()"
}

# ✅ 7. Install ffmpeg
install_system_deps() {
  log "Installing system dependencies..."
  debug "[DEBUG] Test: \$(uname -s) == \"Linux\" → $([[ $(uname -s) == "Linux" ]] && echo true || echo false)"

  if [[ "$(uname -s)" == "Linux" ]]; then
    sudo apt install -y ffmpeg
  fi
  success "System dependencies installed (ffmpeg)."
}

# ✅ 8. Download Required Models
download_models() {
  log "Checking for required models..."
  mkdir -p models

  debug "[DEBUG] Test: ! -f models/inswapper_128.onnx → $([[ ! -f models/inswapper_128.onnx ]] && echo true || echo false)"
#  [[ ! -f models/inswapper_128.onnx ]] && \
#    curl -L https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -o models/inswapper_128.onnx
  if [[ ! -f models/inswapper_128.onnx ]]; then
    log "Downloading inswapper_128.onnx..."
    curl -L https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -o models/inswapper_128.onnx
  else
    debug "Model inswapper_128.onnx already exists."
  fi

  debug "[DEBUG] Test: ! -f models/GFPGANv1.4.pth → $([[ ! -f models/GFPGANv1.4.pth ]] && echo true || echo false)"
#  [[ ! -f models/GFPGANv1.4.pth ]] && \
#    curl -L https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth -o models/GFPGANv1.4.pth
  if [[ ! -f models/GFPGANv1.4.pth ]]; then
    log "Downloading GFPGANv1.4.pth..."
    curl -L https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth -o models/GFPGANv1.4.pth
  else
    debug "Model GFPGANv1.4.pth already exists."
  fi
  success "Model downloads complete."
}

# ✅ 9. Determine Execution Provider
detect_execution_provider() {
  debug "User-supplied execution provider: $1"
  debug "[DEBUG] \"$1\" == \"cuda\" || \"$1\" == \"cpu\" → $([[ "$1" == "cuda" || "$1" == "cpu" ]] && echo true || echo false)"

  if [[ "$1" == "cuda" || "$1" == "cpu" ]]; then
    EXECUTION_PROVIDER="$1"
    log "Execution provider manually set to: $EXECUTION_PROVIDER"
  else
    EXECUTION_PROVIDER=$(python -c 'import onnxruntime as ort; print("cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu")')
    [[ "$EXECUTION_PROVIDER" == "cpu" ]] && warn "CUDA not detected. Defaulting to CPU execution." || log "CUDA detected."
    log "Execution provider set to: $EXECUTION_PROVIDER"
  fi
  export EXECUTION_PROVIDER
  success "Execution provider set to: $EXECUTION_PROVIDER"
}

# ✅ Run command helper
run_face_swap() {
  local description="$1"
  shift
  local log_file="logs/$(date +%F_%T)_${description// /_}.log"
  echo "[🔁] Running: $description..."
  python3 run.py "$@" | tee "$log_file"
}

# ✅ 10. Main Menu
main_menu() {
  log "Starting full setup pipeline..."
  debug "[DEBUG] EXECUTION_PROVIDER in main_menu: $EXECUTION_PROVIDER"

    # ✅ Capture EXECUTION_PROVIDER from argument
    EXECUTION_PROVIDER="$1"

    # ✅ Validate it's set
    if [[ -z "$EXECUTION_PROVIDER" ]]; then
      echo "[❌] EXECUTION_PROVIDER not set. Exiting."
      exit 1
    fi

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

      read -p "👉 Enter number [0-14]: " choice

      case $choice in
        1) run_face_swap "Image (HQ with Enhancer)" \
            --target content/target_image.png \
            --source content/source_image.png \
            --output content/output_image.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer ;;
        2) run_face_swap "Image (Fast, no Enhancer)" \
            --target content/target_image.png \
            --source content/source_image.png \
            --output content/output_image_fast.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper ;;
        3) run_face_swap "Video (HQ)" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps \
            --framewise ;;
        4) run_face_swap "Multi-face Video (custom reference)" \
            --target content/target_multiface_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_multiface.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps \
            --framewise \
            --many-faces \
            --reference-face-position 0 \
            --reference-frame-number 10 ;;
        5) run_face_swap "Compressed Video (HEVC)" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_compressed.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --output-video-encoder libx265 \
            --output-video-quality 40 \
            --keep-fps \
            --framewise ;;
        6) run_face_swap "NVENC Video" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_nvenc.mp4 \
            --execution-provider cuda \
            --frame-processor face_swapper face_enhancer \
            --output-video-encoder h264_nvenc \
            --output-video-quality 30 \
            --keep-fps \
            --framewise ;;
        7) run_face_swap "Debug Video (keep frames, skip audio)" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_debug.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --keep-fps \
            --framewise \
            --keep-frames \
            --skip-audio ;;
        8) run_face_swap "Minimal Example (manual testing)" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_test.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper \
            --keep-fps \
            --framewise ;;
        9) run_face_swap "Image (2 source faces)" \
            --target content/target_multiface_image.png \
            --source "content/source_image1.png;content/source_image1.png" \
            --output content/output_image_multiface.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --many-faces \
            --multi-source ;;
        10) run_face_swap "Video (2 source faces)" \
            --target content/target_multiface_video.mp4 \
            --source "content/source_image1.png;content/source_image1.png" \
            --output content/output_video_multifaces.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps \
            --framewise \
            --many-faces \
            --multi-source ;;
        11) run_face_swap "Video (2→3 targets, ref=0)" \
            --target content/target_3faces_video.mp4 \
            --source "content/source_image1.png;content/source_image2.png" \
            --output content/output_video_multifaces_2sources_3targets_byPosition0.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps \
            --framewise \
            --many-faces \
            --multi-source \
            --reference-face-position 0 \
            --reference-frame-number 0 ;;
        12) run_face_swap "Video (2→3 targets, ref=1)" \
            --target content/target_3faces_video.mp4 \
            --source "content/source_image1.png;content/source_image2.png" \
            --output content/output_video_multifaces_2sources_3targets_byPosition1.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps \
            --framewise \
            --many-faces \
            --multi-source \
            --reference-face-position 1 \
            --reference-frame-number 0 ;;
        13) run_face_swap "Video (3→3 targets, ref=0)" \
            --target content/target_3faces_video.mp4 \
            --source "content/source_image1.png;content/source_image2.png;content/source_image3.png" \
            --output content/output_video_multifaces_3sources_3targets_byPosition0.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps \
            --framewise \
            --many-faces \
            --multi-source \
            --reference-face-position 0 \
            --reference-frame-number 0 ;;
        14) run_face_swap "Video (3→3 targets, ref=1)" \
            --target content/target_3faces_video.mp4 \
            --source "content/source_image1.png;content/source_image2.png;content/source_image3.png" \
            --output content/output_video_multifaces_3sources_3targets_byPosition1.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps \
            --framewise \
            --many-faces \
            --multi-source \
            --reference-face-position 1 \
            --reference-frame-number 0 ;;
        0) echo "[👋] Exiting. Have a nice day!"; break ;;
        *) echo "[❌] Invalid option. Please enter a number between 0 and 14." ;;
      esac

      echo ""
      read -p "🔁 Press [Enter] to continue or [Ctrl+C] to exit..."
      clear
    done
  success "Environment ready. Launching menu..."
}

# ✅ MAIN EXECUTION
main() {
  remove_conflicting_CUDA_packages
  cleanup_env
  setup_cuda
  install_cudnn
  setup_python_env
  check_onnx_gpu
  detect_execution_provider "$1"
  install_system_deps
  download_models

  log "Execution provider set to: $EXECUTION_PROVIDER"
  main_menu "$EXECUTION_PROVIDER"
}
#export EXECUTION_PROVIDER
main "$@"