#!/bin/bash
set -e
export PYTHONUTF8=1

# ✅ Global Variables
VENV=".venv"
PYTHON_VERSION="3.10.13"
CUDA_PATH="/usr/local/cuda-11.8"
ROOT_DIR=$(pwd)

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

# ✅ Remove conflicting CUDA packages
remove_conflicting_CUDA_packages() {
    log "Removing potentially conflicting CUDA packages..."
    sudo apt remove --purge -y nvidia-cuda-toolkit || warn "nvidia-cuda-toolkit not found or already removed."
    #sudo apt remove --purge -y nvidia-cuda-toolkit || true
    success "Conflicting CUDA packages removed (if present)."
}

# ✅ Environment Cleanup
cleanup_env() {
    log "Cleaning up environment variables..."
    unset PYTHONPATH CONDA_PREFIX LD_LIBRARY_PATH
    debug "Unset PYTHONPATH, CONDA_PREFIX, LD_LIBRARY_PATH"
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    success "Environment cleaned."
}

# ✅ CUDA Setup
setup_cuda() {
  log "Setting up CUDA environment..."
  sudo apt remove --purge -y nvidia-cuda-toolkit || true
  export PATH="$CUDA_PATH/bin${PATH:+:${PATH}}"
  export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  debug "Exporting CUDA PATH and LD_LIBRARY_PATH"
  debug "[DEBUG] Test: ! -d \"$CUDA_PATH\" → $([[ ! -d \"$CUDA_PATH\" ]] && echo true || echo false)"

  if [[ ! -d "$CUDA_PATH" ]]; then
    warn "CUDA directory not found at $CUDA_PATH"
    local installer="cuda_11.8.run"
    local url="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"

    wget "$url" -O "$installer"
    chmod +x "$installer"

    # Detect if running on a network or non-native filesystem
    local mount_type
    mount_type=$(df -T . | awk 'NR==2 {print $2}')
    debug "[DEBUG] Current filesystem type: $mount_type"

    if [[ "$mount_type" != "ext4" && "$mount_type" != "xfs" ]]; then
      warn "Non-native filesystem detected ($mount_type). Copying installer to /tmp to avoid stale file handle issues."
      cp "$installer" /tmp/
      chmod +x /tmp/"$installer"
      sudo /tmp/"$installer" --toolkit --silent --override
      rm -f /tmp/"$installer"
    else
      sudo ./"$installer" --toolkit --silent --override
    fi

    rm -f "$installer"
    success "CUDA 11.8 installed successfully."
  else
    success "CUDA already installed."
  fi

  success "Setup CUDA completed."
}

# ✅ cuDNN Setup
install_cudnn() {
  log "Checking for cuDNN installation..."
  debug "[DEBUG] Test: ! -f $CUDA_PATH/lib64/libcudnn.so.8 → $([[ ! -f \"$CUDA_PATH/lib64/libcudnn.so.8\" ]] && echo true || echo false)"

  if [[ ! -f "$CUDA_PATH/lib64/libcudnn.so.8" ]]; then
    warn "cuDNN not found. Installing now..."
    log "Installing cuDNN v8.9.5.29 for CUDA 11.8..."

    local TARBALL="cudnn-linux-x86_64-8.9.5.29_cuda11-archive.tar.xz"
    local URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/$TARBALL"

    # Check current filesystem type
    local mount_type
    mount_type=$(df -T . | awk 'NR==2 {print $2}')
    debug "[DEBUG] Current filesystem type: $mount_type"

    local extract_dir

    if [[ "$mount_type" != "ext4" && "$mount_type" != "xfs" ]]; then
      warn "Non-native filesystem detected ($mount_type). Downloading and extracting in /tmp..."
      extract_dir="/tmp/cudnn_temp"
      mkdir -p "$extract_dir"
      cd "$extract_dir"
    else
      extract_dir="$(pwd)"
    fi

    wget "$URL" -O "$extract_dir/$TARBALL"
    tar -xf "$extract_dir/$TARBALL" -C "$extract_dir"

    local DIR
    DIR=$(find "$extract_dir" -type d -name "cudnn-linux-x86_64*" | head -n 1)

    sudo cp -P "$DIR/include/"* "$CUDA_PATH/include/"
    sudo cp -P "$DIR/lib/"* "$CUDA_PATH/lib64/"
    sudo ldconfig

    rm -rf "$extract_dir/$TARBALL" "$DIR"
    success "cuDNN installed successfully."
  else
    success "cuDNN already present."
  fi

  success "install_cudnn completed"
}

install_cmake() {
  log "Installing compatible CMake 3.24.4..."
  sudo apt remove -y cmake || true
  wget https://github.com/Kitware/CMake/releases/download/v3.24.4/cmake-3.24.4-linux-x86_64.sh
  chmod +x cmake-3.24.4-linux-x86_64.sh
  sudo ./cmake-3.24.4-linux-x86_64.sh --prefix=/usr/local --skip-license
  export PATH="/usr/local/bin:$PATH"
  which cmake
  cmake --version
  ldd $(which cmake) | grep libc
  rm -f cmake-3.24.4-linux-x86_64.sh
  success "CMake installed and validated."
}

install_system_deps() {
  log "Installing system dependencies..."

  if [[ "$(uname -s)" == "Linux" ]]; then
    sudo apt update
    sudo apt install -y \
      ffmpeg build-essential cmake \
      libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev libboost-all-dev \
      libbz2-dev libreadline-dev libssl-dev libsqlite3-dev \
      libncursesw5-dev libffi-dev zlib1g-dev liblzma-dev \
      tk-dev libgdbm-dev uuid-dev libnss3-dev libgdbm-compat-dev
  fi

  success "System dependencies installed (ffmpeg, Python build libs, C++ toolchain, etc.)"
}

# ✅ pyenv + venv setup
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

  cd "$ROOT_DIR"
  pip install --upgrade pip
#  pip install -r requirements.txt
  pip install --use-pep517 -r requirements.txt || warn "Initial install failed, continuing..."

  # ✅ Try wheel first
  if ! python -c "import dlib" 2>/dev/null; then
    warn "Trying to install precompiled dlib wheel..."
    pip install dlib --prefer-binary || {
      warn "Precompiled wheel failed, trying to build from source with CXXFLAGS..."
      CXXFLAGS="-std=c++14" pip install dlib --no-binary dlib || {
        error "❌ Failed to install dlib even with fallback."
        exit 1
      }
    }
  fi

  python -c "import dlib; print('dlib version:', dlib.__version__)"
  success "Python environment set up successfully."
}

# ✅ ONNX GPU Check
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

# ✅ Download Required Models
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

  # ✅ shape_predictor_68_face_landmarks.dat
  debug "[DEBUG] Test: ! -f models/shape_predictor_68_face_landmarks.dat → $([[ ! -f models/shape_predictor_68_face_landmarks.dat ]] && echo true || echo false)"
  # ✅ Download from the official Dlib mirror if missing or corrupted
  if [[ ! -f models/shape_predictor_68_face_landmarks.dat ]] || ! file models/shape_predictor_68_face_landmarks.dat | grep -q "data"; then
    log "Downloading valid shape_predictor_68_face_landmarks.dat from official source..."
    curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -o models/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 -f models/shape_predictor_68_face_landmarks.dat.bz2
    success "Downloaded and extracted valid shape_predictor_68_face_landmarks.dat"
  else
    debug "Model shape_predictor_68_face_landmarks.dat already exists and seems valid."
  fi

  success "Model downloads complete."
}

# ✅ Determine Execution Provider
detect_execution_provider() {
  debug "User-supplied execution provider: $1"
  debug "[DEBUG] \"$1\" == \"cuda\" || \"$1\" == \"cpu\" → $([[ \"$1\" == \"cuda\" || \"$1\" == \"cpu\" ]] && echo true || echo false)"

  if [[ "$1" == "cuda" || "$1" == "cpu" ]]; then
    EXECUTION_PROVIDER="$1"
    log "Execution provider manually set to: $EXECUTION_PROVIDER"
  else
    # Check if NVIDIA driver is loaded
    if ! nvidia-smi &>/dev/null; then
      warn "nvidia-smi failed: NVIDIA driver not loaded or no compatible GPU. Forcing CPU execution."
      EXECUTION_PROVIDER="cpu"
    else
      # Let ONNXRuntime decide
      EXECUTION_PROVIDER=$(python -c 'import onnxruntime as ort; print("cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu")')
      [[ "$EXECUTION_PROVIDER" == "cpu" ]] && warn "CUDAExecutionProvider not available in ONNXRuntime. Using CPU." || log "CUDA detected."
    fi
  fi

  export EXECUTION_PROVIDER
  success "Execution provider set to: $EXECUTION_PROVIDER"
}

# ✅ Run command helper
run_face_swap() {
  local description="$1"
  shift
  local log_file="logs/$(date +%F_%T)_${description// /_}.log"

  log "Running: $description"
  local start_time=$(date +%s)

  {
    python run.py "$@" 2>&1

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    local formatted_time
    formatted_time=$(printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds")

    echo -e "${GREEN}[✅ SUCCESS]${RESET} Finished in ${duration} seconds (${formatted_time})"
  } | tee "$log_file"
}

# ✅ Main Menu
main_menu() {
  log "Starting full setup pipeline..."
  debug "[DEBUG] EXECUTION_PROVIDER in main_menu: $EXECUTION_PROVIDER"

    # ✅ Capture EXECUTION_PROVIDER from argument
    EXECUTION_PROVIDER="$1"

    # ✅ Validate it's set
    if [[ -z "$EXECUTION_PROVIDER" ]]; then
      error "[❌] EXECUTION_PROVIDER not set. Exiting."
      exit 1
    fi

    mkdir -p logs

    while true; do
      echo ""
      echo "🟢 Choose a face swap mode to run:"
      echo "1) Face Swap - Image (HQ with Enhancer)"
      echo "2) Face Swap - Image (Fast, no Enhancer)"
      echo "3) Face Swap - Image (1 source faces onto 2 target faces, ref by position 0)"
      echo "4) Face Swap - Image (1 source faces onto 2 target faces, ref by position 1)"
      echo "5) Face Swap - Video (HQ)"
      echo "6) Face Swap - Video (HQ) (1 source faces onto 2 target faces, ref by position 0)"
      echo "7) Face Swap - Video (HQ) (1 source faces onto 2 target faces, ref by position 1)"
      echo "8) Face Swap - Video (HQ) with --keep-fps"
      echo "9) Face Swap - Video (HQ) with --framewise"
      echo "10) Face Swap - Video (HQ) with --keep-fps and --framewise"
      echo "11) Face Swap - Multi-face Video (custom reference)"
      echo "12) Face Swap - Compressed Video (HEVC, lower quality)"
      echo "13) Face Swap - NVENC Video (fast encoding)"
      echo "14) Face Swap - Debug (keep temp frames, no audio)"
      echo "15) Face Swap - Minimal Example (manual testing)"
      echo "16) Face Swap - Image (2 source faces onto 2 target faces)"
      echo "17) Face Swap - Image (2 source faces onto 3 target faces, ref by position 0)"
      echo "18) Face Swap - Image (2 source faces onto 3 target faces, ref by position 1)"
      echo "19) Face Swap - Video (2 source faces onto 2 target faces)"
      echo "20) Face Swap - Video (2 source faces onto 3 target faces, ref by position 0)"
      echo "21) Face Swap - Video (2 source faces onto 3 target faces, ref by position 1)"
      echo "22) Face Swap - Video (3 source faces onto 3 target faces, ref by position 0)"
      echo "23) Face Swap - Video (3 source faces onto 3 target faces, ref by position 1)"
      echo "24) Face Swap - Image (HQ with Enhancer) with --preserve-expressions"
      echo "25) Face Swap - Image (HQ with Enhancer) with --preserve-expressions (1 source faces onto 2 target faces, ref by position 0)"
      echo "26) Face Swap - Image (HQ with Enhancer) with --preserve-expressions (1 source faces onto 2 target faces, ref by position 1)"
      echo "27) Face Swap - Video (HQ) with --preserve-expressions"
      echo "28) Face Swap - Video (HQ) with --preserve-expressions (1 source faces onto 2 target faces, ref by position 0)"
      echo "29) Face Swap - Video (HQ) with --preserve-expressions (1 source faces onto 2 target faces, ref by position 1)"
      echo "0) ❌ Exit"

      read -p "👉 Enter number [0-19]: " choice

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
        3) run_face_swap "Image (1 source faces onto 2 target faces, ref by position 0)" \
            --target content/target_multiface_image.png \
            --source content/source_image.png \
            --output content/output_image_1sources_2targets_byPosition0.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --reference-face-position 0 ;;
        4) run_face_swap "Image (1 source faces onto 2 target faces, ref by position 1)" \
            --target content/target_multiface_image.png \
            --source content/source_image.png \
            --output content/output_image_1sources_2targets_byPosition1.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --reference-face-position 1 ;;
        5) run_face_swap "Video (HQ)" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 ;;
        6) run_face_swap "Video (HQ) (1 source faces onto 2 target faces, ref by position 0)" \
            --target content/target_multiface_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_1sources_2targets_byPosition0.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --reference-face-position 0 ;;
        7) run_face_swap "Video (HQ) (1 source faces onto 2 target faces, ref by position 1)" \
            --target content/target_multiface_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_1sources_2targets_byPosition1.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --reference-face-position 1 ;;
        8) run_face_swap "Video (HQ) with --keep-fps" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_with_keep_fps.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps ;;
        9) run_face_swap "Video (HQ) with --framewise" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_with_framewise.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --framewise ;;
        10) run_face_swap "Video (HQ) with --keep-fps --framewise" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_with_keep_fps_and_framewise.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps \
            --framewise ;;
        11) run_face_swap "Multi-face Video (custom reference)" \
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
        12) run_face_swap "Compressed Video (HEVC)" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_compressed.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --output-video-encoder libx265 \
            --output-video-quality 40 \
            --keep-fps \
            --framewise ;;
        13) run_face_swap "NVENC Video" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_nvenc.mp4 \
            --execution-provider cuda \
            --frame-processor face_swapper face_enhancer \
            --output-video-encoder h264_nvenc \
            --output-video-quality 30 \
            --keep-fps \
            --framewise ;;
        14) run_face_swap "Debug Video (keep frames, skip audio)" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_debug.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --keep-fps \
            --framewise \
            --keep-frames \
            --skip-audio ;;
        15) run_face_swap "Minimal Example (manual testing)" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_test.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper \
            --keep-fps \
            --framewise ;;
        16) run_face_swap "Image (2 source faces onto 2 target faces)" \
            --target content/target_multiface_image.png \
            --source "content/source_image1.png;content/source_image2.png" \
            --output content/output_image_multiface.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --many-faces \
            --multi-source ;;
        17) run_face_swap "Image (2 source faces onto 3 target faces, ref by position 0)" \
            --target content/target_multiface_image.png \
            --source "content/source_image1.png;content/source_image2.png" \
            --output content/output_image_multifaces_2sources_3targets_byPosition0.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --many-faces \
            --multi-source \
            --reference-frame-number 0 ;;
        18) run_face_swap "Image (2 source faces onto 3 target faces, ref by position 1)" \
            --target content/target_multiface_image.png \
            --source "content/source_image1.png;content/source_image2.png" \
            --output content/output_image_multifaces_2sources_3targets_byPosition1.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --many-faces \
            --multi-source \
            --reference-frame-number 1 ;;
        19) run_face_swap "Video (2 source faces onto 2 target faces" \
            --target content/target_multiface_video.mp4 \
            --source "content/source_image1.png;content/source_image2.png" \
            --output content/output_video_multifaces.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --keep-fps \
            --framewise \
            --many-faces \
            --multi-source ;;
        20) run_face_swap "Video (2 source faces onto 3 target faces, ref by position 0)" \
            --target content/target_multiface_video.mp4 \
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
        21) run_face_swap "Video (2 source faces onto 3 target faces, ref by position 1)" \
            --target content/target_multiface_video.mp4 \
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
        22) run_face_swap "Video (3 source faces onto 3 target faces, ref by position 0)" \
            --target content/target_multiface_video.mp4 \
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
        23) run_face_swap "Video (3 source faces onto 3 target faces, ref by position 1)" \
            --target content/target_multiface_video.mp4 \
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
        24) run_face_swap "Image (HQ with Enhancer) with --preserve-expressions" \
            --target content/target_image.png \
            --source content/source_image.png \
            --output content/output_image_with_preserve_expressions.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --preserve-expressions ;;
        25) run_face_swap "Image (HQ with Enhancer) with --preserve-expressions (1 source faces onto 2 target faces, ref by position 0)" \
            --target content/target_multiface_image.png \
            --source content/source_image.png \
            --output content/output_image_with_preserve_expressions_1sources_2targets_byPosition0.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --preserve-expressions \
            --reference-face-position 0 ;;
        26) run_face_swap "Image (HQ with Enhancer) with --preserve-expressions (1 source faces onto 2 target faces, ref by position 1)" \
            --target content/target_multiface_image.png \
            --source content/source_image.png \
            --output content/output_image_with_preserve_expressions_1sources_2targets_byPosition1.png \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --preserve-expressions \
            --reference-face-position 1 ;;
        27) run_face_swap "Video (HQ) with --preserve-expressions" \
            --target content/target_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_with_preserve_expressions.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --preserve-expressions ;;
        28) run_face_swap "Video (HQ) with --preserve-expressions (1 source faces onto 2 target faces, ref by position 0)" \
            --target content/target_multiface_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_with_preserve_expressions_1sources_2targets_byPosition0.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --preserve-expressions \
            --reference-face-position 0 ;;
        29) run_face_swap "Video (HQ) with --preserve-expressions (1 source faces onto 2 target faces, ref by position 1)" \
            --target content/target_multiface_video.mp4 \
            --source content/source_image.png \
            --output content/output_video_with_preserve_expressions_1sources_2targets_byPosition1.mp4 \
            --execution-provider "$EXECUTION_PROVIDER" \
            --frame-processor face_swapper face_enhancer \
            --execution-threads 8 \
            --preserve-expressions \
            --reference-face-position 1 ;;
        0) log "[👋] Exiting. Have a nice day!"; break ;;
        *) error "[❌] Invalid option. Please enter a number between 0 and 14." ;;
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
  install_cmake
  install_system_deps
  setup_python_env
  check_onnx_gpu
  download_models
  detect_execution_provider "$1"

  log "Execution provider set to: $EXECUTION_PROVIDER"
  main_menu "$EXECUTION_PROVIDER"
}
#export EXECUTION_PROVIDER
main "$@"