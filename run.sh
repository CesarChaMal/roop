#!/bin/bash
set -e
export PYTHONUTF8=1

# ✅ Global Variables
VENV=".venv"
PYTHON_VERSION="3.10.13"
CUDA_PATH="/usr/local/cuda-11.8"

# ✅ Utilities
log() { echo "[INFO] $1"; }
warn() { echo "[⚠️] $1"; }
error() { echo "[❌] $1" >&2; }

# ✅ 1. Environment Cleanup
cleanup_env() {
  unset PYTHONPATH CONDA_PREFIX LD_LIBRARY_PATH
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
}

# ✅ 2. CUDA Setup
setup_cuda() {
  sudo apt remove --purge -y nvidia-cuda-toolkit || true
  export PATH="$CUDA_PATH/bin${PATH:+:${PATH}}"
  export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  if [[ ! -d "$CUDA_PATH" ]]; then
    log "CUDA 11.8 not found. Installing..."
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O cuda_11.8.run
    chmod +x cuda_11.8.run
    sudo ./cuda_11.8.run --toolkit --silent --override
    rm -f cuda_11.8.run
  fi
}

# ✅ 3. cuDNN Setup
install_cudnn() {
  if [[ ! -f "$CUDA_PATH/lib64/libcudnn.so.8" ]]; then
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
  fi
}

# ✅ 4. pyenv + venv setup
setup_python_env() {
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
}

# ✅ 5. ONNX GPU Check
check_onnx_gpu() {
  python3 -c '
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"[DEBUG] Providers available: {providers}")
print(f"[DEBUG] Execution device selected: {ort.get_device()}")
if "CUDAExecutionProvider" in providers:
    print("[✅] GPU (CUDA) is available and will be used.")
else:
    print("[⚠️] CUDAExecutionProvider not found. Running on CPU.")
' || true
}

# ✅ 6. Install ffmpeg
install_system_deps() {
  if [[ "$(uname -s)" == "Linux" ]]; then
    sudo apt install -y ffmpeg
  fi
}

# ✅ 7. Download Required Models
download_models() {
  mkdir -p models
  [[ ! -f models/inswapper_128.onnx ]] && \
    curl -L https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -o models/inswapper_128.onnx
  [[ ! -f models/GFPGANv1.4.pth ]] && \
    curl -L https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth -o models/GFPGANv1.4.pth
}

# ✅ 8. Determine Execution Provider
detect_execution_provider() {
  if [[ "$1" == "cuda" || "$1" == "cpu" ]]; then
    EXECUTION_PROVIDER="$1"
    log "Execution provider manually set to: $EXECUTION_PROVIDER"
  else
    EXECUTION_PROVIDER=$(python3 -c 'import onnxruntime as ort; print("cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu")')
    [[ "$EXECUTION_PROVIDER" == "cpu" ]] && warn "CUDA not detected. Defaulting to CPU execution." || log "CUDA detected."
  fi
  export EXECUTION_PROVIDER
}

# ✅ 9. Main Menu
main_menu() {
  source ./run_menu.sh "$EXECUTION_PROVIDER"
}

# ✅ MAIN EXECUTION
main() {
  cleanup_env
  setup_cuda
  install_cudnn
  setup_python_env
  check_onnx_gpu
  install_system_deps
  download_models
  detect_execution_provider "$1"
  main_menu
}

main "$@"
