#!/bin/bash
set -e  # Exit on any error
export PYTHONUTF8=1

# Detect platform
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     platform=linux;;
    Darwin*)    platform=mac;;
    MINGW*|MSYS*) platform=windows;;
    *)          platform="unknown"
esac

# Use python3 on Linux/macOS, python on Windows
if [[ "$platform" == "linux" || "$platform" == "mac" ]]; then
    PYTHON_BIN="python3"
    VENV_ACTIVATE=".venv/bin/activate"
else
    PYTHON_BIN="python"
    VENV_ACTIVATE=".venv/Scripts/activate"
fi

# Step 1: Clone the GitHub repo (optional)
# git clone https://github.com/CesarChaMal/roop.git
# cd roop

# Step 2: Create and activate virtual environment
$PYTHON_BIN -m venv .venv
source $VENV_ACTIVATE

# Step 3: Install dependencies
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Step 4: Download ONNX model if not already present
MODEL_PATH="models/inswapper_128.onnx"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "📥 Downloading ONNX model..."
    mkdir -p models
    curl -L https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -o "$MODEL_PATH"
else
    echo "✅ ONNX model already exists."
fi

# Step 5a: Run face swap for IMAGE
$PYTHON_BIN run.py \
  --target content/target_image.png \
  --source content/source_image.png \
  -o content/swapped_image.png \
  --execution-provider cuda \
  --frame-processor face_swapper face_enhancer

# Step 5b: Run face swap for VIDEO
$PYTHON_BIN run.py \
  --target content/target_video.mp4 \
  --source content/source_image.jpeg \
  -o content/swapped_video.mp4 \
  --execution-provider cuda \
  --frame-processor face_swapper face_enhancer
