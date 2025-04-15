#!/bin/bash
# Step 1: Clone the GitHub repository (uncomment if starting fresh)
# git clone https://github.com/CesarChaMal/roop.git
# cd roop

# Step 2: Create env and Install Python dependencies
python -m venv .venv
source .venv/scripts/activate
pip install -r requirements.txt

# Step 3: Download the ONNX model and move it to the models directory
# wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O inswapper_128.onnx
curl -L https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -o inswapper_128.onnx

mkdir -p models
mv inswapper_128.onnx ./models/

# Step 4a: Run face swap for IMAGE
python run.py \
  --target /content/target_image.png \
  --source /content/source_image.png \
  -o /content/swapped_image.png \
  --execution-provider cuda \
  --frame-processor face_swapper face_enhancer

# Step 4b: Run face swap for VIDEO
python run.py \
  --target /content/target_video.mp4 \
  --source /content/source_image.jpeg \
  -o /content/swapped_video.mp4 \
  --execution-provider cuda \
  --frame-processor face_swapper face_enhancer
