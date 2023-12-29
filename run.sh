#!/bin/bash

# Step 1: Clone the GitHub repository
#git clone https://github.com/CesarChaMal/roop.git
#cd roop

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Download the model and move it to the 'models' directory
#wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O inswapper_128.onnx
curl -L https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -o inswapper_128.onnx

mkdir -p models
mv inswapper_128.onnx ./models

# Step 4: Run the Python script
# Replace '/content/video.mp4' and '/content/image.jpeg' with your actual file paths
python run.py --target /video.mp4 --source /image.png -o /swapped.mp4 --execution-provider cuda --frame-processor face_swapper face_enhancer
