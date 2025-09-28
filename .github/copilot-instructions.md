# Copilot Instructions for Roop

## Overview
Roop is a Python-based application for face-swapping in images and videos. It uses machine learning models and libraries like TensorFlow, ONNX, and InsightFace to process media. The project supports both GUI and headless modes, with a modular architecture for frame processing.

## Architecture
- **Core Components**:
  - `roop/core.py`: Entry point for the application. Handles argument parsing, resource management, and orchestrates the processing pipeline.
  - `roop/globals.py`: Stores global configuration and runtime variables.
  - `roop/ui.py`: Implements the GUI using `customtkinter`.
  - `roop/utilities.py`: Utility functions for video and image processing.

- **Frame Processors**:
  - Located in `roop/processors/frame/`.
  - Examples: `face_swapper.py`, `face_enhancer.py`.
  - Each processor implements a standard interface (`pre_check`, `process_image`, `process_video`, etc.).

- **Face Analysis**:
  - `roop/face_analyser.py`: Detects and analyzes faces in frames.
  - `roop/face_reference.py`: Manages reference faces for swapping.

- **Predictors**:
  - `roop/predictor.py`: Uses `opennsfw2` to filter inappropriate content.

## Developer Workflows

### Running the Application
- **Headless Mode**:
  ```bash
  python run.py -s <source_image> -t <target_image_or_video> -o <output_path>
  ```
- **GUI Mode**:
  Simply run `python run.py` without arguments.

### Testing
- Tests are defined in the GitHub Actions workflow (`.github/workflows/ci.yml`).
- To manually test:
  ```bash
  python run.py -s .github/examples/source.jpg -t .github/examples/target.mp4 -o .github/examples/output.mp4
  ```

### Linting and Type Checking
- Linting: `flake8 run.py roop`
- Type Checking: `mypy run.py roop`

### Debugging
- Debug output is stored in the `debug_output/` directory.
- Use `--keep-frames` to retain intermediate frames for inspection.

## Conventions
- **Frame Processors**:
  - Add new processors in `roop/processors/frame/`.
  - Ensure the processor implements the required interface (`pre_check`, `process_image`, etc.).

- **Global Variables**:
  - Use `roop/globals.py` for runtime configurations.

- **Logging**:
  - Use `update_status` for status updates.
  - Debugging logs are printed to the console.

## Integration Points
- **External Dependencies**:
  - Models are downloaded dynamically (e.g., `inswapper_128.onnx`, `GFPGANv1.4.pth`).
  - Ensure `ffmpeg` is installed and available in the system PATH.

- **Pre-trained Models**:
  - Stored in the `models/` directory.
  - Downloaded automatically if missing.

- **Third-party Libraries**:
  - Key dependencies: `tensorflow`, `onnxruntime`, `insightface`, `gfpgan`.
  - See `requirements.txt` and `requirements-headless.txt` for details.

## Examples
- **Face Swapping**:
  ```bash
  python run.py -s source_image.png -t target_video.mp4 -o output_video.mp4
  ```
- **Enhancing Faces**:
  ```bash
  python run.py -s source_image.png -t target_image.png -o output_image.png --frame-processor face_enhancer
  ```

## Notes
- The project is discontinued and no longer maintained. Use responsibly and adhere to ethical guidelines.
- Refer to the [documentation](https://github.com/s0md3v/roop/wiki) for more details.