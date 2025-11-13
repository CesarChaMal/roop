# CLAUDE.md - AI Assistant Guide for Roop

> **Last Updated**: 2025-11-13
> **Project Version**: 1.3.2
> **Status**: Discontinued (still functional, no active development)

## Project Overview

**Roop** is a face-swapping application that takes a video and replaces faces in it with a face of your choice. It requires only one image of the desired face - no dataset or training required.

**Ethical Note**: This software is designed for legitimate creative purposes (character animation, clothing models, etc.). The project has implemented measures to prevent inappropriate content creation. Users are expected to follow local laws and use the software responsibly.

### Key Technologies
- **Python 3.9+** - Primary language
- **ONNX Runtime** - Model inference (CPU/GPU)
- **InsightFace** - Face detection and recognition
- **PyTorch** - Deep learning framework
- **OpenCV** - Video/image processing
- **CustomTkinter** - GUI framework
- **FFmpeg** - Video encoding/decoding

---

## Repository Structure

```
roop/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline (linting, testing)
├── debug_output/                  # Debug output directory
├── roop/                          # Main package directory
│   ├── __init__.py
│   ├── core.py                    # Core application logic & entry point (298 lines)
│   ├── globals.py                 # Global configuration state (26 lines)
│   ├── typing.py                  # Type definitions (7 lines)
│   ├── metadata.py                # Project name & version (2 lines)
│   ├── ui.py                      # GUI implementation (290 lines)
│   ├── utilities.py               # Helper functions for video/audio (390 lines)
│   ├── face_analyser.py           # Face detection & analysis (54 lines)
│   ├── face_reference.py          # Face reference management (21 lines)
│   ├── predictor.py               # NSFW content detection (43 lines)
│   ├── capturer.py                # Face capture utilities (22 lines)
│   ├── landmark_utils.py          # Facial landmark processing (145 lines)
│   ├── status_utils.py            # Status reporting (3 lines)
│   └── processors/
│       └── frame/
│           ├── __init__.py
│           ├── core.py            # Frame processor interface & execution
│           ├── face_swapper.py    # Face swapping processor
│           ├── face_enhancer.py   # Face enhancement processor
│           └── face_enhancer_original.py  # Original enhancer implementation
├── run.py                         # Application entry point
├── run.sh                         # Shell script for running
├── requirements.txt               # Python dependencies (with CUDA support)
├── requirements-headless.txt      # Headless mode dependencies
├── .flake8                        # Flake8 linting configuration
├── mypy.ini                       # MyPy type checking configuration
├── .python-version                # Python version specification
├── README.md                      # User documentation
└── CONTRIBUTING.md                # Contribution guidelines
```

---

## Architecture & Design Patterns

### Core Principles

1. **Functional Programming Only**
   - No Object-Oriented Programming (OOP)
   - No classes (except for type definitions)
   - Functions and modules over objects
   - Pure functions where possible

2. **Plugin-Based Frame Processors**
   - Frame processors are dynamically loaded modules
   - Each processor must implement a standard interface:
     - `pre_check()` - Validate dependencies before execution
     - `pre_start()` - Initialize before processing
     - `process_frame()` - Process a single frame
     - `process_frames()` - Batch process frames
     - `process_image()` - Process static images
     - `process_video()` - Process video files
     - `post_process()` - Cleanup after processing

3. **Global State Management**
   - Configuration is stored in `roop.globals`
   - Module-level variables (not class instances)
   - Set once during argument parsing

### Application Flow

```
run.py
  └─> core.run()
       ├─> parse_args()              # Parse CLI arguments → set globals
       ├─> pre_check()               # Validate Python version & ffmpeg
       ├─> limit_resources()         # Set memory limits
       └─> start() OR ui.init()      # Headless or GUI mode
            └─> start()
                 ├─> load frame processors
                 ├─> pre_start() checks
                 ├─> Image → Image processing
                 │    └─> process_image()
                 └─> Image → Video processing
                      ├─> create_temp()
                      ├─> extract_frames()
                      ├─> process_video()
                      ├─> compile_video_from_frames()
                      ├─> add_audio_to_video()
                      └─> clean_temp()
```

### Key Modules Explained

#### `core.py`
- **Purpose**: Main application orchestration
- **Key Functions**:
  - `parse_args()` - CLI argument parsing and globals initialization
  - `start()` - Main processing pipeline
  - `pre_check()` - System requirements validation
  - `limit_resources()` - Memory and GPU configuration
  - `update_status()` - Status message handling (console/UI)

#### `globals.py`
- **Purpose**: Application-wide configuration state
- **Pattern**: Module-level variables (not constants)
- **Key Variables**:
  - `source_path`, `target_path`, `output_path` - File paths
  - `frame_processors` - Active processors list
  - `execution_providers` - ONNX runtime providers (CPU/CUDA)
  - `many_faces`, `preserve_expressions` - Processing flags
  - `multi_source_paths` - Support for multiple source images

#### `utilities.py`
- **Purpose**: Video/audio/frame manipulation helpers
- **Key Functions**:
  - `extract_frames()` - Extract video frames to temp directory
  - `compile_video_from_frames()` - Create video from frame sequence
  - `add_audio_to_video()` - Merge audio track with video
  - `detect_fps()` - Detect video frame rate
  - `create_temp()`, `clean_temp()` - Temporary directory management

#### `processors/frame/`
- **Purpose**: Pluggable frame processing modules
- **Current Processors**:
  - `face_swapper` - Swap faces using InsightFace models
  - `face_enhancer` - Enhance face quality using GFPGAN
- **Interface Contract**: All processors must implement the 7 standard methods

---

## Development Workflows

### Setting Up Development Environment

```bash
# 1. Clone repository
git clone <repository-url>
cd roop

# 2. Install Python 3.9+
# Check .python-version for exact version

# 3. Install dependencies
pip install -r requirements.txt          # With GPU support
# OR
pip install -r requirements-headless.txt # Headless only

# 4. Install development tools
pip install flake8 mypy

# 5. Verify ffmpeg is installed
ffmpeg -version
```

### Running the Application

```bash
# GUI mode
python run.py

# Headless mode (image to image)
python run.py -s source.jpg -t target.jpg -o output.jpg

# Headless mode (image to video)
python run.py -s source.jpg -t target.mp4 -o output.mp4 \
  --frame-processor face_swapper \
  --keep-fps \
  --execution-provider cuda

# Multiple source images
python run.py --multi-source -s "face1.jpg;face2.jpg" -t target.mp4 -o output.mp4

# Preserve expressions
python run.py -s source.jpg -t target.mp4 -o output.mp4 --preserve-expressions
```

### Testing

```bash
# Run linting
flake8 run.py roop

# Run type checking
mypy run.py roop

# Run integration test
python run.py -s .github/examples/source.jpg \
              -t .github/examples/target.mp4 \
              -o test_output.mp4
```

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and PR:

1. **Lint Job** (Ubuntu only):
   - Python 3.9
   - Flake8 linting
   - MyPy type checking

2. **Test Job** (Multi-OS):
   - Runs on: macOS, Ubuntu, Windows
   - Sets up ffmpeg
   - Installs headless dependencies
   - Processes example video
   - Validates output quality with PSNR comparison

---

## Code Quality Standards

### Linting: Flake8

Configuration in `.flake8`:
```ini
[flake8]
select = E3, E4, F
per-file-ignores = roop/core.py:E402,F401
```

- **E3**: Whitespace and indentation errors
- **E4**: Import errors
- **F**: PyFlakes errors (undefined names, unused imports, etc.)
- **Exceptions**: `core.py` has relaxed import rules due to environment setup

### Type Checking: MyPy

Configuration in `mypy.ini`:
```ini
[mypy]
check_untyped_defs = True
disallow_any_generics = True
disallow_untyped_calls = True
disallow_untyped_defs = True
ignore_missing_imports = True
strict_optional = False
```

**Requirements**:
- All functions must have type annotations
- No `Any` types for generics
- All function calls must be typed
- Missing third-party stubs are ignored

### Type Definitions

Located in `roop/typing.py`:
```python
from insightface.app.common import Face
import numpy

Face = Face                          # InsightFace face object
Frame = numpy.ndarray[Any, Any]      # Video/image frame
```

---

## Contribution Guidelines (from CONTRIBUTING.md)

### DO:
- ✅ Fix bugs over adding features
- ✅ One pull request per feature/improvement
- ✅ Consult about implementation details before coding
- ✅ Test thoroughly before submission
- ✅ Resolve all CI pipeline failures

### DON'T:
- ❌ Introduce fundamental architecture changes
- ❌ Introduce OOP - functional programming only
- ❌ Ignore given requirements or work around them
- ❌ Submit code without consulting maintainers
- ❌ Submit massive code changes at once
- ❌ Submit proof-of-concept code
- ❌ Use undocumented or private APIs
- ❌ Try to solve third-party library issues in this project
- ❌ Comment what your code does - use proper naming instead

---

## Key Conventions for AI Assistants

### When Making Changes

1. **Preserve Functional Style**
   - Do not introduce classes or OOP patterns
   - Keep functions pure and side-effect-free where possible
   - Use module-level state via `roop.globals` for configuration

2. **Follow Type Annotations**
   - All new functions must have full type hints
   - Use `from typing import List, Optional` etc.
   - Avoid `Any` unless absolutely necessary
   - Reference `roop.typing.Face` and `roop.typing.Frame` for domain types

3. **Maintain Frame Processor Interface**
   - New processors go in `roop/processors/frame/`
   - Must implement all 7 required methods
   - Follow naming convention: `method_name.py`
   - Register in frame processor loading system

4. **Use Proper Naming**
   - Self-documenting function and variable names
   - No comments explaining what code does (only why if necessary)
   - Snake_case for functions and variables
   - ALL_CAPS for module-level constants

5. **Handle Paths Correctly**
   - Support multi-source mode via `roop.globals.multi_source_paths`
   - Use `os.path.join()` for path construction
   - Normalize paths with `normalize_output_path()`

6. **Memory Management**
   - Respect `roop.globals.max_memory` limits
   - Clean up temporary frames via `clean_temp()`
   - Use `limit_resources()` constraints

### Testing Your Changes

```bash
# Before committing, always run:
flake8 run.py roop
mypy run.py roop

# Test with example data:
python run.py -s .github/examples/source.jpg \
              -t .github/examples/target.mp4 \
              -o output.mp4
```

### Common Pitfalls to Avoid

1. **Don't break headless mode** - Many users run without GUI
2. **Don't assume GPU availability** - Support CPU-only execution
3. **Don't skip audio handling** - Preserve or explicitly skip audio
4. **Don't ignore FPS** - Use `--keep-fps` logic correctly
5. **Don't leave temp files** - Always call `clean_temp()`

---

## Dependencies & Hardware Acceleration

### Core Dependencies

```python
# Deep Learning & Models
torch==2.1.2+cu118              # PyTorch with CUDA 11.8
onnxruntime-gpu==1.16.3         # ONNX Runtime with GPU
insightface==0.7.3              # Face detection/recognition
gfpgan==1.3.8                   # Face enhancement
tensorflow==2.14.0              # TensorFlow (for some models)

# Computer Vision
opencv-python==4.8.0.74         # OpenCV
pillow==10.0.0                  # Image processing

# GUI
customtkinter==5.2.0            # Modern Tkinter UI
tkinterdnd2                     # Drag & drop (platform-specific)

# Utilities
numpy==1.26.4
tqdm==4.65.0                    # Progress bars
psutil==5.9.5                   # System monitoring
```

### Execution Providers (ONNX Runtime)

Available providers (auto-detected):
- `cpu` - CPU-only execution (default fallback)
- `cuda` - NVIDIA GPU acceleration (requires CUDA 11.8)
- `coreml` - macOS Core ML (Intel Macs)
- `silicon` - macOS Silicon (M1/M2 Macs)

Set via `--execution-provider` argument.

---

## File Processing Pipeline Details

### Image-to-Image Mode

```python
# roop/core.py:181-207
1. Copy target to output path
2. For each frame processor:
   a. process_image(sources, target, output_path)
   b. post_process()
3. Validate output is valid image
```

### Image-to-Video Mode

```python
# roop/core.py:209-277
1. create_temp(target_path)                    # Create temp directory
2. extract_frames(target, fps)                 # Extract all frames
3. For each frame processor:
   a. process_video(sources, target, frames)   # Batch process frames
   b. post_process()
4. rename_frames_sequentially()                # Normalize frame names
5. compile_video_from_frames(frames, output)   # Encode video
6. add_audio_to_video(output, target)          # Restore audio track
7. clean_temp(target_path)                     # Remove temp files
```

### Multi-Threading

Frame processing uses `ThreadPoolExecutor`:
- Threads controlled by `--execution-threads` (default: 1 for CPU, 8 for GPU)
- Frames distributed across thread pool
- Progress tracked with `tqdm`
- Memory usage monitored via `psutil`

---

## Debugging & Development Tips

### Debug Output

- Debug directory: `debug_output/`
- Intermediate results saved when debugging
- Check `temp_swap_result.png` for intermediate face swaps

### Enable Debug Logging

```python
# core.py includes debug statements:
print("[DEBUG] source_path =", roop.globals.source_path)
print("[DEBUG] multi_source_paths =", roop.globals.multi_source_paths)
```

### Common Issues

1. **Import errors**: Check environment variables set in `core.py:6-10`
2. **Memory errors**: Reduce `--max-memory` or `--execution-threads`
3. **FFmpeg errors**: Ensure ffmpeg is installed and in PATH
4. **GPU errors**: Check CUDA version matches PyTorch build (11.8)
5. **Face detection fails**: Check `--similar-face-distance` threshold

### Environment Variables

Set before importing torch/tensorflow:
```python
os.environ['OMP_NUM_THREADS'] = '1'          # Optimize CUDA performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'     # Reduce TF logging
```

---

## Project Status & Maintenance

**Current Status**: Discontinued (as of July-August 2023)
- Software remains functional and usable
- No new features or updates planned
- Community forks may continue development

**When Helping Users**:
- The software is feature-complete but unmaintained
- Focus on bug fixes and stability over new features
- Direct users to community Discord for installation support
- Respect the ethical guidelines and intended use cases

---

## Quick Reference

### File Locations

| Purpose | Location |
|---------|----------|
| Entry point | `run.py` |
| Main logic | `roop/core.py` |
| Configuration | `roop/globals.py` |
| Types | `roop/typing.py` |
| GUI | `roop/ui.py` |
| Video/Audio utils | `roop/utilities.py` |
| Frame processors | `roop/processors/frame/*.py` |
| Linting config | `.flake8` |
| Type checking | `mypy.ini` |
| CI/CD | `.github/workflows/ci.yml` |

### Important Globals

```python
roop.globals.source_path          # Source face image
roop.globals.target_path          # Target video/image
roop.globals.output_path          # Output file path
roop.globals.frame_processors     # Active processors list
roop.globals.execution_providers  # CPU/CUDA/etc
roop.globals.many_faces          # Process all faces vs first
roop.globals.preserve_expressions # Preserve facial expressions
roop.globals.multi_source_paths  # Multiple source images
```

### Command Examples

```bash
# Basic face swap
python run.py -s face.jpg -t video.mp4 -o result.mp4

# GPU acceleration
python run.py -s face.jpg -t video.mp4 -o result.mp4 --execution-provider cuda

# Process all faces
python run.py -s face.jpg -t video.mp4 -o result.mp4 --many-faces

# High quality with enhancement
python run.py -s face.jpg -t video.mp4 -o result.mp4 \
  --frame-processor face_swapper face_enhancer \
  --keep-fps --output-video-quality 80

# Multiple sources
python run.py --multi-source -s "face1.jpg;face2.jpg" -t video.mp4 -o result.mp4
```

---

## Summary for AI Assistants

When working with this codebase:

1. **Architecture**: Functional programming only, no OOP
2. **State**: Use `roop.globals` for configuration
3. **Types**: Full type hints required (MyPy strict mode)
4. **Linting**: Pass flake8 with configured rules
5. **Processors**: Follow 7-method interface pattern
6. **Testing**: Run CI checks before submitting
7. **Naming**: Self-documenting names, minimal comments
8. **Ethics**: Respect intended use cases and legal constraints

The codebase is well-structured, type-safe, and follows consistent patterns. Focus on maintaining the functional style and existing conventions when making changes.
