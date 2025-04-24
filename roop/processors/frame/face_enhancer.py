from typing import Any, List, Callable
import cv2
import threading
from gfpgan.utils import GFPGANer
import subprocess
import os
import glob

import roop.globals
import roop.processors.frame.core
from roop.status_utils import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video, get_temp_directory_path
FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'

class FaceEnhancer:
    @staticmethod
    def pre_check() -> bool:
        return pre_check()

    @staticmethod
    def pre_start() -> bool:
        return pre_start()

    @staticmethod
    def post_process() -> None:
        post_process()

    @staticmethod
    def process_image(source_path: str, target_path: str, output_path: str) -> None:
        process_image(source_path, target_path, output_path)

    @staticmethod
    def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
        process_video(source_path, temp_frame_paths)

def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def clear_face_enhancer() -> None:
    global FACE_ENHANCER

    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, temp_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=True
            )
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def compile_video_from_frames(frame_dir: str, output_video_path: str) -> None:
    print(f"[INFO] Compiling video from frames in {frame_dir} to {output_video_path}")

    # Ensure directory exists
    if not os.path.exists(frame_dir):
        print(f"[ERROR] Frame directory does not exist: {frame_dir}")
        return

    # Check PNGs exist
    frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
    if not frame_files:
        print(f"[ERROR] No PNG frames found in directory: {frame_dir}")
        return

    # Format check
    first_frame = os.path.basename(frame_files[0])
    if not first_frame.startswith("00000000"):
        print(f"[WARN] Frame filenames might not follow '%08d.png' format. Expected: 00000000.png ...")

    # Run ffmpeg
    command = [
        'ffmpeg', '-y', '-framerate', '24', '-i', f'{frame_dir}/%08d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video_path
    ]
    try:
        subprocess.run(command, check=True)
        print("[INFO] Video compilation complete")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed: {e}")


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)

    # Confirm the target folder exists
    frame_dir = get_temp_directory_path(roop.globals.target_path)
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)

    # compile_video_from_frames('/content/temp/target_video', '/content/swapped_video.mp4')
    # compile_video_from_frames(roop.globals.temp_directory, roop.globals.output_path)
    compile_video_from_frames(frame_dir, roop.globals.output_path)
