import cv2
import glob
import os
import subprocess
import threading
from gfpgan.utils import GFPGANer
from typing import Any, List, Callable
import numpy as np
import roop.globals
import roop.processors.frame.core
from roop.face_analyser import get_many_faces, get_one_face
from roop.processors.frame.core import core_process_video
from roop.status_utils import update_status
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

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
    def process_video(source_path: str, target_path: str, frame_paths: list[str]) -> None:
        process_video(source_path, target_path, frame_paths)


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
    conditional_download(download_directory_path,
                         ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    return True


# def pre_start() -> bool:
#     if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
#         update_status('Select an image or video for target path.', NAME)
#         return False
#     return True

def pre_start() -> bool:
    print("[DEBUG] pre_start face_enhancer source_path =", roop.globals.source_path)
    print("[DEBUG] pre_start face_enhancer multi_source_paths =", roop.globals.multi_source_paths)
    source_paths = roop.globals.multi_source_paths if roop.globals.multi_source else [roop.globals.source_path]

    valid_sources = []
    for path in source_paths:
        if not is_image(path):
            update_status(f'Invalid image path: {path}', NAME)
            continue
        image = cv2.imread(path)
        if image is None:
            update_status(f'Could not load image: {path}', NAME)
            continue
        if not get_one_face(image):
            update_status(f'No face found in source image: {path}', NAME)
            continue
        valid_sources.append(path)

    if not valid_sources:
        update_status('Select at least one valid source image with a detectable face.', NAME)
        return False

    # Check target path validity
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False

    return True

def post_process() -> None:
    clear_face_enhancer()


def enhance_face_for(face: Face, frame: Frame, crop: np.ndarray = None) -> Frame:
    if crop is None:
        start_x, start_y, end_x, end_y = map(int, face['bbox'])
        padding_x = int((end_x - start_x) * 0.5)
        padding_y = int((end_y - start_y) * 0.5)

        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = min(frame.shape[1], end_x + padding_x)
        end_y = min(frame.shape[0], end_y + padding_y)

        crop = frame[start_y:end_y, start_x:end_x]

    if crop is None or crop.size == 0:
        print("[WARN] Crop is None or empty.")
        return frame

    if crop.shape[0] < 16 or crop.shape[1] < 16:
        print(f"[WARN] Crop too small: shape={crop.shape}")
        return frame

    with THREAD_SEMAPHORE:
        try:
            enhancer = get_face_enhancer()
            print(f"[DEBUG] enhancer={enhancer}, type={type(enhancer)}")
            print(f"[DEBUG] crop.shape={crop.shape}, crop.dtype={crop.dtype}")
            _, _, restored = enhancer.enhance(crop, paste_back=False)
        except Exception as e:
            print(f"[ERROR] Face enhancer threw exception: {e}")
            return frame

    if restored is not None:
        start_x, start_y = face['bbox'][:2]
        end_x = start_x + restored.shape[1]
        end_y = start_y + restored.shape[0]
        frame[start_y:end_y, start_x:end_x] = restored
        return frame
    else:
        print(f"[WARN] Face enhancer returned None for crop shape={crop.shape}")
        return frame


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

    # ✅ Safely rename all frames to %08d.png format
    from roop.utilities import rename_frames_sequentially
    rename_frames_sequentially(temp_frame_paths)


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    print("[FaceEnhancer] Starting process_image")
    print(f"[DEBUG] source_paths = {source_paths}")  # not used, just for consistency
    print(f"[DEBUG] target_path = {target_path}")

    image = cv2.imread(target_path)
    if image is None:
        update_status(f"Could not read target image: {target_path}", NAME)
        return

    # Enhance all faces in the image
    faces = get_many_faces(image)
    print(f"[DEBUG] Detected {len(faces)} face(s) to enhance")

    result = process_frame(None, None, image)

    # Save enhanced image
    cv2.imwrite(output_path, result)
    print(f"[FaceEnhancer] Saved enhanced image to {output_path}")

def process_video(source_paths: List[str], target_path: str, temp_frame_paths: List[str]) -> None:
    print("[FaceEnhancer] Starting process_video")
    print(f"[DEBUG] source_paths = {source_paths}")
    print(f"[DEBUG] number of frames = {len(temp_frame_paths)}")

    def enhance(_: str, __: str, frame: Frame) -> Frame:
        # Detect and enhance all faces in the frame
        target_faces = get_many_faces(frame)
        # print(f"[DEBUG] Enhancing {len(target_faces)} faces in frame")
        return process_frame(None, None, frame)

    core_process_video(
        source_paths[0],  # not used by `enhance`, still required by signature
        target_path,
        temp_frame_paths,
        enhance,
        is_framewise=True
    )

    print("[FaceEnhancer] Finished enhancing video frames")
