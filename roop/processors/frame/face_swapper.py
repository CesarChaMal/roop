from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.status_utils import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video
from roop.processors.frame.core import core_process_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

class FaceSwapper:
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

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


# def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
#     if roop.globals.many_faces:
#         many_faces = get_many_faces(temp_frame)
#         if many_faces:
#             for target_face in many_faces:
#                 temp_frame = swap_face(source_face, target_face, temp_frame)
#     else:
#         target_face = find_similar_face(temp_frame, reference_face)
#         if target_face:
#             temp_frame = swap_face(source_face, target_face, temp_frame)
#     return temp_frame

def process_frame(source_faces: List[Face], reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        target_faces = get_many_faces(temp_frame)
        for i, target_face in enumerate(target_faces):
            if i < len(source_faces):
                temp_frame = swap_face(source_faces[i], target_face, temp_frame)
            else:
                temp_frame = swap_face(source_faces[0], target_face, temp_frame)  # fallback to first
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_faces[0], target_face, temp_frame)
    return temp_frame

# def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
#     image = cv2.imread(source_path)
#     if image is None:
#         raise ValueError(f"[ERROR] Cannot read image from source_path: {source_path}")
#     source_face = get_one_face(image)
#     reference_face = None if roop.globals.many_faces else get_face_reference()
#     for temp_frame_path in temp_frame_paths:
#         temp_frame = cv2.imread(temp_frame_path)
#         result = process_frame(source_face, reference_face, temp_frame)
#         cv2.imwrite(temp_frame_path, result)
#         if update:
#             update()

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_faces = []

    # 🔄 Handle multiple source paths split by ";"
    for path in source_path.split(';'):
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Cannot read image from {path}")
            continue
        faces = get_many_faces(image)
        if faces:
            source_faces.extend(faces)
        else:
            print(f"[WARN] No faces found in {path}")

    if not source_faces:
        raise ValueError("[ERROR] No source faces loaded. Aborting.")

    # 📌 Use reference face only if not doing many-faces mode
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(f"[WARN] Could not read frame {temp_frame_path}")
            continue
        result = process_frame(source_faces, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


# def process_image(source_path: str, target_path: str, output_path: str) -> None:
#     source_face = get_one_face(cv2.imread(source_path))
#     target_frame = cv2.imread(target_path)
#     reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
#     result = process_frame(source_face, reference_face, target_frame)
#     cv2.imwrite(output_path, result)

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    # Read the source and target images
    source_face = get_one_face(cv2.imread(source_path))
    print(f"Source Image Read: {'Success' if source_face is not None else 'Failed'}")
    target_frame = cv2.imread(target_path)
    print(f"Target Image Read: {'Success' if target_frame is not None else 'Failed'}")

    # Check if processing many faces or just a specific one
    if roop.globals.many_faces:
        # Process every face found in the target image
        many_faces = get_many_faces(target_frame)
        if many_faces:
            for target_face in many_faces:
                target_frame = swap_face(source_face, target_face, target_frame)
    else:
        # Process a specific face based on the reference face position
        reference_face = get_one_face(target_frame, roop.globals.reference_face_position)
        print(f"Faces detected in Source: {'Yes' if reference_face is not None else 'No'}")
        if reference_face:
            target_frame = swap_face(source_face, reference_face, target_frame)

    # Save the result to the output path
    cv2.imwrite(output_path, target_frame)

# def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
#     if not roop.globals.many_faces and not get_face_reference():
#         reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
#         reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
#         set_face_reference(reference_face)
#     roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

# def process_video(source_path: str, target_path: str, temp_frame_paths: List[str]) -> None:
#     print("[FaceSwapper] Starting process_video")
#     print(f"Source Path: {source_path}")
#     print(f"Number of frames to process: {len(temp_frame_paths)}")
#     print(f"Many Faces Mode: {roop.globals.many_faces}")
#
#     if not roop.globals.many_faces and not get_face_reference():
#         print("[FaceSwapper] No face reference found, extracting from reference frame.")
#         reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
#         reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
#         set_face_reference(reference_face)
#         print(f"[FaceSwapper] Reference Face Set: {'Success' if reference_face is not None else 'Failed'}")
#
#     print("[FaceSwapper] Processing frames...")
#
#     for temp_frame_path in temp_frame_paths:
#         temp_frame = cv2.imread(temp_frame_path)
#         result = process_frame(get_one_face(cv2.imread(source_path)), get_face_reference(), temp_frame)
#         cv2.imwrite(temp_frame_path, result)
#
#     print("[FaceSwapper] Finished processing frames")

def process_video(source_path: str, target_path: str, temp_frame_paths: List[str]) -> None:
    print("[FaceSwapper] Starting process_video")
    print(f"Source Path: {source_path}")
    print(f"Number of frames to process: {len(temp_frame_paths)}")

    # Load source face once
    # source_image = cv2.imread(source_path)
    # source_face = get_one_face(source_image)

    # if not source_face:
    #     print("[FaceSwapper] No face found in source image!")
    #     return

    # Load source face Multiple
    source_paths = source_path.split(';')  # semi-colon delimited multiple paths
    source_faces = []

    for spath in source_paths:
        image = cv2.imread(spath)
        faces = get_many_faces(image)
        source_faces.extend(faces)

    if not source_faces:
        print("No source faces found!")
        return

    # Set face reference if not already set
    if not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        if reference_face is not None:
            set_face_reference(reference_face)
            print("[FaceSwapper] Reference Face Set: Success")
        else:
            print("[FaceSwapper] Reference Face Set: Failed")
            return

    # Get reference face
    ref_face = get_face_reference()

    print("[FaceSwapper] Processing frames...")

    # for temp_frame_path in temp_frame_paths:
    #     temp_frame = cv2.imread(temp_frame_path)
    #     result = process_frame(source_face, ref_face, temp_frame)
    #     cv2.imwrite(temp_frame_path, result)

    # def face_swap(source_path: str, target_path: str, frame: Frame) -> Frame:
    #     return process_frame(source_face, ref_face, frame)

    # core_process_video(source_path, target_path, temp_frame_paths, face_swap, is_framewise=True)

    def face_swap(source_path: str, target_path: str, frame: Frame) -> Frame:
        return process_frame(source_faces, ref_face, frame)

    core_process_video(source_path, target_path, temp_frame_paths, face_swap, is_framewise=True)

    print("[FaceSwapper] Finished processing frames")
