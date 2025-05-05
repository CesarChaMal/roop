from typing import Any, List, Callable
import cv2
import os
import datetime
import insightface
import threading
import numpy as np
import roop.globals
import roop.processors.frame.core
from roop.status_utils import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video
from roop.processors.frame.core import core_process_video
from roop.landmark_utils import warp_expression, extract_landmarks, is_clone_successful, create_face_hull_mask, match_histogram_colors

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

def unwrap_array(obj):
    while isinstance(obj, tuple):
        obj = obj[0]
    return obj

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def swap_face_with_expression(source_face: Face, target_face: Face, original_frame: Frame, temp_frame: Frame, index: int = 0, prefix: str = "") -> Frame:
    print("[DEBUG] Warping expression from source to match target...")

    source_face.image = original_frame
    target_face.image = temp_frame
    warped_image = warp_expression(source_face, target_face, debug_name=f"{prefix}_{index:04d}")

    warped_source_face = Face(source_face)
    warped_source_face.image = warped_image

    warped_landmarks = extract_landmarks(warped_source_face, fallback_image=warped_image)
    if warped_landmarks is not None:
        warped_source_face.landmarks = warped_landmarks.tolist()
    else:
        print("[DEBUG] warped_source_face.landmarks is still None after extraction.")

    print("[DEBUG] Performing face swap with expression-preserved source...")
    swapped = swap_face(warped_source_face, target_face, temp_frame)

    os.makedirs("debug_output", exist_ok=True)

    try:
        # Immediately after face swapping, re-extract accurate landmarks from the swapped face
        # swapped_face_obj = Face(target_face)
        # swapped_face_obj.image = swapped
        # swapped_landmarks = extract_landmarks(swapped_face_obj, fallback_image=swapped)

        swapped_face_obj = Face()
        swapped_face_obj.image = swapped
        swapped_landmarks = extract_landmarks(swapped_face_obj, fallback_image=swapped)

        if swapped_landmarks is not None:
            mask = create_face_hull_mask(swapped, swapped_landmarks)
            mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
            cv2.imwrite(f"debug_output/{prefix}_{index:04d}_mask_debug.png", mask)

            # Debug: Draw landmarks directly to verify visually
            debug_landmarks_img = swapped.copy()
            for (x, y) in swapped_landmarks:
                cv2.circle(debug_landmarks_img, (int(x), int(y)), 2, (0,255,0), -1)
            cv2.imwrite(f"debug_output/{prefix}_{index:04d}_landmarks_debug.png", debug_landmarks_img)

            matched_swapped = match_histogram_colors(swapped, temp_frame, mask)

            # Verify and ensure data types and shapes explicitly
            matched_swapped = np.array(matched_swapped, dtype=np.uint8)
            temp_frame = np.array(temp_frame, dtype=np.uint8)
            mask = np.array(mask, dtype=np.uint8)

            print(f"[DEBUG] matched_swapped shape: {matched_swapped.shape}")
            print(f"[DEBUG] temp_frame shape: {temp_frame.shape}")
            print(f"[DEBUG] mask shape: {mask.shape}")

            # Use robust bounding rectangle method to find center
            x, y, w, h = cv2.boundingRect(mask)
            center = (x + w // 2, y + h // 2)
            # center = tuple(np.mean(swapped_landmarks, axis=0).astype(np.int32))

            result = cv2.seamlessClone(matched_swapped, temp_frame, mask, center, cv2.MIXED_CLONE)

            cv2.imwrite(f"debug_output/{prefix}_{index:04d}_swapped_face.png", matched_swapped)
            cv2.imwrite(f"debug_output/{prefix}_{index:04d}_clone.png", result)

            if is_clone_successful(matched_swapped, result, mask):
                print("[DEBUG] Seamless cloning successful with convex hull mask.")
            else:
                print("[WARN] SeamlessClone produced minimal or invalid result, using matched_swapped.")
                result = matched_swapped
        else:
            print("[ERROR] Couldn't extract landmarks from swapped face!")
            result = swapped

    except Exception as e:
        print(f"[WARN] SeamlessClone failed ({e}), returning direct swap result.")
        result = swapped

    cv2.imwrite(f"debug_output/{prefix}_{index:04d}_result.png", result)
    print("[DEBUG] Expression-preserving face swap complete.")
    return result

def process_frame(source_faces: List[Face], reference_face: Face, temp_frame: Frame, frame_index: int = 0) -> Frame:
    original_frame = cv2.imread(roop.globals.source_path)
    if original_frame is None:
        print("[ERROR] Failed to read original source frame.")
        return temp_frame  # or raise an Exception if critical

    if roop.globals.many_faces:
        target_faces = get_many_faces(temp_frame)
        for i, target_face in enumerate(target_faces):
            index = frame_index * 100 + i
            src_face = source_faces[i] if i < len(source_faces) else source_faces[0]

            temp_frame = swap_face_with_expression(
                src_face, target_face,
                original_frame=original_frame,
                temp_frame=temp_frame,
                index=index,
                prefix="video"
            ) if roop.globals.preserve_expressions else swap_face(src_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face_with_expression(
                source_faces[0], target_face,
                original_frame=original_frame,
                temp_frame=temp_frame,
                index=frame_index,
                prefix="video"
            ) if roop.globals.preserve_expressions else swap_face(source_faces[0], target_face, temp_frame)
    return temp_frame

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

    for frame_index, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(f"[WARN] Could not read frame {temp_frame_path}")
            continue
        result = process_frame(source_faces, reference_face, temp_frame, frame_index=frame_index)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    # Read the source and target images
    source_image = cv2.imread(source_path)
    source_face = get_one_face(source_image)
    print(f"Source Image Read: {'Success' if source_face is not None else 'Failed'}")

    target_frame = cv2.imread(target_path)
    print(f"Target Image Read: {'Success' if target_frame is not None else 'Failed'}")

    if roop.globals.many_faces:
        many_faces = get_many_faces(target_frame)
        if many_faces:
            for target_face in many_faces:
                if roop.globals.preserve_expressions:
                    target_frame = swap_face_with_expression(
                        source_face,
                        target_face,
                        original_frame=source_image,
                        temp_frame=target_frame,
                        index=0,
                        prefix="image"
                    )
                else:
                    target_frame = swap_face(source_face, target_face, target_frame)
    else:
        reference_face = get_one_face(target_frame, roop.globals.reference_face_position)
        print(f"Faces detected in Target: {'Yes' if reference_face is not None else 'No'}")
        if reference_face:
            if roop.globals.preserve_expressions:
                target_frame = swap_face_with_expression(
                    source_face,
                    reference_face,
                    original_frame=source_image,
                    temp_frame=target_frame,
                    index=0,
                    prefix="image"
                )
            else:
                target_frame = swap_face(source_face, reference_face, target_frame)

    # Save the result
    cv2.imwrite(output_path, target_frame)

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
    def face_swap(source_path: str, target_path: str, frame: Frame) -> Frame:
        return process_frame(source_faces, ref_face, frame)

    core_process_video(source_path, target_path, temp_frame_paths, face_swap, is_framewise=True)

    print("[FaceSwapper] Finished processing frames")
