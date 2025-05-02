import os
import cv2
import dlib
import numpy as np
from roop.typing import Frame
from roop.face_analyser import get_many_faces
# Resolve absolute path
PREDICTOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'shape_predictor_68_face_landmarks.dat'))

if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"Missing landmark model at {PREDICTOR_PATH}")

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None
    shape = predictor(gray, rects[0])
    return np.array([[p.x, p.y] for p in shape.parts()])

def extract_landmarks(face):
    if hasattr(face, 'landmarks_2d') and face.landmarks_2d is not None:
        return np.array(face.landmarks_2d, dtype=np.float32)
    if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
        return np.array(face.landmark_3d_68[:, :2], dtype=np.float32)
    return None

def warp_expression(frame: Frame, original: Frame) -> Frame:
    target_faces = get_many_faces(frame)
    source_faces = get_many_faces(original)

    if not target_faces or not source_faces:
        print("[WARN] warp_expression: No faces found in frame or original.")
        return frame

    target = target_faces[0]
    source = source_faces[0]

    target_landmarks = extract_landmarks(target)
    source_landmarks = extract_landmarks(source)

    if target_landmarks is None or source_landmarks is None:
        print("[WARN] warp_expression: No valid landmarks found.")
        return frame

    if target_landmarks.ndim != 2 or source_landmarks.ndim != 2:
        print(f"[WARN] warp_expression: Landmarks are not 2D. Shapes: {target_landmarks.shape}, {source_landmarks.shape}")
        return frame
    if target_landmarks.shape[1] != 2 or source_landmarks.shape[1] != 2:
        print(f"[WARN] warp_expression: Landmark points are not (x,y) format.")
        return frame
    if target_landmarks.shape != source_landmarks.shape or target_landmarks.shape[0] < 3:
        print(f"[WARN] warp_expression: Invalid shape {target_landmarks.shape} vs {source_landmarks.shape}")
        return frame

    print("[DEBUG] Source landmarks:", source_landmarks[:5])
    print("[DEBUG] Target landmarks:", target_landmarks[:5])

    # Get convex hull indices
    hull_indices = cv2.convexHull(source_landmarks, returnPoints=False).flatten()
    src_points = source_landmarks[hull_indices]
    tgt_points = target_landmarks[hull_indices]

    # Compute affine transformation
    matrix, _ = cv2.estimateAffinePartial2D(tgt_points, src_points)
    if matrix is None:
        print("[WARN] warp_expression: Failed to compute affine transform.")
        return frame

    # Warp the frame
    warped = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))

    # Create a mask
    mask = np.zeros_like(frame[:, :, 0])
    cv2.fillConvexPoly(mask, np.int32(target_landmarks), 255)
    mask = cv2.merge([mask] * 3)


    # Blend the warped face
    if hasattr(target, 'bbox') and isinstance(target.bbox, (list, tuple, np.ndarray)) and len(target.bbox) >= 2:
        center_x, center_y = map(int, target.bbox[:2])
    else:
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2

    result = cv2.seamlessClone(warped, frame, mask, (center_x, center_y), cv2.NORMAL_CLONE)

    # Debug images
    try:
        def draw_landmarks(image, landmarks, color=(0, 255, 0)):
            for (x, y) in landmarks.astype(int):
                cv2.circle(image, (x, y), 2, color, -1)

        debug_original = original.copy()
        debug_frame = frame.copy()
        debug_result = result.copy()

        draw_landmarks(debug_original, source_landmarks, color=(255, 0, 0))
        draw_landmarks(debug_frame, target_landmarks, color=(0, 255, 0))
        draw_landmarks(debug_result, target_landmarks, color=(0, 0, 255))

        cv2.imwrite("debug_original_source.png", debug_original)
        cv2.imwrite("debug_before_expression.png", debug_frame)
        cv2.imwrite("debug_after_expression.png", debug_result)
    except Exception as e:
        print(f"[WARN] Debug drawing failed: {e}")

    return result
