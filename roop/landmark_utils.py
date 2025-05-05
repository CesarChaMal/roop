import dlib
import numpy as np
from roop.typing import Frame, Face
from roop.face_analyser import get_many_faces
import datetime
import os
import cv2

# Resolve absolute path
PREDICTOR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'models', 'shape_predictor_68_face_landmarks.dat'))

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

def extract_landmarks(face, fallback_image: Frame = None):
    if hasattr(face, 'landmarks_2d') and face.landmarks_2d is not None:
        return np.array(face.landmarks_2d, dtype=np.float32)
    if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
        return np.array(face.landmark_3d_68[:, :2], dtype=np.float32)
    if fallback_image is not None:
        return get_landmarks(fallback_image)
    return None

def create_face_hull_mask(target_image, landmarks):
    mask = np.zeros(target_image.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.array(landmarks, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def match_histogram_colors(source, reference, mask=None):
    matched = source.copy()
    for i in range(3):
        src = source[:, :, i]
        ref = reference[:, :, i]
        if mask is not None:
            src = src[mask > 0]  # mask is single-channel, directly use mask > 0
            ref = ref[mask > 0]
        src_mean, src_std = np.mean(src), np.std(src)
        ref_mean, ref_std = np.mean(ref), np.std(ref)
        if src_std == 0:
            src_std = 1
        matched[:, :, i] = np.clip((source[:, :, i] - src_mean) * (ref_std / src_std) + ref_mean, 0, 255)
    return matched.astype(np.uint8)

def is_clone_successful(original: np.ndarray, cloned: np.ndarray, mask: np.ndarray, threshold: float = 15.0) -> bool:
    """
    Check if seamless cloning had a meaningful impact by comparing only the masked area.
    Improved: uses mean of all 3 channels and logs more details.
    """
    if original.shape != cloned.shape or original.shape[:2] != mask.shape[:2]:
        print("[WARN] Shape mismatch in clone success check.")
        return False

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Absolute difference across channels
    diff = cv2.absdiff(original, cloned)
    mean_vals = cv2.mean(diff, mask=mask)
    masked_diff = np.mean(mean_vals[:3])  # average over B, G, R

    # Optional: check standard deviation for more insight
    std_vals = [np.std(cv2.mean(cv2.absdiff(original[:, :, i], cloned[:, :, i]), mask=mask)) for i in range(3)]
    std_avg = np.mean(std_vals)

    print(f"[DEBUG] Mean pixel diff over mask (RGB avg): {masked_diff:.2f}, Std Dev: {std_avg:.2f}")

    return masked_diff < threshold

def warp_expression(source_face: Face, target_face: Face, debug_name: str = "") -> np.ndarray:
    source_img = source_face.image
    target_img = target_face.image

    src_landmarks = extract_landmarks(source_face, fallback_image=source_img)
    tgt_landmarks = extract_landmarks(target_face, fallback_image=target_img)

    if src_landmarks is None or tgt_landmarks is None:
        print("[WARN] warp_expression_face_only: Failed to extract landmarks.")
        return target_img.copy()

    warped = np.copy(target_img)

    def apply_affine_transform(src, src_tri, dst_tri, size):
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        return cv2.warpAffine(src, warp_mat, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def delaunay_triangulation(landmarks, size):
        subdiv = cv2.Subdiv2D((0, 0, size[1], size[0]))
        for p in landmarks:
            subdiv.insert((int(p[0]), int(p[1])))
        triangle_list = subdiv.getTriangleList()
        delaunay_tri = []
        for t in triangle_list:
            pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            indices = []
            for pt in pts:
                distances = np.linalg.norm(landmarks - np.array(pt), axis=1)
                idx = np.argmin(distances)
                if distances[idx] < 5.0:
                    indices.append(idx)
            if len(indices) == 3:
                delaunay_tri.append(tuple(indices))
        return delaunay_tri

    triangles = delaunay_triangulation(tgt_landmarks, target_img.shape)

    for tri in triangles:
        t1 = [src_landmarks[i] for i in tri]
        t2 = [tgt_landmarks[i] for i in tri]
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        t1_rect = [(p[0] - r1[0], p[1] - r1[1]) for p in t1]
        t2_rect = [(p[0] - r2[0], p[1] - r2[1]) for p in t2]

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

        src_crop = source_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        warped_patch = apply_affine_transform(src_crop, t1_rect, t2_rect, (r2[2], r2[3]))

        warped_area = warped[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]].astype(np.float32)
        blended = warped_area * (1 - mask) + warped_patch.astype(np.float32) * mask
        warped[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = np.clip(blended, 0, 255).astype(np.uint8)

    if debug_name:
        os.makedirs("debug_output", exist_ok=True)
        cv2.imwrite(f"debug_output/{debug_name}_source_face.png", source_img)
        cv2.imwrite(f"debug_output/{debug_name}_target_face.png", target_img)
        cv2.imwrite(f"debug_output/{debug_name}_warped_face_only.png", warped)

    return warped
