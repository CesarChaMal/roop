import os
import cv2
import dlib
import numpy as np

# Resolve absolute path
PREDICTOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'shape_predictor_68_face_landmarks.dat'))

# ❗️Fail early if not found
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"Missing landmark model at {PREDICTOR_PATH}")

# Load model
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None
    shape = predictor(gray, rects[0])
    return np.array([[p.x, p.y] for p in shape.parts()])

def warp_expression(source, target):
    src_landmarks = get_landmarks(source)
    tgt_landmarks = get_landmarks(target)
    if src_landmarks is None or tgt_landmarks is None:
        return source

    warp_matrix = cv2.estimateAffinePartial2D(src_landmarks, tgt_landmarks)[0]
    warped = cv2.warpAffine(source, warp_matrix, (target.shape[1], target.shape[0]))
    return warped
