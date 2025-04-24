from .face_swapper import FaceSwapper
from .face_enhancer import FaceEnhancer

PROCESSORS = {
    "face_swapper": FaceSwapper,
    "face_enhancer": FaceEnhancer,
}
