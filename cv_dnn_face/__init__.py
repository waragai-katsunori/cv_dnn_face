from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent

DETECT_WEIGHTS = BASE_DIR / "model" / "yunet.onnx"
RECOG_WEIGHTS = BASE_DIR / "model" / "face_recognizer_fast.onnx"

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128

class InitalizeFailedException(Exception):
    def __init__(self, arg=""):
        self.arg = arg

@dataclass
class YunetFaceDetector:
    """
    cv2.FaceDetectorYN
    cv2.FaceRecognizerSF
    base face recognition
    """
    detect_weights = DETECT_WEIGHTS

    def __post_init__(self):
        if self.detect_weights.is_file():
            self.face_detector = cv2.FaceDetectorYN.create(str(self.detect_weights), "", (0, 0))
            self.face_recognizer = cv2.FaceRecognizerSF_create(str(RECOG_WEIGHTS), "")

        else:
            print(f"error: missing {self.detect_weights}")
            print("Please download model onnx files")
            raise InitalizeFailedException

    def detect(self, img: np.ndarray):
        img = as_bgr(img)
        height, width, _ = img.shape
        self.face_detector.setInputSize((width, height))

        return self.face_detector.detect(img)

    def object_parser(self, obj):
        return _parse_yunet_face(obj)

    def alignCrop(self, img, face):
        return self.face_recognizer.alignCrop(img, face)

    def get_feature(self, img, face):
        aligned_face = self.face_recognizer.alignCrop(img, face)
        return self.face_recognizer.feature(aligned_face)

    def match(self, feature1, feature2):
        score = self.face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        return score

    def search(self, feature1: np.ndarray, face_db: np.ndarray):
        """
        return isSamePerson, (user_id_string, score)
        """
        for element in face_db:
            user_id, feature2 = element
            score = self.face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if score > COSINE_THRESHOLD:
                return True, (user_id, score)
        return False, ("", 0.0)


def _parse_yunet_face(face):
    bbox = [int(a) for a in face[0:4]]
    landmarks = [int(a) for a in face[4:14]]
    landmarks = np.array_split(landmarks, len(landmarks) / 2)
    confidence = face[14]
    return bbox, landmarks, confidence


def as_bgr(img: np.ndarray) -> np.ndarray:
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def enlarge(top: int, right: int, bottom: int, left: int, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    returns enlarged region.
    [top, right, bottom, left]
    width and height is twice of original region.
    """
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    w = right - left
    h = bottom - top
    new_top = max(int(cy - h), 0)
    new_left = max(int(cx - w), 0)
    new_bottom = min(new_top + 2 * h, shape[0])
    new_right = min(new_left + 2 * w, shape[1])
    return new_top, new_right, new_bottom, new_left


def as_top_right_bottom_left(bboxes: List) -> List[Tuple[int, int, int, int]]:
    r = []
    for x, y, w, h in bboxes:
        top = int(y)
        right = int(x + w)
        bottom = int(y + h)
        left = int(x)
        r.append((top, right, bottom, left))
    return r
