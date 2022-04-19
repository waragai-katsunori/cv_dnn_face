from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent

DETECT_WEIGHTS = BASE_DIR / "model" / "yunet.onnx"
RECOG_WEIGHTS = BASE_DIR / "model" / "face_recognizer_fast.onnx"


def parse_yunet_face(face):
    bbox = [int(a) for a in face[0:4]]
    landmarks = [int(a) for a in face[4:14]]
    landmarks = np.array_split(landmarks, len(landmarks) / 2)
    confidence = face[14]
    return bbox, landmarks, confidence

def parse_yunet_faces(faces):
    if faces is None:
        return [], [], []
    bboxes = [face[0:4] for face in faces]
    landmarks = [face[4:14] for face in faces]
    confidences = [face[14] for face in faces]
    return bboxes, landmarks, confidences


@dataclass
class YunetFaceDetector:
    detect_weights = DETECT_WEIGHTS

    def __post_init__(self):
        if self.detect_weights.is_file():
            self.face_detector = cv2.FaceDetectorYN.create(str(self.detect_weights), "", (0, 0))
        else:
            print(f"error: missing {self.detect_weights}")
            print("Please download model onnx files")
            exit()

    def detect(self, image: np.ndarray):
        image = as_bgr(image)
        height, width, _ = image.shape
        self.face_detector.setInputSize((width, height))

        return self.face_detector.detect(image)


def as_bgr(image: np.ndarray) -> np.ndarray:
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


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
