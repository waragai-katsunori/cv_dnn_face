from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).parent
from common import enlarge, parse_yunet_faces, YunetFaceDetector, as_top_right_bottom_left

global yunet_face_detector
yunet_face_detector = YunetFaceDetector()


def yunet_face_locations(image: np.ndarray):
    _, faces = yunet_face_detector.detect(image)
    bboxes, landmarks, confidences = parse_yunet_faces(faces)
    return as_top_right_bottom_left(bboxes)


def face_crop(src_dir: Path, dst_dir: Path, clockwise=False, recursive=True):
    """
    Crop faces in globed image files in src_dir.
    Faces are saved in dst_dir.
    If clockwise is True, rotate image clockwise before face detection.
    If recursive, glob image files recursive.
    """
    if recursive:
        names = sorted(
            (list(src_dir.glob("**/*.jpg")) + list(src_dir.glob("**/*.png")))
        )
    else:
        names = sorted((list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))))
    for p in names:
        img = cv2.imread(str(p))
        if clockwise:
            img = np.rot90(img, 3)

        face_bounding_boxes = yunet_face_locations(img)

        for i, (top, right, bottom, left) in enumerate(face_bounding_boxes):
            top, right, bottom, left = enlarge(top, right, bottom, left, img.shape)
            face = img[top:bottom, left:right, :]
            print(top, right, bottom, left)
            tmpname = dst_dir / p.relative_to(src_dir)
            dstname = tmpname.parent / f"{tmpname.stem}_{i}.jpg"

            dstname.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(dstname), face)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="face cropper from images.")
    parser.add_argument("dir", help="image source dir")
    parser.add_argument(
        "--clockwise", action="store_true", help="rotate image clockwize"
    )
    args = parser.parse_args()

    src_dir = Path(args.dir)
    dst_dir = src_dir.parent / f"{src_dir.name}_crop"
    clockwise = args.clockwise
    print(src_dir, dst_dir, clockwise)
    face_crop(src_dir, dst_dir, clockwise=clockwise)
