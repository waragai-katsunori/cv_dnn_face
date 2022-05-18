from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).parent
from cv_dnn_face import enlarge, YunetFaceDetector

global yunet_face_detector
face_detector = YunetFaceDetector()


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
        print(p)
        img = cv2.imread(str(p))
        if clockwise:
            img = np.rot90(img, 3)

        _, faces = face_detector.detect(img)
        for i, face in enumerate(faces):
            box, landmarks, confidence = face_detector.object_parser(face)
            x, y, w, h = box
            top, right, bottom, left = y, x + w, y + h, x
            top, right, bottom, left = enlarge(top, right, bottom, left, img.shape)
            face = img[top:bottom, left:right, :]
            print(p, top, right, bottom, left)
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
    print(f"saved to {dst_dir}")
