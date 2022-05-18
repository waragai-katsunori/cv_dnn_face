from pathlib import Path


import cv2
import numpy as np


from cv_dnn_face import enlarge, YunetFaceDetector

face_detector = YunetFaceDetector()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="face cropper from images.")
    parser.add_argument("src_dir", help="image source dir")
    parser.add_argument("dst_dir", help="image source dir")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir) / f"{src_dir.name}_crop"

    names = sorted(list(src_dir.glob("**/*.jpg")) + list(src_dir.glob("**/*.png")))

    for i, p in enumerate(names):
        print(i, p)
        img = cv2.imread(str(p))
        img = np.rot90(img, 3)
        h, w = img.shape[:2]

        img = np.array(img[int(0.3 * h) :, :, :]).copy()

        _, faces = face_detector.detect(img)
        for i, face in enumerate(faces):
            box, landmarks, confidence = face_detector.object_parser(face)
            x, y, w, h = box
            top, right, bottom, left = y, x + w, y + h, x
            top, right, bottom, left = enlarge(top, right, bottom, left, img.shape)
            face = img[top:bottom, left:right, :]
            print(top, right, bottom, left)
            tmpname = dst_dir / p.relative_to(src_dir)
            dstname = tmpname.parent / f"{tmpname.stem}_{i}.jpg"

            dstname.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(dstname), face)
