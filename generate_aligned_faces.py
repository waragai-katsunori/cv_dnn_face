import argparse
from pathlib import Path

import numpy as np
import cv2

BASE_DIR = Path(__file__).parent
from cv_dnn_face import YunetFaceDetector

def main():
    parser = argparse.ArgumentParser("generate aligned face images from an image")
    parser.add_argument("image", help="input image file path (./image.jpg)")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if args.recursive:
        paths = list([p for p in Path(args.image).glob("**/*.jpg")])
    else:
        paths = [p for p in Path(args.image).glob("*.jpg")]

    face_detector = YunetFaceDetector()

    for p in paths:
        img = cv2.imread(str(p))
        print(p)
        if img is None:
            continue

        label = p.parent.name
        height, width, _ = img.shape
        _, faces = face_detector.detect(img)

        for i, face in enumerate(faces):
            aligned_face = face_detector.alignCrop(img, face)
            cv2.imshow("aligned_face", aligned_face)
            outname = BASE_DIR / "aligned_faces" / label / f"{p.stem}_{(i+1):03d}.jpg"
            outname.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(outname), aligned_face)
            face_feature = face_detector.get_feature(img, face)
            feature_file = outname.parent / outname.stem
            np.save(str(feature_file), face_feature)
            print(f"saved {feature_file}")

        cv2.waitKey(10)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
