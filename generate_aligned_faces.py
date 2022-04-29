import argparse
from pathlib import Path

import numpy as np
import cv2

BASE_DIR = Path(__file__).parent
from common import RECOG_WEIGHTS, YunetFaceDetector

def main():
    parser = argparse.ArgumentParser("generate aligned face images from an image")
    parser.add_argument("image", help="input image file path (./image.jpg)")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if args.recursive:
        paths = list(Path(args.image).glob("**/*.jpg"))
    else:
        paths = [Path(args.image)]

    face_detector = YunetFaceDetector()
    face_recognizer = cv2.FaceRecognizerSF_create(str(RECOG_WEIGHTS), "")

    for p in paths:
        image = cv2.imread(str(p))
        if image is None:
            continue

        label = p.parent.name
        height, width, _ = image.shape
        _, faces = face_detector.detect(image)

        aligned_faces = (
            [face_recognizer.alignCrop(image, face) for face in faces]
            if faces is not None
            else []
        )

        for i, aligned_face in enumerate(aligned_faces):
            cv2.imshow("aligned_face", aligned_face)
            outname = BASE_DIR / "aligned_faces" / label / f"{p.stem}_{(i+1):03d}.jpg"
            outname.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(outname), aligned_face)

            face_feature = face_recognizer.feature(aligned_face)
            feature_file = outname.parent / outname.stem
            np.save(str(feature_file), face_feature)
            print(f"saved {feature_file}")

        cv2.waitKey(10)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
