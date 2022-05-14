from pathlib import Path
import shutil

import cv2

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128

BASE_DIR = Path(__file__).parent
from cv_dnn_face import YunetFaceDetector

face_detector = YunetFaceDetector()


def slim_by_distance(target_dir: Path, dst_dir: Path, th: float, recursive=False):
    if recursive:
        names = sorted(
            list([p for p in target_dir.glob("**/*.png")]) + list([p for p in target_dir.glob("**/*.jpg")])
        )
    else:
        names = sorted(
            list([p for p in target_dir.glob("*.png")]) + list([p for p in target_dir.glob("*.jpg")])
        )  # random.shuffle(names)
    name_list = []
    encodings_list = []
    for i, p in enumerate(names):
        print(p)
        try:
            img = cv2.imread(str(p))
            if img is None:
                continue
            height, width, _ = img.shape
            result, faces = face_detector.detect(img)
            faces = faces if faces is not None else []

            encs = [face_detector.get_feature(img, face) for face in faces]
            #    print(encs)
            if encs:
                print(encs[0].shape)
                dist = [1.0 - face_detector.match(e, encs[0]) for e in encodings_list]
                print(p, len(dist) * "*")
                if len(dist) == 0:
                    encodings_list.append(encs[0])
                    name_list.append(p)
                else:
                    min_dist = min(dist)
                    if min_dist > th:
                        encodings_list.append(encs[0])
                        name_list.append(p)
                    else:
                        print("skip", p)
                        shutil.move(str(p), str(dst_dir))
        except cv2.error:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="face cropper from images.")
    parser.add_argument("src_dir", help="image source dir")
    parser.add_argument("dst_dir", help="destination dir")
    parser.add_argument(
        "--th", default=0.20, help="if face_distance is smaller than threshold, skips "
    )
    parser.add_argument("-r", action="store_true", help="recursive file search")
    args = parser.parse_args()

    target_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir) / f"{target_dir.name}_similar"
    dst_dir.mkdir(exist_ok=True, parents=True)

    assert target_dir.is_dir()

    th = float(args.th)
    recursive = args.r
    slim_by_distance(target_dir, dst_dir, th, recursive=recursive)
