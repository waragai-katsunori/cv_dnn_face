from pathlib import Path

import numpy as np
import cv2


from cv_dnn_face import YunetFaceDetector


BASE_DIR = Path(__file__).parent.parent
assert BASE_DIR.is_dir()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="face_recognizer")
    parser.add_argument("img_file", help="image_file, video_file or camera_number")
    args = parser.parse_args()

    name = args.img_file
    if Path(name).suffix in (".jpg", ".jpeg", ".png"):
        capture = cv2.VideoCapture(name)
    elif Path(name).suffix in (".mp4", ".avi", ".webm"):
        capture = cv2.VideoCapture(name)
    else:
        num = int(args.img_file)
        capture = cv2.VideoCapture(num)
    if not capture.isOpened():
        exit()

    # read face_db in aligned_faces
    # List[(user_id, feature)]
    face_db = [
        (file.stem, np.load(file))
        for file in (BASE_DIR / "aligned_faces").glob("*.npy")
    ]

    face_db2 = [
        (file.parent.name, np.load(file))
        for file in (BASE_DIR / "aligned_faces").glob("*/*.npy")
    ]

    face_db.extend(face_db2)

    face_detector = YunetFaceDetector()

    while True:
        result, img = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        img = cv2.flip(img, 1)
        height, width, _ = img.shape
        t1 = cv2.getTickCount()
        result, faces = face_detector.detect(img)
        t2 = cv2.getTickCount()
        used = 1000 * (t2 - t1) / cv2.getTickFrequency()  # [sec]
        faces = faces if faces is not None else []

        cv2.namedWindow("face recognition", cv2.WINDOW_NORMAL)

        used_recog = 0
        for face in faces:
            box, landmarks, confidence = face_detector.object_parser(face)
            t1 = cv2.getTickCount()
            feature = face_detector.get_feature(img, face)

            result, user = face_detector.search(feature, face_db)
            t2 = cv2.getTickCount()
            used_recog += 1000 * (t2 - t1) / cv2.getTickFrequency()  # [sec]

            # box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv2.rectangle(img, box, color, thickness, cv2.LINE_AA)

            # landmarks = list(map(int, face[4 : len(face) - 1]))

            for landmark in landmarks:
                radius = 5
                thickness = 2
                landmark = tuple(landmark)
                cv2.circle(img, landmark, radius, color, thickness, cv2.LINE_AA)

            id, score = user if result else ("unknown", 0.0)
            text = "{0} ({1:.2f})".format(id, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(img, text, position, font, scale, color, thickness, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img,
            f"detect recog[s] {used:.1f} {used_recog:.1f}",
            (int(0.05 * width), int(0.9 * height)),
            font,
            1.0,
            (0, 0, 255),
            3,
        )
        cv2.imshow("face recognition", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
