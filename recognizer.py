from pathlib import Path

import numpy as np
import cv2

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128


BASE_DIR = Path(__file__).parent
from common import RECOG_WEIGHTS, YunetFaceDetector, parse_yunet_face


def match(recognizer, feature1: np.ndarray, face_db: np.ndarray):
    """
    return isSamePerson, (user_id_string, score)
    """
    for element in face_db:
        user_id, feature2 = element
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score > COSINE_THRESHOLD:
            return True, (user_id, score)
    return False, ("", 0.0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="face_recognizer")
    parser.add_argument("img_file", help="image_file or camera_number")
    args = parser.parse_args()

    name = args.img_file
    if Path(name).suffix in (".jpg", ".jpeg", ".png"):
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

    face_detector = YunetFaceDetector()

    face_recognizer = cv2.FaceRecognizerSF_create(str(RECOG_WEIGHTS), "")

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        image = cv2.flip(image, 1)
        height, width, _ = image.shape
        t1 = cv2.getTickCount()
        result, faces = face_detector.detect(image)
        t2 = cv2.getTickCount()
        used = 1000 * (t2 - t1) / cv2.getTickFrequency()  # [sec]
        faces = faces if faces is not None else []

        cv2.namedWindow("face recognition", cv2.WINDOW_NORMAL)

        used_recog = 0
        for face in faces:
            box, landmarks, confidence = parse_yunet_face(face)
            t1 = cv2.getTickCount()
            aligned_face = face_recognizer.alignCrop(image, face)
            feature = face_recognizer.feature(aligned_face)

            result, user = match(face_recognizer, feature, face_db)
            t2 = cv2.getTickCount()
            used_recog += 1000 * (t2 - t1) / cv2.getTickFrequency()  # [sec]

            # box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # landmarks = list(map(int, face[4 : len(face) - 1]))

            for landmark in landmarks:
                radius = 5
                thickness = 2
                landmark = tuple(landmark)
                cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)

            id, score = user if result else ("unknown", 0.0)
            text = "{0} ({1:.2f})".format(id, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(
                image, text, position, font, scale, color, thickness, cv2.LINE_AA
            )

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image,
            f"detect recog[s] {used:.1f} {used_recog:.1f}",
            (int(0.05 * width), int(0.9 * height)),
            font,
            1.0,
            (0, 0, 255),
            3,
        )
        cv2.imshow("face recognition", image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
