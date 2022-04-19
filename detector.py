from pathlib import Path

import cv2

from common import YunetFaceDetector, parse_yunet_face


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

    face_detector = YunetFaceDetector()

    cv2.namedWindow("face detection", cv2.WINDOW_NORMAL)

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        image = cv2.flip(image, 1)

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []
        # draw detections
        for face in faces:
            box, landmarks, confidence = parse_yunet_face(face)
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # landmark（right_eye, left_eye, nose, right corner of the mouth, left corner of the mouth）
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)

            confidence_str = f"{confidence:.2f}"
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            cv2.putText(
                image, confidence_str, position, font, scale, color, thickness, cv2.LINE_AA
            )

        cv2.imshow("face detection", image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()