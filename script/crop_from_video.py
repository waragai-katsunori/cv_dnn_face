from pathlib import Path
import cv2

from cv_dnn_face import enlarge, YunetFaceDetector

global yunet_face_detector
face_detector = YunetFaceDetector()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="crop face from video")
    parser.add_argument("name", help="video file")
    parser.add_argument("--dst_dir", default="face_cropped", help="dst dir")
    parser.add_argument("--interval", default=4, help="frame interval to process")
    args = parser.parse_args()

    name = Path(args.name)
    interval = int(args.interval)

    cap = cv2.VideoCapture(str(name))

    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)

    counter = -1
    frame_num = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % interval == 0:
            _, faces = face_detector.detect(frame)
            faces = [] if faces is None else faces
            for i, face in enumerate(faces):
                box, landmarks, confidence = face_detector.object_parser(face)
                x, y, w, h = box
                top, right, bottom, left = y, x + w, y + h, x
                top, right, bottom, left = enlarge(top, right, bottom, left, frame.shape)
                counter += 1
                top, right, bottom, left = enlarge(
                    top, right, bottom, left, frame.shape
                )
                face = frame[top:bottom, left:right, :]
                dst_name = dst_dir / f"{name.stem}_{counter:04d}_{i}.png"
                cv2.imwrite(str(dst_name), face)
