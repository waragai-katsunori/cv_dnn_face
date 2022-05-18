from pathlib import Path

import cv2

from cv_dnn_face import YunetFaceDetector
face_detector = YunetFaceDetector()

test_image_dir = Path("image")
out_dir = Path("out")

def write_detected_images(test_image_dir: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)

    for p in test_image_dir.glob("**/*.jpg"):
        outname = out_dir / p.relative_to(test_image_dir)
        outname.parent.mkdir(exist_ok=True)
        image = cv2.imread(str(p))
        _, faces = face_detector.detect(image)
        for face in faces:
            box, landmarks, confidence = face_detector.object_parser(face)
            print(box, landmarks, confidence)
            x, y, w, h = box

            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

            # landmark（right_eye, left_eye, nose, right corner of the mouth, left corner of the mouth）
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)

        cv2.imwrite(str(outname), image)


def test_detection(p, num_faces):
    img = cv2.imread(str(p))
    _, faces = face_detector.detect(img)
    faces = [] if faces is None else faces
    print(p, faces)
    assert len(faces) == num_faces

    for obj in faces:
        box, landmarks, confidence = face_detector.object_parser(obj)
        print(box, landmarks, confidence)
        x, y, w, h = box
        assert x < x + w < img.shape[1]
        assert y < y + h < img.shape[0]

        assert len(landmarks) == 5
        assert 0.0 <= confidence <= 1.0


def test_detection_feature(p, num_faces):
    img = cv2.imread(str(p))
    _, faces = face_detector.detect(img)
    faces = [] if faces is None else faces
    print(p, faces)
    assert len(faces) == num_faces

    for obj in faces:
        box, landmarks, confidence = face_detector.object_parser(obj)
        print(box, landmarks, confidence)
        x, y, w, h = box
        assert x < x + w < img.shape[1]
        assert y < y + h < img.shape[0]

        assert len(landmarks) == 5
        assert 0.0 <= confidence <= 1.0

        feature = face_detector.get_feature(img, obj)
        assert feature is not None
        print(len(feature))
        print(len(feature[0]))
        print(feature.shape)

conditions = [
        ["image/biden.jpg", 1],
        ["image/obama.jpg", 1],
        ["image/trump.jpg", 1],
    ]


for p, num_faces in conditions:
    assert Path(p).is_file()
    test_detection(p, num_faces)


write_detected_images(test_image_dir, out_dir)

for p, num_faces in conditions:
    assert Path(p).is_file()
    test_detection_feature(p, num_faces)