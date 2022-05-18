from pathlib import Path
from typing import List, Tuple
from math import sqrt
import os

import numpy as np

import FaceMe.FaceMePython3SDK as FaceMe


from FaceMe.FaceMeSDK import FaceMeSDK, FR_FAILED, FR_SUCC
from FaceMePythonSample import initialize_SDK


def read_licensekey():
    name = ".licensekey"
    return open(name, "rt").read().strip()


BASE_DIR = Path(__file__).parent


licensekey = read_licensekey()
faceMe_sdk = FaceMeSDK()
app_bundle_path = str(BASE_DIR)
app_cache_path = os.path.join(os.getenv("HOME"), ".cache")
options = {}

# print(licensekey)
app_data_path = app_data_path = os.path.join(os.getenv("HOME"), ".local", "share")

faceMe_sdk.initialize(
    licensekey, app_bundle_path, app_cache_path, app_data_path, options
)


def get_feature(image_paths: List[Path]):
    features = []
    paths = []
    for p in image_paths:
        ret, image1 = faceMe_sdk.convert_to_faceMe_image(str(p))

        if FR_FAILED(ret):
            print("convert to Faceme Image failed", p)
            continue

        images = [
            image1,
        ]
        options = {"extractOptions": FaceMe.FR_FEATURE_OPTION_ALL}
        ret, recognize_results = faceMe_sdk.recognize_faces(images, options)
        if FR_FAILED(ret):
            print("recognize face failed, return: ", ret)
            continue
        else:
            for result in recognize_results:
                features.append(result["faceFeatureStruct"])
                paths.append(p)

    return features, paths


def eye_distance(points):
    left_x, left_y = points[0]
    right_x, right_y = points[1]
    return sqrt((right_x - left_x) ** 2 + (right_y - right_y) ** 2)


def get_feature_in(image_dir: Path, recursive=True, quality_check=True) -> Tuple[List, List, List]:
    """
    以下のようなディレクトリ構造を想定している。
    test/horinouchi
    test/numaguchi
    test/waragai
    valid/horinouchi
    valid/numaguchi
    valid/waragai
    """

    options = {"checkMode": FaceMe.FR_QUALITY_CHECK_MODE_ALL_FAILURE}

    dirs = list([p for p in image_dir.glob("*") if p.is_dir()])
    dirs.sort()

    features = []
    paths = []
    labels = []
    failed_paths = []
    eye_distances = []
    for d in dirs:
        if recursive:
            names = sorted(list(d.glob("**/*.jpg")) + list(d.glob("**/*.png")))
        else:
            names = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
        for p in names:
            ret, image1 = faceMe_sdk.convert_to_faceMe_image(str(p))

            if FR_FAILED(ret):
                print("convert to Faceme Image failed", p)
                failed_paths.append(p)
                continue

            if quality_check:
                ret, detect_results = faceMe_sdk.detect_image_quality(image1, options)
                if FR_FAILED(ret) or is_bad_quality(detect_results):
                    failed_paths.append(p)
                    continue

            images = [
                image1,
            ]
            options = {"extractOptions": FaceMe.FR_FEATURE_OPTION_ALL}
            ret, recognize_results = faceMe_sdk.recognize_faces(images, options)
            if FR_FAILED(ret):
                print("recognize face failed, return: ", ret)
                failed_paths.append(p)
                continue
            else:
                for result in recognize_results:
                    # print(result["faceLandmark"])
                    points = result["faceLandmark"]
                    features.append(result["faceFeatureStruct"])
                    paths.append(p)
                    labels.append(d.name)
                    eye_distances.append(eye_distance(points))

    return features, paths, labels, failed_paths, eye_distances


def calc_similarities(features):
    return calc_xsimilarities(features, features)


def calc_xsimilarities(features1, features2):
    m = len(features1)
    n = len(features2)
    xsimilarities = np.zeros((m, n))
    xconfidences = np.zeros((m, n))
    xis_sames_pred = np.zeros((m, n), dtype=np.bool_)
    for i, f1 in enumerate(features1):
        for j, f2 in enumerate(features2):
            cmp_result = faceMe_sdk.compare_face_feature(f1, f2)
            xsimilarities[i, j] = cmp_result[1]["similarity"]
            xconfidences[i, j] = cmp_result[1]["confidence"]
            xis_sames_pred[i, j] = cmp_result[1]["isSamePerson"]

    return xsimilarities, xconfidences, xis_sames_pred
