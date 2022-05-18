import os
import re
from pathlib import Path
from pprint import pprint
from typing import List, Tuple
from math import sqrt
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve
import cv2

import faceme_depend
from faceme_depend import faceMe_sdk
from faceme_depend import get_feature, get_feature_in, calc_xsimilarities
from nonst import (
    get_TPR_at_FAR,
    calc_xmatched,
    plot_roc,
    plot_similarity_hist,
    plot_similarity,
)


def get_distance_part(p: Path):
    f = p.stem.replace("-", "_").split("_")
    pattern = re.compile(r"\d+\.\d+m")
    for a in f:
        if pattern.match(a):
            return float(a.replace("m", ""))

    return None

def eval_face_idenification(master_dir: Path, eval_dir: Path, csvname: Path, quality_check=True):
    """
    master_dir : master data for face recognition
    valid_dir: data for evaluate
    """
    with open(csvname, "wt") as f:
        labels = [p.name for p in master_dir.glob("*") if p.is_dir()]
        assert len(labels) > 0
        features, paths, labels, failed_paths, eye_distances = get_feature_in(master_dir, recursive=True, quality_check=quality_check)
        print(paths)
        print(f"len(features) {len(features)}")
        print(f"len(paths) len(failed_paths): {len(paths)}, {len(failed_paths)}")
        features2, paths2, labels2, failed_paths2, eye_distances2 = get_feature_in(eval_dir, recursive=True, quality_check=quality_check)

        print(paths2)
        print(f"len(features2) {len(features2)}")
        print(f"len(paths2) len(failed_paths2): {len(paths2)}, {len(failed_paths2)}")

        for i, (f1, p1) in enumerate(zip(features, paths)):
            label1 = p1.parent.name
            for j, (f2, p2, eye_distance2) in enumerate(zip(features2, paths2, eye_distances2)):
                label2 = p2.parent.name
                if label1 == label2:
                    cmp_result = faceMe_sdk.compare_face_feature(f1, f2)
                    similarities = cmp_result[1]["similarity"]
                    confidences = cmp_result[1]["confidence"]
                    is_sames_pred = cmp_result[1]["isSamePerson"]
                    distance = get_distance_part(p2)
                    line = f"{is_sames_pred}, {similarities}, {confidences}, {distance}, {eye_distance2}, {label1}, {label2}, {p2} \n"
                    print(line, end="")
                    f.write(line)


if __name__ == "__main__":
    """
    カメラモックのsubカメラ画像に対して顔の同一性の判定を実施する。
    利用手順：
    顔照合のマスター画像・評価画像とを
    git clone しておく
    cd ~
    git clone git@bitbucket.org:groove-x/faces-20220407-proto-cam.git
    cd faces-20220407-proto-cam ; git lfs pull

    その後、このスクリプトを実行する。

    マスターデータ：照合用の基準となる登録画像。
    評価データ：評価のための画像群。

    出力：

    """

    import argparse

    parser = argparse.ArgumentParser(description="evaluate face idenification")
    parser.add_argument("master_dir", help="master data dir")
    parser.add_argument("eval_dir", help="evaluation data dir")
    parser.add_argument("csvname", help="report csv file name")
    parser.add_argument("--disable_quality_check", action="store_true", help="disable_quality_check")
    args = parser.parse_args()

    master_dir = Path(args.master_dir)
    eval_dir = Path(args.eval_dir)
    csvname = Path(args.csvname)  #"faces-20220407-proto-cam.csv"
    quality_check = not args.disable_quality_check

    assert master_dir.is_dir()

    eval_face_idenification(master_dir, eval_dir, csvname, quality_check=quality_check)
