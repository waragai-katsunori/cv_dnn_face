import os
from pathlib import Path
from pprint import pprint
from typing import List, Tuple
from math import sqrt
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve
import cv2

from faceme_depend import get_feature, get_feature_in, calc_xsimilarities
from nonst import (
    get_TPR_at_FAR,
    calc_xmatched,
    plot_roc,
    plot_similarity_hist,
    plot_similarity,
)

from feature_type import FeatureType


def xmain(test_dir: Path, valid_dir, report_name, quality_check=True):
    """
      test_dir の階層の下にあるラベルごとのフォルダについて、画像特徴量を抽出したうえで、同一人物かどうかの判定を行なう。
    同一ラベルの中の人物は同一人物、異なるラベルの人物は他人である。
    """
    qualty_str = "QC_enabled" if quality_check else "QC_disabled"
    report_dir = Path(f"report_{report_name}_{qualty_str}")
    report_dir.mkdir(exist_ok=True)

    t1 = cv2.getTickCount()
    features, paths, labels, failed_paths, eye_distances = get_feature_in(
        test_dir, recursive=True, quality_check=quality_check
    )

    features2, paths2, labels2, failed_paths2, eye_distances2 = get_feature_in(
        valid_dir, recursive=True, quality_check=quality_check
    )
    if features:
        print("features[0].featureType", features[0].featureType)
        print("features[0].featureType", FeatureType(features[0].featureType))

    t2 = cv2.getTickCount()
    used = (t2 - t1) / cv2.getTickFrequency()
    num_features = len(features) + len(features2)

    print(f"len(features): {len(features)}")
    print(f"len(features2): {len(features2)}")
    print(f"len(paths): {len(paths)}")
    print(f"len(paths2): {len(paths2)}")
    print(f"len(failed_paths): {len(failed_paths)}")
    print(f"len(failed_paths2): {len(failed_paths2)}")
    detection_ratio = (len(paths) + len(paths2)) / (len(paths) + len(paths2) + len(failed_paths) + len(failed_paths2))
    print(f"detection (or picked_ratio) : {detection_ratio:.3f}")

    print("start calc_similarities()")
    t3 = cv2.getTickCount()
    xsimilarities, xconfidences, xis_sames_pred = calc_xsimilarities(
        features, features2
    )
    t4 = cv2.getTickCount()
    used_similarity = (t4 - t3) / cv2.getTickFrequency()
    used_similarity_per_pair = used_similarity / (xsimilarities.shape[0] * xsimilarities.shape[1])

    xmatched = calc_xmatched(labels, labels2)
    print("passed calc_similarities()")

    print(xsimilarities)
    print(xmatched)
    print(xis_sames_pred)
    plot_similarity(
        xsimilarities,
        pngname=(report_dir / "similarities.png"),
        title=f"similarities {report_name}",
    )
    # np.savetxt(report_dir / "similarities.txt", xsimilarities)
    with open(report_dir / "similarities.pkl", "wb") as f:
        pickle.dump((xsimilarities, labels, eye_distances), f)

    cmatrix = confusion_matrix(
        xmatched.flatten(),
        xis_sames_pred.flatten(),
    )
    print(cmatrix)
    print(f1_score(xmatched.flatten(), xis_sames_pred.flatten()))
    print(classification_report(xmatched.flatten(), xis_sames_pred.flatten()))

    fpr, tpr, thresholds = roc_curve(xmatched.flatten(), xsimilarities.flatten())
    curve = (fpr, tpr, thresholds)

    with open(report_dir / "report.txt", "wt") as log:
        log.write(classification_report(xmatched.flatten(), xis_sames_pred.flatten()))
        log.write("--------------\n")
        log.write(str(cmatrix) + "\n")
        log.write(f"detection_ratio: {detection_ratio}\n")
        log.write(f"len(paths): {len(paths)}\n")
        log.write(f"len(failed_paths): {len(failed_paths)}\n")
        log.write(f"extract_feature / num_feature: {(used / num_features):.3f} [sec]\n")
        log.write(f"used_similarity_per_pair: {used_similarity_per_pair:.3f} [sec]\n")
        far_tar = get_TPR_at_FAR(curve)
        log.write("FAR TAR\n")
        for x, y in far_tar:
            print(f"{x:.5f} {y:.4f}")
            log.write(f"{x:.5f} {y:.4f}\n")

    # plot roc curve
    plot_roc(
        curve, pngname=report_dir / "roc-curve.png", title=f"ROC-curve {report_name}"
    )

    # hist for similarites
    plot_similarity_hist(
        xsimilarities,
        xmatched,
        pngname=(report_dir / "similarity_hist.png"),
        title=f"similarities {report_name}",
    )

    m = len(eye_distances)
    n = len(eye_distances2)

    min_eye_distance = np.zeros((m, n), dtype=np.float)
    for i in range(m):
        for j in range(n):
            min_eye_distance[i, j] = min(eye_distances[i], eye_distances2[j])

    # hist for eye_distances
    max_x = np.max(eye_distances)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.hist(eye_distances, bins=100, label="all")
    plt.legend()
    plt.title(f"eye_distances {report_name}")
    plt.xlabel("[pixel]")
    plt.ylabel("freq")
    plt.grid(True)
    plt.xlim([0, max_x])
    plt.subplot(3, 1, 2)
    plt.hist(min_eye_distance[xmatched == xis_sames_pred], bins=100, label="succeeded")
    plt.legend()
    plt.ylabel("freq")
    plt.xlim([0, max_x])
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.hist(min_eye_distance[xmatched != xis_sames_pred], bins=100, label="failed")
    plt.legend()
    plt.ylabel("freq")
    plt.grid(True)
    plt.xlim([0, max_x])
    plt.xlabel("[pixel]")
    plt.savefig(report_dir / "eye_distance_succeed.png")

    # hist for confidence
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.hist(xconfidences.flatten(), bins=100)
    plt.ylabel("freq")
    plt.grid(True)
    plt.xlim([0, 1])
    plt.title(f"confidence {report_name}")
    plt.subplot(3, 1, 2)
    plt.hist(xconfidences[xmatched == xis_sames_pred], bins=100, label="succeeded")
    plt.legend()
    plt.ylabel("freq")
    plt.xlim([0, 1])
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.hist(xconfidences[xmatched != xis_sames_pred], bins=100, label="failed")
    plt.legend()
    plt.ylabel("freq")
    plt.xlim([0, 1])
    plt.grid(True)
    plt.savefig(report_dir / "confidence.png")

    failed_pairs = set([])
    failed = xmatched != xis_sames_pred
    for i in range(failed.shape[0]):
        for j in range(failed.shape[1]):
            if failed[i, j]:
                a, b = min(i, j), max(i, j)
                failed_pairs.add((a, b))

    with (report_dir / "failed_pairs.html").open("wt") as html:
        html.write("<html>\n")
        m = 100
        counter = 0
        for a, b in failed_pairs:
            if counter >= m:
                break
            html.write(f"""<img src="{paths[a]}"> <img src="{paths2[b]}"> <p>\n""")
            m += 1
        html.write("<html>\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="cross roc curve")
    parser.add_argument("test_dir")
    parser.add_argument("valid_dir")
    parser.add_argument("report_name")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    valid_dir = Path(args.valid_dir)
    assert test_dir.is_dir()
    assert valid_dir.is_dir()
    report_name = Path(args.report_name)

    for quality_check in (True, False):
        if quality_check:
            report_name2 = f"{report_name}_QC_enabled"
        else:
            report_name2 = f"{report_name}_QC_disabled"
        xmain(test_dir, valid_dir, report_name, quality_check=quality_check)
