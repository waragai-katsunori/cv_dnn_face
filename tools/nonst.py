from matplotlib import pyplot as plt
import numpy as np

def plot_similarity(similarities: np.ndarray, pngname="similarities.png", title="similarities"):
    """
    類似度を表示する。
    :param similarities:　類似性の行列
    :param pngname:
    :param title:
    :return:
    """
    #    plt.jet()
    plt.clf()
    plt.imshow(similarities)
    plt.colorbar()
    plt.title(title)
    # plt.show()
    plt.savefig(pngname)


def plot_similarity_hist(
    scores: np.ndarray, matched: np.ndarray, pngname="similarity_hist.png", title="similarities"
):
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.hist(scores[matched == 1], bins=100, label="matched")
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylabel("freq")
    plt.title(title)
    plt.subplot(2, 1, 2)
    plt.hist(scores[matched == 0], bins=100, label="matched==0")
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylabel("freq")
    plt.xlabel("similarity")
    # plt.show()
    plt.savefig(pngname)


def plot_roc(curve: np.ndarray, pngname="roc-curve.png", title="ROC-curve"):
    """
    ROCカーブを表示する。
    :param curve:  roc_curve() の戻り値
    :param pngname:
    :param title:
    :return:
    """
    plt.clf()
    plt.plot(curve[0], curve[1])
    plt.grid(True)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if 0:
        plt.xlim([0, 0.1])
        plt.ylim([0.9, 1.0])
    plt.title(title)
    # plt.show()
    plt.savefig(pngname)


def plot_eye_distances(
    total_eye_distances, pngname="eye_distance.png", title="eye-distance"
):
    """
    目間画素数の分布を表示する。
    :param total_eye_distances:
    :param pngname:
    :param title:
    :return:
    """
    plt.clf()
    plt.hist(total_eye_distances, bins=100)
    plt.xlabel("eye distance [pixel]")
    plt.ylabel("freq")
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.title(title)
    #    plt.show()
    plt.savefig(pngname)


def get_TPR_at_FAR(curve: np.ndarray, fars=(1e-4, 1e-3, 1e-2)):
    """
    :param curve: roc_curve() の戻り値
    :param fars:　本人受け入れ率を計算する条件の他人受け入れ率
    :return:
    """
    far_tar = []  # 他人受け入れ率と本人受け入れ率
    for far in fars:
        for x, y in zip(curve[0], curve[1]):
            if x >= far:
                far_tar.append((x, y))
                break
    return far_tar


def calc_matched(total_labels) -> np.ndarray:
    xmatched = [
        [1 if label1 == label2 else 0 for label2 in total_labels]
        for label1 in total_labels
    ]
    return np.array(xmatched)


def calc_xmatched(total_labels1, total_labels2) -> np.ndarray:
    xmatched = [
        [1 if label1 == label2 else 0 for label2 in total_labels2]
        for label1 in total_labels1
    ]
    return np.array(xmatched)