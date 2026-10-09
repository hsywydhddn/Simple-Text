import os

import numpy as np


CEFR_Descriptors = {
    1: "Can understand very short, simple texts a single phrase at a time, picking up familiar names, words and basic phrases and rereading as required.",
    2: "Can understand short, simple texts containing the highest frequency vocabulary, including a proportion of shared international vocabulary items.",
    3: "Can read straightforward factual texts on subjects related to their field of interest with a satisfactory level of comprehension.",
    4: "Can read with a large degree of independence, adapting style and speed of reading to different texts and purposes, and using appropriate reference sources selectively. Has a broad active reading vocabulary, but may experience some difficulty with low-frequency idioms.",
    5: "Can understand a wide variety of texts including literary writings, newspaper or magazine articles, and specialised academic or professional publications, provided there are opportunities for rereading and they have access to reference tools. Can understand in detail lengthy, complex texts, whether or not these relate to their own area of speciality, provided they can reread difficult sections.",
    6: "Can understand a wide range of long and complex texts, appreciating subtle distinctions of style and implicit as well as explicit meaning. Can understand virtually all types of texts including abstract, structurally complex, or highly colloquial literary and non-literary writings.",
}


CEFRMAP = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}


BERTSCORE_LOCAL_PATH = os.getenv("BERTSCORE_LOCAL_PATH", "")


READABERT_LOCAL_PATH = os.getenv("READABERT_LOCAL_PATH", "")


STS_LOCAL_PATH = os.getenv("STS_LOCAL_PATH", "")


def exact_accuracy(true_labels, pred_labels):
    """计算 CEFR 等级精确匹配准确率。

    Args:
        true_labels: 参数说明。
        pred_labels: 参数说明。

    Returns:
        返回值说明。
    """
    success_count = 0

    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            success_count += 1

    accuracy = success_count / len(true_labels)

    return accuracy


def calculate_rmse(true_labels, pred_labels):
    """计算预测等级与真实等级的均方根误差。

    Args:
        true_labels: 参数说明。
        pred_labels: 参数说明。

    Returns:
        返回值说明。
    """
    true_labels = np.array(true_labels)

    pred_labels = np.array(pred_labels)

    mse = np.mean((true_labels - pred_labels) ** 2)

    rmse = np.sqrt(mse)

    return rmse


def adjacency_accuracy(true_labels, pred_labels):
    """计算 CEFR 等级相邻容忍（|diff|<=1）准确率。

    Args:
        true_labels: 参数说明。
        pred_labels: 参数说明。

    Returns:
        返回值说明。
    """
    success_count = 0

    for true, pred in zip(true_labels, pred_labels):
        if abs(true - pred) <= 1:
            success_count += 1

    accuracy = success_count / len(true_labels)

    return accuracy
