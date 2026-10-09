"""评估器工具：STS 语义模型、ReadMe 可读性模型与 CEFR 预测。

从原 improve.py 拆分出的公共模块，提供句向量语义相似度计算
与 ReadMe 可读性等级预测（predict_cefr 已从占位实现补全为真实推理）。
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_sts_model(model_name: str) -> SentenceTransformer:
    """加载句向量语义模型（STS）。

    Args:
        model_name: 本地模型路径或 HuggingFace 模型名。

    Returns:
        SentenceTransformer 实例。
    """
    return SentenceTransformer(model_name)


def get_readability_predictor(lang: str):
    """加载 ReadMe 可读性评估器。

    Args:
        lang: 语言代码，如 en。

    Returns:
        ReadMe 预测器实例；模型缺失时返回 None。
    """
    try:
        from readmepp import ReadMe
    except ImportError:
        return None
    try:
        return ReadMe(lang)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def predict_cefr(readability_predictor, text: str, fallback_level: int) -> int:
    """预测句子所属 CEFR 等级（1-6）。

    调用 ReadMe 可读性模型对输入句子做等级预测，兼容模型返回
    int / float / str（如 "A2"）三种形态；预测失败或结果非法时
    回退到 fallback_level，保证主流程不被单个句子中断。

    Args:
        readability_predictor: ReadMe 预测器实例。
        text: 输入句子文本。
        fallback_level: 预测失败时的兜底等级。

    Returns:
        预测的 CEFR 等级（1-6）。
    """
    if readability_predictor is None:
        return fallback_level

    try:
        pred = readability_predictor.predict(text)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return fallback_level

    if isinstance(pred, (int, np.integer)):
        level = int(pred)
        if 1 <= level <= 6:
            return level
        return fallback_level

    if isinstance(pred, float):
        level = round(pred)
        if 1 <= level <= 6:
            return level
        return fallback_level

    if isinstance(pred, str):
        mapping = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
        token = pred.strip().upper()
        if token in mapping:
            return mapping[token]
        # 兼容 "Level 3" / "3" 形态
        digits = "".join(ch for ch in pred if ch.isdigit())
        if digits:
            level = int(digits)
            if 1 <= level <= 6:
                return level

    return fallback_level


def sts_similarity(sts_model, sent1: str, sent2: str) -> float:
    """计算两个句子的语义相似度（余弦相似度）。

    Args:
        sts_model: SentenceTransformer 实例。
        sent1: 句子 1。
        sent2: 句子 2。

    Returns:
        余弦相似度分数（0~1）。
    """
    emb1 = sts_model.encode([sent1], convert_to_tensor=True)
    emb2 = sts_model.encode([sent2], convert_to_tensor=True)
    return float(torch.nn.functional.cosine_similarity(emb1, emb2).item())
