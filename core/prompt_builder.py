"""Prompt 构建与生成结果清洗工具。

从原 improve.py 拆分出的公共模块，包含 CEFR 等级映射、
few-shot 示例加载、负面示例构建、推理 prompt 组装以及
LLM 生成结果的清洗与后处理。
"""

import os
import re
from pathlib import Path

from utils import CEFRMAP


def clean_llm_generation(generation):
    """清洗 LLM 生成的简化句。

    去除 "Simple:" 等前缀标记、模型自述式废话（"Sure! Here's..."）、
    占位符文本（下划线密集、无字母、尾冒号短句等）以及元信息行。

    Args:
        generation: 模型原始输出字符串。

    Returns:
        清洗后的单句简化结果；无有效内容时返回 None。
    """
    if generation is None:
        return None

    text = str(generation).strip()
    if text == "":
        return None

    simple_markers = ["Simple:", "simple:", "Simplified:", "simplified:", "简化句："]
    for marker in simple_markers:
        if marker in text:
            text = text.split(marker)[-1].strip()

    text = text.replace("**Simple:**", "").replace("**Complex:**", "").strip()
    text = text.replace("**简化句：**", "").replace("**复杂句：**", "").strip()

    sentlist = [line.strip() for line in text.split("\n") if line.strip() != ""]
    if len(sentlist) == 0:
        return None

    bad_prefixes = (
        "sure! here's",
        "here's a simplified version",
        "the simplified version",
        "revised simplified sentence",
        "original complex sentence",
        "the user wants",
        "here is the simplified",
        "please provide",
    )

    def is_placeholder_text(text):
        """判断文本是否为占位/空壳输出（LLM 未真正改写时常见）。

        Args:
            text: LLM 生成文本。

        Returns:
            若文本为空、下划线占位或仅复述提示词则返回 True。
        """
        compact = text.replace(" ", "")
        if compact == "":
            return True
        underscore_ratio = compact.count("_") / len(compact)
        if underscore_ratio >= 0.6:
            return True
        alpha_count = sum(ch.isalpha() for ch in compact)
        if alpha_count == 0:
            return True
        lower = text.strip().lower()

        if "please provide" in lower:
            return True
        if lower.startswith(("please provide the", "please provide the target")):
            return True
        if lower.startswith("please help"):
            return True
        if lower.startswith(("original sentence:", "please provide the original")):
            return True

        words = text.strip().split()
        if text.strip().endswith(":") and len(words) <= 10:
            return True
        return lower.startswith("here is the simplified") and len(words) <= 8

    meta_markers = (
        "reviewer feedback",
        "[critical]",
        "predcefr=",
        "targetcefr=",
        "local sts=",
        "global sts=",
        "your task",
        "reference examples",
        "bad-complex:",
        "bad-simple:",
        "reason:",
        "complex:",
        "simple:",
        "复杂句：",
        "<example>",
        "</example>",
        "<target>",
        "</target>",
    )

    for sent in sentlist:
        lower = sent.lower()
        if lower.startswith(("okay", "let me", "i need to")):
            continue
        if lower.startswith(("the user", "first,")):
            continue
        if any(lower.startswith(prefix) for prefix in bad_prefixes):
            continue
        if any(marker in lower for marker in meta_markers):
            continue
        if "cefr level learner" in lower or "simplified version" in lower:
            continue
        if is_placeholder_text(sent):
            continue
        sent = sent.strip("\"'` ")
        if sent != "":
            return sent

    for sent in sentlist:
        fallback = sent.strip("\"'` ")
        lower = fallback.lower()
        if (
            fallback != ""
            and not any(prefix in lower for prefix in bad_prefixes)
            and not any(marker in lower for marker in meta_markers)
            and not is_placeholder_text(fallback)
        ):
            return fallback
    return None


def padding_case(case_dict_expert: dict) -> dict:
    """为缺失的 (起始级, 目标级) 键补上空列表。

    同时覆盖 CEFR 等级（1-6）与 HSK 等级（101-107，汉语场景）。

    Args:
        case_dict_expert: 专家示例字典，键形如 "63" 或 "107106"。

    Returns:
        补齐后的字典。
    """
    for begin_cefr in [6, 5, 4, 3, 2]:
        for target_cefr in range(begin_cefr - 1, 0, -1):
            if f"{begin_cefr}{target_cefr}" not in case_dict_expert:
                case_dict_expert[f"{begin_cefr}{target_cefr}"] = []
    for begin_cefr in [107, 106, 105, 104, 103, 102]:
        for target_cefr in range(begin_cefr - 1, 100, -1):
            key = f"{begin_cefr}{target_cefr}"
            if key not in case_dict_expert:
                case_dict_expert[key] = []
    return case_dict_expert


def resolve_split_file(
    corpus: str, lang: str, split: str, data_root: str | Path | None = None
) -> str:
    """按语料/语言/划分解析数据集文件路径。

    依次尝试多种命名约定（含下划线、连字符、语言后缀等），
    返回第一个存在的数据文件路径。

    Args:
        corpus: 语料名，如 README / CEFR-SP / CEFR-SP_whole。
        lang: 语言代码，如 en / fr。
        split: 数据划分，train / dev / test。
        data_root: 数据目录；省略时自动使用仓库根目录下的 data/。

    Returns:
        匹配到的数据集文件绝对路径。

    Raises:
        FileNotFoundError: 所有候选路径均不存在。
    """
    root = (
        Path(data_root).expanduser()
        if data_root is not None
        else Path(__file__).resolve().parents[1] / "data"
    )
    data_lang_dir = root / lang
    candidates = [
        data_lang_dir / f"{corpus}_{split}.csv",
        data_lang_dir / f"{corpus}_{lang}_{split}.csv",
        data_lang_dir / f"{corpus.lower()}_{lang}_{split}.csv",
        data_lang_dir / f"{corpus.replace('-', '_')}_{lang}_{split}.csv",
        data_lang_dir / f"{corpus.replace('-', '').lower()}_{lang}_{split}.csv",
    ]

    if corpus == "README":
        candidates.append(data_lang_dir / f"readme_{lang}_{split}.csv")
    if corpus in ["CEFR-SP", "CEFR-SP_whole"]:
        candidates.append(data_lang_dir / f"CEFR-SP_{lang}_{split}.csv")

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    tried = "\n".join(candidates)
    raise FileNotFoundError(
        f"No dataset file found for corpus={corpus}, lang={lang}, split={split}. Tried:\n{tried}"
    )


def make_bad_simplification(complex_text: str, simple_text: str) -> str:
    """构造负面简化样本。

    通过把专有名词替换为 "someone"、数字替换为 "some" 并截断句子，
    模拟丢失关键实体的错误简化，用于构造负面示例。

    Args:
        complex_text: 原复杂句（保留用于上下文一致性检查）。
        simple_text: 正确简化句。

    Returns:
        负面简化句。
    """
    bad = simple_text
    bad = re.sub(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "someone", bad)
    bad = re.sub(r"\b\d+\b", "some", bad)
    bad = bad.strip()

    words = bad.split()
    if len(words) > 10:
        bad = " ".join(words[:10]).strip() + "."

    if bad == "" or bad.lower() == simple_text.strip().lower():
        bad = "This says something happened but loses key details."

    return bad


def build_negative_examples(prompt_case_pair: list, negative_case_num: int) -> list:
    """从正例对中构建负面示例列表。

    Args:
        prompt_case_pair: (复杂句, 简化句) 示例对列表。
        negative_case_num: 需要的负面示例数量。

    Returns:
        (复杂句, 负面简化句, 原因说明) 三元组列表。
    """
    negatives = []
    for complex_text, simple_text in prompt_case_pair[:negative_case_num]:
        negatives.append(
            (
                complex_text,
                make_bad_simplification(complex_text, simple_text),
                "Bad because it drops key entities/details and may distort meaning.",
            )
        )
    return negatives


def get_prompt(
    prompt_case_pair: list,
    prompt_case_single: list,
    complex_sentence: str,
    CEFR_LEVEL: int,
    CEFR_Description: str,
    use_negative_examples: bool = False,
    negative_case_num: int = 1,
    history_original: str | None = None,
    history_current: str | None = None,
) -> str:
    """构建简化任务的完整 prompt。

    统一示例与目标的格式，可选注入负面示例（MUST AVOID）与历史简化轨迹，
    用于零样本/少样本渐进式简化推理。

    Args:
        prompt_case_pair: (复杂句, 简化句) 正例对。
        prompt_case_single: 仅目标等级可读句列表（当无成对示例时使用）。
        complex_sentence: 待简化的复杂句。
        CEFR_LEVEL: 目标 CEFR 等级数字（1-6）。
        CEFR_Description: 目标等级学习者能力描述。
        use_negative_examples: 是否注入负面示例。
        negative_case_num: 负面示例数量。
        history_original: 历史原句（多轮渐进简化时使用）。
        history_current: 当前已简化句。

    Returns:
        组装好的 prompt 字符串。
    """
    CEFR_LEVEL = CEFRMAP[CEFR_LEVEL]
    content = (
        f"Please rewrite the following complex sentence in order to make it easier to understand by {CEFR_LEVEL} CEFR level language learners. "
        f"{CEFR_LEVEL} level language learner {CEFR_Description} "
        "You can do so by replacing complex words with simpler synonyms, deleting unimportant information, and/or splitting a long complex sentence into several simpler ones. "
        "The final simplified sentence needs to be grammatical, fluent, and retain the main ideas of its original counterpart without altering its meaning.\n\n"
        "### REFERENCE EXAMPLES ###\n"
        "The following examples are for style only.\n"
    )

    if len(prompt_case_pair) > 0:
        for i, (complex_text, simple_text) in enumerate(prompt_case_pair):
            content += (
                f"Example {i + 1}:\nComplex: {complex_text}\nSimple: {simple_text}\n\n"
            )
    else:
        for i in range(len(prompt_case_single)):
            content += (
                f"Example {i + 1}:\nReadable sentence: {prompt_case_single[i]}\n\n"
            )

    if use_negative_examples and len(prompt_case_pair) > 0:
        negatives = build_negative_examples(prompt_case_pair, negative_case_num)
        if len(negatives) > 0:
            content += "### BAD EXAMPLES (MUST AVOID) ###\n"
            for complex_text, bad_text, reason in negatives:
                content += (
                    f"Bad-Complex: {complex_text}\n"
                    f"Bad-Simple: {bad_text}\n"
                    f"Reason: {reason}\n\n"
                )
            content += "Do not over-compress, do not replace named entities with ambiguous pronouns, and do not drop key facts.\n"

    if isinstance(history_original, str) and history_original.strip() != "":
        content += "### HISTORY ###\n"
        content += f"Original sentence: {history_original}\n"
        if (
            isinstance(history_current, str)
            and history_current.strip() != ""
            and history_current != history_original
        ):
            content += f"Current simplification: {history_current}\n"

    content += (
        "### YOUR TASK ###\n"
        "Simplify the following complex sentence. Output ONLY the simplified sentence. Do NOT include phrases like 'Here is the simplified sentence' or 'Please provide'.\n"
        f"Complex: {complex_sentence}\n"
        "Simple:"
    )
    return content
