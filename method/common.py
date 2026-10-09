"""method 公共实现模块。

存放完整方法及其消融、敏感性实验入口共享的工具函数。
共享的常量与工具函数，避免跨文件重复定义。

包含：
- 常量：TARGET_COMPLEXITY / TARGET_LENGTH_RATIO
- 工具函数：clamp01 / word_count / TextComplexityAnalyzer /
  build_compact_memory /
  difficulty_rules / get_dynamic_prompt / parse_fact_lines /
  extract_facts_heuristic / extract_facts_llm / trim_chat_history /
  generate_candidates

路径规划已统一收敛至 core/path_planner.py（唯一实现），
原 ContextAwareRewardMatrix / build_context_aware_path 已删除，
各入口统一使用 core.path_planner.build_multi_objective_path。
"""

import json
import re
from pathlib import Path

from core.prompt_builder import clean_llm_generation
from utils import CEFRMAP, CEFR_Descriptors

TARGET_COMPLEXITY = {
    1: 0.20,
    2: 0.30,
    3: 0.45,
    4: 0.60,
    5: 0.75,
    6: 0.90,
}


TARGET_LENGTH_RATIO = {}


def clamp01(value: float) -> float:
    """将输入限制在 [0,1] 区间。

    Args:
        value: 参数说明。

    Returns:
        返回值说明。
    """
    return max(0.0, min(1.0, value))


def word_count(text: str) -> int:
    """统计文本单词数。

    Args:
        text: 参数说明。

    Returns:
        返回值说明。
    """
    return len(text.split())


class TextComplexityAnalyzer:
    """文本复杂度分析器：输出 0~1 复杂度分数。

    Args:

    Returns:
        返回值说明。
    """

    @classmethod
    def compute_features(cls, text: str) -> tuple[float, float, float]:
        words = text.split()
        length_score = min(len(words) / 100.0, 1.0)
        vocab_score = min(
            len({word.lower() for word in words}) / max(len(words), 1), 1.0
        )
        sub_markers = ["because", "although", "while", "when", "which", "that", "i", "since"]
        lower_text = text.lower()
        clause_count = 1 + sum(
            lower_text.count(f" {marker} ") for marker in sub_markers
        )
        syntax_score = min((clause_count - 1) / 3.0, 1.0)
        return length_score, vocab_score, syntax_score

    @classmethod
    def compute_overall_complexity(
        cls, text: str, weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    ) -> float:
        """compute_overall_complexity 功能。

        Args:
            text: 参数说明。

        Returns:
            返回值说明。
        """
        features = cls.compute_features(text)
        return sum(weight * value for weight, value in zip(weights, features))


def load_iaps_calibration(path: str | Path) -> dict:
    """Load calibrated complexity anchors and parallel-corpus length ratios."""
    with open(path, "r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    required = ("weights", "mean", "std", "beta", "level_complexity", "length_ratio")
    missing = [key for key in required if key not in calibration]
    if missing:
        raise ValueError(f"Calibration missing keys: {', '.join(missing)}")
    calibration["level_complexity"] = {
        int(key): float(value)
        for key, value in calibration["level_complexity"].items()
    }
    calibration["length_ratio"] = {
        int(key): float(value)
        for key, value in calibration["length_ratio"].items()
    }
    return calibration


def length_ratios_from_parallel_cases(case_dict: dict) -> dict[int, float]:
    """Estimate rho(level) from expert parallel complex/simple pairs."""
    totals: dict[int, list[float]] = {}
    for key, pairs in case_dict.items():
        if not key.isdigit() or len(key) != 2:
            continue
        target = int(key[1])
        for pair in pairs:
            if len(pair) != 2:
                continue
            source, simplified = (str(part) for part in pair)
            totals.setdefault(target, []).append(
                word_count(simplified) / max(word_count(source), 1)
            )
    ratios = {level: sum(values) / len(values) for level, values in totals.items() if values}
    ratios.setdefault(6, 1.0)
    return ratios


def build_iaps_calibration(case_dict: dict) -> dict:
    """Build a runnable calibration from the supplied expert parallel corpus.

    The length ratios always come from the parallel pairs. Complexity anchors
    remain explicit defaults until a validation-set fitting file is supplied.
    """
    length_ratio = length_ratios_from_parallel_cases(case_dict)
    level_complexity = dict(TARGET_COMPLEXITY)
    anchor_values = list(level_complexity.values())
    mean = sum(anchor_values) / len(anchor_values)
    variance = sum((value - mean) ** 2 for value in anchor_values) / len(anchor_values)
    return {
        "weights": [0.4, 0.3, 0.3],
        "mean": mean,
        "std": max(variance**0.5, 1e-8),
        "beta": 0.5,
        "level_complexity": level_complexity,
        "length_ratio": length_ratio,
        "length_ratio_source": "expert parallel case pairs",
        "calibration_note": "Use validation-fitted weights, anchors, and beta for final results.",
    }


def build_compact_memory(
    original: str,
    facts: list[str],
    current_draft: str | None,
    target_cefr: int,
) -> str:
    """构建压缩记忆，保留关键事实供后续步骤使用。

    Args:
        original: 参数说明。
        facts: 参数说明。
        current_draft: 参数说明。
        target_cefr: 参数说明。

    Returns:
        返回值说明。
    """
    lines = ["ORIGINAL SENTENCE:", original]

    if facts:
        lines.append("FACTS TO KEEP:")

        lines.extend([f"{idx}. {fact}" for idx, fact in enumerate(facts, 1)])

    if current_draft and current_draft != original:
        lines.append("CURRENT DRAFT:")

        lines.append(current_draft)

    lines.append(f"TARGET LEVEL: {CEFRMAP[target_cefr]}")

    return "\n".join(lines)


def difficulty_rules(target_cefr: int) -> list[str]:
    """难度语义动态检索规则：基于目标等级选择合适示例。

    Args:
        target_cefr: 参数说明。

    Returns:
        返回值说明。
    """
    if target_cefr <= 2:
        return [
            "Use 1-2 short sentences.",
            "Avoid subordinate clauses and rare words.",
            "Prefer 12 words or fewer per sentence.",
        ]

    if target_cefr == 3:
        return [
            "Use 1-2 sentences and simple grammar.",
            "Prefer 18 words or fewer per sentence.",
        ]

    return [
        "Keep sentences clear and direct.",
        "Prefer 25 words or fewer per sentence.",
    ]


def get_dynamic_prompt(
    test_sentence,
    target_cefr,
    selected_exemplars,
    sims,
    memory_block: str | None = None,
):
    """组装带动态示例与历史轨迹的推理 prompt。

    Args:
        test_sentence: 参数说明。
        target_cefr: 参数说明。
        selected_exemplars: 参数说明。
        sims: 参数说明。
        memory_block: 参数说明。

    Returns:
        返回值说明。
    """
    cefr_level = CEFRMAP[target_cefr]

    content = (
        "You are an expert linguistic simplifier. "
        f"Your task is to simplify text for {cefr_level} level language learners "
        f"({CEFR_Descriptors[target_cefr]}).\n\n"
    )

    content += "CONSTRAINTS:\n"

    content += "- Use simple vocabulary and short sentences.\n"

    content += "- Do not copy complex words or long phrases from the original.\n"

    content += "- Keep only the main ideas; drop minor details if needed.\n"

    for rule in difficulty_rules(target_cefr):
        content += f"- {rule}\n"

    if memory_block:
        content += f"\n### MEMORY ###\n{memory_block}\n\n"

    content += "### REFERENCE EXAMPLES ###\n"

    for idx, (exemplar, sim) in enumerate(zip(selected_exemplars, sims), 1):
        complex_sent, simple_sent = exemplar

        content += (
            f"Example {idx} (sim={sim:.3f}):\n"
            f"Complex: {complex_sent}\n"
            f"Simple: {simple_sent}\n\n"
        )

    content += "### YOUR TASK ###\n"

    content += f"TARGET SENTENCE:\n{test_sentence}\n\n"

    content += "Return only the final simplified sentence in plain text."

    return content


def parse_fact_lines(raw_text: str, max_facts: int) -> list[str]:
    """解析模型输出中的事实行列表。

    Args:
        raw_text: 参数说明。
        max_facts: 参数说明。

    Returns:
        返回值说明。
    """
    facts: list[str] = []

    for line in raw_text.splitlines():
        cleaned = line.strip()

        if not cleaned:
            continue

        cleaned = re.sub(r"^[-*\d+.()\s]+", "", cleaned)

        if cleaned and cleaned not in facts:
            facts.append(cleaned)

        if len(facts) >= max_facts:
            break

    return facts


def extract_facts_heuristic(text: str, max_facts: int) -> list[str]:
    """基于启发式规则从句子中提取原子事实。

    Args:
        text: 参数说明。
        max_facts: 参数说明。

    Returns:
        返回值说明。
    """
    parts = re.split(r"[.;:!?]\s+", text.strip())

    facts: list[str] = []

    for part in parts:
        cleaned = part.strip()

        if cleaned:
            facts.append(cleaned)

        if len(facts) >= max_facts:
            break

    return facts


def extract_facts_llm(generator, text: str, max_facts: int) -> list[str]:
    """调用 LLM 从句子中提取原子事实。

    Args:
        generator: 参数说明。
        text: 参数说明。
        max_facts: 参数说明。

    Returns:
        返回值说明。
    """
    prompt = (
        "Extract 3-5 key facts from the sentence. "
        "Write each fact on a new line. "
        "Keep facts short and concrete.\n\n"
        f"Sentence: {text}\n"
        "Facts:"
    )

    raw = generator.generate_text([{"role": "user", "content": prompt}])

    facts = parse_fact_lines(raw, max_facts)

    if not facts:
        return extract_facts_heuristic(text, max_facts)

    return facts


def trim_chat_history(
    history: list[dict[str, str]], max_turns: int
) -> list[dict[str, str]]:
    """裁剪聊天历史，控制上下文长度。

    Args:
        history: 参数说明。
        max_turns: 参数说明。

    Returns:
        返回值说明。
    """
    if max_turns <= 0:
        return []

    max_messages = max_turns * 2

    return history[-max_messages:]


def generate_candidates(generator, payload, num_candidates: int) -> list[str]:
    """为复杂句生成多个简化候选。

    Args:
        generator: 参数说明。
        payload: 参数说明。
        num_candidates: 参数说明。

    Returns:
        返回值说明。
    """
    candidates: list[str] = []

    seen = set()

    for _ in range(max(1, num_candidates)):
        raw_text = generator.generate_text(payload)

        clean_text = (clean_llm_generation(raw_text) or "").strip()

        if clean_text and clean_text not in seen:
            candidates.append(clean_text)

            seen.add(clean_text)

    return candidates
