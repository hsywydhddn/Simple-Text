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
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

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
    required = (
        "weights",
        "mean",
        "std",
        "beta",
        "level_complexity",
        "length_ratio",
        "retrieval_weights",
        "rerank_weights",
    )
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
    for key in ("retrieval_weights", "rerank_weights"):
        values = calibration[key]
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(float(value) < 0 for value in values)
            or not np.isclose(sum(float(value) for value in values), 1.0)
        ):
            raise ValueError(
                f"Calibration field {key} must contain three non-negative "
                "weights summing to one."
            )
        calibration[key] = [float(value) for value in values]
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


def _simplex_weights(step: float = 0.05) -> list[tuple[float, float, float]]:
    """Enumerate non-negative weights that sum to one."""
    count = int(round(1.0 / step))
    return [
        (i / count, j / count, (count - i - j) / count)
        for i in range(count + 1)
        for j in range(count - i + 1)
    ]


def _token_similarity(left: str, right: str) -> float:
    """Return a deterministic lexical similarity used during calibration."""
    left_tokens = left.lower().split()
    right_tokens = right.lower().split()
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(set(left_tokens) & set(right_tokens))
    union = len(set(left_tokens) | set(right_tokens))
    jaccard = overlap / max(union, 1)
    sequence = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return 0.5 * jaccard + 0.5 * sequence


def _validation_pair_records(
    validation_df,
    case_dict: dict,
    weights: np.ndarray,
    level_complexity: dict[int, float],
    length_ratio: dict[int, float],
    max_rows: int = 256,
) -> list[dict[str, object]]:
    """Create validation records for fitting retrieval and reranking weights.

    The expert pair is an independently sourced reference for the requested
    transition. It supplies the validation objective; the fitted weights only
    control which pair is selected.
    """
    records: list[dict[str, object]] = []
    rows = validation_df.head(max_rows)
    for _, row in rows.iterrows():
        try:
            source_level = int(row["Rating"])
            source_text = str(row["Sentence"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if source_level <= 1 or source_level > 6 or not source_text:
            continue
        source_complexity = float(
            np.dot(weights, TextComplexityAnalyzer.compute_features(source_text))
        )
        for target_level in range(source_level - 1, 0, -1):
            pairs = case_dict.get(f"{source_level}{target_level}", [])
            if not pairs:
                continue
            candidates = []
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                complex_text, simple_text = map(str, pair)
                target_complexity = level_complexity.get(target_level, 0.0)
                semantic_distance = 1.0 - _token_similarity(
                    source_text, complex_text
                )
                source_distance = abs(
                    source_complexity
                    - float(
                        np.dot(
                            weights,
                            TextComplexityAnalyzer.compute_features(complex_text),
                        )
                    )
                )
                level_distance = abs(
                    float(
                        np.dot(
                            weights,
                            TextComplexityAnalyzer.compute_features(simple_text),
                        )
                    )
                    - target_complexity
                )
                expected_ratio = length_ratio.get(target_level, 1.0)
                actual_ratio = len(simple_text.split()) / max(
                    len(complex_text.split()), 1
                )
                length_distance = abs(actual_ratio - expected_ratio)
                validation_loss = (
                    semantic_distance + level_distance + length_distance
                )
                candidates.append(
                    {
                        "complex_text": complex_text,
                        "simple_text": simple_text,
                        "semantic_distance": semantic_distance,
                        "source_distance": source_distance,
                        "level_distance": level_distance,
                        "length_distance": length_distance,
                        "validation_loss": validation_loss,
                    }
                )
            if candidates:
                records.append(
                    {
                        "source_level": source_level,
                        "target_level": target_level,
                        "source_text": source_text,
                        "candidates": candidates,
                    }
                )
    return records


def _fit_selection_weights(
    records: list[dict[str, object]],
    feature_names: tuple[str, str, str],
    top_k: int = 3,
) -> tuple[list[float], float]:
    """Fit selector weights against the independent validation-pair loss."""
    if not records:
        raise ValueError("Cannot fit selection weights without validation records.")
    best_weights: tuple[float, float, float] | None = None
    best_loss = float("inf")
    for weights in _simplex_weights():
        total_loss = 0.0
        count = 0
        for record in records:
            candidates = record["candidates"]
            ranked = sorted(
                candidates,
                key=lambda item: sum(
                    weight * float(item[name])
                    for weight, name in zip(weights, feature_names)
                ),
            )[:top_k]
            total_loss += sum(
                float(item["validation_loss"]) for item in ranked
            ) / len(ranked)
            count += 1
        mean_loss = total_loss / max(count, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_weights = weights
    assert best_weights is not None
    return list(best_weights), best_loss


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


def fit_iaps_calibration_from_validation(
    training_df,
    validation_df,
    case_dict: dict,
    transition_observations: list[tuple[tuple[float, float, float], int, int, int]]
    | None = None,
) -> dict:
    """Fit IAPS calibration with train anchors and validation transitions.

    Training data fit the complexity model and level anchors. Validation
    transition observations fit beta. Expert parallel pairs and validation
    sentences provide the held-out objective for fitting retrieval and
    reranking weights.
    """
    features = []
    labels = []
    ratings = []
    for _, row in training_df.iterrows():
        try:
            rating = int(row["Rating"])
            text = str(row["Sentence"])
        except (KeyError, TypeError, ValueError):
            continue
        if rating < 1 or rating > 6 or not text.strip():
            continue
        features.append(TextComplexityAnalyzer.compute_features(text))
        labels.append((rating - 1) / 5.0)
        ratings.append(rating)

    if not features:
        raise ValueError("Training split has no usable Rating/Sentence rows.")

    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    weights, *_ = np.linalg.lstsq(feature_array, label_array, rcond=None)
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0:
        weights = np.array([0.4, 0.3, 0.3], dtype=float)
    weights /= weights.sum()

    complexity_values = feature_array @ weights
    level_values: dict[int, list[float]] = {level: [] for level in range(1, 7)}
    for complexity, level in zip(complexity_values, ratings):
        if level in level_values:
            level_values[level].append(float(complexity))
    level_complexity = {
        level: (
            float(np.mean(values))
            if values
            else float((level - 1) / 5.0)
        )
        for level, values in level_values.items()
    }

    mean = float(np.mean(complexity_values))
    std = max(float(np.std(complexity_values)), 1e-8)
    beta = 0.5
    observations = transition_observations or []
    if observations:
        grouped: dict[tuple[int, int], list[int]] = {}
        for _, source, target, success in observations:
            grouped.setdefault((source, target), []).append(success)
        best_loss = float("inf")
        for candidate in np.linspace(0.0, 3.0, 61):
            loss = 0.0
            for feature_values, source, target, success in observations:
                complexity = float(np.dot(weights, feature_values))
                z = (complexity - mean) / std
                rates = grouped[(source, target)]
                base_rate = (sum(rates) + 1.0) / (len(rates) + 2.0)
                probability = 1.0 / (
                    1.0
                    + np.exp(
                        -(
                            np.log(base_rate / (1.0 - base_rate))
                            - candidate * z * (source - target)
                        )
                    )
                )
                loss -= success * np.log(max(probability, 1e-8))
                loss -= (1 - success) * np.log(max(1 - probability, 1e-8))
            if loss < best_loss:
                best_loss = loss
                beta = float(candidate)

    length_ratio = length_ratios_from_parallel_cases(case_dict)
    validation_records = _validation_pair_records(
        validation_df,
        case_dict,
        weights,
        level_complexity,
        length_ratio,
    )
    retrieval_weights, retrieval_loss = _fit_selection_weights(
        validation_records,
        ("semantic_distance", "source_distance", "level_distance"),
    )
    rerank_weights, rerank_loss = _fit_selection_weights(
        validation_records,
        ("semantic_distance", "level_distance", "length_distance"),
        top_k=1,
    )
    return {
        "weights": weights.tolist(),
        "mean": mean,
        "std": std,
        "beta": beta,
        "level_complexity": level_complexity,
        "length_ratio": length_ratio,
        "length_ratio_source": "expert parallel case pairs",
        "retrieval_weights": retrieval_weights,
        "rerank_weights": rerank_weights,
        "retrieval_fit_loss": retrieval_loss,
        "rerank_fit_loss": rerank_loss,
        "validation_pair_count": len(validation_records),
        "retrieval_weights_source": "validation pair-loss optimization",
        "rerank_weights_source": "validation pair-loss optimization",
        "fit_source": (
            "training complexity fit; validation transition, retrieval, "
            "and reranking fit"
        ),
        "r_base_success_mode": "predicted_level <= target_level",
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
