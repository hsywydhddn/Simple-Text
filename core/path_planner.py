"""DP 路径规划与奖励矩阵加载。

从原 improve.py 拆分出的公共模块，实现基于奖励矩阵的多目标
路径规划（复杂度/长度/语义三目标加权），用于生成 CEFR 等级
渐进式简化策略。
"""

import json
import math


def _get_key(start: int, target: int) -> str:
    """将 (起始级, 目标级) 转为矩阵键字符串。

    Args:
        start: 起始 CEFR 等级。
        target: 目标 CEFR 等级。

    Returns:
        形如 "6_3" 的键。
    """
    return f"{start}_{target}"


def load_reward_matrix(reward_matrix_path: str) -> dict:
    """从 JSON 文件加载奖励矩阵。

    Args:
        reward_matrix_path: 奖励矩阵 JSON 路径。

    Returns:
        奖励矩阵 dict，含 'matrix' 键，内部键形如 "6_3"。
    """
    with open(reward_matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matrix = data.get("matrix", data)
    return matrix


def load_iaps_calibration(calibration_path: str) -> dict:
    """Load instance-adaptive path calibration parameters."""
    with open(calibration_path, "r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    required = ("weights", "mean", "std", "beta", "level_complexity", "length_ratio")
    missing = [key for key in required if key not in calibration]
    if missing:
        raise ValueError(
            f"Calibration file {calibration_path} is missing keys: {', '.join(missing)}"
        )
    return calibration


def _transition_probability(
    start: int,
    target: int,
    complexity: float,
    r_base: dict,
    calibration: dict,
) -> float:
    """Calculate p(i,j|S) using the calibrated logit interaction model."""
    raw_probability = r_base.get(_get_key(start, target))
    if isinstance(raw_probability, dict):
        raw_probability = raw_probability.get(
            "smoothed_rate", raw_probability.get("raw_rate")
        )
    if raw_probability is None:
        raise KeyError(f"Missing R_base transition {_get_key(start, target)}")

    probability = min(max(float(raw_probability), 1e-6), 1.0 - 1e-6)
    logit_base = math.log(probability / (1.0 - probability))
    beta = max(0.0, float(calibration["beta"]))
    logit_value = logit_base - beta * complexity * (start - target)
    if logit_value >= 0:
        exp_neg = math.exp(-logit_value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(logit_value)
    return exp_pos / (1.0 + exp_pos)


def build_instance_adaptive_path(
    start: int,
    target: int,
    sentence_features: tuple[float, float, float],
    r_base: dict,
    calibration: dict,
) -> list:
    """Find the maximum-probability CEFR path for one input sentence."""
    if start <= target:
        return [target]

    weights = calibration["weights"]
    mean = float(calibration["mean"])
    std = max(float(calibration["std"]), 1e-8)
    complexity_score = sum(
        float(weight) * float(value)
        for weight, value in zip(weights, sentence_features)
    )
    complexity = (complexity_score - mean) / std

    costs = {target: 0.0}
    paths = {target: [target]}
    for level in range(target + 1, start + 1):
        best = None
        for nxt in range(target, level):
            if nxt not in costs:
                continue
            probability = _transition_probability(
                level, nxt, complexity, r_base, calibration
            )
            candidate = -math.log(probability) + costs[nxt]
            if best is None or candidate < best[0]:
                best = (candidate, [level] + paths[nxt])
        if best is not None:
            costs[level], paths[level] = best

    return paths.get(start, [start, target])


def build_multi_objective_path(
    start: int,
    target: int,
    reward_matrix: dict,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> list:
    """构建多目标最优简化路径。

    使用动态规划在奖励矩阵上搜索从 start 到 target 的路径，
    路径每一步的综合奖励为：语义奖励 * alpha + 复杂度奖励 * beta +
    长度奖励 * gamma（权重可通过参数调整）。

    Args:
        start: 起始 CEFR 等级（1-6）。
        target: 目标 CEFR 等级（1-6，须小于 start）。
        reward_matrix: 奖励矩阵 dict，键形如 "6_3"。
        alpha: 语义奖励权重。
        beta: 复杂度奖励权重。
        gamma: 长度奖励权重。

    Returns:
        等级列表，形如 [6, 5, 3]，其中末尾为目标等级。
    """
    if start <= target:
        return [target]

    # 对每个中间等级计算累计最优
    # dp[level] = (score, path)
    dp = {start: (0.0, [start])}

    # 按等级递减顺序松弛
    for level in range(start, target, -1):
        if level not in dp:
            continue
        cur_score, cur_path = dp[level]
        for nxt in range(level - 1, target - 1, -1):
            key = _get_key(level, nxt)
            if key not in reward_matrix:
                continue
            step_reward = reward_matrix[key]

            # 分解 reward：语义(总奖励)、复杂度、长度 由调用方编码
            semantic = step_reward
            complexity = 1.0 - abs((nxt / 6.0) - (target / 6.0))
            length_ratio = nxt / level
            score = (
                alpha * semantic + beta * complexity + gamma * min(length_ratio, 1.0)
            )
            new_score = cur_score + score
            new_path = cur_path + [nxt]
            if nxt not in dp or new_score > dp[nxt][0]:
                dp[nxt] = (new_score, new_path)

    if target not in dp:
        return [target]
    return dp[target][1]


def build_path_from_matrix(
    start: int,
    target: int,
    matrix: dict,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    max_hops: int = 4,
) -> list:
    """基于奖励矩阵搜索最优路径（带跳数上限）。

    Args:
        start: 起始 CEFR 等级。
        target: 目标 CEFR 等级。
        matrix: 奖励矩阵 dict，键形如 "6_3"。
        alpha: 语义奖励权重。
        beta: 复杂度奖励权重。
        gamma: 长度奖励权重。
        max_hops: 最大跳数。

    Returns:
        最优路径等级列表，保证末位为目标等级。
    """
    best_path = [target]

    def dfs(level: int, path: list, acc: float) -> None:
        """深度优先搜索多目标打分的最优降级路径。

        Args:
            level: 当前等级。
            path: 已走过的等级序列。
            acc: 累计得分。

        Returns:
            无返回值，结果写入闭包变量 best_path。
        """
        nonlocal best_path
        if level == target:
            if acc > 0 or len(best_path) == 1:
                best_path = list(path)
            return
        if len(path) - 1 >= max_hops:
            return
        for nxt in range(level - 1, target - 1, -1):
            key = _get_key(level, nxt)
            if key not in matrix:
                continue
            step = matrix[key]
            semantic = step
            complexity = 1.0 - abs((nxt / 6.0) - (target / 6.0))
            length_ratio = nxt / level
            score = (
                alpha * semantic + beta * complexity + gamma * min(length_ratio, 1.0)
            )
            dfs(nxt, path + [nxt], acc + score)

    dfs(start, [start], 0.0)
    return best_path


def get_strategy(
    start: int,
    target: int,
    reward_matrix: dict,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> list:
    """获取单条 (起始级, 目标级) 的简化策略路径。

    Args:
        start: 起始 CEFR 等级。
        target: 目标 CEFR 等级。
        reward_matrix: 奖励矩阵 dict。
        alpha: 语义奖励权重。
        beta: 复杂度奖励权重。
        gamma: 长度奖励权重。

    Returns:
        路径等级列表。
    """
    return build_multi_objective_path(start, target, reward_matrix, alpha, beta, gamma)
