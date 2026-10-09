"""奖励矩阵构建与 R_base 基础转移成功率（统一单文件实现）。

由原 code.md 中两个矩阵构建脚本（matrix / matrix_CEFR）及其共享模块
matrix/common.py 合并而来，matrix/ 目录仅保留本文件：

- 语料切换不再使用 --corpus 参数，改为直接修改本文件顶部的
  数据集 / 模型 / 输出路径变量（README-EN 或 CEFR-SP 换路径即可）；
- 奖励矩阵键统一为 "{起始级}_{目标级}" 数字格式（如 "6_3"），
  与 core/path_planner.py 的 _get_key 完全一致，可直接被路径规划消费；
- 新增论文 3.2 节 R_base(i, j) 基础转移成功率实现（--mode r_base）。
"""

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from core.llm_client import OpenAICompatGenerator, load_dotenv_file

# ============================================================
# 路径配置：切换语料 / 模型时只需修改以下变量
# ============================================================

# vLLM 本地模型路径（生成简化句用，README / CEFR-SP 共用）
MODEL_PATH = ""

# 训练集路径（README-EN 训练集 或 CEFR-SP 训练集，按语料切换）
TRAIN_DATA_PATH = ""

# 验证集路径（README-EN 验证集 283 条 或 CEFR-SP 验证集 1366 条，按语料切换）
VAL_DATA_PATH = ""

# STS 模型：README 语料用在线模型名 "all-MiniLM-L6-v2"，CEFR-SP 语料用本地模型路径
STS_MODEL = "all-MiniLM-L6-v2"

# 是否为本地 STS 模型路径（True 时提示文案与显存释放逻辑按本地处理）
STS_IS_LOCAL = False

# 奖励矩阵输出路径（按语料切换：README 用 gemma 矩阵名，CEFR-SP 用 llama 矩阵名）
OUTPUT_PATH = ""

# R_base 基础转移成功率输出路径
R_BASE_OUTPUT_PATH = "r_base_matrix.json"

# readabert-en 本地模型路径（句子等级预测用，与 evaluation/eval_res.py 一致）
READABERT_LOCAL_PATH = ""

# ============================================================
# 常量
# ============================================================

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

LEVEL_TO_INT = {level: idx for idx, level in enumerate(CEFR_LEVELS)}

RATING_TO_CEFR = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}

TOP_K_EXEMPLARS = 3

# R_base 考虑的初始等级集合：所有仍有更低目标等级的 CEFR 等级。
R_BASE_SOURCE_LEVELS = [6, 5, 4, 3, 2]

# R_base 基础提示词：不使用任何示例，要求将句子直接降级到目标等级。
BASE_PROMPT_TEMPLATE = (
    "Please simplify the following complex sentence to make it "
    "suitable for {target_level} CEFR level learners.\n\n"
    "Use simpler vocabulary, shorter sentences, and clearer grammar "
    "structures while preserving the core meaning.\n\n"
    "Now simplify this sentence:\nComplex: {sentence}\nSimple:"
)


def get_reward_score(pred_level, target_level):
    """根据真实等级与生成等级计算奖励分数。

    Args:
        pred_level: 模型预测的 CEFR 等级（字符串）。
        target_level: 期望的目标 CEFR 等级（字符串）。

    Returns:
        奖励分数：完全匹配 1.0，相邻 0.5，其余 -1.0。
    """
    pred_idx = LEVEL_TO_INT.get(pred_level, -1)

    target_idx = LEVEL_TO_INT.get(target_level, -1)

    if pred_idx < 0 or target_idx < 0:
        return -1.0

    diff = abs(pred_idx - target_idx)

    if diff == 0:
        return 1.0

    elif diff == 1:
        return 0.5

    else:
        return -1.0


def generate_with_vllm(prompts):
    """使用 vLLM 批量生成简化句。

    Args:
        prompts: 提示词列表。

    Returns:
        生成的简化句列表。
    """
    print(f">>> 生成 {len(prompts)} 个简化版本...")

    server_url = os.getenv("OPENAI_API_BASE_URL", "").strip()
    server_model = os.getenv("OPENAI_API_MODEL", "").strip() or MODEL_PATH
    if server_url:
        if not server_model or "替换为" in server_model:
            raise ValueError(
                "Set OPENAI_API_MODEL to the exact model id returned by "
                "GET /v1/models."
            )
        generator = OpenAICompatGenerator(
            api_base_url=server_url,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            model_name=server_model,
            temperature=0.0,
            top_p=0.95,
            max_tokens=128,
            timeout=300,
        )
        return [text.strip() for text in generator.generate_texts(prompts)]

    from vllm import LLM, SamplingParams

    print("\n>>> 加载本地 vLLM...")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128, top_p=0.95)
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, enforce_eager=False)
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [out.outputs[0].text.strip() for out in outputs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return generated_texts


def build_transition_prompts(
    df_val, df_train, sts_model=STS_MODEL, sts_is_local=STS_IS_LOCAL
):
    """使用张量化操作极速构建动态 Few-shot 提示。

    Args:
        df_val: 验证集 DataFrame。
        df_train: 训练集 DataFrame。
        sts_model: STS 模型名或本地模型路径（README 在线 / CEFR-SP 本地）。
        sts_is_local: 是否为本地路径（影响提示文案）。

    Returns:
        prompts 列表与对应 metadata 列表。
    """
    try:
        from sentence_transformers import SentenceTransformer, util

    except ImportError:
        print(" 请先运行：pip install sentence-transformers")

        return None, None

    if sts_is_local:
        print(f"\n>>> [1/2] 从本地加载 STS 模型: {sts_model} (极速模式)...")

    else:
        print("\n>>> [1/2] 加载 STS 模型并计算训练集全局特征矩阵 (极速模式)...")

    sts_model = SentenceTransformer(sts_model)

    sts_model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_sentences_by_level = defaultdict(list)

    for _, row in df_train.iterrows():
        level = RATING_TO_CEFR.get(int(row["Rating"]), None)

        if level:
            train_sentences_by_level[level].append(row["Sentence"])

    train_embeddings_by_level = {}

    for level, sents in train_sentences_by_level.items():
        train_embeddings_by_level[level] = sts_model.encode(
            sents, convert_to_tensor=True
        )

    prompts = []

    metadata = []

    print(">>> [2/2] 正在为验证集匹配相似句并组装 Prompts...")

    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc="组装进度"):
        src_level = RATING_TO_CEFR.get(int(row["Rating"]), None)

        val_sentence = row["Sentence"]

        if src_level not in ["C2", "C1", "B2"]:
            continue

        val_emb = sts_model.encode(val_sentence, convert_to_tensor=True)

        target_levels = [
            tgt for tgt in CEFR_LEVELS if LEVEL_TO_INT[tgt] < LEVEL_TO_INT[src_level]
        ]

        for tgt_level in target_levels:
            exemplar_sents = []

            if (
                tgt_level in train_embeddings_by_level
                and len(train_sentences_by_level[tgt_level]) > 0
            ):
                target_corpus_embs = train_embeddings_by_level[tgt_level]

                cos_scores = util.pytorch_cos_sim(val_emb, target_corpus_embs)[0]

                k = min(TOP_K_EXEMPLARS, len(cos_scores))

                top_k_indices = torch.topk(cos_scores, k=k).indices.tolist()

                exemplar_sents = [
                    train_sentences_by_level[tgt_level][i] for i in top_k_indices
                ]

            few_shot_prompt = f"""

Please simplify the following complex sentence to make it suitable for {tgt_level} CEFR level learners.



Use simpler vocabulary, shorter sentences, and clearer grammar structures while preserving the core meaning.



Examples of {tgt_level} level simplifications:"""

            for idx, exemplar in enumerate(exemplar_sents, 1):
                few_shot_prompt += f"\nExample {idx}: {exemplar}"

            few_shot_prompt += (
                f"\n\nNow simplify this sentence:\nComplex: {val_sentence}\nSimple:"
            )

            prompts.append(few_shot_prompt)

            metadata.append({"src_level": src_level, "tgt_level": tgt_level})

    print(">>> 释放 STS 模型显存...")

    del sts_model

    del train_embeddings_by_level

    gc.collect()

    torch.cuda.empty_cache()

    return prompts, metadata


def evaluate_with_readme(generated_texts):
    """使用 ReadMe 评估生成句的可读性等级（奖励矩阵构建用）。

    Args:
        generated_texts: 生成句列表。

    Returns:
        predictions 等级列表。
    """
    from readmepp import ReadMe

    print("\n>>> 加载 ReadMe 预测器...")

    predictor = ReadMe(lang="en")

    predictions = []

    for gen_text in tqdm(generated_texts, desc="评估进度"):
        try:
            pred_val = predictor.predict(gen_text)

            if isinstance(pred_val, (int, np.integer, float)):
                pred_level = RATING_TO_CEFR.get(int(pred_val), "A1")

            elif isinstance(pred_val, str):
                pred_level = pred_val if pred_val in CEFR_LEVELS else "A1"

            else:
                pred_level = "A1"

            predictions.append(pred_level)

        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            predictions.append("A1")

    return predictions


def predict_readabert_en(generated_texts):
    """使用 readabert-en 本地模型预测句子 CEFR 等级（R_base 用）。

    与 evaluation/eval_res.py 的加载方式一致：从本地路径加载
    transformers 序列分类模型，输出 logits 取 argmax 后 +1 映射为 1-6 等级。

    Args:
        generated_texts: 生成句列表。

    Returns:
        预测等级整数列表（1-6，对应 A1-C2）。
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"\n>>> 加载 readabert-en 评估模型: {READABERT_LOCAL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        READABERT_LOCAL_PATH, local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        READABERT_LOCAL_PATH, local_files_only=True
    )

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    predictions = []

    for gen_text in tqdm(generated_texts, desc="readabert-en 评估进度"):
        try:
            inputs = tokenizer(
                gen_text, return_tensors="pt", truncation=True, max_length=512
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits

            pred_cefr = logits.argmax(dim=-1).item() + 1

            predictions.append(pred_cefr)

        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            predictions.append(None)

    return predictions


def build_reward_matrix(predictions, metadata):
    """构建完整奖励矩阵（数字键格式，与 core/path_planner 对齐）。

    Args:
        predictions: ReadMe 预测等级列表（CEFR 字符串）。
        metadata: 每条样本的 (src_level, tgt_level) 元信息列表。

    Returns:
        归一化奖励矩阵 dict，键形如 "6_3"（起始级_目标级，数字）。
    """
    transition_scores = defaultdict(list)

    print("\n>>> 计算奖励分...")

    for pred_level, meta in zip(predictions, metadata):
        score = get_reward_score(pred_level, meta["tgt_level"])

        transition_scores[(meta["src_level"], meta["tgt_level"])].append(score)

    mean_rewards = {k: np.mean(v) for k, v in transition_scores.items()}

    print("\n>>> 原始奖励（未归一化）：")

    for (src, tgt), val in sorted(mean_rewards.items()):
        print(
            f"  R[{src}  {tgt}] = {val:.4f}  (n={len(transition_scores[(src, tgt)])})"
        )

    min_val, max_val = min(mean_rewards.values()), max(mean_rewards.values())

    normalized_matrix = {}

    for (src, tgt), v in mean_rewards.items():
        key = f"{LEVEL_TO_INT[src] + 1}_{LEVEL_TO_INT[tgt] + 1}"

        normalized_matrix[key] = (
            0.5 if max_val == min_val else (v - min_val) / (max_val - min_val)
        )

    print(f"\n>>> 归一化后的奖励矩阵（min={min_val:.4f}, max={max_val:.4f}）：")

    for (src, tgt), val in sorted(mean_rewards.items()):
        key = f"{LEVEL_TO_INT[src] + 1}_{LEVEL_TO_INT[tgt] + 1}"

        print(f"  R[{key}] = {normalized_matrix[key]:.4f}")

    return normalized_matrix


def save_matrix(matrix, output_path, label="README"):
    """将奖励矩阵写入 JSON 文件。

    Args:
        matrix: 奖励矩阵字典（键形如 "6_3"）。
        output_path: 输出文件路径。
        label: 语料标识（README / CEFR-SP）。
    """
    matrix_dict = {
        "matrix": matrix,
        "levels": CEFR_LEVELS,
        "note": f"Generated with dynamic semantic STS top-3 exemplars for {label}",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matrix_dict, f, indent=2)

    print(f"\n 矩阵已保存到：{output_path}")


def build_r_base_prompts(df_val):
    """从验证集构造 R_base 基础提示词（无示例）与 (i, j) 组合。

    Args:
        df_val: 验证集 DataFrame（含 Rating 1-6 与 Sentence 列）。

    Returns:
        prompts 列表与对应 (src_int, tgt_int) 组合列表。
    """
    prompts = []

    combos = []

    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc="R_base 提示词组装"):
        src_int = int(row["Rating"])

        if src_int not in R_BASE_SOURCE_LEVELS:
            continue

        sentence = str(row["Sentence"]).strip()

        src_level = RATING_TO_CEFR[src_int]

        for tgt_int in range(src_int - 1, 0, -1):
            tgt_level = RATING_TO_CEFR[tgt_int]

            if LEVEL_TO_INT[tgt_level] >= LEVEL_TO_INT[src_level]:
                continue

            prompt = BASE_PROMPT_TEMPLATE.format(
                target_level=tgt_level, sentence=sentence
            )

            prompts.append(prompt)

            combos.append((src_int, tgt_int))

    return prompts, combos


def compute_r_base(df_val, laplace_alpha=1.0):
    """计算论文 3.2 节 R_base(i, j) 基础转移成功率矩阵。

    流程：从验证集提取初始等级为 i 的句子 -> 使用最基础提示词要求模型
    直接降级到目标等级 j -> vLLM 生成简化句 -> readabert-en 预测等级 ->
    预测等级等于目标等级 j 记为成功 -> 成功数 / 总数。

    Args:
        df_val: 验证集 DataFrame。
        成功口径固定为预测等级严格等于目标等级。

    Returns:
        R_base 矩阵 dict，键形如 "6_3"，值为成功率（0-1）。
    """
    print("\n>>> [1/3] 组装 R_base 基础提示词...")

    prompts, combos = build_r_base_prompts(df_val)

    if not prompts:
        print(" 验证集中没有匹配的初始等级样本，请检查 VAL_DATA_PATH。")

        return {}

    print(f">>> [2/3] vLLM 基础提示词生成（共 {len(prompts)} 条）...")

    generated_texts = generate_with_vllm(prompts)

    print(">>> [3/3] readabert-en 等级预测...")

    pred_levels = predict_readabert_en(generated_texts)

    success_by_key = defaultdict(int)

    total_by_key = defaultdict(int)

    for (src_int, tgt_int), pred_int in zip(combos, pred_levels):
        total_by_key[f"{src_int}_{tgt_int}"] += 1

        if pred_int is None:
            ok = False
        else:
            ok = pred_int == tgt_int

        if ok:
            success_by_key[f"{src_int}_{tgt_int}"] += 1

    r_base = {}

    print("\n>>> R_base(i, j) 基础转移成功率：")

    for key in sorted(
        total_by_key, key=lambda k: (int(k.split("_")[0]), -int(k.split("_")[1]))
    ):
        n = total_by_key[key]

        raw_rate = success_by_key[key] / n
        r_base[key] = {
            "success": success_by_key[key],
            "total": n,
            "raw_rate": raw_rate,
            "smoothed_rate": (success_by_key[key] + laplace_alpha)
            / (n + 2.0 * laplace_alpha),
        }

        print(
            f"  R_base[{key}] = {r_base[key]['smoothed_rate']:.4f} "
            f"(raw={raw_rate:.4f}, n={n}, success_mode=exact)"
        )

    return r_base


def run_r_base():
    """R_base 主流程：读取验证集 -> 计算 R_base 矩阵 -> 保存 JSON。"""
    print("=" * 60)

    print(" 计算论文 3.2 节 R_base(i, j) 基础转移成功率")

    print("=" * 60)

    if not Path(VAL_DATA_PATH).exists():
        print(" 找不到验证集文件，请检查 VAL_DATA_PATH！")

        print(f" 期望路径: {VAL_DATA_PATH}")

        return

    df_val = pd.read_csv(VAL_DATA_PATH)

    r_base = compute_r_base(df_val, laplace_alpha=1.0)

    if r_base:
        save_matrix(r_base, R_BASE_OUTPUT_PATH, label="R_base")


def main():
    """奖励矩阵构建主流程（统一单文件入口）。"""
    global MODEL_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, STS_MODEL
    global STS_IS_LOCAL, OUTPUT_PATH, R_BASE_OUTPUT_PATH, READABERT_LOCAL_PATH

    parser = argparse.ArgumentParser(description="Build Reward Matrix / R_base")

    parser.add_argument(
        "--mode",
        type=str,
        default="reward",
        choices=["reward", "r_base"],
        help="reward=构建奖励矩阵, r_base=计算基础转移成功率",
    )

    parser.add_argument("--corpus", choices=["README", "CEFR-SP"], default="README")
    parser.add_argument("--model_path", default=os.getenv("VLLM_MODEL_PATH", ""))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--train_path", default=None)
    parser.add_argument("--val_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--r_base_output_path", default=None)
    parser.add_argument("--readabert_path", default=os.getenv("READABERT_LOCAL_PATH", ""))
    parser.add_argument("--sts_model", default=STS_MODEL)
    parser.add_argument("--sts_is_local", action="store_true")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_file(repo_root / "local_config.env")
    data_root = Path(args.data_root).expanduser() if args.data_root else repo_root / "data" / "en"
    if args.corpus == "README":
        default_train = data_root / "readme_en_train.csv"
        default_val = data_root / "readme_en_val.csv"
    else:
        default_train = data_root / "CEFR-SP_en_train.csv"
        default_val = data_root / "CEFR-SP_en_dev.csv"

    MODEL_PATH = args.model_path or os.getenv("OPENAI_API_MODEL", "").strip()
    TRAIN_DATA_PATH = args.train_path or str(default_train)
    VAL_DATA_PATH = args.val_path or str(default_val)
    STS_MODEL = args.sts_model
    STS_IS_LOCAL = args.sts_is_local
    corpus_name = args.corpus.lower().replace("-", "_")
    OUTPUT_PATH = args.output_path or str(
        repo_root / "LLMGeneration" / "matrices" / f"reward_matrix_{corpus_name}.json"
    )
    R_BASE_OUTPUT_PATH = args.r_base_output_path or str(
        repo_root / "LLMGeneration" / "matrices" / f"r_base_{corpus_name}.json"
    )
    READABERT_LOCAL_PATH = args.readabert_path

    if not MODEL_PATH:
        raise ValueError(
            "Missing --model_path/VLLM_MODEL_PATH or OPENAI_API_MODEL."
        )
    if args.mode == "r_base" and not READABERT_LOCAL_PATH:
        raise ValueError("Missing --readabert_path or READABERT_LOCAL_PATH.")

    if args.mode == "r_base":
        run_r_base()

        return

    print("=" * 60)

    print(" 构建奖励矩阵（语料由顶部路径变量决定）")

    print("=" * 60)

    if not Path(TRAIN_DATA_PATH).exists() or not Path(VAL_DATA_PATH).exists():
        print(" 找不到训练集或验证集文件，请检查路径！")

        print(f" 期望路径:\n- {TRAIN_DATA_PATH}\n- {VAL_DATA_PATH}")

        return

    df_train = pd.read_csv(TRAIN_DATA_PATH)

    df_val = pd.read_csv(VAL_DATA_PATH)

    prompts, metadata = build_transition_prompts(
        df_val, df_train, sts_model=STS_MODEL, sts_is_local=STS_IS_LOCAL
    )

    if not prompts:
        return

    generated_texts = generate_with_vllm(prompts)

    predictions = evaluate_with_readme(generated_texts)

    matrix = build_reward_matrix(predictions, metadata)

    save_matrix(matrix, OUTPUT_PATH, label="README" if not STS_IS_LOCAL else "CEFR-SP")


if __name__ == "__main__":
    main()
