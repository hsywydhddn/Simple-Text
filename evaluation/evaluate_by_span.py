import math
import os
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from bert_score import score as calc_bert_score
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import BERTSCORE_LOCAL_PATH, READABERT_LOCAL_PATH, STS_LOCAL_PATH

os.environ["HF_HUB_OFFLINE"] = "1"

os.environ["TRANSFORMERS_OFFLINE"] = "1"


REPO_ROOT = Path(__file__).resolve().parents[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
readme_tokenizer = None
readme_model = None
sts_model = None


def load_models():
    """在真正开始评估时加载模型，避免导入模块就触发模型加载。"""
    global readme_tokenizer, readme_model, sts_model

    if not READABERT_LOCAL_PATH or not STS_LOCAL_PATH:
        raise ValueError(
            "Set READABERT_LOCAL_PATH and STS_LOCAL_PATH before evaluation."
        )

    print("=" * 80)
    print(" 初始化环境与模型 (多指标全面评估)")
    print("=" * 80)
    print(f"-> 正在加载 ReadMe CEFR 分类器: {READABERT_LOCAL_PATH}")

    readme_tokenizer = AutoTokenizer.from_pretrained(
        READABERT_LOCAL_PATH, local_files_only=True
    )
    readme_model = AutoModelForSequenceClassification.from_pretrained(
        READABERT_LOCAL_PATH, local_files_only=True
    )
    readme_model.eval()
    readme_model.to(device)

    print(f"-> 正在加载 STS 语义模型: {STS_LOCAL_PATH}")
    sts_model = SentenceTransformer(STS_LOCAL_PATH)
    sts_model.to(device)
    print(" 模型加载完毕！开始切片算分...\n")


def calculate_sts(sources, targets):
    """计算语义保留度 STS"""

    if not sources or not targets:
        return 0.0

    with torch.no_grad():
        embs_src = sts_model.encode(
            sources, batch_size=64, convert_to_tensor=True, show_progress_bar=False
        )

        embs_pred = sts_model.encode(
            targets, batch_size=64, convert_to_tensor=True, show_progress_bar=False
        )

        sts_scores = F.cosine_similarity(embs_src, embs_pred)

        return sts_scores.mean().item() * 100


def calculate_bs_f1(sources, targets):
    """计算 BERTScore F1"""

    if not sources or not targets:
        return 0.0

    _, _, F1 = calc_bert_score(
        targets,
        sources,
        model_type=BERTSCORE_LOCAL_PATH,
        num_layers=17,
        device=device,
        verbose=False,
    )

    return F1.mean().item() * 100


def get_cefr_metrics(generated_texts, target_cefr):
    """联合计算 AdjAcc, ExactAcc, RMSE"""

    if not generated_texts:
        return 0.0, 0.0, 0.0

    total_exact_hits = 0

    total_adj_hits = 0

    sum_squared_error = 0

    total_valid = len(generated_texts)

    for pred in generated_texts:
        inputs = readme_tokenizer(
            pred, return_tensors="pt", truncation=True, max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = readme_model(**inputs).logits

            pred_cefr = logits.argmax(dim=-1).item() + 1

        diff = abs(pred_cefr - target_cefr)

        if diff == 0:
            total_exact_hits += 1

        if diff <= 1:
            total_adj_hits += 1

        sum_squared_error += diff**2

    exact_acc = (total_exact_hits / total_valid) * 100

    adj_acc = (total_adj_hits / total_valid) * 100

    rmse = math.sqrt(sum_squared_error / total_valid)

    return adj_acc, exact_acc, rmse


def evaluate_span(dir_path, begin_cefr, target_cefr, model_name):
    """读取数据并整合所有指标"""

    file_path = os.path.join(dir_path, f"en_few-shot_beginCEFR{begin_cefr}.csv")

    if not os.path.exists(file_path):
        print(f" [跳过] 找不到文件: {file_path}")

        return None

    df = pd.read_csv(file_path)

    target_col = f"llm_gene_CEFR{target_cefr}"

    if target_col not in df.columns:
        print(f" [跳过] 找不到目标列: {target_col}")

        return None

    valid_data = df.dropna(subset=["Sentence", target_col])

    sources = valid_data["Sentence"].tolist()

    gens = valid_data[target_col].tolist()

    if len(sources) == 0:
        return None

    sts_score = calculate_sts(sources, gens)

    bs_f1_score = calculate_bs_f1(sources, gens)

    adj_score, exact_score, rmse_score = get_cefr_metrics(gens, target_cefr)

    print(f"[{model_name}] 跨度 {begin_cefr} -> {target_cefr} | 样本数: {len(sources)}")

    print(
        f"    => AdjAcc: {adj_score:5.2f}% | ExactAcc: {exact_score:5.2f}% | RMSE: {rmse_score:5.2f} | BS_F1: {bs_f1_score:5.2f} | STS: {sts_score:5.2f}\n"
    )

    return {
        "Model": model_name.strip(),
        "Begin_CEFR": begin_cefr,
        "Target_CEFR": target_cefr,
        "AdjAcc()": round(adj_score, 2),
        "ExactAcc()": round(exact_score, 2),
        "RMSE()": round(rmse_score, 4),
        "BS_F1()": round(bs_f1_score, 2),
        "STS()": round(sts_score, 2),
    }


def main():
    """分跨度评估主流程。"""
    parser = argparse.ArgumentParser(description="Evaluate results by CEFR span")
    parser.add_argument(
        "--baseline_dir",
        default=str(REPO_ROOT / "LLMGeneration" / "README" / "zero_shot_baseline_gemma_2_9b_it"),
    )
    parser.add_argument(
        "--ours_dir",
        default=str(REPO_ROOT / "LLMGeneration" / "README" / "full_method"),
    )
    parser.add_argument(
        "--output_csv",
        default=str(REPO_ROOT / "LLMGeneration" / "evaluation_by_span.csv"),
    )
    args = parser.parse_args()
    load_models()

    spans = [
        ("小跨度 (B2 -> B1)", 4, 3),
        ("中跨度 (C2 -> B1)", 6, 3),
        ("大跨度 (C2 -> A1)", 6, 1),
    ]

    all_results = []

    for span_name, begin, target in spans:
        print(f"========== 正在评估 {span_name} ==========")

        res_base = evaluate_span(args.baseline_dir, begin, target, "Baseline ")

        if res_base:
            res_base["Span"] = span_name

            all_results.append(res_base)

        res_ours = evaluate_span(args.ours_dir, begin, target, "Ours")

        if res_ours:
            res_ours["Span"] = span_name

            all_results.append(res_ours)

    if all_results:
        columns_order = [
            "Span",
            "Model",
            "Begin_CEFR",
            "Target_CEFR",
            "AdjAcc()",
            "ExactAcc()",
            "RMSE()",
            "BS_F1()",
            "STS()",
        ]

        df_results = pd.DataFrame(all_results)[columns_order]

        save_path = args.output_csv

        df_results.to_csv(save_path, index=False)

        print(f" 所有评估指标已计算完毕，并保存至: {save_path}")

        print("\n最终结果概览:")

        print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
