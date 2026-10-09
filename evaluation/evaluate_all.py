import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bert_score import score as calculate_bert_score
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import BERTSCORE_LOCAL_PATH, READABERT_LOCAL_PATH, STS_LOCAL_PATH

os.environ["HF_HUB_OFFLINE"] = "1"

os.environ["TRANSFORMERS_OFFLINE"] = "1"


REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate_global():
    """全量多维评估：输出论文六维指标（rho/AdjAcc/ExactAcc/RMSE/BS_F1/STS）。"""
    print("=" * 65)

    print(" 开始全量多维评估：(完整对齐论文 6 维指标)")

    print("=" * 65)

    parser = argparse.ArgumentParser(description="Evaluate all generated results")
    parser.add_argument(
        "--data_dir",
        default=str(REPO_ROOT / "LLMGeneration" / "README" / "full_method"),
        help="Directory containing en_few-shot_beginCEFR*.csv files.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/4] 正在从本地加载 ReadMe 评估模型: {READABERT_LOCAL_PATH}")

    readme_tokenizer = AutoTokenizer.from_pretrained(
        READABERT_LOCAL_PATH, local_files_only=True
    )

    readme_model = AutoModelForSequenceClassification.from_pretrained(
        READABERT_LOCAL_PATH, local_files_only=True
    )

    readme_model.eval()

    readme_model.to(device)

    print(f"[2/4] 正在从本地加载 STS 语义模型: {STS_LOCAL_PATH}")

    sts_model = SentenceTransformer(STS_LOCAL_PATH)

    sts_model.to(device)

    print("[3/4] 正在从 CSV 提取所有数据对...")

    all_sources = []

    all_preds = []

    all_targets = []

    for begin_cefr in [6, 5, 4]:
        file_path = os.path.join(data_dir, f"en_few-shot_beginCEFR{begin_cefr}.csv")

        if not os.path.exists(file_path):
            print(f" 找不到文件: {file_path}，跳过。")

            continue

        df = pd.read_csv(file_path)

        for target_cefr in [3, 2, 1]:
            col_name = f"llm_gene_CEFR{target_cefr}"

            if col_name not in df.columns:
                continue

            for idx, row in df.iterrows():
                src = str(row["Sentence"]).strip()

                pred = str(row[col_name]).strip() if pd.notna(row[col_name]) else ""

                if not pred:
                    continue

                all_sources.append(src)

                all_preds.append(pred)

                all_targets.append(target_cefr)

    total_valid = len(all_preds)

    if total_valid == 0:
        print(f" 没有找到有效数据，请检查 --data_dir: {data_dir}")

        return

    print(f" 共提取到 {total_valid} 条测试样本。")

    print(" 正在计算 ReadMe 预测阶数 (用于计算 ρ, AdjAcc, ExactAcc, RMSE)...")

    all_pred_cefrs = []

    total_exact_hits = 0

    total_adj_hits = 0

    for pred, target_cefr in tqdm(
        zip(all_preds, all_targets), total=total_valid, desc="ReadMe 评估"
    ):
        inputs = readme_tokenizer(
            pred, return_tensors="pt", truncation=True, max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = readme_model(**inputs).logits

            pred_cefr = logits.argmax(dim=-1).item() + 1

            all_pred_cefrs.append(pred_cefr)

        diff = abs(pred_cefr - target_cefr)

        if diff == 0:
            total_exact_hits += 1

        if diff <= 1:
            total_adj_hits += 1

    targets_np = np.array(all_targets)

    preds_np = np.array(all_pred_cefrs)

    rho, _ = spearmanr(targets_np, preds_np)

    rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))

    overall_exact = (total_exact_hits / total_valid) * 100

    overall_adj = (total_adj_hits / total_valid) * 100

    print(" 正在计算 STS 句级语义相似度 (GPU 批量编码)...")

    with torch.no_grad():
        embs_src = sts_model.encode(
            all_sources, batch_size=64, convert_to_tensor=True, show_progress_bar=False
        )

        embs_pred = sts_model.encode(
            all_preds, batch_size=64, convert_to_tensor=True, show_progress_bar=False
        )

        sts_scores = F.cosine_similarity(embs_src, embs_pred)

        overall_sts = sts_scores.mean().item() * 100

    print(" 正在计算 BERTScore Token 级语义保真度 (纯离线模式)...")

    _, _, F1 = calculate_bert_score(
        all_preds,
        all_sources,
        model_type=BERTSCORE_LOCAL_PATH,
        num_layers=17,
        device=device,
        rescale_with_baseline=True,
        lang="en",
        verbose=False,
    )

    overall_bs_f1 = F1.mean().item() * 100

    print("\n[4/4] 评估完成！最终全局成绩单如下：")

    print("=" * 75)

    print(f" 评估样本总数 (Total Samples) : {total_valid}")

    print("-" * 75)

    print(f" Spearman (ρ)      : {rho:.4f}")

    print(f" 全局 Adj Acc (%)   : {overall_adj:.2f}")

    print(f" 全局 Exact Acc (%) : {overall_exact:.2f}")

    print(f" RMSE               : {rmse:.4f}")

    print(f" BERTScore (BS_F1)  : {overall_bs_f1:.2f}")

    print(f" STS 语义保留 (%)   : {overall_sts:.2f}")

    print("=" * 75)


if __name__ == "__main__":
    evaluate_global()
