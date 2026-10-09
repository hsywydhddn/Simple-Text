import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core.llm_client import VLLMGenerator
from core.prompt_builder import clean_llm_generation, resolve_split_file
from utils import CEFRMAP


def get_zeroshot_prompt(test_sentence: str, target_cefr: int) -> str:
    """构建零样本简化 prompt。

    Args:
        test_sentence: 参数说明。
        target_cefr: 参数说明。

    Returns:
        返回值说明。
    """
    cefr_level = CEFRMAP[target_cefr]

    prompt = (
        "You are an expert linguistic simplifier. "
        f"Your task is to simplify text for {cefr_level} level language learners.\n\n"
        "CONSTRAINTS:\n"
        "- Use simple vocabulary and short sentences.\n"
        "- Do not copy complex words or long phrases from the original.\n"
        "- Keep only the main ideas; drop minor details if needed.\n\n"
        "### YOUR TASK ###\n"
        f"TARGET SENTENCE:\n{test_sentence}\n\n"
        "Return only the final simplified sentence in plain text."
    )

    return prompt


def main():
    """零样本基线推理主流程。"""
    parser = argparse.ArgumentParser(description="Vanilla Zero-shot Baseline")

    parser.add_argument(
        "--corpus", type=str, default="README", choices=["README", "CEFR-SP"]
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("VLLM_MODEL_PATH", ""),
        help="Local model path used by vLLM.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--sample_size", type=int, default=0)

    args = parser.parse_args()

    lang = "en"

    repo_root = Path(__file__).resolve().parents[1]
    if not args.model_path:
        raise ValueError("Missing --model_path or VLLM_MODEL_PATH.")

    model_label = Path(args.model_path).name.replace("-", "_")
    save_dir = repo_root / "LLMGeneration" / args.corpus / (
        f"zero_shot_baseline_{model_label}"
    )

    os.makedirs(save_dir, exist_ok=True)

    test_file = resolve_split_file(args.corpus, lang, "test")

    generator = VLLMGenerator(
        model_name=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.0,
        max_tokens=256,
    )

    for begin_cefr in [6, 5, 4]:
        original_df = pd.read_csv(test_file)

        df = original_df[original_df["Rating"] == begin_cefr].reset_index(drop=True)
        if args.sample_size > 0:
            df = df.head(args.sample_size)

        if len(df) == 0:
            continue

        print(
            f"\n[Zero-shot] Dataset: {args.corpus} | Model: {args.model_path} | Begin CEFR: {begin_cefr} ({len(df)} sents)"
        )

        for target_cefr in [3, 2, 1]:
            generated_list = []

            for _, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Target {CEFRMAP[target_cefr]}"
            ):
                src_sentence = row["Sentence"]

                prompt = get_zeroshot_prompt(src_sentence, target_cefr)

                payload = [{"role": "user", "content": prompt}]

                try:
                    raw_text = generator.generate_text(payload)

                    clean_text = (clean_llm_generation(raw_text) or "").strip()

                except (OSError, RuntimeError, TypeError, ValueError) as e:
                    print(f"Error generating text: {e}")

                    clean_text = src_sentence

                generated_list.append(clean_text)

            df[f"llm_gene_CEFR{target_cefr}"] = generated_list

        out_file = f"{save_dir}/{lang}_few-shot_beginCEFR{begin_cefr}.csv"

        df.to_csv(out_file, index=False)

        print(f"Saved zero-shot results to: {out_file}")


if __name__ == "__main__":
    main()
