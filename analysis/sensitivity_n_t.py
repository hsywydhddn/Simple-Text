import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core.llm_client import VLLMGenerator
from core.path_planner import build_multi_objective_path
from core.prompt_builder import padding_case, resolve_split_file
from method.full_method import (
    SemanticExemplarSelector,
    TextComplexityAnalyzer,
    build_compact_memory,
    extract_facts_heuristic,
    extract_facts_llm,
    generate_candidates,
    get_dynamic_prompt,
    rerank_candidates,
    run_preflight_checks,
    trim_chat_history,
)


def main():
    """N/T 敏感性分析主流程：扫描示例数与步数超参。"""
    parser = argparse.ArgumentParser(description="Hyperparameter Sensitivity Analysis")

    parser.add_argument(
        "--num_candidates", type=int, default=3, help="Number of candidates (N)"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Sampling temperature (T)"
    )

    parser.add_argument(
        "--sample_size", type=int, default=100, help="Subset size to speed up testing"
    )

    parser.add_argument("--max_tokens", type=int, default=512)

    parser.add_argument("--fact_max_tokens", type=int, default=128)

    parser.add_argument("--fact_max_items", type=int, default=5)

    parser.add_argument("--max_history_turns", type=int, default=2)
    parser.add_argument("--model_path", default=os.getenv("VLLM_MODEL_PATH", ""))
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--retrieval_model", default=os.getenv("RETRIEVAL_MODEL", ""))

    args = parser.parse_args()

    print("=" * 50)

    print(
        f" Running Sensitivity Test: N={args.num_candidates}, T={args.temperature}, SampleSize={args.sample_size}"
    )

    print("=" * 50)

    repo_root = Path(__file__).resolve().parents[1]
    save_dir = repo_root / "LLMGeneration" / "Sensitivity" / (
        f"full_method_N{args.num_candidates}_T{args.temperature}"
    )

    os.makedirs(save_dir, exist_ok=True)

    matrix_path = Path(
        os.getenv(
            "REWARD_MATRIX_PATH",
            str(repo_root / "LLMGeneration" / "matrices" / "reward_matrix_readme.json"),
        )
    )

    lang, corpus = "en", "README"

    case_file = Path(
        os.getenv(
            "EXPERT_CASE_FILE",
            str(repo_root / "data" / "prompt_expert" / "en_expert_case_readme.json"),
        )
    )

    run_preflight_checks(case_file, matrix_path)

    if not args.model_path:
        raise ValueError("Missing --model_path or VLLM_MODEL_PATH.")
    if not args.retrieval_model:
        raise ValueError("Missing --retrieval_model or RETRIEVAL_MODEL.")
    selector = SemanticExemplarSelector(args.retrieval_model)

    analyzer = TextComplexityAnalyzer()

    generator = VLLMGenerator(
        model_name=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=0.95,
        repetition_penalty=1.0,
        max_tokens=args.max_tokens,
    )

    fact_generator = VLLMGenerator(
        model_name=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_tokens=args.fact_max_tokens,
    )

    with open(matrix_path, "r") as handle:
        base_reward_matrix = json.load(handle).get("matrix", {})

    with open(case_file) as handle:
        case_dict_expert = padding_case(json.load(handle))

    test_file = resolve_split_file(corpus, lang, "test")

    fact_cache: dict[str, list[str]] = {}

    for begin_cefr in [6, 5]:
        df = pd.read_csv(test_file)

        df = (
            df[df["Rating"] == begin_cefr].reset_index(drop=True).head(args.sample_size)
        )

        print(
            f"\nProcessing begin CEFR {begin_cefr}, total {len(df)} sentences (Sub-sampled)"
        )

        for target_cefr in [3, 1]:
            final_generated_list = []

            for _, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Target {target_cefr}"
            ):
                original_src = row["Sentence"]

                strategy = build_multi_objective_path(
                    begin_cefr, target_cefr, base_reward_matrix
                )

                if strategy and strategy[0] == begin_cefr:
                    strategy = strategy[1:]

                current_text = original_src

                chat_history: list[dict[str, str]] = []

                prev_target = begin_cefr

                cached_facts = fact_cache.get(original_src)

                if cached_facts is None:
                    try:
                        cached_facts = extract_facts_llm(
                            fact_generator, original_src, args.fact_max_items
                        )

                    except (OSError, RuntimeError, TypeError, ValueError):
                        cached_facts = extract_facts_heuristic(
                            original_src, args.fact_max_items
                        )

                    fact_cache[original_src] = cached_facts

                facts = cached_facts

                for step_target in strategy:
                    step_source = (
                        begin_cefr if step_target == strategy[0] else prev_target
                    )

                    candidates = case_dict_expert[f"{step_source}{step_target}"]

                    selected, sims = selector.select_topk_exemplars(
                        current_text,
                        candidates,
                        k=3,
                        target_cefr=step_target,
                        complexity_analyzer=analyzer,
                    )

                    memory_block = build_compact_memory(
                        original_src, facts, current_text, step_target
                    )

                    user_prompt = get_dynamic_prompt(
                        current_text,
                        step_target,
                        selected,
                        sims,
                        memory_block=memory_block,
                    )

                    payload = chat_history + [{"role": "user", "content": user_prompt}]

                    candidates_text = generate_candidates(
                        generator, payload, args.num_candidates
                    )

                    clean_candidates = [cand for cand in candidates_text if cand]

                    if not clean_candidates:
                        clean_candidates = [current_text]

                    clean_text = rerank_candidates(
                        clean_candidates, original_src, step_target, selector, analyzer
                    )

                    chat_history.extend(
                        [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": clean_text},
                        ]
                    )

                    chat_history = trim_chat_history(
                        chat_history, args.max_history_turns
                    )

                    current_text, prev_target = clean_text, step_target

                final_generated_list.append(current_text)

            df[f"llm_gene_CEFR{target_cefr}"] = final_generated_list

        out_file = f"{save_dir}/{lang}_few-shot_beginCEFR{begin_cefr}.csv"

        df.to_csv(out_file, index=False)

        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
