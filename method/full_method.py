import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.llm_client import OpenAICompatGenerator, VLLMGenerator, load_dotenv_file
from core.path_planner import build_instance_adaptive_path
from core.prompt_builder import padding_case, resolve_split_file
from method.common import (
    TARGET_COMPLEXITY,
    TARGET_LENGTH_RATIO,
    TextComplexityAnalyzer,
    build_compact_memory,
    extract_facts_heuristic,
    extract_facts_llm,
    load_iaps_calibration as load_method_calibration,
    generate_candidates,
    get_dynamic_prompt,
    trim_chat_history,
    word_count,
)

STATIC_DP_STRATEGY = {
    "63": [5, 3],
    "62": [5, 3, 2],
    "61": [5, 3, 2, 1],
    "53": [3],
    "52": [3, 2],
    "51": [3, 2, 1],
    "43": [3],
    "42": [3, 2],
    "41": [3, 2, 1],
}


class SemanticExemplarSelector:
    """语义示例选择器：按语义相似度为输入句挑选 top-k 参考示例。

    Args:

    Returns:
        返回值说明。
    """

    def __init__(self, model_path: str | None = None):
        """__init__ 功能。

        Args:
            model_path: 参数说明。

        Returns:
            返回值说明。
        """
        from sentence_transformers import SentenceTransformer

        print(f"Loading local retrieval model: {model_path}")

        self.model = SentenceTransformer(model_path)

        self._complexity_cache: dict[str, float] = {}

    def _get_complexity(self, analyzer, text: str) -> float:
        """_get_complexity 功能。

        Args:
            analyzer: 参数说明。
            text: 参数说明。

        Returns:
            返回值说明。
        """
        cached = self._complexity_cache.get(text)

        if cached is not None:
            return cached

        score = analyzer.compute_overall_complexity(text)

        self._complexity_cache[text] = score

        return score

    def compute_batch_similarity(
        self, reference: str, candidates: list[str]
    ) -> list[float]:
        """compute_batch_similarity 功能。

        Args:
            reference: 参数说明。
            candidates: 参数说明。

        Returns:
            返回值说明。
        """
        if not candidates:
            return []

        embeddings = self.model.encode(
            [reference] + candidates, convert_to_tensor=True, normalize_embeddings=True
        )

        ref_emb = embeddings[0]

        cand_embs = embeddings[1:]

        return [float((ref_emb @ cand_embs[i]).item()) for i in range(len(candidates))]

    def select_topk_exemplars(
        self,
        test_sentence: str,
        candidate_pairs: list[tuple[str, str]],
        k: int = 3,
        target_cefr: int | None = None,
        complexity_analyzer=None,
        weights: tuple[float, float, float] = (0.6, 0.2, 0.2),
        target_complexities: dict[int, float] | None = None,
    ):
        """select_topk_exemplars 功能。

        Args:
            test_sentence: 参数说明。
            candidate_pairs: 参数说明。
            k: 参数说明。
            target_cefr: 参数说明。
            complexity_analyzer: 参数说明。
            weights: 参数说明。

        Returns:
            返回值说明。
        """
        if not candidate_pairs:
            return [], []

        complex_sentences = [pair[0] for pair in candidate_pairs]

        embeddings = self.model.encode(
            [test_sentence] + complex_sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        test_emb = embeddings[0]

        cand_embs = embeddings[1:]

        semantic_sims = [
            float((test_emb @ cand_embs[i]).item())
            for i in range(len(complex_sentences))
        ]

        if target_cefr is None or complexity_analyzer is None:
            top_k_indices = np.argsort(semantic_sims)[-k:][::-1]

            return [candidate_pairs[i] for i in top_k_indices], [
                semantic_sims[i] for i in top_k_indices
            ]

        test_complexity = self._get_complexity(complexity_analyzer, test_sentence)

        target_complexity = (target_complexities or TARGET_COMPLEXITY).get(
            target_cefr, 0.45
        )

        scores = []

        for idx, (complex_sent, simple_sent) in enumerate(candidate_pairs):
            source_deviation = abs(
                test_complexity
                - self._get_complexity(complexity_analyzer, complex_sent)
            )
            target_deviation = abs(
                target_complexity
                - self._get_complexity(complexity_analyzer, simple_sent)
            )
            score = (
                weights[0] * (1.0 - semantic_sims[idx])
                + weights[1] * source_deviation
                + weights[2] * target_deviation
            )

            scores.append(score)

        top_k_indices = np.argsort(scores)[:k]

        return [candidate_pairs[i] for i in top_k_indices], [
            semantic_sims[i] for i in top_k_indices
        ]


def rerank_candidates(
    candidates: list[str],
    original: str,
    current_text: str,
    current_cefr: int,
    target_cefr: int,
    selector: SemanticExemplarSelector,
    analyzer: TextComplexityAnalyzer,
    weights: tuple[float, float, float] = (0.55, 0.30, 0.15),
) -> str:
    """基于语义保持度与等级适配度对候选重排。

    Args:
        candidates: 参数说明。
        original: 参数说明。
        target_cefr: 参数说明。
        selector: 参数说明。
        analyzer: 参数说明。

    Returns:
        返回值说明。
    """
    if not candidates:
        return ""

    if len(candidates) == 1:
        return candidates[0]

    sims = selector.compute_batch_similarity(original, candidates)

    target_complexity = TARGET_COMPLEXITY.get(target_cefr, 0.45)
    current_complexity = TARGET_LENGTH_RATIO.get(current_cefr, 1.0)
    target_length = TARGET_LENGTH_RATIO.get(target_cefr, current_complexity)
    expected_ratio = target_length / max(current_complexity, 1e-8)
    current_len = max(word_count(current_text), 1)

    scores: list[float] = []

    for idx, cand in enumerate(candidates):
        level_deviation = abs(
            analyzer.compute_overall_complexity(cand) - target_complexity
        )
        actual_ratio = word_count(cand) / current_len
        length_deviation = abs(actual_ratio - expected_ratio)
        score = (
            weights[0] * (1.0 - sims[idx])
            + weights[1] * level_deviation
            + weights[2] * length_deviation
        )

        scores.append(score)

    best_idx = int(np.argmin(scores))

    return candidates[best_idx]


def run_preflight_checks(case_file, matrix_path):
    """运行前置检查，确认模型/矩阵/数据文件就绪。

    Args:
        case_file: 参数说明。
        matrix_path: 参数说明。

    Returns:
        返回值说明。
    """
    print("=" * 50)

    print("Running preflight safety checks...")

    if not os.path.exists(case_file):
        print(f"[Error] Expert case file not found at: {case_file}")

        print(
            "Hint: set EXPERT_CASE_FILE or place the expected file under data/prompt_expert/."
        )

        sys.exit(1)

    if not os.path.exists(matrix_path):
        print(f"[Error] Reward matrix not found at: {matrix_path}")

        sys.exit(1)

    try:
        with open(case_file) as handle:
            test_dict = json.load(handle)

        sample_key = next(iter(test_dict))

        sample_pool = test_dict[sample_key]

        if len(sample_pool) > 0:
            first_item = sample_pool[0]

            if not isinstance(first_item, (list, tuple)) or len(first_item) != 2:
                print(f"[Error] Invalid data format in {case_file}!")

                print(f"Found: {type(first_item)} -> {first_item}")

                sys.exit(1)

    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[Error] Failed to parse {case_file}: {exc}")

        sys.exit(1)

    print("All preflight checks passed.")

    print("=" * 50)


def main():
    """主方法推理流程（通用合并版，支持 README / CEFR-SP 语料与全部消融开关）。"""
    parser = argparse.ArgumentParser(
        description="Text Simplification Ablation Study (merged universal)"
    )

    parser.add_argument(
        "--corpus",
        type=str,
        default="README",
        choices=["README", "CEFR-SP"],
        help="Corpus name",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Subdirectory name for saving results (auto default per corpus/ablation)",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root; defaults to the repository data/ directory.",
    )

    parser.add_argument(
        "--matrix_path",
        type=str,
        default=None,
        help="Reward matrix JSON path; defaults to LLMGeneration/matrices/.",
    )

    parser.add_argument(
        "--calibration_path",
        type=str,
        default=None,
        help="IAPS calibration JSON; rho is derived from expert pairs when omitted.",
    )

    parser.add_argument(
        "--expert_case_file",
        type=str,
        default=None,
        help="Expert case JSON path; defaults to data/prompt_expert/.",
    )

    parser.add_argument(
        "--retrieval_model",
        type=str,
        default=None,
        help="Sentence-transformers model path or model ID; can use RETRIEVAL_MODEL.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local model path used by vLLM; can use VLLM_MODEL_PATH.",
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs used by vLLM tensor parallelism.",
    )

    parser.add_argument(
        "--disable_history",
        action="store_true",
        help="Turn off multi-turn chat history",
    )

    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max tokens for LLM generation"
    )

    parser.add_argument(
        "--num_candidates", type=int, default=3, help="Number of candidates per step"
    )

    parser.add_argument(
        "--disable_rerank", action="store_true", help="Disable candidate reranking"
    )

    parser.add_argument(
        "--disable_fact_memory", action="store_true", help="Disable fact memory block"
    )

    parser.add_argument(
        "--disable_dynamic_path",
        action="store_true",
        help="Disable dynamic path and use static DP strategy",
    )

    parser.add_argument(
        "--disable_dynamic_retrieval",
        action="store_true",
        help="Disable semantic retrieval, use random exemplars (ablation)",
    )

    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Sampling temperature"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Subset size per begin CEFR (0 = all)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override LLM model name; CEFR-SP can discover it from the configured API.",
    )

    parser.add_argument(
        "--l1", type=float, default=0.6, help="Retrieval weight: Semantic"
    )

    parser.add_argument(
        "--l2", type=float, default=0.2, help="Retrieval weight: Source align"
    )

    parser.add_argument(
        "--l3", type=float, default=0.2, help="Retrieval weight: Target align"
    )

    parser.add_argument(
        "--ws", type=float, default=0.55, help="Rerank weight: Semantic"
    )

    parser.add_argument(
        "--wc", type=float, default=0.30, help="Rerank weight: Complexity"
    )

    parser.add_argument("--wl", type=float, default=0.15, help="Rerank weight: Length")

    parser.add_argument(
        "--fact_source",
        type=str,
        default="llm",
        choices=["llm", "heuristic"],
        help="Fact extraction source",
    )

    parser.add_argument("--fact_max_tokens", type=int, default=128)

    parser.add_argument("--fact_max_items", type=int, default=5)

    parser.add_argument("--max_history_turns", type=int, default=2)

    args = parser.parse_args()

    print("=" * 50)

    print(
        "Running configuration: "
        f"Corpus={args.corpus}, "
        f"History={'OFF' if args.disable_history else 'ON'}, "
        f"DynamicPath={'OFF (Static)' if args.disable_dynamic_path else 'ON'}, "
        f"DynamicRetrieval={'OFF (Random)' if args.disable_dynamic_retrieval else 'ON'}, "
        f"MaxTokens={args.max_tokens}, Candidates={args.num_candidates}"
    )

    print("=" * 50)

    lang = "en"

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_file(repo_root / "local_config.env")
    data_root = (
        Path(args.data_root).expanduser() if args.data_root else repo_root / "data"
    )
    result_root = repo_root / "LLMGeneration"

    if args.corpus == "CEFR-SP":
        default_matrix = result_root / "matrices" / "reward_matrix_cefr_sp.json"
        matrix_path = (
            Path(args.matrix_path).expanduser() if args.matrix_path else default_matrix
        )
        default_calibration = result_root / "matrices" / "iaps_calibration_cefr_sp.json"
        case_file = (
            Path(args.expert_case_file).expanduser()
            if args.expert_case_file
            else data_root / "prompt_expert" / "en_expert_case_cefrsp.json"
        )

        save_subdir = args.save_dir if args.save_dir else "ours_finall"

        save_dir = result_root / "CEFR-SP" / save_subdir

    else:
        default_matrix = result_root / "matrices" / "reward_matrix_readme.json"
        matrix_path = (
            Path(args.matrix_path).expanduser() if args.matrix_path else default_matrix
        )
        default_calibration = result_root / "matrices" / "iaps_calibration_readme.json"
        case_file = (
            Path(args.expert_case_file).expanduser()
            if args.expert_case_file
            else data_root / "prompt_expert" / "en_expert_case_readme.json"
        )

        if args.save_dir:
            save_subdir = args.save_dir

        elif args.disable_rerank:
            save_subdir = "no_rerank"

        elif args.disable_fact_memory:
            save_subdir = "no_fact"

        elif args.disable_dynamic_retrieval:
            save_subdir = "no_retrieval"

        else:
            save_subdir = "full_method"

        save_dir = result_root / "README" / save_subdir

    calibration_path = (
        Path(args.calibration_path).expanduser()
        if args.calibration_path
        else default_calibration
    )

    os.makedirs(save_dir, exist_ok=True)

    print(f"[*] 正在使用的奖励矩阵: {matrix_path}")

    print(f"[*] 正在使用的案例池: {case_file}")

    run_preflight_checks(case_file, matrix_path)

    retrieval_model = args.retrieval_model or os.getenv("RETRIEVAL_MODEL", "").strip()
    if not retrieval_model:
        raise ValueError("Missing --retrieval_model or RETRIEVAL_MODEL.")

    selector = SemanticExemplarSelector(retrieval_model)

    analyzer = TextComplexityAnalyzer()

    server_url = os.getenv("OPENAI_API_BASE_URL", "").strip()
    server_model = (
        args.model_name
        or os.getenv("OPENAI_API_MODEL", "").strip()
        or args.model_path
    )
    if server_url:
        if not server_model or "替换为" in server_model:
            raise ValueError(
                "Missing valid OPENAI_API_MODEL. Set it to the model id returned by "
                "GET /v1/models."
            )
        generator = OpenAICompatGenerator(
            api_base_url=server_url,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            model_name=server_model,
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=args.max_tokens,
            timeout=300,
        )
    else:
        model_path = (
            args.model_path or os.getenv("VLLM_MODEL_PATH", "").strip()
        )
        if not model_path:
            raise ValueError(
                "Missing --model_path or VLLM_MODEL_PATH when "
                "OPENAI_API_BASE_URL is not configured."
            )
        generator = VLLMGenerator(
            model_name=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            temperature=args.temperature,
            top_p=0.95,
            repetition_penalty=1.0,
            max_tokens=args.max_tokens,
        )

    fact_generator = generator

    if not args.disable_fact_memory and args.fact_source == "llm":
        if server_url:
            fact_generator = OpenAICompatGenerator(
                api_base_url=server_url,
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                model_name=server_model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=args.fact_max_tokens,
                timeout=300,
            )
        else:
            fact_generator = VLLMGenerator(
                model_name=model_path,
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

    if calibration_path.exists():
        calibration = load_method_calibration(calibration_path)
    else:
        raise FileNotFoundError(
            "Validation-fitted calibration is required for the main experiment: "
            f"{calibration_path}. Run matrix/build_reward_matrix.py --mode r_base "
            "with train/validation paths first."
        )

    TARGET_COMPLEXITY.update(calibration["level_complexity"])
    TARGET_LENGTH_RATIO.update(calibration["length_ratio"])
    retrieval_weights = tuple(calibration["retrieval_weights"])
    rerank_weights = tuple(calibration["rerank_weights"])
    print(f"[*] IAPS calibration: {calibration_path}")

    test_file = resolve_split_file(args.corpus, lang, "test", data_root=data_root)

    fact_cache: dict[str, list[str]] = {}

    for begin_cefr in [6, 5, 4]:
        df = pd.read_csv(test_file)

        df = df[df["Rating"] == begin_cefr].reset_index(drop=True)

        if args.sample_size > 0:
            df = df.head(args.sample_size)

        print(f"\nProcessing begin CEFR {begin_cefr}, total {len(df)} sentences")

        for target_cefr in [3, 2, 1]:
            final_generated_list = []

            for _, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Target {target_cefr}"
            ):
                original_src = row["Sentence"]

                if args.disable_dynamic_path:
                    strategy = STATIC_DP_STRATEGY[f"{begin_cefr}{target_cefr}"]

                else:
                    strategy = build_instance_adaptive_path(
                        begin_cefr,
                        target_cefr,
                        analyzer.compute_features(original_src),
                        base_reward_matrix,
                        calibration,
                    )

                    if strategy and strategy[0] == begin_cefr:
                        strategy = strategy[1:]

                current_text = original_src

                chat_history: list[dict[str, str]] = []

                prev_target = begin_cefr

                if args.disable_fact_memory:
                    facts = []

                else:
                    cached_facts = fact_cache.get(original_src)

                    if cached_facts is None:
                        if args.fact_source == "llm":
                            try:
                                cached_facts = extract_facts_llm(
                                    fact_generator, original_src, args.fact_max_items
                                )

                            except (OSError, RuntimeError, TypeError, ValueError):
                                cached_facts = extract_facts_heuristic(
                                    original_src, args.fact_max_items
                                )

                        else:
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

                    if args.disable_dynamic_retrieval:
                        selected = random.sample(candidates, min(3, len(candidates)))

                        sims = [0.0] * len(selected)

                    else:
                        selected, sims = selector.select_topk_exemplars(
                            current_text,
                            candidates,
                            k=3,
                            target_cefr=step_target,
                            complexity_analyzer=analyzer,
                            weights=retrieval_weights,
                            target_complexities=TARGET_COMPLEXITY,
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

                    payload = [{"role": "user", "content": user_prompt}]

                    if not args.disable_history:
                        payload = chat_history + payload

                    candidates_text = generate_candidates(
                        generator, payload, args.num_candidates
                    )

                    clean_candidates = [cand for cand in candidates_text if cand]

                    if not clean_candidates:
                        clean_candidates = [current_text]

                    if args.disable_rerank:
                        clean_text = clean_candidates[0]

                    else:
                        clean_text = rerank_candidates(
                            clean_candidates,
                            original_src,
                            current_text,
                            prev_target,
                            step_target,
                            selector,
                            analyzer,
                            weights=rerank_weights,
                        )

                    if not args.disable_history:
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
