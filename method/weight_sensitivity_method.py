import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core.llm_client import VLLMGenerator
from core.path_planner import build_multi_objective_path
from core.prompt_builder import padding_case, resolve_split_file
from method.common import (
    TARGET_COMPLEXITY,
    TARGET_LENGTH_RATIO,
    TextComplexityAnalyzer,
    build_compact_memory,
    clamp01,
    extract_facts_heuristic,
    extract_facts_llm,
    generate_candidates,
    get_dynamic_prompt,
    trim_chat_history,
    word_count,
)


class SemanticExemplarSelector:
    """语义示例选择器（参数搜索版）。

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

        model_path = model_path or os.getenv("RETRIEVAL_MODEL", "").strip()
        if not model_path:
            raise ValueError("Missing retrieval model. Set RETRIEVAL_MODEL.")
        print(f"Loading local retrieval model: {model_path}")
        self.model = SentenceTransformer(model_path)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
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
        target_complexity = TARGET_COMPLEXITY.get(target_cefr, 0.45)

        scores = []
        for idx, (complex_sent, simple_sent) in enumerate(candidate_pairs):
            comp_score = 1.0 - abs(
                test_complexity
                - self._get_complexity(complexity_analyzer, complex_sent)
            )
            simple_score = 1.0 - abs(
                target_complexity
                - self._get_complexity(complexity_analyzer, simple_sent)
            )
            score = (
                weights[0] * semantic_sims[idx]
                + weights[1] * clamp01(comp_score)
                + weights[2] * clamp01(simple_score)
            )
            scores.append(score)

        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [candidate_pairs[i] for i in top_k_indices], [
            semantic_sims[i] for i in top_k_indices
        ]


def rerank_candidates(
    candidates: list[str],
    original: str,
    target_cefr: int,
    selector: SemanticExemplarSelector,
    analyzer: TextComplexityAnalyzer,
    weights: tuple[float, float, float] = (0.55, 0.30, 0.15),
) -> str:
    """对候选重排。

    Args:
        candidates: 参数说明。
        original: 参数说明。
        target_cefr: 参数说明。
        selector: 参数说明。
        analyzer: 参数说明。
        weights: 参数说明。

    Returns:
        返回值说明。
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    sims = selector.compute_batch_similarity(original, candidates)
    target_complexity = TARGET_COMPLEXITY.get(target_cefr, 0.45)
    target_len_ratio = TARGET_LENGTH_RATIO.get(target_cefr, 0.65)
    original_len = max(word_count(original), 1)

    scores: list[float] = []
    for idx, cand in enumerate(candidates):
        comp_score = 1.0 - abs(
            analyzer.compute_overall_complexity(cand) - target_complexity
        )
        len_ratio = word_count(cand) / original_len
        len_score = 1.0 - min(abs(len_ratio - target_len_ratio), 1.0)

        score = (
            weights[0] * sims[idx]
            + weights[1] * clamp01(comp_score)
            + weights[2] * clamp01(len_score)
        )
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return candidates[best_idx]


def run_preflight_checks(case_file, matrix_path):
    """运行前置检查。

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
        sys.exit(1)
    if not os.path.exists(matrix_path):
        print(f"[Error] Reward matrix not found at: {matrix_path}")
        sys.exit(1)
    print("All preflight checks passed.")
    print("=" * 50)


def main():
    """参数敏感性搜索主流程（l1/l2/l3/ws/wc/wl 权重扫描）。"""
    parser = argparse.ArgumentParser(
        description="Text Simplification Weight Sensitivity"
    )
    parser.add_argument("--disable_history", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--disable_rerank", action="store_true")
    parser.add_argument("--disable_fact_memory", action="store_true")
    parser.add_argument(
        "--fact_source", type=str, default="llm", choices=["llm", "heuristic"]
    )
    parser.add_argument("--fact_max_tokens", type=int, default=128)
    parser.add_argument("--fact_max_items", type=int, default=5)
    parser.add_argument("--max_history_turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.3)

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
        "--save_folder", type=str, default="default_run", help="Save folder name"
    )
    parser.add_argument(
        "--sample_size", type=int, default=100, help="Subset size for sensitivity test"
    )
    parser.add_argument("--model_path", default=os.getenv("VLLM_MODEL_PATH", ""))
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--retrieval_model", default=os.getenv("RETRIEVAL_MODEL", ""))
    args = parser.parse_args()

    print("=" * 50)
    print(f"Retrieval Weights: ({args.l1}, {args.l2}, {args.l3})")
    print(f"Rerank Weights: ({args.ws}, {args.wc}, {args.wl})")
    print("=" * 50)

    repo_root = Path(__file__).resolve().parents[1]
    save_dir = repo_root / "LLMGeneration" / "parameter_rerank" / args.save_folder
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

    fact_generator = generator
    if not args.disable_fact_memory and args.fact_source == "llm":
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
                strategy = build_multi_objective_path(
                    begin_cefr, target_cefr, base_reward_matrix
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

                    selected, sims = selector.select_topk_exemplars(
                        current_text,
                        candidates,
                        k=3,
                        target_cefr=step_target,
                        complexity_analyzer=analyzer,
                        weights=(args.l1, args.l2, args.l3),
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
                            step_target,
                            selector,
                            analyzer,
                            weights=(args.ws, args.wc, args.wl),
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
