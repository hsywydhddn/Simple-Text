"""Core 公共模块：LLM 客户端、Prompt 构建、路径规划与评估器。"""
from core.llm_client import OpenAICompatGenerator, VLLMGenerator, load_dotenv_file
from core.path_planner import (
    build_multi_objective_path,
    build_path_from_matrix,
    get_strategy,
    load_reward_matrix,
)
from core.prompt_builder import (
    CEFRMAP,
    build_negative_examples,
    clean_llm_generation,
    get_prompt,
    make_bad_simplification,
    padding_case,
    resolve_split_file,
)

__all__ = [
    "CEFRMAP",
    "OpenAICompatGenerator",
    "VLLMGenerator",
    "build_multi_objective_path",
    "build_negative_examples",
    "build_path_from_matrix",
    "clean_llm_generation",
    "get_prompt",
    "get_readability_predictor",
    "get_strategy",
    "get_sts_model",
    "load_dotenv_file",
    "load_reward_matrix",
    "make_bad_simplification",
    "padding_case",
    "predict_cefr",
    "resolve_split_file",
    "sts_similarity",
]


def __getattr__(name):
    """按需加载评估器，避免基础推理依赖评估模型库。"""
    if name in {
        "get_readability_predictor",
        "get_sts_model",
        "predict_cefr",
        "sts_similarity",
    }:
        from core import evaluator

        return getattr(evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
