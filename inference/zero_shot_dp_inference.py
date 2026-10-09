import argparse
import copy
import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from core.prompt_builder import clean_llm_generation
from utils import CEFRMAP, CEFR_Descriptors


def padding_case(case_dict_expert):
    """为缺失的 (起始级, 目标级) 键补上空列表。

    Args:
        case_dict_expert: 参数说明。

    Returns:
        返回值说明。
    """
    for begin_cefr in [6, 5, 4, 3, 2]:
        for target_cefr in range(begin_cefr - 1, 0, -1):
            if f"{begin_cefr}{target_cefr}" not in case_dict_expert:
                case_dict_expert[f"{begin_cefr}{target_cefr}"] = []

    return case_dict_expert


def get_prompt(
    prompt_case_pair, prompt_case_single, complex_sentence, CEFR_LEVEL, CEFR_Description
):
    """构建零样本/少样本简化 prompt。

    Args:
        prompt_case_pair: 参数说明。
        prompt_case_single: 参数说明。
        complex_sentence: 参数说明。
        CEFR_LEVEL: 参数说明。
        CEFR_Description: 参数说明。

    Returns:
        返回值说明。
    """
    CEFR_LEVEL = CEFRMAP[CEFR_LEVEL]

    content = (
        f"Please rewrite the following complex sentence in order to make it easier to understand by {CEFR_LEVEL} CEFR level language learners. "
        "{CEFR_LEVEL} level language learner {CEFR_Description}"
        "You can do so by replacing "
        "complex words with simpler synonyms (i.e. paraphrasing), deleting unimportant information (i.e. compression), and/or splitting a long "
        "complex sentence into several simpler ones. The final simplified sentence needs to be grammatical, fluent, and retain the main ideas "
        "of its original counterpart without altering its meaning."
    )

    if len(prompt_case_pair) > 0:
        content += f"The following are examples of simplification for {CEFR_LEVEL} CEFR level language learners:\n"

        for i in range(len(prompt_case_pair)):
            content += (
                f"Complex:{prompt_case_pair[i][0]}\nSimple:{prompt_case_pair[i][1]}\n\n"
            )

    else:
        content += f"The following are examples of sentences readable by {CEFR_LEVEL} CEFR level language learners:\n"

        for i in range(len(prompt_case_single)):
            content += f"Example {i + 1}:{prompt_case_single[i]}\n\n"

    content += f"Complex:{complex_sentence}\nSimple:"

    return content


DP_STARTEGY = {
    "en": {
        "63": [5, 3],
        "62": [5, 3, 2],
        "61": [5, 3, 2, 1],
        "53": [3],
        "52": [3, 2],
        "51": [3, 2, 1],
        "43": [3],
        "42": [3, 2],
        "41": [3, 2, 1],
    },
}

COSPUS_LANG = {"CEFR-SP": ["en"], "CEFR-SP_whole": ["en"], "README": ["en"]}


def resolve_split_file(corpus, lang, split):
    """按语料/语言/划分解析数据集文件路径。

    Args:
        corpus: 参数说明。
        lang: 参数说明。
        split: 参数说明。

    Returns:
        返回值说明。
    """
    data_lang_dir = Path(__file__).resolve().parents[1] / "data" / lang

    candidates = [
        data_lang_dir / f"{corpus}_{split}.csv",
        data_lang_dir / f"{corpus}_{lang}_{split}.csv",
        data_lang_dir / f"{corpus.lower()}_{lang}_{split}.csv",
        data_lang_dir / f"{corpus.replace('-', '_')}_{lang}_{split}.csv",
        data_lang_dir / f"{corpus.replace('-', '').lower()}_{lang}_{split}.csv",
    ]

    if corpus == "README":
        candidates.append(data_lang_dir / f"readme_{lang}_{split}.csv")

    if corpus in ["CEFR-SP", "CEFR-SP_whole"]:
        candidates.append(data_lang_dir / f"CEFR-SP_{lang}_{split}.csv")

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    tried = "\n".join(candidates)

    raise FileNotFoundError(
        f"No dataset file found for corpus={corpus}, lang={lang}, split={split}. Tried:\n{tried}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference param")

    parser.add_argument("--infer_bs", type=int, default=5)

    parser.add_argument("--case_num", type=int, default=3)

    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=0,
        help="Tensor parallel size for vLLM; 0 means auto-detect from visible GPUs",
    )

    parser.add_argument("--save_dir", type=str, default="few-shot_cefrsp_28")

    parser.add_argument(
        "--corpus",
        type=str,
        default="CEFR-SP",
        choices=["CEFR-SP", "README", "CEFR-SP_whole"],
    )

    parser.add_argument(
        "--langs",
        type=str,
        default="",
        help="Comma-separated language list to run, e.g. 'en' or 'en,fr'. Empty means corpus default.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    model_name = args.model_name

    llmtype = os.path.basename(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=256
    )

    available_gpus = torch.cuda.device_count()

    if available_gpus < 1:
        raise RuntimeError("No CUDA device is available for vLLM inference.")

    tensor_parallel_size = (
        args.tensor_parallel_size
        if args.tensor_parallel_size > 0
        else min(4, available_gpus)
    )

    tensor_parallel_size = min(tensor_parallel_size, available_gpus)

    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)

    infer_bs = args.infer_bs

    case_num = args.case_num

    corpus = args.corpus

    if args.langs.strip() == "":
        selected_langs = COSPUS_LANG[corpus]

    else:
        selected_langs = [
            lang.strip() for lang in args.langs.split(",") if lang.strip() != ""
        ]

    for lang in selected_langs:
        if lang not in COSPUS_LANG[corpus]:
            raise ValueError(
                f"lang={lang} is not supported for corpus={corpus}. Supported: {COSPUS_LANG[corpus]}"
            )

        if lang not in DP_STARTEGY:
            raise ValueError(
                f"No DP strategy configured for lang={lang}. Available: {list(DP_STARTEGY.keys())}"
            )

    save_dir = f"LLMGeneration/{corpus}/{args.save_dir}"

    os.makedirs(save_dir, exist_ok=True)

    for lang in selected_langs:
        expert_case_name = (
            "en_expert_case_cefrsp.json"
            if corpus.startswith("CEFR-SP")
            else "en_expert_case_readme.json"
        )
        case_file = repo_root / "data" / "prompt_expert" / expert_case_name
        with open(case_file, encoding="utf-8") as handle:
            case_dict_expert = json.load(handle)

        case_dict_expert = padding_case(case_dict_expert)

        train_file = resolve_split_file(corpus, lang, "train")

        test_file = resolve_split_file(corpus, lang, "test")

        casedf = pd.read_csv(train_file)

        case_dict = {i: [] for i in range(1, 7)}

        for rid, row in casedf.iterrows():
            case_dict[int(row["Rating"])].append(row["Sentence"])

        for begin_cefr in [6, 5, 4]:
            df = pd.read_csv(test_file)

            df = df[df["Rating"] == begin_cefr].reset_index(drop=True)

            for target_cefr in [1, 2, 3]:
                complex_sentence = df["Sentence"].tolist()

                strategy = DP_STARTEGY[lang][f"{begin_cefr}{target_cefr}"]

                assert strategy[-1] == target_cefr

                for sid, c in enumerate(strategy):
                    generated_list = []

                    if sid == 0:
                        bid = begin_cefr

                    else:
                        bid = strategy[sid - 1]

                    for i, sent in enumerate(complex_sentence[::infer_bs]):
                        srclist = complex_sentence[i * infer_bs : (i + 1) * infer_bs]

                        input_prompt_ = [
                            get_prompt(
                                case_dict_expert[f"{bid}{target_cefr}"][:case_num],
                                case_dict[c][:case_num],
                                src,
                                c,
                                CEFR_Descriptors[c],
                            )
                            for src in srclist
                        ]

                        outputs = llm.generate(input_prompt_, sampling_params)

                        for output in outputs:
                            generated_text = output.outputs[0].text

                            generated_list.append(clean_llm_generation(generated_text))

                    complex_sentence = copy.deepcopy(generated_list)

                df[f"llm_gene_CEFR{target_cefr}"] = generated_list

            df.to_csv(
                f"{save_dir}/{lang}_few-shot_beginCEFR{begin_cefr}.csv",
                index=False,
                sep=",",
            )
