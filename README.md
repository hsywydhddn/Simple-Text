# simplify-step-by-step（改进版）

本项目实现论文中的渐进式文本简化流程：**奖励矩阵构建 → 路径规划 → LLM 逐步简化 → 多指标评估**。输入复杂句子后，系统按照 CEFR 等级路径逐步生成更易理解的句子，同时尽量保持原意和关键事实。

## 论文模块与代码对应

| 论文模块/实验 | 代码入口或实现 | 功能 |
| --- | --- | --- |
| 奖励矩阵与 R_base | [`matrix/build_reward_matrix.py`](matrix/build_reward_matrix.py) | 根据训练/验证数据计算等级转移奖励，并计算基础转移成功率。 |
| 复杂度感知路径规划 | [`core/path_planner.py`](core/path_planner.py) | 根据奖励矩阵搜索起始等级到目标等级的渐进式路径。 |
| LLM 客户端 | [`core/llm_client.py`](core/llm_client.py) | 封装 vLLM 和 OpenAI 兼容 API。 |
| Prompt 与示例构建 | [`core/prompt_builder.py`](core/prompt_builder.py) | 构造简化提示词、加载示例、清洗生成结果。 |
| 原子事实提取与事实记忆 | [`method/common.py`](method/common.py) | 提取关键事实并在多步简化中保留原句信息。 |
| 难度语义动态检索 | [`method/full_method.py`](method/full_method.py) | 按语义相似度、源等级和目标等级选择示例。 |
| 多候选生成与重排 | [`method/full_method.py`](method/full_method.py)、[`core/evaluator.py`](core/evaluator.py) | 生成多个候选句，并依据语义、复杂度和长度排序。 |
| 完整方法实验 | [`method/full_method.py`](method/full_method.py) | 统一入口，支持动态路径、动态检索、事实记忆和候选重排开关。 |
| 动态检索消融 | [`method/ablation_no_dynamic_retrieval.py`](method/ablation_no_dynamic_retrieval.py) | 关闭动态示例检索，使用随机示例进行对比。 |
| 检索/重排权重敏感性 | [`method/weight_sensitivity_method.py`](method/weight_sensitivity_method.py) | 扫描 `l1/l2/l3` 和 `ws/wc/wl` 权重。 |
| 零样本基线 | [`baseline/zero_shot_baseline.py`](baseline/zero_shot_baseline.py) | 不使用渐进路径和示例，直接要求 LLM 简化。 |
| 全量评估 | [`evaluation/evaluate_all.py`](evaluation/evaluate_all.py) | 计算 Spearman ρ、AdjAcc、ExactAcc、RMSE、BERTScore F1 和 STS。 |
| 分跨度评估 | [`evaluation/evaluate_by_span.py`](evaluation/evaluate_by_span.py) | 比较小跨度、中跨度和大跨度等级转换。 |
| N/T 参数分析 | [`analysis/sensitivity_n_t.py`](analysis/sensitivity_n_t.py) | 分析候选数量 N 与采样温度 T 对结果的影响。 |
| Zero-shot DP 推理 | [`inference/zero_shot_dp_inference.py`](inference/zero_shot_dp_inference.py) | 旧版/独立的 zero-shot + 静态路径推理入口。 |

## 目录和文件说明

```text
simplify-step-by-step/
├── core/                  # 被多个实验入口复用的核心实现
│   ├── llm_client.py      # vLLM / OpenAI 兼容 API 客户端
│   ├── prompt_builder.py  # Prompt、示例和生成结果处理
│   ├── path_planner.py    # 奖励矩阵上的路径规划
│   └── evaluator.py       # STS、可读性预测和 CEFR 预测工具
├── matrix/
│   └── build_reward_matrix.py       # 奖励矩阵与 R_base 构建实验
├── method/
│   ├── full_method.py               # 完整方法主实验
│   ├── common.py                    # 方法变体共享的事实、检索、重排工具
│   ├── ablation_no_dynamic_retrieval.py # 关闭动态检索的消融实验
│   └── weight_sensitivity_method.py # 检索/重排权重敏感性实验
├── baseline/
│   └── zero_shot_baseline.py        # 零样本基线生成
├── evaluation/
│   ├── evaluate_all.py              # 全量六指标评估
│   └── evaluate_by_span.py          # 按等级跨度评估
├── analysis/
│   └── sensitivity_n_t.py           # N/T 敏感性分析
├── inference/
│   └── zero_shot_dp_inference.py    # 独立 zero-shot DP 推理入口
├── utils/
│   ├── utils.py                     # CEFR 描述符、等级映射和基础指标
│   └── __init__.py                  # 公共工具导出接口
├── data/                            # 数据集与专家示例
│   ├── en/                          # 英语 README / CEFR-SP 的训练、验证和测试数据
│   ├── ar/、fr/、hi/、ru/           # README 多语言训练/验证/测试数据
│   └── prompt_expert/               # 各语言、各语料的专家简化示例 JSON
├── model/                           # 预留的本地模型目录，目前为空
├── .vscode/settings.json             # VS Code 工作区设置
└── README.md
```

### 数据文件命名

- `*_train.csv`：训练数据，用于构建示例池或奖励矩阵。
- `*_val.csv` / `*_dev.csv`：验证或开发数据，用于矩阵和实验调参。
- `*_test.csv`：测试数据，用于生成结果和最终评估。
- `data/prompt_expert/*_expert_case_*.json`：专家提供的等级转换示例，供 few-shot Prompt 使用。
- 每条 CSV 数据通常包含 `Sentence` 和 `Rating` 字段，分别表示句子和 CEFR 等级。

各目录中的 `__init__.py` 只负责将目录声明为 Python 包；其中 `core/__init__.py` 和 `utils/__init__.py` 还集中导出公共函数和类，便于其他模块统一导入。

## 依赖

- Python 3.10+
- `numpy`, `pandas`, `torch`, `transformers`, `tqdm`, `scipy`, `bert-score`, `readmepp`
- `vllm`（本地 LLM 推理）
- `sentence-transformers`（语义检索与 STS 矩阵构建，可选）
- `pyflakes`（静态检查，开发用）

安装：`pip install -r requirements.txt`

### 配置本地模型

代码不包含本地模型和服务器地址。运行需要通过环境变量或命令行参数提供：

```bash
export VLLM_MODEL_PATH="请输入本地 LLM 模型路径"
export RETRIEVAL_MODEL="请输入 sentence-transformers 模型路径或模型名"
export READABERT_LOCAL_PATH="请输入 ReadMe/ReadBERT 模型路径"
export STS_LOCAL_PATH="请输入 STS 模型路径"
export BERTSCORE_LOCAL_PATH="请输入 BERTScore 模型路径"
```

也可以直接编辑项目根目录的 [`local_config.env`](</mnt/c/Users/whany/Desktop/simplify-step-by-step/local_config.env>)，
将其中的 `/absolute/path/to/...` 替换为本地模型目录，然后在每次打开新终端后执行：

```bash
source local_config.env
```

检查配置是否生效：

```bash
printf '%s\n' "$VLLM_MODEL_PATH"
printf '%s\n' "$RETRIEVAL_MODEL"
printf '%s\n' "$READABERT_LOCAL_PATH"
```


## 运行方式

> 数据集默认从仓库内的 `data/` 读取；本地模型路径通过环境变量或命令行参数提供。
>
> **统一运行方式**：所有入口脚本一律采用 `PYTHONPATH=. python xxx.py` 或 `python -m xxx.yyy` 方式运行，保证 `from core.xxx`、`from utils import ...` 等相对包导入可用；不要在仓库根目录之外直接执行脚本文件。

1. **构建奖励矩阵**（默认从仓库 `data/en/` 读取数据，并将矩阵保存到 `LLMGeneration/matrices/`）
   ```bash
   PYTHONPATH=. python matrix/build_reward_matrix.py --model_path "请输入模型路径"
   ```
2. **计算 R_base(i, j) 基础转移成功率**（论文 3.2 节，验证集 + 基础提示词 + readabert-en 等级预测）
   ```bash
   PYTHONPATH=. python matrix/build_reward_matrix.py --mode r_base --model_path "请输入模型路径" --readabert_path "请输入评估模型路径"
   ```
   R_base 输出 JSON 键格式与奖励矩阵一致（如 `"6_3"`），可直接用于路径规划对比。
3. **LLM 推理**（渐进式简化）
   ```bash
   # 完整方法（默认 README 语料 + 全部机制开启）
   PYTHONPATH=. python method/full_method.py --corpus README --retrieval_model "请输入检索模型"
   PYTHONPATH=. python method/full_method.py --corpus CEFR-SP --retrieval_model "请输入检索模型"
   # 消融与专项
   PYTHONPATH=. python method/full_method.py --corpus README --disable_dynamic_retrieval   # 禁用动态检索（随机示例）
   PYTHONPATH=. python method/full_method.py --corpus README --disable_rerank              # 禁用多候选重排
   PYTHONPATH=. python method/full_method.py --corpus README --disable_fact_memory         # 禁用事实记忆
   PYTHONPATH=. python method/full_method.py --corpus README --disable_dynamic_path        # 禁用动态路径（静态 DP）
   PYTHONPATH=. python method/ablation_no_dynamic_retrieval.py ... # 动态检索消融
   PYTHONPATH=. python method/weight_sensitivity_method.py ...      # 检索/重排权重敏感性
   PYTHONPATH=. python inference/zero_shot_dp_inference.py ...      # zero-shot 推理
   ```
   
   主实验使用 IAPS 实例自适应路径。正式运行时应将严格命中的
   `R_base` 矩阵通过 `--matrix_path` 传入；如果未提供
   `--calibration_path`，程序会从专家平行示例统计 `rho(l)` 并写出临时校准文件，
   最终论文结果应替换为验证集拟合的校准参数。

   最小连通性测试（每个起始等级只跑 1 条，仍会调用 LLM 服务）：
   ```bash
   PYTHONPATH=. python method/full_method.py \
      --corpus README \
      --model_path "$VLLM_MODEL_PATH" \
      --retrieval_model "$RETRIEVAL_MODEL" \
      --matrix_path LLMGeneration/matrices/r_base_readme_exact.json \
      --sample_size 1 \
      --num_candidates 1 \
      --temperature 0 \
      --fact_source heuristic \
      --save_dir smoke_test_iaps
   ```

   成功标准：命令正常结束，并在
   `LLMGeneration/README/smoke_test_iaps/` 下生成
   `en_few-shot_beginCEFR4.csv`、`en_few-shot_beginCEFR5.csv` 和
   `en_few-shot_beginCEFR6.csv`。
4. **评估**
   ```bash
    PYTHONPATH=. python evaluation/evaluate_all.py --data_dir LLMGeneration/README/full_method
    PYTHONPATH=. python evaluation/evaluate_by_span.py \
       --baseline_dir LLMGeneration/README/zero_shot_baseline_gemma_2_9b_it \
       --ours_dir LLMGeneration/README/full_method
   ```
5. **参数分析与基线实验**
   ```bash
   PYTHONPATH=. python analysis/sensitivity_n_t.py
   PYTHONPATH=. python baseline/zero_shot_baseline.py
   ```

各脚本均支持 `python xxx.py --help` 查看参数。请以仓库根目录为工作目录运行，IAPS 主实验通过 `core/path_planner.py` 的实例自适应 DP 规划路径。

### 生成结果位置

- 完整方法默认写入 `LLMGeneration/<corpus>/full_method/`。
- 其中生成文件为 `en_few-shot_beginCEFR4.csv`、`en_few-shot_beginCEFR5.csv` 和 `en_few-shot_beginCEFR6.csv`。
- 零样本基线默认写入 `LLMGeneration/<corpus>/zero_shot_baseline_<model>/`，文件命名规则相同。
- `evaluate_all.py` 的 `--data_dir` 接收一个结果目录，并自动读取其中的 `en_few-shot_beginCEFR*.csv`。
- `evaluate_by_span.py` 的 `--ours_dir` 和 `--baseline_dir` 也接收结果目录，不在代码中写死具体 CSV 文件。

### 语料与消融参数（method/full_method.py）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--corpus` | `README` | `README` / `CEFR-SP`；自动选择奖励矩阵、案例池与输出目录 |
| `--model_path` | 环境变量 | 本地 LLM 路径，传给 vLLM；也可使用 `VLLM_MODEL_PATH` |
| `--tensor_parallel_size` | 1 | vLLM 使用的 GPU 张量并行数 |
| `--matrix_path` | 自动 | 严格命中的 IAPS `R_base` 矩阵路径 |
| `--calibration_path` | 自动 | IAPS 校准文件；缺失时从专家平行示例统计 `rho(l)` |
| `--disable_dynamic_path` | 关 | 禁用动态路径，使用静态 DP 策略 |
| `--disable_dynamic_retrieval` | 关 | 禁用语义检索，随机选取示例 |
| `--disable_rerank` | 关 | 禁用多候选重排 |
| `--disable_fact_memory` | 关 | 禁用事实记忆块 |
| `--sample_size` | 0 | 每个起始等级只处理前 N 句（0 = 全部） |
| `--l1 / --l2 / --l3` | 0.6 / 0.2 / 0.2 | 语义检索权重（语义 / 源对齐 / 目标对齐） |
| `--ws / --wc / --wl` | 0.55 / 0.30 / 0.15 | 多候选重排权重（语义 / 复杂度 / 长度） |
| `--model_name` | 自动 | README 使用默认模型名；CEFR-SP 从配置的 OpenAI 兼容服务获取模型名 |
| `--temperature` | 0.3 | 采样温度 |
