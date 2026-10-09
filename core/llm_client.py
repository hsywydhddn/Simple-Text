"""LLM 推理客户端封装。

从原 improve.py 拆分出的公共模块，提供 vLLM 与 OpenAI 兼容 API 两种生成后端，
以及 .env 环境变量加载工具。
"""

import importlib
import os
from pathlib import Path


def load_dotenv_file(env_path: Path, override: bool = False) -> None:
    """从本地 .env 文件加载环境变量。

    Args:
        env_path: .env 文件路径，文件不存在或非文件时静默返回。
        override: 是否覆盖已存在的环境变量，默认为 False。
    """
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "" or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "":
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value


class VLLMGenerator:
    """基于 vLLM 的批量文本生成器。

    使用 vLLM 引擎加载本地模型，支持张量并行推理。
    generate_texts 会对每个 prompt 应用官方 Chat Template，
    避免模型因格式错乱产生下划线等伪影。
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
    ):
        """初始化 vLLM 生成器。

        Args:
            model_name: 本地模型路径或 HuggingFace 模型名。
            tensor_parallel_size: 张量并行大小。
            temperature: 采样温度。
            top_p: 核采样概率。
            repetition_penalty: 重复惩罚系数。
            max_tokens: 最大生成 token 数。
        """
        vllm = importlib.import_module("vllm")
        llm_cls = vllm.LLM
        sampling_params_cls = vllm.SamplingParams

        self.sampling_params = sampling_params_cls(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )
        self.llm = llm_cls(model=model_name, tensor_parallel_size=tensor_parallel_size)

    def generate_text(self, prompt: str) -> str:
        """生成单条文本。

        Args:
            prompt: 输入 prompt 字符串。

        Returns:
            模型生成的文本。
        """
        if isinstance(prompt, list):
            tokenizer = self.llm.get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        output = self.llm.generate([prompt], self.sampling_params)[0]
        return output.outputs[0].text

    def generate_texts(self, prompts) -> list:
        """批量生成文本，自动应用 Chat Template 防止格式错乱。

        Args:
            prompts: prompt 字符串列表。

        Returns:
            生成文本列表，顺序与输入一致。
        """
        tokenizer = self.llm.get_tokenizer()
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted)

        outputs = self.llm.generate(formatted_prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


class OpenAICompatGenerator:
    """OpenAI 兼容 API 生成器。

    支持任意实现 OpenAI Chat Completions 协议的本地或远程服务
    （如 vLLM server、TGI、Xirang 等），通过 base_url 指定服务地址。
    """

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: int,
        system_prompt: str = "",
    ):
        """初始化 OpenAI 兼容客户端。

        Args:
            api_base_url: API 服务地址，建议通过 OPENAI_API_BASE_URL 配置。
            api_key: API 密钥。
            model_name: 服务端模型名称。
            temperature: 采样温度。
            top_p: 核采样概率。
            max_tokens: 最大生成 token 数。
            timeout: 请求超时秒数。
            system_prompt: 可选的 system prompt，为空则不注入。
        """
        from openai import OpenAI

        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url=api_base_url, api_key=api_key, timeout=timeout)

    def generate_text(self, prompt) -> str:
        """生成单条文本。

        Args:
            prompt: 输入 prompt，可以是字符串或完整 messages 列表。

        Returns:
            模型生成的文本；若 content 为空则尝试回退到 reasoning_content。
        """
        import openai

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
                extra_body={"enable_thinking": False},
            )
            message = res.choices[0].message
            content = message.content

            if not content:
                reasoning_content = getattr(message, "reasoning_content", None)
                if (
                    not reasoning_content
                    and hasattr(message, "model_extra")
                    and isinstance(message.model_extra, dict)
                ):
                    reasoning_content = message.model_extra.get("reasoning_content")
                if (
                    isinstance(reasoning_content, str)
                    and reasoning_content.strip() != ""
                ):
                    content = reasoning_content

            if isinstance(content, str):
                return content
            raise RuntimeError(f"API response missing assistant content: {res}")
        except openai.APIStatusError as e:
            raise RuntimeError(
                f"APIStatusError: {e.status_code}, {e.message}, {e.body}"
            )
        except openai.APIError as e:
            raise RuntimeError(f"APIError: {e.body}")
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise RuntimeError(f"Exception: {e}")

    def generate_texts(self, prompts) -> list:
        """批量生成文本。

        Args:
            prompts: prompt 列表。

        Returns:
            生成文本列表，顺序与输入一致。
        """
        return [self.generate_text(prompt) for prompt in prompts]
