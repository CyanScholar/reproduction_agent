"""
API Provider - 面向 API 的模型调用抽象层
v0.5: AgentProvider 系统 (ClaudeAgentProvider / CodexAgentProvider / KimiAgentProvider)
      支持 tool_use ReAct 循环、结构化输出、多步推理
v0.4: 多 Provider 支持 (OpenAI-compatible / Anthropic / Agent-style)
      每个 Provider 自行实现 parse_file() 处理文件上传差异
"""

import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """模型响应"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    raw_response: Any = None


@dataclass
class AgentResult:
    """Agent 任务执行结果"""
    raw_output: str               # The raw text output from the agent
    usage: dict               # Token counts {input_tokens, output_tokens, cached_tokens}
    cost: float               # USD cost
    trajectory: list          # List of reasoning/tool-call steps
    raw_response: Any = None


class ModelProvider(ABC):
    """模型提供者抽象类"""

    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, Any]],
        output_format: str = "text",
        temperature: float = 1.0,
        max_tokens: int = 16000,
    ) -> ModelResponse:
        pass

    @abstractmethod
    def parse_file(
        self,
        file_path: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> ModelResponse:
        """
        上传文件给 API 并返回 LLM 的解析结果

        不同 Provider 实现不同的文件上传策略：
        - OpenAI-compatible: files API 上传 / base64 inline
        - Anthropic: base64 document inline
        - Agent-style: 未来实现
        """
        pass

    @abstractmethod
    def get_cost(self, response: ModelResponse) -> float:
        """计算调用成本（美元）"""
        pass


class OpenAICompatibleProvider(ModelProvider):
    """
    OpenAI-compatible API 提供者

    兼容所有实现了 OpenAI Chat Completions API 的服务：
    GPT, Kimi, Minimax, DeepSeek, vLLM 等

    parse_file() 按顺序尝试 3 种策略：
    1. files API 上传 + 文本提取 (Kimi/Moonshot: purpose=file-extract)
    2. files API 上传 + file_id 引用 (OpenAI: purpose=assistants)
    3. base64 inline file_data (OpenAI 新版)
    """

    KNOWN_COSTS = {
        "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o3-mini-high": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "gpt-4-turbo": {"input": 10.00, "cached_input": 5.00, "output": 30.00},
        "deepseek-chat": {"input": 0.14, "cached_input": 0.07, "output": 0.28},
        "deepseek-reasoner": {"input": 0.55, "cached_input": 0.14, "output": 2.19},
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None

    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = OpenAI(**kwargs)
        return self._client

    def call(
        self,
        messages: List[Dict[str, Any]],
        output_format: str = "text",
        temperature: float = 1.0,
        max_tokens: int = 16000,
    ) -> ModelResponse:

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # 处理推理模型（o3/o4 系列）
        if self.model.startswith("o3") or self.model.startswith("o4"):
            kwargs["reasoning_effort"] = "high"
        else:
            kwargs["temperature"] = temperature

        # 处理输出格式
        if output_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        # 提取缓存 token（如果有的话）
        cached_tokens = 0
        if hasattr(response.usage, 'prompt_tokens_details'):
            cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0

        return ModelResponse(
            content=response.choices[0].message.content,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "cached_tokens": cached_tokens,
            },
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response,
        )

    def parse_file(
        self,
        file_path: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> ModelResponse:
        """
        上传文件给 API，3 策略自动降级：
        1. files API (file-extract + files.content) — Kimi/Moonshot
        2. files API (assistants + file_id 引用) — OpenAI
        3. base64 inline file_data — OpenAI 新版格式
        """
        errors = []

        # 策略 1: files API 上传 + 文本提取 (Kimi/Moonshot)
        try:
            return self._parse_via_file_extract(file_path, system_prompt, user_prompt)
        except Exception as e:
            errors.append(f"Strategy 1 (file-extract): {e}")
            logger.debug(f"parse_file strategy 1 failed: {e}")

        # 策略 2: files API 上传 + file_id 引用 (OpenAI)
        try:
            return self._parse_via_file_id(file_path, system_prompt, user_prompt)
        except Exception as e:
            errors.append(f"Strategy 2 (file_id): {e}")
            logger.debug(f"parse_file strategy 2 failed: {e}")

        # 策略 3: base64 inline (file_data 格式)
        try:
            return self._parse_via_base64(file_path, system_prompt, user_prompt)
        except Exception as e:
            errors.append(f"Strategy 3 (base64): {e}")
            logger.debug(f"parse_file strategy 3 failed: {e}")

        # 全部失败
        error_detail = "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(
            f"所有文件上传策略均失败:\n{error_detail}\n"
            f"请检查 API 是否支持文件解析，或尝试其他 API endpoint。"
        )

    def _parse_via_file_extract(
        self, file_path: Path, system_prompt: str, user_prompt: str
    ) -> ModelResponse:
        """策略 1: files.create(purpose=file-extract) + files.content()"""
        print("  - Trying: files API (file-extract)...")
        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="file-extract")

        # 提取文本
        content_resp = self.client.files.content(file_id=file_obj.id)
        if hasattr(content_resp, "text"):
            file_text = content_resp.text
        elif hasattr(content_resp, "read"):
            file_text = content_resp.read().decode("utf-8", errors="replace")
        else:
            file_text = str(content_resp)

        print(f"  - File text extracted ({len(file_text)} chars)")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"以下是论文全文内容:\n\n{file_text}"},
            {"role": "user", "content": user_prompt},
        ]
        return self.call(messages, output_format="json", temperature=0.1, max_tokens=32000)

    def _parse_via_file_id(
        self, file_path: Path, system_prompt: str, user_prompt: str
    ) -> ModelResponse:
        """策略 2: files.create(purpose=assistants) + file_id 引用"""
        print("  - Trying: files API (file_id reference)...")
        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="assistants")

        print(f"  - Uploaded file_id: {file_obj.id}")
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_id": file_obj.id}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self.call(messages, output_format="json", temperature=0.1, max_tokens=32000)

    def _parse_via_base64(
        self, file_path: Path, system_prompt: str, user_prompt: str
    ) -> ModelResponse:
        """策略 3: base64 inline file_data"""
        print("  - Trying: base64 inline...")
        file_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": file_path.name,
                            "file_data": f"data:application/pdf;base64,{file_b64}",
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self.call(messages, output_format="json", temperature=0.1, max_tokens=32000)

    def get_cost(self, response: ModelResponse) -> float:
        """计算调用成本"""
        costs = None
        for known_model, known_costs in self.KNOWN_COSTS.items():
            if self.model.startswith(known_model):
                costs = known_costs
                break
        if costs is None:
            costs = self.KNOWN_COSTS["gpt-4o"]  # fallback

        input_cost = (response.usage["input_tokens"] - response.usage["cached_tokens"]) * costs["input"] / 1_000_000
        cached_cost = response.usage["cached_tokens"] * costs["cached_input"] / 1_000_000
        output_cost = response.usage["output_tokens"] * costs["output"] / 1_000_000

        return input_cost + cached_cost + output_cost


class AnthropicProvider(ModelProvider):
    """
    Anthropic Claude 原生 SDK Provider

    使用 anthropic Python SDK 直接调用 Claude API。
    PDF 通过 base64 document inline 发送。
    """

    KNOWN_COSTS = {
        "claude-sonnet-4": {"input": 3.00, "cached_input": 1.50, "output": 15.00},
        "claude-3-5-sonnet": {"input": 3.00, "cached_input": 1.50, "output": 15.00},
        "claude-3-5-haiku": {"input": 0.80, "cached_input": 0.40, "output": 4.00},
        "claude-haiku-4": {"input": 0.80, "cached_input": 0.40, "output": 4.00},
        "claude-opus-4": {"input": 15.00, "cached_input": 7.50, "output": 75.00},
        "claude-3-opus": {"input": 15.00, "cached_input": 7.50, "output": 75.00},
    }

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("请安装 anthropic: pip install anthropic")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def call(
        self,
        messages: List[Dict[str, Any]],
        output_format: str = "text",
        temperature: float = 1.0,
        max_tokens: int = 16000,
    ) -> ModelResponse:
        """
        调用 Claude API

        自动将 OpenAI 格式的 messages 转换为 Anthropic 格式：
        - role=system 的消息提取为顶层 system 参数
        - 其余保留为 messages
        """
        system_parts = []
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                anthropic_messages.append(msg)

        # Anthropic 没有 response_format，通过 system prompt 指示
        system_text = "\n\n".join(system_parts)
        if output_format == "json":
            system_text += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no extra text."

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text

        response = self.client.messages.create(**kwargs)

        # 提取文本内容
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        # 提取缓存 token
        cached_tokens = 0
        if hasattr(response.usage, "cache_read_input_tokens"):
            cached_tokens = response.usage.cache_read_input_tokens or 0

        return ModelResponse(
            content=content,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cached_tokens": cached_tokens,
            },
            model=response.model,
            finish_reason=response.stop_reason or "end_turn",
            raw_response=response,
        )

    def parse_file(
        self,
        file_path: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> ModelResponse:
        """通过 base64 document inline 发送 PDF 给 Claude"""
        file_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")

        # 判断 MIME 类型
        suffix = file_path.suffix.lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = mime_map.get(suffix, "application/pdf")

        # 选择 content block 类型
        if media_type.startswith("image/"):
            content_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": file_b64,
                },
            }
        else:
            content_block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": file_b64,
                },
            }

        system_text = system_prompt + "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no extra text."

        print(f"  - Sending to Claude ({len(file_b64)} base64 chars, {media_type})...")
        response = self.client.messages.create(
            model=self.model,
            system=system_text,
            max_tokens=32000,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": [
                    content_block,
                    {"type": "text", "text": user_prompt},
                ],
            }],
        )

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        cached_tokens = 0
        if hasattr(response.usage, "cache_read_input_tokens"):
            cached_tokens = response.usage.cache_read_input_tokens or 0

        return ModelResponse(
            content=content,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cached_tokens": cached_tokens,
            },
            model=response.model,
            finish_reason=response.stop_reason or "end_turn",
            raw_response=response,
        )

    def get_cost(self, response: ModelResponse) -> float:
        """计算调用成本"""
        costs = None
        for known_model, known_costs in self.KNOWN_COSTS.items():
            if self.model.startswith(known_model):
                costs = known_costs
                break
        if costs is None:
            costs = self.KNOWN_COSTS["claude-sonnet-4"]  # fallback

        input_cost = (response.usage["input_tokens"] - response.usage["cached_tokens"]) * costs["input"] / 1_000_000
        cached_cost = response.usage["cached_tokens"] * costs["cached_input"] / 1_000_000
        output_cost = response.usage["output_tokens"] * costs["output"] / 1_000_000

        return input_cost + cached_cost + output_cost


class AgentProvider(ABC):
    """
    Agent 风格的模型调用抽象类

    与 ModelProvider 的区别：Agent 风格不是简单的 prompt → response，
    而是给定任务描述后，Agent 会自主执行多步操作（工具调用、推理等），
    最终返回结构化的 AgentResult。
    """

    @abstractmethod
    def run_task(
        self,
        task: str,
        system_prompt: str = "",
        attachments: Optional[List[str]] = None,
    ) -> AgentResult:
        """Execute a complete task and return raw text output."""
        pass


class ClaudeAgentProvider(AgentProvider):
    """
    Anthropic Claude Agent Provider

    使用 Anthropic Messages API 配合 tool_use 实现 ReAct 循环。
    支持本地工具注册表 (ToolRegistry) 进行工具执行。
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        tool_registry=None,
        max_iterations: int = 20,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.tool_registry = tool_registry  # Optional ToolRegistry for local tool execution
        self.max_iterations = max_iterations
        self._client = None

    @property
    def client(self):
        """延迟初始化 Anthropic 客户端"""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("请安装 anthropic: pip install anthropic")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def run_task(
        self,
        task: str,
        system_prompt: str = "",
        attachments: Optional[List[str]] = None,
    ) -> AgentResult:
        # Build tools list from registry
        tools = []
        if self.tool_registry:
            tools = self.tool_registry.get_all_schemas()

        # Build messages
        messages = [{"role": "user", "content": self._build_content(task, attachments)}]

        trajectory = []
        total_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

        # ReAct loop
        for i in range(self.max_iterations):
            kwargs = {
                "model": self.model,
                "max_tokens": 16000,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            if tools:
                kwargs["tools"] = tools

            try:
                response = self.client.messages.create(**kwargs)
            except Exception as e:
                logger.error(f"ClaudeAgentProvider API call failed at iteration {i}: {e}")
                raise

            # Accumulate usage
            total_usage["input_tokens"] += response.usage.input_tokens
            total_usage["output_tokens"] += response.usage.output_tokens
            if hasattr(response.usage, "cache_read_input_tokens"):
                total_usage["cached_tokens"] += (response.usage.cache_read_input_tokens or 0)

            # Check for tool_use
            if response.stop_reason == "tool_use":
                # Process tool calls
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                tool_results = []

                for block in tool_use_blocks:
                    trajectory.append({"type": "tool_call", "name": block.name, "input": block.input})

                    if self.tool_registry:
                        try:
                            result = self.tool_registry.execute(block.name, **block.input)
                            result_str = result.to_context_string() if result.success else f"Error: {result.error}"
                        except Exception as e:
                            result_str = f"Error executing tool {block.name}: {e}"
                    else:
                        result_str = f"Tool {block.name} not available locally"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                    trajectory.append({"type": "tool_result", "name": block.name, "output": result_str[:500]})

                # Append assistant response and tool results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            # end_turn - extract final text
            final_text = "".join(b.text for b in response.content if hasattr(b, "text"))
            trajectory.append({"type": "final_response", "length": len(final_text)})

            cost = self._calculate_cost(total_usage)
            return AgentResult(
                raw_output=final_text,
                usage=total_usage,
                cost=cost,
                trajectory=trajectory,
                raw_response=response,
            )

        # Max iterations reached
        raise RuntimeError(f"Agent exceeded {self.max_iterations} iterations")

    def _build_content(self, task: str, attachments: Optional[List[str]]):
        """
        构建用户消息内容。
        如果附件包含 PDF/图片文件，通过 base64 document blocks 发送。
        """
        if not attachments:
            return task

        content_blocks = []
        for file_path_str in attachments:
            try:
                file_path = Path(file_path_str)
                if not file_path.exists():
                    logger.warning(f"Attachment not found: {file_path_str}")
                    continue

                file_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
                suffix = file_path.suffix.lower()
                mime_map = {
                    ".pdf": "application/pdf",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                media_type = mime_map.get(suffix, "application/pdf")

                if media_type.startswith("image/"):
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": file_b64,
                        },
                    })
                else:
                    content_blocks.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": file_b64,
                        },
                    })
            except Exception as e:
                logger.warning(f"Failed to process attachment {file_path_str}: {e}")

        content_blocks.append({"type": "text", "text": task})
        return content_blocks

    def _calculate_cost(self, usage: dict) -> float:
        """使用 AnthropicProvider.KNOWN_COSTS 计算成本"""
        costs = None
        for known_model, known_costs in AnthropicProvider.KNOWN_COSTS.items():
            if self.model.startswith(known_model):
                costs = known_costs
                break
        if costs is None:
            costs = AnthropicProvider.KNOWN_COSTS["claude-sonnet-4"]  # fallback

        input_cost = (usage["input_tokens"] - usage["cached_tokens"]) * costs["input"] / 1_000_000
        cached_cost = usage["cached_tokens"] * costs["cached_input"] / 1_000_000
        output_cost = usage["output_tokens"] * costs["output"] / 1_000_000

        return input_cost + cached_cost + output_cost


class CodexAgentProvider(AgentProvider):
    """
    OpenAI Codex Agent Provider

    使用 OpenAI Responses API（带内置工具支持），
    如果 Responses API 不可用则降级到 Chat Completions API。
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tool_registry=None,
        max_iterations: int = 20,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self._client = None

    @property
    def client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def run_task(
        self,
        task: str,
        system_prompt: str = "",
        attachments: Optional[List[str]] = None,
    ) -> AgentResult:
        tools = [{"type": "web_search_preview"}]  # Built-in, no local execution needed

        # Add custom function tools from registry
        if self.tool_registry:
            for schema in self.tool_registry.get_all_schemas():
                tools.append({
                    "type": "function",
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema.get("input_schema", {}),
                })

        # Build input
        input_text = task
        if system_prompt:
            input_text = f"System: {system_prompt}\n\n{task}"

        # Try Responses API first, fallback to Chat Completions
        try:
            return self._run_via_responses_api(input_text, tools)
        except Exception as e:
            logger.info(f"Responses API failed ({e}), falling back to Chat Completions")
            return self._run_via_chat_completions(input_text, tools)

    def _run_via_responses_api(
        self, input_text: str, tools: list
    ) -> AgentResult:
        """通过 OpenAI Responses API 执行任务"""
        resp_kwargs = {
            "model": self.model,
            "input": input_text,
            "tools": tools,
        }

        response = self.client.responses.create(**resp_kwargs)

        trajectory = []
        total_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

        for i in range(self.max_iterations):
            # Accumulate usage
            if hasattr(response, "usage") and response.usage:
                total_usage["input_tokens"] += getattr(response.usage, "input_tokens", 0)
                total_usage["output_tokens"] += getattr(response.usage, "output_tokens", 0)

            # Check for function_call items that need local execution
            function_calls = [item for item in response.output if item.type == "function_call"]

            if not function_calls:
                break  # No more tool calls, extract result

            tool_outputs = []
            for call in function_calls:
                trajectory.append({"type": "tool_call", "name": call.name, "input": call.arguments})

                if self.tool_registry:
                    try:
                        args = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
                        result = self.tool_registry.execute(call.name, **args)
                        result_str = result.to_context_string() if result.success else f"Error: {result.error}"
                    except Exception as e:
                        result_str = f"Error executing tool {call.name}: {e}"
                else:
                    result_str = f"Tool {call.name} not available"

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result_str,
                })

            response = self.client.responses.create(
                model=self.model,
                input=tool_outputs,
                previous_response_id=response.id,
                tools=tools,
            )

        # Extract final text from output
        final_text = ""
        for item in response.output:
            if hasattr(item, "text"):
                final_text = item.text
                break
            elif item.type == "message":
                for content in item.content:
                    if hasattr(content, "text"):
                        final_text = content.text
                        break
                if final_text:
                    break

        trajectory.append({"type": "final_response", "length": len(final_text)})
        cost = self._calculate_cost(total_usage)
        return AgentResult(
            raw_output=final_text,
            usage=total_usage,
            cost=cost,
            trajectory=trajectory,
            raw_response=response,
        )

    def _run_via_chat_completions(
        self, input_text: str, tools: list
    ) -> AgentResult:
        """降级到标准 Chat Completions API 实现函数调用"""
        messages = [{"role": "user", "content": input_text}]

        # Convert tools to chat completions format (filter out built-in types)
        chat_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                chat_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                })

        trajectory = []
        total_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

        for i in range(self.max_iterations):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 16000,
            }
            if chat_tools:
                kwargs["tools"] = chat_tools

            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.error(f"CodexAgentProvider Chat Completions call failed at iteration {i}: {e}")
                raise

            # Accumulate usage
            if response.usage:
                total_usage["input_tokens"] += response.usage.prompt_tokens
                total_usage["output_tokens"] += response.usage.completion_tokens

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)

                for tc in choice.message.tool_calls:
                    fn_name = tc.function.name
                    trajectory.append({"type": "tool_call", "name": fn_name, "input": tc.function.arguments})

                    if self.tool_registry:
                        try:
                            args = json.loads(tc.function.arguments)
                            result = self.tool_registry.execute(fn_name, **args)
                            result_str = result.to_context_string() if result.success else f"Error: {result.error}"
                        except Exception as e:
                            result_str = f"Error executing tool {fn_name}: {e}"
                    else:
                        result_str = f"Tool {fn_name} not available"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
                continue

            # Done
            final_text = choice.message.content or ""
            trajectory.append({"type": "final_response", "length": len(final_text)})
            cost = self._calculate_cost(total_usage)
            return AgentResult(
                raw_output=final_text,
                usage=total_usage,
                cost=cost,
                trajectory=trajectory,
                raw_response=response,
            )

        raise RuntimeError(f"Agent exceeded {self.max_iterations} iterations")

    def _calculate_cost(self, usage: dict) -> float:
        """使用 OpenAICompatibleProvider.KNOWN_COSTS 计算成本"""
        costs = None
        for known_model, known_costs in OpenAICompatibleProvider.KNOWN_COSTS.items():
            if self.model.startswith(known_model):
                costs = known_costs
                break
        if costs is None:
            costs = OpenAICompatibleProvider.KNOWN_COSTS["gpt-4o"]  # fallback

        input_cost = (usage["input_tokens"] - usage["cached_tokens"]) * costs["input"] / 1_000_000
        cached_cost = usage["cached_tokens"] * costs["cached_input"] / 1_000_000
        output_cost = usage["output_tokens"] * costs["output"] / 1_000_000

        return input_cost + cached_cost + output_cost


class KimiAgentProvider(AgentProvider):
    """
    Kimi Agent Provider

    使用 OpenAI-compatible API 配合 Kimi 的文件理解能力。
    支持通过 files API 上传 PDF 进行内容提取，
    并支持 Kimi 内置的 $web_search 工具。
    """

    def __init__(
        self,
        model: str = "kimi-k2.5",
        api_key: Optional[str] = None,
        base_url: str = "https://api.moonshot.cn/v1",
        tool_registry=None,
        max_iterations: int = 20,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        self.base_url = base_url
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self._client = None

    @property
    def client(self):
        """延迟初始化 OpenAI 兼容客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def run_task(
        self,
        task: str,
        system_prompt: str = "",
        attachments: Optional[List[str]] = None,
    ) -> AgentResult:
        # Step 1: If attachments contain PDF, upload via files API
        file_text = ""
        if attachments:
            try:
                file_text = self._extract_file_content(attachments[0])
            except Exception as e:
                logger.warning(f"KimiAgentProvider file extraction failed: {e}")

        # Step 2: Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = task
        if file_text:
            user_content = f"Paper content:\n{file_text}\n\n{task}"
        messages.append({"role": "user", "content": user_content})

        # Step 3: Build tools
        tools = [{"type": "builtin_function", "function": {"name": "$web_search"}}]
        if self.tool_registry:
            for schema in self.tool_registry.get_all_schemas():
                tools.append({
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema.get("description", ""),
                        "parameters": schema.get("input_schema", {}),
                    },
                })

        # Step 4: Chat completion loop
        trajectory = []
        total_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

        for i in range(self.max_iterations):
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools

            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.error(f"KimiAgentProvider API call failed at iteration {i}: {e}")
                raise

            # Accumulate usage
            if response.usage:
                total_usage["input_tokens"] += response.usage.prompt_tokens
                total_usage["output_tokens"] += response.usage.completion_tokens

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)  # Append assistant message with tool_calls

                for tc in choice.message.tool_calls:
                    fn_name = tc.function.name
                    trajectory.append({"type": "tool_call", "name": fn_name, "input": tc.function.arguments})

                    if fn_name.startswith("$"):
                        # Built-in function, handled by Kimi server-side
                        continue

                    if self.tool_registry:
                        try:
                            args = json.loads(tc.function.arguments)
                            result = self.tool_registry.execute(fn_name, **args)
                            result_str = result.to_context_string() if result.success else f"Error: {result.error}"
                        except Exception as e:
                            result_str = f"Error executing tool {fn_name}: {e}"
                    else:
                        result_str = f"Tool {fn_name} not available"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
                continue

            # Done
            final_text = choice.message.content or ""
            trajectory.append({"type": "final_response", "length": len(final_text)})
            cost = self._calculate_cost(total_usage)
            return AgentResult(
                raw_output=final_text,
                usage=total_usage,
                cost=cost,
                trajectory=trajectory,
                raw_response=response,
            )

        raise RuntimeError(f"Agent exceeded {self.max_iterations} iterations")

    def _extract_file_content(self, file_path: str) -> str:
        """通过 Kimi files API 上传文件并提取文本内容"""
        try:
            with open(file_path, "rb") as f:
                file_obj = self.client.files.create(file=f, purpose="file-extract")
            content_resp = self.client.files.content(file_id=file_obj.id)
            if hasattr(content_resp, "text"):
                return content_resp.text
            elif hasattr(content_resp, "read"):
                return content_resp.read().decode("utf-8", errors="replace")
            else:
                return str(content_resp)
        except Exception as e:
            logger.warning(f"File extraction via Kimi files API failed: {e}")
            raise

    def _calculate_cost(self, usage: dict) -> float:
        """Kimi 成本估算（使用类似 GPT-4o-mini 的价格作为近似）"""
        # Kimi pricing is approximate; use a conservative estimate
        costs = {"input": 0.15, "cached_input": 0.075, "output": 0.60}

        input_cost = (usage["input_tokens"] - usage["cached_tokens"]) * costs["input"] / 1_000_000
        cached_cost = usage["cached_tokens"] * costs["cached_input"] / 1_000_000
        output_cost = usage["output_tokens"] * costs["output"] / 1_000_000

        return input_cost + cached_cost + output_cost


# =========================================================================
# Factory
# =========================================================================

def create_provider(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_style: str = "openai_compatible",
    tool_registry=None,
):
    """
    工厂函数：根据 api_style 创建 Provider

    Args:
        model: 模型名称（传给 API 的 model 参数）
        api_key: API Key
        base_url: 自定义 API endpoint
        api_style: API 风格
            - "openai_compatible"（默认）: OpenAI-compatible Chat Completions API
            - "anthropic": Anthropic Claude 原生 SDK
            - "agent_claude": Claude Agent (Anthropic Messages API + tool_use ReAct)
            - "agent_codex": Codex Agent (OpenAI Responses API + built-in tools)
            - "agent_kimi": Kimi Agent (OpenAI-compatible API + Kimi file understanding)
        tool_registry: 可选的工具注册表，仅 agent_* 风格使用
    """
    if api_style == "agent_claude":
        return ClaudeAgentProvider(model=model, api_key=api_key, tool_registry=tool_registry)
    elif api_style == "agent_codex":
        return CodexAgentProvider(model=model, api_key=api_key, base_url=base_url, tool_registry=tool_registry)
    elif api_style == "agent_kimi":
        return KimiAgentProvider(
            model=model, api_key=api_key,
            base_url=base_url or "https://api.moonshot.cn/v1",
            tool_registry=tool_registry,
        )
    elif api_style == "openai_compatible":
        return OpenAICompatibleProvider(model=model, api_key=api_key, base_url=base_url)
    elif api_style == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown api_style: {api_style}")


# =========================================================================
# Cost Tracker
# =========================================================================

class CostTracker:
    """成本追踪器"""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.total_cost = 0.0

    def add_entry(
        self,
        response: ModelResponse,
        stage: str,
        provider: ModelProvider,
    ):
        """添加成本记录"""
        cost = provider.get_cost(response)
        entry = {
            "stage": stage,
            "model": response.model,
            "input_tokens": response.usage["input_tokens"],
            "output_tokens": response.usage["output_tokens"],
            "cached_tokens": response.usage["cached_tokens"],
            "cost": cost,
        }
        self.entries.append(entry)
        self.total_cost += cost

    def summary(self) -> str:
        """生成成本摘要"""
        lines = [
            "=" * 50,
            "Cost Summary",
            "=" * 50,
        ]
        for entry in self.entries:
            lines.append(
                f"{entry['stage']}: ${entry['cost']:.4f} "
                f"(in: {entry['input_tokens']}, out: {entry['output_tokens']}, "
                f"cached: {entry['cached_tokens']})"
            )
        lines.append("=" * 50)
        lines.append(f"Total: ${self.total_cost:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "entries": self.entries,
            "total_cost": self.total_cost,
        }
