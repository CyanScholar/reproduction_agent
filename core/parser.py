"""
Paper Parser - 论文解析器（调度层）
v0.4: 支持文件路径和 URL，调用 provider.parse_file() 完成解析
"""

import json
import re
import tempfile
from typing import List, Dict, Any
from pathlib import Path

from .models import (
    PaperContext,
    PaperFormat,
    Section,
)

# paper-parsing skill 的默认 prompt（当 SkillLoader 不可用时）
_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert academic paper analyst. Your task is to read a research paper "
    "and extract its structured content into a precise JSON format.\n\n"
    "Key principles:\n"
    "1. Extract ALL sections faithfully — do not summarize or omit content.\n"
    "2. Preserve the original text as much as possible.\n"
    "3. For methodology sections, include full algorithmic details, equations, and hyperparameters.\n"
    "4. If something is unclear or missing, note it explicitly."
)

_DEFAULT_USER_PROMPT = (
    'Please analyze the paper and extract its content into the following JSON structure.\n\n'
    '**IMPORTANT**: Return ONLY valid JSON, no markdown code blocks, no extra text.\n\n'
    '{\n'
    '  "paper_id": "string — use filename or DOI if visible",\n'
    '  "title": "string — exact paper title",\n'
    '  "authors": ["list of author names"],\n'
    '  "abstract": "string — full abstract text",\n'
    '  "sections": [\n'
    '    {\n'
    '      "title": "section title",\n'
    '      "content": "full section text",\n'
    '      "section_number": "e.g. 1, 2, 3.1",\n'
    '      "subsections": []\n'
    '    }\n'
    '  ],\n'
    '  "methodology_keywords": ["list of key method/model names mentioned"],\n'
    '  "code_url": "GitHub/GitLab URL if mentioned, otherwise null",\n'
    '  "key_contributions": ["1-3 sentence summary of each contribution"]\n'
    '}\n\n'
    'Extract the content thoroughly. Include ALL sections from Introduction through References.'
)

# 方法论相关的章节关键词
METHODOLOGY_KEYWORDS = [
    "method", "methodology", "approach", "model", "architecture",
    "algorithm", "framework", "design", "implementation",
    "方法", "模型", "算法", "框架", "设计", "实现",
    "encoder", "decoder", "attention", "transformer",
]


class PaperParser:
    """
    论文解析器 — 调度层

    职责：
    1. 判断输入类型（本地文件 / URL）
    2. 如果是 URL → 下载到临时文件
    3. 加载 paper-parsing skill prompt
    4. 调用 provider.parse_file() 完成解析
    5. 解析返回的 JSON → 构建 PaperContext

    文件上传的具体实现由 Provider 负责（各 API 方式不同）。
    """

    def __init__(self, provider=None):
        """
        Args:
            provider: ModelProvider 实例。
                      如果为 None，需要在 parse() 时传入。
        """
        self.provider = provider

    def parse(self, source: str, provider=None) -> PaperContext:
        """
        解析论文

        Args:
            source: 论文来源，支持：
                - 本地文件路径 (e.g. "paper.pdf")
                - URL (e.g. "https://arxiv.org/pdf/2401.xxxxx")
            provider: 可选，覆盖初始化时的 provider

        Returns:
            PaperContext: 结构化的论文内容
        """
        active_provider = provider or self.provider
        if active_provider is None:
            raise ValueError(
                "PaperParser 需要一个 provider 来调用 API 解析论文。"
                "请在初始化时传入 provider 或在 parse() 时传入。"
            )

        # 判断输入类型
        file_path, is_temp = self._resolve_source(source)

        try:
            # 加载 prompt
            system_prompt, user_prompt = self._load_skill_prompts()

            # 调用 provider.parse_file()
            suffix = file_path.suffix.lower()
            file_size = file_path.stat().st_size
            print(f"  - Sending to API for parsing ({file_size} bytes, {suffix})...")

            response = active_provider.parse_file(file_path, system_prompt, user_prompt)

            # 解析 JSON 响应
            parsed = self._parse_response(response.content)

            # 构建 PaperContext
            return self._to_paper_context(parsed, file_path, suffix)

        finally:
            # 清理临时文件
            if is_temp and file_path.exists():
                file_path.unlink()

    def _resolve_source(self, source: str) -> tuple:
        """
        解析输入来源

        Returns:
            (file_path: Path, is_temp: bool)
        """
        # URL 检测
        if source.startswith("http://") or source.startswith("https://"):
            return self._download_url(source), True

        # 本地文件
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {source}")
        return file_path, False

    def _download_url(self, url: str) -> Path:
        """下载 URL 到临时文件"""
        try:
            import requests
        except ImportError:
            raise ImportError("下载 URL 需要 requests 库: pip install requests")

        print(f"  - Downloading: {url}")
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()

        # 从 URL 或 Content-Type 推断后缀
        suffix = ".pdf"  # 默认 PDF
        if "content-type" in response.headers:
            ct = response.headers["content-type"].lower()
            if "html" in ct:
                suffix = ".html"
            elif "png" in ct:
                suffix = ".png"
            elif "jpeg" in ct or "jpg" in ct:
                suffix = ".jpg"

        # 保存到临时文件
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.close()

        print(f"  - Downloaded to: {tmp.name} ({Path(tmp.name).stat().st_size} bytes)")
        return Path(tmp.name)

    def _load_skill_prompts(self) -> tuple:
        """返回论文解析的 prompt"""
        return _DEFAULT_SYSTEM_PROMPT, _DEFAULT_USER_PROMPT

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON"""
        # 直接尝试解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"API 返回的内容无法解析为 JSON。\n"
            f"返回内容前 500 字符: {content[:500]}"
        )

    def _to_paper_context(
        self, data: Dict[str, Any], file_path: Path, suffix: str
    ) -> PaperContext:
        """将 API 解析结果转换为 PaperContext"""
        paper_id = data.get("paper_id", file_path.stem)
        title = data.get("title", "")
        authors = data.get("authors", [])
        abstract = data.get("abstract", "")

        # 解析 sections
        sections = []
        for sec_data in data.get("sections", []):
            subsections = []
            for sub_data in sec_data.get("subsections", []):
                subsections.append(Section(
                    title=sub_data.get("title", ""),
                    content=sub_data.get("content", ""),
                    section_number=sub_data.get("section_number"),
                ))
            sections.append(Section(
                title=sec_data.get("title", ""),
                content=sec_data.get("content", ""),
                section_number=sec_data.get("section_number"),
                subsections=subsections,
            ))

        # 提取方法论章节
        methodology = _extract_methodology(sections)

        # 提取代码 URL
        code_url = data.get("code_url")

        # 判断格式
        fmt = PaperFormat.PDF if suffix == ".pdf" else PaperFormat.JSON

        return PaperContext(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            methodology=methodology,
            code_url=code_url,
            raw_content=data,
            format=fmt,
        )


def _extract_methodology(sections: List[Section]) -> List[Section]:
    """提取方法论相关的章节"""
    methodology = []
    for section in sections:
        title_lower = section.title.lower()
        if any(kw in title_lower for kw in METHODOLOGY_KEYWORDS):
            methodology.append(section)
            continue
        for sub in section.subsections:
            if any(kw in sub.title.lower() for kw in METHODOLOGY_KEYWORDS):
                methodology.append(section)
                break
    return methodology


def parse_paper(source: str, provider=None) -> PaperContext:
    """便捷函数：解析论文"""
    parser = PaperParser(provider=provider)
    return parser.parse(source)
