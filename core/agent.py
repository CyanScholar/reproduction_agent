"""
Reproduction Agent - 复现计划生成 Agent 主流程
v0.4: 直接输出 Markdown，去掉 JSON 中间层
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List

from .api_provider import (
    AgentProvider,
    AgentResult,
    CostTracker,
    create_provider,
)
from .models import FeedbackMode
from .task_prompt import (
    AGENT_SYSTEM_PROMPT,
    build_task_prompt,
)

# Tool registry (optional, used by agent providers for tool_use)
try:
    from ..skills.tools import create_default_registry
    _HAS_TOOLS = True
except (ImportError, ValueError):
    _HAS_TOOLS = False

# Paper parser (only needed for non-agent mode)
try:
    from .parser import PaperParser
    _HAS_PARSER = True
except ImportError:
    _HAS_PARSER = False


class ReproductionAgent:
    """
    复现计划生成 Agent

    两种运行模式：
    1. Agent模式 (agent_claude/agent_codex/agent_kimi):
       单次委派给 Agent API，Agent 自主完成全部工作，直接输出 Markdown。
    2. 经典模式 (openai_compatible/anthropic):
       简化的 LLM 调用管道，直接输出 Markdown。
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_style: str = "openai_compatible",
    ):
        # 初始化工具注册表
        tool_registry = None
        if _HAS_TOOLS:
            try:
                tool_registry = create_default_registry()
            except Exception:
                pass

        self.provider = create_provider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_style=api_style,
            tool_registry=tool_registry,
        )
        self._is_agent_mode = isinstance(self.provider, AgentProvider)
        self.cost_tracker = CostTracker()
        self.trajectories: List[Dict] = []

        # Parser only for non-agent mode
        if not self._is_agent_mode and _HAS_PARSER:
            self.parser = PaperParser(provider=self.provider)
        else:
            self.parser = None

    def generate_plan(
        self,
        paper_source: str,
        output_dir: Optional[str] = None,
        feedback_mode: FeedbackMode = FeedbackMode.NONE,
        feedback_callback: Optional[Callable] = None,
        tier: str = "both",
    ) -> str:
        """
        生成复现计划

        Args:
            paper_source: 论文来源 (文件路径 / URL / 搜索关键词)
            output_dir: 输出目录
            feedback_mode: 人工反馈模式
            feedback_callback: 回调函数
            tier: 计划层级 ("mvp", "full", "both")

        Returns:
            生成的 Markdown 文本
        """
        print("=" * 60)
        print("Reproduction Plan Generator v0.4.0")
        print(f"Mode: {'Agent' if self._is_agent_mode else 'Classic'}")
        print("=" * 60)

        if self._is_agent_mode:
            markdown = self._generate_via_agent(paper_source, tier)
        else:
            markdown = self._generate_via_llm(
                paper_source, tier, feedback_mode, feedback_callback,
            )

        # 保存输出
        if output_dir:
            self._save_outputs(markdown, output_dir)

        return markdown

    # =========================================================================
    # Agent Mode: 单次委派
    # =========================================================================

    def _generate_via_agent(self, paper_source: str, tier: str) -> str:
        """Agent 模式：构建任务描述，单次委派给 Agent，直接返回 Markdown"""
        # Step 1: 读取论文内容
        paper_content = self._read_paper_content(paper_source)
        title, authors = self._extract_basic_info(paper_content)

        print(f"\n[Step 1] Paper loaded ({len(paper_content)} chars)")
        print(f"  - Title: {title}")

        # Step 2: 构建任务描述
        print("\n[Step 2] Building task prompt...")
        task = build_task_prompt(paper_content, title, authors, tier=tier)
        print(f"  - Task prompt: {len(task)} chars")

        # Step 3: 委派给 Agent
        print("\n[Step 3] Delegating to Agent...")
        attachments = []
        source_path = Path(paper_source)
        if source_path.exists() and source_path.suffix.lower() == ".pdf":
            attachments = [str(source_path)]

        result: AgentResult = self.provider.run_task(
            task=task,
            system_prompt=AGENT_SYSTEM_PROMPT,
            attachments=attachments,
        )

        print(f"  - Agent completed ({len(result.trajectory)} steps)")
        print(f"  - Cost: ${result.cost:.4f}")
        print(f"  - Tokens: in={result.usage.get('input_tokens', 0)}, out={result.usage.get('output_tokens', 0)}")

        self.trajectories.append({
            "mode": "agent",
            "trajectory": result.trajectory,
            "usage": result.usage,
        })

        markdown = result.raw_output
        print(f"\n[Step 4] Markdown output received ({len(markdown)} chars)")
        return markdown

    # =========================================================================
    # Classic Mode: 简化 LLM 管道
    # =========================================================================

    def _generate_via_llm(
        self,
        paper_source: str,
        tier: str,
        feedback_mode: FeedbackMode,
        feedback_callback: Optional[Callable],
    ) -> str:
        """经典模式：解析论文 + LLM 调用，直接输出 Markdown"""
        # Step 1: 解析论文来源
        resolved_source = self._resolve_paper_source(paper_source)

        # Step 2: 解析论文
        print("\n[Step 1] Parsing paper...")
        paper_context = self.parser.parse(resolved_source)
        print(f"  - Title: {paper_context.title}")
        print(f"  - Sections: {len(paper_context.sections)}")

        paper_content = paper_context.get_full_text()
        authors_str = ", ".join(paper_context.authors[:3])

        # Step 3: 生成 Markdown
        print("\n[Step 2] Generating reproduction plan...")
        task = build_task_prompt(paper_content, paper_context.title, authors_str, tier=tier)

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        print("  - Calling LLM for plan generation...")
        response = self.provider.call(messages, max_tokens=16000)
        self.cost_tracker.add_entry(response, f"plan_{tier}", self.provider)

        self.trajectories.append({
            "mode": "classic",
            "tier": tier,
            "response_preview": response.content[:500],
        })

        markdown = response.content

        # 反馈检查点
        markdown = self._checkpoint(
            f"plan_{tier}", markdown, feedback_mode, feedback_callback,
            hint="Review the generated plan.",
        )

        print("\n" + self.cost_tracker.summary())
        return markdown

    # =========================================================================
    # Shared Helpers
    # =========================================================================

    def _read_paper_content(self, paper_source: str) -> str:
        """读取论文内容为文本"""
        source_path = Path(paper_source)

        # URL — 下载并提取
        if paper_source.startswith("http://") or paper_source.startswith("https://"):
            try:
                import requests
                resp = requests.get(paper_source, timeout=60)
                resp.raise_for_status()
                return resp.text[:200000]
            except Exception as e:
                return f"[Paper from URL: {paper_source}]\n(Could not download: {e})"

        # 本地文件
        if source_path.exists():
            if source_path.suffix.lower() == ".pdf":
                # For agent mode, the PDF will be sent as attachment
                # Return a placeholder — the agent will read the PDF via attachment
                return f"[Paper: {source_path.name}]\n(PDF content will be provided as attachment)"
            else:
                return source_path.read_text(encoding="utf-8", errors="replace")[:200000]

        # 搜索关键词
        return f"[Search query: {paper_source}]\nPlease search for this paper and analyze it."

    def _extract_basic_info(self, paper_content: str) -> tuple:
        """从论文内容中提取基本信息"""
        lines = paper_content.strip().split("\n")
        title = lines[0].strip("# ").strip() if lines else "Unknown"
        authors = ""
        for line in lines[1:10]:
            if "author" in line.lower() or "@" in line or "," in line:
                authors = line.strip()
                break
        return title, authors

    def _resolve_paper_source(self, paper_source: str) -> str:
        """解析论文来源（仅经典模式使用）"""
        if paper_source.startswith("http://") or paper_source.startswith("https://"):
            print(f"\n[Step 0] Paper source: URL — {paper_source}")
            return paper_source

        if Path(paper_source).exists():
            print(f"\n[Step 0] Paper source: local file — {paper_source}")
            return paper_source

        print(f"\n[Step 0] Paper source: search query — {paper_source}")
        return self._search_and_download(paper_source)

    def _search_and_download(self, query: str) -> str:
        """通过 arXiv 搜索论文并下载"""
        try:
            from ..skills.tools import ArxivSearchTool
            tool = ArxivSearchTool()
            result = tool.execute(query=query, action="search", limit=3)
            if result.success and result.data:
                top = result.data[0]
                print(f"  - Found: {top['title']}")
                arxiv_id = top.get("arxiv_id", "")
                if arxiv_id:
                    dl = tool.execute(arxiv_id=arxiv_id, action="download")
                    if dl.success:
                        print(f"  - Downloaded: {dl.data['file_path']}")
                        return dl.data["file_path"]
        except Exception as e:
            print(f"  - Search failed: {e}")

        raise RuntimeError(
            f"无法搜索论文 '{query}'。请提供论文文件路径或 URL。"
        )

    # =========================================================================
    # Feedback Checkpoint
    # =========================================================================

    def _checkpoint(
        self, phase_name: str, output: Any,
        feedback_mode: FeedbackMode, callback: Optional[Callable],
        hint: str = "",
    ) -> Any:
        """在步骤间插入人工反馈"""
        if feedback_mode == FeedbackMode.NONE:
            return output

        if feedback_mode == FeedbackMode.INTERACTIVE:
            return self._interactive_feedback(phase_name, output, hint)

        if feedback_mode == FeedbackMode.CALLBACK and callback:
            modified = callback(phase_name, output)
            return modified if modified is not None else output

        return output

    def _interactive_feedback(self, phase_name: str, output: Any, hint: str) -> Any:
        """交互式反馈"""
        print(f"\n{'=' * 60}")
        print(f"[Checkpoint: {phase_name}] {hint}")
        print(f"{'=' * 60}")

        if isinstance(output, str):
            print(output[:3000])

        if not sys.stdin.isatty():
            print("  (Non-interactive, skipping)")
            return output

        user_input = input("Feedback (Enter to continue): ").strip()
        return output

    # =========================================================================
    # Output
    # =========================================================================

    def _save_outputs(self, markdown: str, output_dir: str):
        """保存输出文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Markdown
        with open(output_path / "reproduction_plan.md", "w", encoding="utf-8") as f:
            f.write(markdown)

        # Trajectories
        with open(output_path / "trajectories.json", "w", encoding="utf-8") as f:
            json.dump(self.trajectories, f, indent=2, ensure_ascii=False, default=str)

        # Cost
        with open(output_path / "cost_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.cost_tracker.to_dict(), f, indent=2)

        print(f"\nOutputs saved to: {output_path}")
