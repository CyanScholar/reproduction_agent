"""
论文复现计划示例

使用方式:
    # Agent 模式 — Claude (推荐)
    export ANTHROPIC_API_KEY="your-anthropic-key"
    python3 mvp_example.py --paper paper.pdf --api-style agent_claude

    # Agent 模式 — Kimi
    export OPENAI_API_KEY="your-kimi-key"
    python3 mvp_example.py --paper paper.pdf --api-style agent_kimi --base-url https://api.moonshot.cn/v1

    # Agent 模式 — Codex/OpenAI
    export OPENAI_API_KEY="your-openai-key"
    python3 mvp_example.py --paper paper.pdf --api-style agent_codex

    # 经典模式 — OpenAI-compatible
    export OPENAI_API_KEY="your-key"
    python3 mvp_example.py --paper paper.pdf --api-style openai_compatible --model gpt-4o

    # 经典模式 — Anthropic 原生
    export ANTHROPIC_API_KEY="your-key"
    python3 mvp_example.py --paper paper.pdf --api-style anthropic --model claude-sonnet-4-20250514

    # URL 输入
    python3 mvp_example.py --paper "https://arxiv.org/pdf/2401.00001" --api-style agent_claude
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ReproductionAgent


def main():
    import argparse

    parser = argparse.ArgumentParser(description="论文复现计划生成")
    parser.add_argument(
        "--paper", type=str, required=True,
        help="论文来源：文件路径、URL 或搜索关键词"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).parent / "output"),
        help="输出目录"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型名称（如未指定，根据 api-style 使用默认模型）"
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="自定义 API endpoint（如 https://api.moonshot.cn/v1）"
    )
    parser.add_argument(
        "--api-style", type=str, default="agent_claude",
        choices=["openai_compatible", "anthropic", "agent_claude", "agent_codex", "agent_kimi"],
        help="API 风格（默认: agent_claude）"
    )
    parser.add_argument(
        "--tier", type=str, default="both",
        choices=["mvp", "full", "both"],
        help="计划层级（默认: both）"
    )

    args = parser.parse_args()

    # API Key 检查
    if args.api_style in ("anthropic", "agent_claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("请设置 ANTHROPIC_API_KEY 环境变量:")
            print("  export ANTHROPIC_API_KEY='your-key'")
            sys.exit(1)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("请设置 OPENAI_API_KEY 环境变量:")
            print("  export OPENAI_API_KEY='your-key'")
            sys.exit(1)

    # 默认模型
    model = args.model
    if not model:
        model_defaults = {
            "agent_claude": "claude-sonnet-4-20250514",
            "anthropic": "claude-sonnet-4-20250514",
            "agent_codex": "gpt-4o",
            "agent_kimi": "kimi-k2.5",
            "openai_compatible": "gpt-4o",
        }
        model = model_defaults.get(args.api_style, "gpt-4o")

    # 生成计划
    agent = ReproductionAgent(
        model=model,
        api_key=api_key,
        base_url=args.base_url,
        api_style=args.api_style,
    )
    markdown = agent.generate_plan(
        paper_source=args.paper,
        output_dir=args.output,
        tier=args.tier,
    )

    print("\n" + "=" * 60)
    print("复现计划生成完成!")
    print("=" * 60)
    print(f"Output length: {len(markdown)} chars")
    print(f"\n输出目录: {args.output}")


if __name__ == "__main__":
    main()
