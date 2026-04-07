"""
CLI Entry Point - 命令行入口
v0.3: Agent-centric 架构，支持 agent_claude/agent_codex/agent_kimi
"""

import argparse
import os
import sys
from pathlib import Path

from .core import ReproductionAgent
from .core.models import FeedbackMode
from .core.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Reproduction Plan Generator - 看完paper后可直接得到复现方案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Agent 模式 — Claude (推荐)
  export ANTHROPIC_API_KEY="your-key"
  python -m reproduction paper.pdf -o output/ --api-style agent_claude

  # Agent 模式 — Kimi
  export OPENAI_API_KEY="your-kimi-key"
  python -m reproduction paper.pdf -o output/ --api-style agent_kimi --base-url https://api.moonshot.cn/v1

  # Agent 模式 — Codex/OpenAI
  export OPENAI_API_KEY="your-key"
  python -m reproduction paper.pdf -o output/ --api-style agent_codex

  # 经典模式 — OpenAI-compatible
  python -m reproduction paper.pdf -o output/ --api-style openai_compatible --model gpt-4o

  # 经典模式 — Anthropic 原生
  python -m reproduction paper.pdf -o output/ --api-style anthropic --model claude-sonnet-4-20250514

  # 从 URL 解析
  python -m reproduction "https://arxiv.org/pdf/2401.xxxxx" -o output/

  # 搜索关键词自动下载
  python -m reproduction "attention is all you need" -o output/

  # 交互模式
  python -m reproduction paper.pdf -o output/ --interactive
        """
    )

    parser.add_argument(
        "paper", type=str,
        help="论文来源：文件路径、URL 或搜索关键词"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="输出目录路径"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型名称（默认从配置读取）"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API Key（默认从环境变量读取）"
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="API Base URL（自定义 endpoint）"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径"
    )
    parser.add_argument(
        "--api-style", type=str, default=None,
        choices=["openai_compatible", "anthropic", "agent_claude", "agent_codex", "agent_kimi"],
        help="API 风格"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="交互模式：步骤间暂停，允许人工审核"
    )
    parser.add_argument(
        "--tier", type=str, default="both",
        choices=["mvp", "full", "both"],
        help="计划层级（默认: both）"
    )

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # CLI 参数覆盖配置
    model = args.model or config.model_name
    api_key = args.api_key or config.api_key
    base_url = args.base_url or config.base_url
    api_style = args.api_style or config.api_style

    # API Key 检查
    if not api_key:
        if api_style in ("anthropic", "agent_claude"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Error: ANTHROPIC_API_KEY not set.")
                sys.exit(1)
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY not set.")
                sys.exit(1)

    # 默认模型
    if not model:
        model_defaults = {
            "agent_claude": "claude-sonnet-4-20250514",
            "anthropic": "claude-sonnet-4-20250514",
            "agent_codex": "gpt-4o",
            "agent_kimi": "kimi-k2.5",
            "openai_compatible": "gpt-4o",
        }
        model = model_defaults.get(api_style, "gpt-4o")

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    feedback_mode = FeedbackMode.INTERACTIVE if args.interactive else FeedbackMode.NONE

    try:
        agent = ReproductionAgent(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_style=api_style,
        )

        markdown = agent.generate_plan(
            paper_source=args.paper,
            output_dir=args.output,
            feedback_mode=feedback_mode,
            tier=args.tier,
        )

        print("\n" + "=" * 60)
        print("Reproduction plan generated successfully!")
        print("=" * 60)
        print(f"\nOutput length: {len(markdown)} chars")

        if args.output:
            print(f"Outputs saved to: {args.output}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
