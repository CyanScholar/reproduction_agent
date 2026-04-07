"""
Reproduction Planner Agent
复现计划生成器 - 看完paper后可直接得到"先写什么、先跑什么、哪里最容易失败"
"""

__version__ = "0.4.0"

from .core import (
    # Models
    PaperContext,
    ExperimentPlan,
    TieredExperimentPlan,
    Risk,
    RiskMatrix,
    Gap,
    GapReport,
    AcceptanceTest,
    AcceptanceCriteria,
    # Providers
    OpenAICompatibleProvider,
    AnthropicProvider,
    AgentProvider,
    ClaudeAgentProvider,
    CodexAgentProvider,
    KimiAgentProvider,
    CostTracker,
    create_provider,
    # Agent
    ReproductionAgent,
)

__all__ = [
    "PaperContext",
    "ExperimentPlan",
    "TieredExperimentPlan",
    "Risk",
    "RiskMatrix",
    "Gap",
    "GapReport",
    "AcceptanceTest",
    "AcceptanceCriteria",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "AgentProvider",
    "ClaudeAgentProvider",
    "CodexAgentProvider",
    "KimiAgentProvider",
    "CostTracker",
    "create_provider",
    "ReproductionAgent",
]
