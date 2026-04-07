"""
Core module for reproduction planner.
整合所有核心组件：models, api_provider, agent, task_prompt, template, config
"""

from .models import (
    # Paper models
    PaperContext,
    PaperFormat,
    Section,
    Algorithm,
    Hyperparameter,
    Experiment,
    # Plan models
    ExperimentPlan,
    TieredExperimentPlan,
    PlanTier,
    FeedbackMode,
    Module,
    Risk,
    RiskMatrix,
    RiskCategory,
    Severity,
    Gap,
    GapReport,
    AcceptanceTest,
    AcceptanceCriteria,
)
from .api_provider import (
    ModelProvider,
    OpenAICompatibleProvider,
    AnthropicProvider,
    AgentProvider,
    AgentResult,
    ClaudeAgentProvider,
    CodexAgentProvider,
    KimiAgentProvider,
    CostTracker,
    create_provider,
)
from .agent import ReproductionAgent
from .task_prompt import AGENT_SYSTEM_PROMPT, build_task_prompt
from .config import AgentConfig, load_config

__all__ = [
    # Paper models
    "PaperContext",
    "PaperFormat",
    "Section",
    "Algorithm",
    "Hyperparameter",
    "Experiment",
    # Plan models
    "ExperimentPlan",
    "TieredExperimentPlan",
    "PlanTier",
    "FeedbackMode",
    "Module",
    "Risk",
    "RiskMatrix",
    "RiskCategory",
    "Severity",
    "Gap",
    "GapReport",
    "AcceptanceTest",
    "AcceptanceCriteria",
    # API Provider
    "ModelProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "AgentProvider",
    "AgentResult",
    "ClaudeAgentProvider",
    "CodexAgentProvider",
    "KimiAgentProvider",
    "CostTracker",
    "create_provider",
    # Agent
    "ReproductionAgent",
    # Task Prompt
    "AGENT_SYSTEM_PROMPT",
    "build_task_prompt",
    # Config
    "AgentConfig",
    "load_config",
]
