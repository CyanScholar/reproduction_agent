"""
Config - 配置加载
从 YAML 文件加载配置，支持环境变量替换
"""

import os
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict


@dataclass
class AgentConfig:
    """Agent 配置"""
    # 模型设置
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_style: str = "openai_compatible"  # "openai_compatible", "anthropic", or "agent"

    # 输出设置
    output_format: str = "markdown"
    include_trajectory: bool = True
    include_cost: bool = True

    # 规划设置
    max_paper_length: int = 100000
    phases_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "planning": True,
        "risk_analysis": True,
        "acceptance": True,
    })

    # 成本限制
    max_cost_per_paper: float = 5.0
    warn_cost: float = 2.0


def _substitute_env_vars(value: str) -> str:
    """替换 ${VAR} 或 $VAR 格式的环境变量"""
    if not isinstance(value, str):
        return value

    def replacer(match):
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r'\$\{(\w+)\}|\$(\w+)', replacer, value)


def load_config(config_path: Optional[Path] = None) -> AgentConfig:
    """
    从 YAML 加载配置

    Args:
        config_path: 配置文件路径，默认为包目录下的 config.yaml

    Returns:
        AgentConfig 实例
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        return AgentConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        return AgentConfig()

    # 提取各级配置
    model_cfg = raw.get("model", {})
    output_cfg = raw.get("output", {})
    planning_cfg = raw.get("planning", {})
    limits_cfg = raw.get("limits", {})

    # 构建 phases_enabled
    phases_enabled = {"planning": True, "risk_analysis": True, "acceptance": True}
    for phase in planning_cfg.get("phases", []):
        name = phase.get("name", "")
        phases_enabled[name] = phase.get("enabled", True)

    # API key 环境变量替换
    api_key = _substitute_env_vars(model_cfg.get("api_key", "")) or None
    base_url = model_cfg.get("base_url")
    if isinstance(base_url, str):
        base_url = _substitute_env_vars(base_url) or None

    return AgentConfig(
        model_name=model_cfg.get("name", "gpt-4o"),
        api_key=api_key,
        base_url=base_url,
        api_style=model_cfg.get("api_style", "openai_compatible"),
        output_format=output_cfg.get("format", "markdown"),
        include_trajectory=output_cfg.get("include_trajectory", True),
        include_cost=output_cfg.get("include_cost", True),
        max_paper_length=planning_cfg.get("max_paper_length", 100000),
        phases_enabled=phases_enabled,
        max_cost_per_paper=limits_cfg.get("max_cost_per_paper", 5.0),
        warn_cost=limits_cfg.get("warn_cost", 2.0),
    )
