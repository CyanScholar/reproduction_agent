"""
Skills package - Skill definitions, loaders, and unified tool layer.
"""

from .tools import (
    Tool,
    ToolResult,
    ToolRegistry,
    ArxivSearchTool,
    SemanticScholarTool,
    GitHubSearchTool,
    PapersWithCodeTool,
    create_default_registry,
)

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "ArxivSearchTool",
    "SemanticScholarTool",
    "GitHubSearchTool",
    "PapersWithCodeTool",
    "create_default_registry",
]
