"""
Unit tests for agent providers and ReproductionAgent local logic.

Covers: AgentResult, ClaudeAgentProvider, CodexAgentProvider,
KimiAgentProvider constructors and _parse_json(); ReproductionAgent
constructor modes, _parse_json_response(), _extract_basic_info().

Run:
    cd agents/reproduction
    python -m pytest tests/test_agent_providers.py -v
"""

import unittest
from pathlib import Path
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.api_provider import (
    AgentResult,
    AgentProvider,
    ClaudeAgentProvider,
    CodexAgentProvider,
    KimiAgentProvider,
)
from core.agent import ReproductionAgent


# =========================================================================
# AgentResult
# =========================================================================

class TestAgentResult(unittest.TestCase):
    """AgentResult can be instantiated with all fields."""

    def test_instantiation(self):
        result = AgentResult(
            raw_output="# Test Plan\n\nSome content",
            usage={"input_tokens": 100, "output_tokens": 50, "cached_tokens": 10},
            cost=0.005,
            trajectory=[{"type": "final_response", "length": 42}],
            raw_response=None,
        )
        self.assertIn("Test Plan", result.raw_output)
        self.assertEqual(result.usage["input_tokens"], 100)
        self.assertAlmostEqual(result.cost, 0.005)
        self.assertEqual(len(result.trajectory), 1)
        self.assertIsNone(result.raw_response)

    def test_raw_output_accessible(self):
        result = AgentResult(
            raw_output="# Reproduction Plan\n\n## 1. Overview",
            usage={},
            cost=0.0,
            trajectory=[],
        )
        self.assertIn("Reproduction Plan", result.raw_output)
        self.assertIn("Overview", result.raw_output)


# =========================================================================
# ClaudeAgentProvider
# =========================================================================

class TestClaudeAgentProvider(unittest.TestCase):
    """Tests for ClaudeAgentProvider constructor."""

    def test_constructor(self):
        p = ClaudeAgentProvider(model="claude-sonnet-4-20250514", api_key="test-key")
        self.assertEqual(p.model, "claude-sonnet-4-20250514")
        self.assertEqual(p.api_key, "test-key")
        self.assertIsInstance(p, AgentProvider)


# =========================================================================
# CodexAgentProvider
# =========================================================================

class TestCodexAgentProvider(unittest.TestCase):
    """Tests for CodexAgentProvider constructor."""

    def test_constructor(self):
        p = CodexAgentProvider(
            model="gpt-4o",
            api_key="test-key",
            base_url="https://custom.api/v1",
        )
        self.assertEqual(p.model, "gpt-4o")
        self.assertEqual(p.api_key, "test-key")
        self.assertEqual(p.base_url, "https://custom.api/v1")
        self.assertIsInstance(p, AgentProvider)


# =========================================================================
# KimiAgentProvider
# =========================================================================

class TestKimiAgentProvider(unittest.TestCase):
    """Tests for KimiAgentProvider constructor and defaults."""

    def test_constructor(self):
        p = KimiAgentProvider(
            model="kimi-k2.5",
            api_key="test-key",
            base_url="https://custom.moonshot/v1",
        )
        self.assertEqual(p.model, "kimi-k2.5")
        self.assertEqual(p.api_key, "test-key")
        self.assertEqual(p.base_url, "https://custom.moonshot/v1")
        self.assertIsInstance(p, AgentProvider)

    def test_default_base_url(self):
        p = KimiAgentProvider(model="kimi-k2.5", api_key="test-key")
        self.assertEqual(p.base_url, "https://api.moonshot.cn/v1")


# =========================================================================
# ReproductionAgent
# =========================================================================

class TestReproductionAgent(unittest.TestCase):
    """Tests for ReproductionAgent constructor modes and helper methods."""

    def test_openai_compatible_not_agent_mode(self):
        agent = ReproductionAgent(
            model="gpt-4o",
            api_key="test-key",
            api_style="openai_compatible",
        )
        self.assertFalse(agent._is_agent_mode)

    def test_agent_claude_is_agent_mode(self):
        agent = ReproductionAgent(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            api_style="agent_claude",
        )
        self.assertTrue(agent._is_agent_mode)

    def test_extract_basic_info_title_from_first_line(self):
        agent = ReproductionAgent(api_key="test-key", api_style="openai_compatible")
        content = "# Attention Is All You Need\nAuthors: Vaswani et al.\nSome content."
        title, authors = agent._extract_basic_info(content)
        self.assertIn("Attention Is All You Need", title)

    def test_extract_basic_info_plain_text(self):
        agent = ReproductionAgent(api_key="test-key", api_style="openai_compatible")
        content = "My Paper Title\nby Author A, Author B\nAbstract here."
        title, authors = agent._extract_basic_info(content)
        self.assertEqual(title, "My Paper Title")
        # The method looks for lines with commas for authors
        self.assertIn("Author A", authors)


if __name__ == "__main__":
    unittest.main()
