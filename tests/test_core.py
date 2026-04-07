"""
Unit tests for refactored Reproduction Planner core modules.

Covers: models (from_dict / to_dict round-trips), task_prompt constants
and builder, provider factory, skills/tools layer, and config loading.

Run:
    cd agents/reproduction
    python -m pytest tests/test_core.py -v
"""

import json
import unittest
from pathlib import Path
import sys

# Ensure the project root is on sys.path so "core" / "skills" are importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    Risk,
    RiskCategory,
    Severity,
    RiskMatrix,
    Gap,
    Module,
    AcceptanceTest,
    ExperimentPlan,
    TieredExperimentPlan,
)
from core.task_prompt import (
    AGENT_SYSTEM_PROMPT,
    build_task_prompt,
)
from core.api_provider import (
    OpenAICompatibleProvider,
    AnthropicProvider,
    ClaudeAgentProvider,
    CodexAgentProvider,
    KimiAgentProvider,
    create_provider,
)
from core.config import AgentConfig, load_config
from skills.tools import (
    ToolResult,
    create_default_registry,
)


# =========================================================================
# Models: from_dict / to_dict round-trip
# =========================================================================

class TestRiskRoundTrip(unittest.TestCase):
    """Risk.from_dict() / to_dict() round-trip."""

    def test_round_trip(self):
        original = Risk(
            category=RiskCategory.PAPER_CLARITY,
            severity=Severity.HIGH,
            component="Encoder",
            description="Ambiguous layer norm position",
            paper_reference="Section 3.1",
            mitigation="Try both pre- and post-norm",
            confidence=0.8,
        )
        d = original.to_dict()
        restored = Risk.from_dict(d)

        self.assertEqual(restored.category, original.category)
        self.assertEqual(restored.severity, original.severity)
        self.assertEqual(restored.component, original.component)
        self.assertEqual(restored.description, original.description)
        self.assertEqual(restored.paper_reference, original.paper_reference)
        self.assertEqual(restored.mitigation, original.mitigation)
        self.assertAlmostEqual(restored.confidence, original.confidence)

    def test_from_dict_missing_fields(self):
        """Missing fields should fall back to defaults."""
        r = Risk.from_dict({})
        self.assertEqual(r.category, RiskCategory.IMPLEMENTATION_DIFFICULTY)
        self.assertEqual(r.severity, Severity.MEDIUM)
        self.assertEqual(r.component, "")
        self.assertEqual(r.description, "")

    def test_from_dict_unknown_enum(self):
        """Unknown enum values should fall back to defaults."""
        r = Risk.from_dict({
            "category": "nonexistent_category",
            "severity": "ultra_critical",
        })
        self.assertEqual(r.category, RiskCategory.IMPLEMENTATION_DIFFICULTY)
        self.assertEqual(r.severity, Severity.MEDIUM)


class TestGapRoundTrip(unittest.TestCase):
    """Gap.from_dict() / to_dict() round-trip."""

    def test_round_trip(self):
        original = Gap(
            type="ambiguous",
            section="Section 4",
            description="Unclear augmentation pipeline",
            impact="important",
            suggestion="Try standard ImageNet augmentation",
        )
        d = original.to_dict()
        restored = Gap.from_dict(d)

        self.assertEqual(restored.type, original.type)
        self.assertEqual(restored.section, original.section)
        self.assertEqual(restored.description, original.description)
        self.assertEqual(restored.impact, original.impact)
        self.assertEqual(restored.suggestion, original.suggestion)

    def test_from_dict_missing_fields(self):
        g = Gap.from_dict({})
        self.assertEqual(g.type, "missing")
        self.assertEqual(g.section, "")
        self.assertEqual(g.impact, "important")
        self.assertIsNone(g.suggestion)


class TestModuleRoundTrip(unittest.TestCase):
    """Module.from_dict() / to_dict() round-trip."""

    def test_round_trip(self):
        original = Module(
            name="DataLoader",
            file_path="src/data/loader.py",
            description="Handles dataset loading and preprocessing",
            dependencies=["torch", "torchvision"],
            interfaces=["def load(path: str) -> Dataset"],
            implementation_notes="Use streaming for large datasets",
        )
        d = original.to_dict()
        restored = Module.from_dict(d)

        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.file_path, original.file_path)
        self.assertEqual(restored.description, original.description)
        self.assertEqual(restored.dependencies, original.dependencies)
        self.assertEqual(restored.interfaces, original.interfaces)
        self.assertEqual(restored.implementation_notes, original.implementation_notes)

    def test_from_dict_missing_fields(self):
        m = Module.from_dict({})
        self.assertEqual(m.name, "")
        self.assertEqual(m.file_path, "")
        self.assertEqual(m.dependencies, [])
        self.assertEqual(m.interfaces, [])


class TestAcceptanceTestRoundTrip(unittest.TestCase):
    """AcceptanceTest.from_dict() / to_dict() round-trip."""

    def test_round_trip(self):
        original = AcceptanceTest(
            name="test_output_shape",
            test_type="unit",
            description="Verify model output tensor shape",
            procedure="Run model.forward(dummy_input)",
            expected_outcome="Output shape matches (B, C, H, W)",
            tolerance=0.001,
        )
        d = original.to_dict()
        restored = AcceptanceTest.from_dict(d)

        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.test_type, original.test_type)
        self.assertEqual(restored.description, original.description)
        self.assertEqual(restored.procedure, original.procedure)
        self.assertEqual(restored.expected_outcome, original.expected_outcome)
        self.assertAlmostEqual(restored.tolerance, original.tolerance)

    def test_from_dict_missing_fields(self):
        at = AcceptanceTest.from_dict({})
        self.assertEqual(at.name, "")
        self.assertEqual(at.test_type, "automated")
        self.assertIsNone(at.tolerance)


class TestRiskMatrixFromDict(unittest.TestCase):
    """RiskMatrix.from_dict() with multiple risks."""

    def test_from_dict_multiple_risks(self):
        d = {
            "risks": [
                {
                    "category": "implementation_difficulty",
                    "severity": "high",
                    "component": "A",
                    "description": "Complex math",
                },
                {
                    "category": "paper_clarity",
                    "severity": "low",
                    "component": "B",
                    "description": "Clear enough",
                },
                {
                    "category": "data_availability",
                    "severity": "critical",
                    "component": "C",
                    "description": "Proprietary dataset",
                },
            ],
            "overall_score": 3.0,
        }
        matrix = RiskMatrix.from_dict(d)
        self.assertEqual(len(matrix.risks), 3)
        self.assertEqual(len(matrix.get_by_severity(Severity.HIGH)), 1)
        self.assertEqual(len(matrix.get_by_severity(Severity.LOW)), 1)
        self.assertEqual(len(matrix.get_by_severity(Severity.CRITICAL)), 1)
        # overall_score is recalculated, not taken from the dict
        self.assertGreater(matrix.overall_score(), 1.0)

    def test_from_dict_empty(self):
        matrix = RiskMatrix.from_dict({})
        self.assertEqual(len(matrix.risks), 0)
        self.assertAlmostEqual(matrix.overall_score(), 1.0)


class TestExperimentPlanFromDict(unittest.TestCase):
    """ExperimentPlan.from_dict() with full nested data."""

    def _make_full_dict(self):
        return {
            "paper_id": "2401.00001",
            "paper_title": "A Novel Approach",
            "generated_at": "2026-01-01T00:00:00",
            "generator_version": "0.2.0",
            "tier": "full",
            "overview": "Full reproduction of the novel approach",
            "modules": [
                {
                    "name": "Encoder",
                    "file_path": "src/encoder.py",
                    "description": "Main encoder",
                    "dependencies": ["torch"],
                    "interfaces": ["def forward(x)"],
                    "implementation_notes": "See Section 3",
                },
            ],
            "file_list": ["src/encoder.py", "src/train.py"],
            "architecture_diagram": "classDiagram ...",
            "call_flow": "sequenceDiagram ...",
            "python_packages": ["torch>=2.0", "numpy"],
            "other_dependencies": {"system": ["cuda"]},
            "hardware_requirements": {"gpu": "A100", "gpu_memory_gb": 40},
            "config": {"lr": "3e-4 [PAPER]", "batch_size": "32 [PAPER]"},
            "hyperparameters": [
                {"name": "lr", "value": 3e-4, "source": "explicit", "confidence": 1.0},
            ],
            "risks": {
                "risks": [
                    {
                        "category": "implementation_difficulty",
                        "severity": "high",
                        "component": "Encoder",
                        "description": "Novel attention mechanism",
                        "mitigation": "Reference unofficial impl",
                    },
                ],
                "overall_score": 3.5,
            },
            "gaps": {
                "gaps": [
                    {
                        "type": "missing",
                        "section": "Section 4",
                        "description": "LR schedule not specified",
                        "impact": "critical",
                        "suggestion": "Try cosine annealing",
                    },
                ],
            },
            "acceptance_criteria": {
                "automated_tests": [
                    {
                        "name": "test_accuracy",
                        "test_type": "benchmark",
                        "description": "Check final accuracy",
                        "procedure": "Run eval.py",
                        "expected_outcome": "accuracy >= 0.90",
                        "tolerance": 0.02,
                    },
                ],
                "manual_checks": [],
                "benchmark_targets": {
                    "accuracy": {"reported": 0.95, "min_acceptable": 0.90},
                },
            },
            "reproduce_script": "bash run.sh",
        }

    def test_full_round_trip(self):
        d = self._make_full_dict()
        plan = ExperimentPlan.from_dict(d)

        self.assertEqual(plan.paper_id, "2401.00001")
        self.assertEqual(plan.paper_title, "A Novel Approach")
        self.assertEqual(plan.tier, "full")
        self.assertEqual(len(plan.modules), 1)
        self.assertEqual(plan.modules[0].name, "Encoder")
        self.assertIsNotNone(plan.risks)
        self.assertEqual(len(plan.risks.risks), 1)
        self.assertIsNotNone(plan.gaps)
        self.assertEqual(len(plan.gaps.gaps), 1)
        self.assertIsNotNone(plan.acceptance_criteria)
        self.assertEqual(len(plan.acceptance_criteria.automated_tests), 1)
        self.assertEqual(plan.reproduce_script, "bash run.sh")

        # Serialize back and verify JSON round-trip
        json_str = plan.to_json()
        reparsed = json.loads(json_str)
        self.assertEqual(reparsed["paper_id"], "2401.00001")
        self.assertEqual(reparsed["tier"], "full")

    def test_from_dict_minimal(self):
        """from_dict with minimal/missing fields uses defaults."""
        plan = ExperimentPlan.from_dict({"paper_id": "test", "paper_title": "T"})
        self.assertEqual(plan.paper_id, "test")
        self.assertEqual(plan.tier, "full")
        self.assertEqual(plan.modules, [])
        self.assertIsNone(plan.risks)
        self.assertIsNone(plan.gaps)
        self.assertIsNone(plan.acceptance_criteria)


class TestTieredExperimentPlanFromDict(unittest.TestCase):
    """TieredExperimentPlan.from_dict() with mvp + full plans."""

    def test_round_trip(self):
        d = {
            "paper_id": "test123",
            "paper_title": "Test Paper",
            "generated_at": "2026-01-01T00:00:00",
            "generator_version": "0.2.0",
            "tier_rationale": "Core algorithm first",
            "upgrade_path": ["Add dataset B", "Enable multi-GPU"],
            "mvp_plan": {
                "paper_id": "test123",
                "paper_title": "Test Paper",
                "tier": "mvp",
                "overview": "MVP overview",
            },
            "full_plan": {
                "paper_id": "test123",
                "paper_title": "Test Paper",
                "tier": "full",
                "overview": "Full overview",
            },
        }
        tiered = TieredExperimentPlan.from_dict(d)

        self.assertEqual(tiered.paper_id, "test123")
        self.assertEqual(tiered.paper_title, "Test Paper")
        self.assertEqual(tiered.tier_rationale, "Core algorithm first")
        self.assertEqual(len(tiered.upgrade_path), 2)
        self.assertIsNotNone(tiered.mvp_plan)
        self.assertEqual(tiered.mvp_plan.tier, "mvp")
        self.assertIsNotNone(tiered.full_plan)
        self.assertEqual(tiered.full_plan.tier, "full")

        # Serialize and re-parse
        json_str = tiered.to_json()
        reparsed = json.loads(json_str)
        self.assertEqual(reparsed["mvp_plan"]["tier"], "mvp")
        self.assertEqual(reparsed["full_plan"]["tier"], "full")

    def test_from_dict_no_plans(self):
        tiered = TieredExperimentPlan.from_dict({
            "paper_id": "x",
            "paper_title": "X",
        })
        self.assertIsNone(tiered.mvp_plan)
        self.assertIsNone(tiered.full_plan)


# =========================================================================
# Task Prompt
# =========================================================================

class TestTaskPrompt(unittest.TestCase):
    """Tests for AGENT_SYSTEM_PROMPT and build_task_prompt."""

    def test_system_prompt_nonempty(self):
        self.assertTrue(len(AGENT_SYSTEM_PROMPT) > 100)

    def test_system_prompt_key_phrases(self):
        prompt = AGENT_SYSTEM_PROMPT
        self.assertIn("reproduction", prompt.lower())
        self.assertIn("risk", prompt.lower())
        self.assertIn("gap", prompt.lower())
        self.assertIn("MVP", prompt)

    def test_build_task_prompt_nonempty(self):
        prompt = build_task_prompt(
            paper_content="Some paper content here.",
            title="Test Paper",
            authors="Author A, Author B",
            tier="full",
        )
        self.assertTrue(len(prompt) > 0)
        self.assertIn("Some paper content here.", prompt)
        self.assertIn("Test Paper", prompt)

    def test_build_task_prompt_different_tiers(self):
        mvp_prompt = build_task_prompt(
            paper_content="Content",
            title="Title",
            authors="Authors",
            tier="mvp",
        )
        full_prompt = build_task_prompt(
            paper_content="Content",
            title="Title",
            authors="Authors",
            tier="full",
        )
        both_prompt = build_task_prompt(
            paper_content="Content",
            title="Title",
            authors="Authors",
            tier="both",
        )
        # mvp and full should differ (different tier instructions)
        self.assertNotEqual(mvp_prompt, full_prompt)
        # "both" delegates to tiered prompt builder, should also differ
        self.assertNotEqual(both_prompt, full_prompt)
        self.assertNotEqual(both_prompt, mvp_prompt)
        # mvp prompt should mention MVP-specific scope
        self.assertIn("MVP", mvp_prompt)
        # full prompt should mention Full-specific scope
        self.assertIn("Full", full_prompt)


# =========================================================================
# Provider Factory
# =========================================================================

class TestProviderFactory(unittest.TestCase):
    """Tests for create_provider() factory function."""

    def test_create_openai_compatible(self):
        p = create_provider("gpt-4o", api_key="test", api_style="openai_compatible")
        self.assertIsInstance(p, OpenAICompatibleProvider)
        self.assertEqual(p.model, "gpt-4o")

    def test_create_anthropic(self):
        p = create_provider("claude-sonnet-4", api_key="test", api_style="anthropic")
        self.assertIsInstance(p, AnthropicProvider)
        self.assertEqual(p.model, "claude-sonnet-4")

    def test_create_agent_claude(self):
        p = create_provider("claude-sonnet-4", api_key="test", api_style="agent_claude")
        self.assertIsInstance(p, ClaudeAgentProvider)
        self.assertEqual(p.model, "claude-sonnet-4")

    def test_create_agent_codex(self):
        p = create_provider("gpt-4o", api_key="test", api_style="agent_codex")
        self.assertIsInstance(p, CodexAgentProvider)
        self.assertEqual(p.model, "gpt-4o")

    def test_create_agent_kimi(self):
        p = create_provider("kimi-k2.5", api_key="test", api_style="agent_kimi")
        self.assertIsInstance(p, KimiAgentProvider)
        self.assertEqual(p.model, "kimi-k2.5")

    def test_unknown_api_style_raises(self):
        with self.assertRaises(ValueError) as ctx:
            create_provider("gpt-4o", api_key="test", api_style="nonexistent")
        self.assertIn("Unknown api_style", str(ctx.exception))


# =========================================================================
# Tools (skills/tools.py)
# =========================================================================

class TestToolRegistry(unittest.TestCase):
    """Tests for create_default_registry and tool schemas."""

    def test_default_registry_has_4_tools(self):
        registry = create_default_registry()
        names = registry.list_tools()
        self.assertEqual(len(names), 4)
        self.assertIn("arxiv_search", names)
        self.assertIn("semantic_scholar", names)
        self.assertIn("github_search", names)
        self.assertIn("papers_with_code", names)

    def test_tool_schema_structure(self):
        """Each tool's to_schema() returns a dict with 'name' and 'input_schema'."""
        registry = create_default_registry()
        for tool_name in registry.list_tools():
            tool = registry.get(tool_name)
            schema = tool.to_schema()
            self.assertIsInstance(schema, dict)
            self.assertIn("name", schema)
            self.assertIn("input_schema", schema)
            self.assertEqual(schema["name"], tool_name)
            self.assertIsInstance(schema["input_schema"], dict)

    def test_get_all_schemas(self):
        registry = create_default_registry()
        schemas = registry.get_all_schemas()
        self.assertIsInstance(schemas, list)
        self.assertEqual(len(schemas), 4)
        for s in schemas:
            self.assertIn("name", s)
            self.assertIn("input_schema", s)


class TestToolResult(unittest.TestCase):
    """Tests for ToolResult.to_context_string()."""

    def test_success_context_string(self):
        result = ToolResult(
            tool_name="test_tool",
            success=True,
            data={"key": "value"},
        )
        ctx = result.to_context_string()
        self.assertIn("test_tool", ctx)
        self.assertIn("key", ctx)
        self.assertIn("value", ctx)

    def test_failure_context_string(self):
        result = ToolResult(
            tool_name="test_tool",
            success=False,
            error="Connection timeout",
        )
        ctx = result.to_context_string()
        self.assertIn("test_tool", ctx)
        self.assertIn("Failed", ctx)
        self.assertIn("Connection timeout", ctx)

    def test_list_data_context_string(self):
        result = ToolResult(
            tool_name="search",
            success=True,
            data=[{"title": "Paper A"}, {"title": "Paper B"}],
        )
        ctx = result.to_context_string()
        self.assertIn("2 results", ctx)
        self.assertIn("Paper A", ctx)

    def test_plain_string_data(self):
        result = ToolResult(tool_name="echo", success=True, data="hello world")
        ctx = result.to_context_string()
        self.assertIn("echo", ctx)
        self.assertIn("hello world", ctx)


# =========================================================================
# Config
# =========================================================================

class TestConfig(unittest.TestCase):
    """Tests for load_config."""

    def test_default_config(self):
        config = load_config(Path("/nonexistent/path/config.yaml"))
        self.assertIsInstance(config, AgentConfig)
        self.assertEqual(config.model_name, "gpt-4o")
        self.assertEqual(config.max_paper_length, 100000)
        self.assertAlmostEqual(config.max_cost_per_paper, 5.0)

    def test_load_builtin_config(self):
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            config = load_config(config_path)
            self.assertTrue(len(config.model_name) > 0)


if __name__ == "__main__":
    unittest.main()
