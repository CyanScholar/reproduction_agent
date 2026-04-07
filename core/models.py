"""
Core data models - 核心数据模型
整合了Paper和Plan相关的数据结构
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json


# =============================================================================
# Paper Models - 论文相关
# =============================================================================

class PaperFormat(Enum):
    """论文格式"""
    JSON = "json"  # s2orc-doc2json format
    PDF = "pdf"
    LATEX = "latex"


@dataclass
class Section:
    """论文章节"""
    title: str
    content: str
    section_number: Optional[str] = None
    subsections: List['Section'] = field(default_factory=list)


@dataclass
class Algorithm:
    """论文中的算法描述"""
    name: str
    description: str
    pseudo_code: Optional[str] = None
    equations: List[str] = field(default_factory=list)


@dataclass
class Hyperparameter:
    """超参数"""
    name: str
    value: Any
    source: str = "explicit"  # "explicit", "inferred", "default", "unclear"
    confidence: float = 1.0  # 0.0 - 1.0
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Hyperparameter':
        return cls(
            name=d.get("name", ""),
            value=d.get("value"),
            source=d.get("source", "explicit"),
            confidence=d.get("confidence", 1.0),
            notes=d.get("notes"),
        )


@dataclass
class Experiment:
    """实验配置"""
    name: str
    dataset: str
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hardware_specs: Optional[Dict[str, Any]] = None


@dataclass
class PaperContext:
    """论文统一结构化表示"""
    # 元信息
    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    venue: Optional[str] = None
    year: Optional[int] = None

    # 内容
    abstract: str = ""
    sections: List[Section] = field(default_factory=list)

    # 提取的结构化信息
    methodology: List[Section] = field(default_factory=list)
    algorithms: List[Algorithm] = field(default_factory=list)
    hyperparameters: List[Hyperparameter] = field(default_factory=list)
    experiments: List[Experiment] = field(default_factory=list)

    # 图表
    figures: Dict[str, str] = field(default_factory=dict)
    tables: Dict[str, Any] = field(default_factory=dict)

    # 链接
    code_url: Optional[str] = None
    dataset_urls: Dict[str, str] = field(default_factory=dict)

    # 原始内容
    raw_content: Any = None
    format: PaperFormat = PaperFormat.JSON

    def get_full_text(self) -> str:
        """获取论文全文"""
        parts = [f"# {self.title}\n"]
        parts.append(f"\n## Abstract\n{self.abstract}\n")

        for section in self.sections:
            parts.append(f"\n## {section.title}\n{section.content}\n")
            for sub in section.subsections:
                parts.append(f"\n### {sub.title}\n{sub.content}\n")

        return "\n".join(parts)

    def get_methodology_text(self) -> str:
        """获取方法论部分文本"""
        parts = []
        for section in self.methodology:
            parts.append(f"## {section.title}\n{section.content}\n")
            for sub in section.subsections:
                parts.append(f"### {sub.title}\n{sub.content}\n")
        return "\n".join(parts)


# =============================================================================
# Plan Models - 复现计划相关
# =============================================================================

class RiskCategory(Enum):
    """风险类别"""
    IMPLEMENTATION_DIFFICULTY = "implementation_difficulty"
    PAPER_CLARITY = "paper_clarity"
    ENVIRONMENT_DEPENDENCY = "environment_dependency"
    DATA_AVAILABILITY = "data_availability"
    COMPUTATIONAL_RESOURCE = "computational_resource"


class Severity(Enum):
    """严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Risk:
    """风险项"""
    category: RiskCategory
    severity: Severity
    component: str
    description: str
    paper_reference: Optional[str] = None
    mitigation: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "component": self.component,
            "description": self.description,
            "paper_reference": self.paper_reference,
            "mitigation": self.mitigation,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Risk':
        try:
            category = RiskCategory(d.get("category", "implementation_difficulty"))
        except ValueError:
            category = RiskCategory.IMPLEMENTATION_DIFFICULTY
        try:
            severity = Severity(d.get("severity", "medium"))
        except ValueError:
            severity = Severity.MEDIUM
        return cls(
            category=category,
            severity=severity,
            component=d.get("component", ""),
            description=d.get("description", ""),
            paper_reference=d.get("paper_reference"),
            mitigation=d.get("mitigation"),
            confidence=d.get("confidence", 1.0),
        )


@dataclass
class RiskMatrix:
    """风险矩阵"""
    risks: List[Risk] = field(default_factory=list)

    def get_by_severity(self, severity: Severity) -> List[Risk]:
        return [r for r in self.risks if r.severity == severity]

    def get_by_category(self, category: RiskCategory) -> List[Risk]:
        return [r for r in self.risks if r.category == category]

    def overall_score(self) -> float:
        """计算整体复现难度评分 (1-5)"""
        severity_weights = {
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4,
        }
        if not self.risks:
            return 1.0
        weighted_sum = sum(severity_weights[r.severity] for r in self.risks)
        return min(5.0, weighted_sum / len(self.risks) + 1)

    def to_dict(self) -> Dict:
        return {
            "risks": [r.to_dict() for r in self.risks],
            "overall_score": self.overall_score(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'RiskMatrix':
        risks = [Risk.from_dict(r) for r in d.get("risks", [])]
        return cls(risks=risks)


@dataclass
class Gap:
    """论文描述缺失项"""
    type: str  # "missing", "ambiguous", "contradictory", "omitted"
    section: str
    description: str
    impact: str  # "critical", "important", "minor"
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "section": self.section,
            "description": self.description,
            "impact": self.impact,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Gap':
        return cls(
            type=d.get("type", "missing"),
            section=d.get("section", ""),
            description=d.get("description", ""),
            impact=d.get("impact", "important"),
            suggestion=d.get("suggestion"),
        )


@dataclass
class GapReport:
    """论文缺失报告"""
    gaps: List[Gap] = field(default_factory=list)

    def critical_gaps(self) -> List[Gap]:
        return [g for g in self.gaps if g.impact == "critical"]

    def to_dict(self) -> Dict:
        return {
            "gaps": [g.to_dict() for g in self.gaps],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'GapReport':
        gaps = [Gap.from_dict(g) for g in d.get("gaps", [])]
        return cls(gaps=gaps)


@dataclass
class AcceptanceTest:
    """验收测试项"""
    name: str
    test_type: str  # "automated", "manual", "benchmark"
    description: str
    procedure: str
    expected_outcome: str
    tolerance: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "test_type": self.test_type,
            "description": self.description,
            "procedure": self.procedure,
            "expected_outcome": self.expected_outcome,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'AcceptanceTest':
        return cls(
            name=d.get("name", ""),
            test_type=d.get("test_type", "automated"),
            description=d.get("description", ""),
            procedure=d.get("procedure", ""),
            expected_outcome=d.get("expected_outcome", ""),
            tolerance=d.get("tolerance"),
        )


@dataclass
class AcceptanceCriteria:
    """验收标准"""
    automated_tests: List[AcceptanceTest] = field(default_factory=list)
    manual_checks: List[AcceptanceTest] = field(default_factory=list)
    benchmark_targets: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "automated_tests": [t.to_dict() for t in self.automated_tests],
            "manual_checks": [c.to_dict() for c in self.manual_checks],
            "benchmark_targets": self.benchmark_targets,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'AcceptanceCriteria':
        return cls(
            automated_tests=[AcceptanceTest.from_dict(t) for t in d.get("automated_tests", [])],
            manual_checks=[AcceptanceTest.from_dict(c) for c in d.get("manual_checks", [])],
            benchmark_targets=d.get("benchmark_targets", {}),
        )


@dataclass
class Module:
    """模块定义"""
    name: str
    file_path: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    implementation_notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "description": self.description,
            "dependencies": self.dependencies,
            "interfaces": self.interfaces,
            "implementation_notes": self.implementation_notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Module':
        return cls(
            name=d.get("name", ""),
            file_path=d.get("file_path", ""),
            description=d.get("description", ""),
            dependencies=d.get("dependencies", []),
            interfaces=d.get("interfaces", []),
            implementation_notes=d.get("implementation_notes", ""),
        )


@dataclass
class ExperimentPlan:
    """完整的复现计划"""
    # 元信息
    paper_id: str
    paper_title: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generator_version: str = "0.2.0"

    # 计划层级
    tier: str = "full"  # "mvp" or "full"

    # 核心计划
    overview: str = ""
    modules: List[Module] = field(default_factory=list)
    file_list: List[str] = field(default_factory=list)
    architecture_diagram: str = ""  # Mermaid classDiagram
    call_flow: str = ""  # Mermaid sequenceDiagram

    # 依赖
    python_packages: List[str] = field(default_factory=list)
    other_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)

    # 配置
    config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: List[Hyperparameter] = field(default_factory=list)

    # 风险与差距
    risks: Optional[RiskMatrix] = None
    gaps: Optional[GapReport] = None

    # 验收
    acceptance_criteria: Optional[AcceptanceCriteria] = None

    # 可选
    reproduce_script: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "generated_at": self.generated_at,
            "generator_version": self.generator_version,
            "tier": self.tier,
            "overview": self.overview,
            "modules": [m.to_dict() for m in self.modules],
            "file_list": self.file_list,
            "architecture_diagram": self.architecture_diagram,
            "call_flow": self.call_flow,
            "python_packages": self.python_packages,
            "other_dependencies": self.other_dependencies,
            "hardware_requirements": self.hardware_requirements,
            "config": self.config,
            "hyperparameters": [h.to_dict() for h in self.hyperparameters],
            "risks": self.risks.to_dict() if self.risks else None,
            "gaps": self.gaps.to_dict() if self.gaps else None,
            "acceptance_criteria": self.acceptance_criteria.to_dict() if self.acceptance_criteria else None,
            "reproduce_script": self.reproduce_script,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentPlan':
        risks = RiskMatrix.from_dict(d["risks"]) if d.get("risks") else None
        gaps = GapReport.from_dict(d["gaps"]) if d.get("gaps") else None
        acceptance = AcceptanceCriteria.from_dict(d["acceptance_criteria"]) if d.get("acceptance_criteria") else None
        return cls(
            paper_id=d.get("paper_id", ""),
            paper_title=d.get("paper_title", ""),
            generated_at=d.get("generated_at", datetime.now().isoformat()),
            generator_version=d.get("generator_version", "0.2.0"),
            tier=d.get("tier", "full"),
            overview=d.get("overview", ""),
            modules=[Module.from_dict(m) for m in d.get("modules", [])],
            file_list=d.get("file_list", []),
            architecture_diagram=d.get("architecture_diagram", ""),
            call_flow=d.get("call_flow", ""),
            python_packages=d.get("python_packages", []),
            other_dependencies=d.get("other_dependencies", {}),
            hardware_requirements=d.get("hardware_requirements", {}),
            config=d.get("config", {}),
            hyperparameters=[Hyperparameter.from_dict(h) for h in d.get("hyperparameters", [])],
            risks=risks,
            gaps=gaps,
            acceptance_criteria=acceptance,
            reproduce_script=d.get("reproduce_script"),
        )


# =============================================================================
# Tiered Plan - 两层计划
# =============================================================================

class PlanTier(Enum):
    """计划层级"""
    MVP = "mvp"    # 最小可运行版
    FULL = "full"  # 完整复现版


class FeedbackMode(Enum):
    """人工反馈模式"""
    NONE = "none"              # 直通运行（默认）
    INTERACTIVE = "interactive" # 在 Phase 间暂停等待终端输入
    CALLBACK = "callback"      # 调用回调函数


@dataclass
class TieredExperimentPlan:
    """两层复现计划：MVP + Full"""
    # 元信息
    paper_id: str
    paper_title: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generator_version: str = "0.2.0"

    # 两层计划
    mvp_plan: Optional[ExperimentPlan] = None
    full_plan: Optional[ExperimentPlan] = None

    # 切分说明
    tier_rationale: str = ""
    upgrade_path: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "generated_at": self.generated_at,
            "generator_version": self.generator_version,
            "tier_rationale": self.tier_rationale,
            "upgrade_path": self.upgrade_path,
            "mvp_plan": self.mvp_plan.to_dict() if self.mvp_plan else None,
            "full_plan": self.full_plan.to_dict() if self.full_plan else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TieredExperimentPlan':
        mvp = ExperimentPlan.from_dict(d["mvp_plan"]) if d.get("mvp_plan") else None
        full = ExperimentPlan.from_dict(d["full_plan"]) if d.get("full_plan") else None
        return cls(
            paper_id=d.get("paper_id", ""),
            paper_title=d.get("paper_title", ""),
            generated_at=d.get("generated_at", datetime.now().isoformat()),
            generator_version=d.get("generator_version", "0.2.0"),
            mvp_plan=mvp,
            full_plan=full,
            tier_rationale=d.get("tier_rationale", ""),
            upgrade_path=d.get("upgrade_path", []),
        )
