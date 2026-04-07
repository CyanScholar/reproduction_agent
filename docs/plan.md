# 复现计划生成器Agent实现方案

## Context

用户需要一个"复现计划生成器Agent"，用于看完论文后自动生成复现方案，输出"先写什么、先跑什么、哪里最容易失败"。

**核心需求**：
- 输入：`paper_to_plan(paper, artifacts) -> ExperimentPlan`
- 输出：复现计划文档（模块划分、依赖列表、伪代码、风险点）、论文描述不足清单、复现脚本、验收方案
- 技术约束：独立开发、多模型支持、SKILLS机制、简洁易扩展

---

## 1. 整体架构

### 1.1 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| Agent框架 | 轻量自研 | 参考Paper2Code的轨迹累积模式，无需重型框架 |
| 模型层 | 抽象层+多后端 | 支持OpenAI/Claude/vLLM切换 |
| SKILLS | Prompt模板 + 工具链 | 简洁易扩展，遵循Claude Code Skills规范 |

### 1.2 目录结构（已实现）

```
agents/reproduction/
├── __init__.py              # 包入口，导出核心组件
├── __main__.py              # python -m reproduction 入口
├── cli.py                   # 命令行接口
├── config.yaml              # 配置文件
│
├── core/                    # 核心模块（整合所有组件）
│   ├── __init__.py          # 统一导出
│   ├── models.py            # 数据模型（PaperContext + ExperimentPlan）
│   ├── parser.py            # 论文解析器（s2orc JSON）
│   ├── model_provider.py    # OpenAI调用 + 成本追踪
│   ├── prompts.py           # 3阶段Prompt模板
│   ├── template.py          # Markdown输出模板
│   └── agent.py             # 主Agent流程（3次LLM调用）
│
├── docs/                    # 文档
│   ├── plan.md              # 本文档
│   └── demands.md           # 需求说明
│
├── examples/                # 使用示例
│   ├── README.md            # 示例说明
│   ├── contextpilot_example.py  # ContextPilot论文示例
│   └── output/              # 示例输出
│       ├── contextpilot_plan.md
│       └── contextpilot_plan.json
│
└── tests/                   # 单元测试
    ├── __init__.py
    └── test_core.py         # 核心模块测试
```

### 1.3 执行流程

```
Paper输入 → 论文解析 → Phase 1(规划+架构) → Phase 2(风险+差距) → Phase 3(验收) → 输出生成
```

**3次LLM调用**：
1. **Phase 1**: 整体规划 + 架构设计（Mermaid图）
2. **Phase 2**: 风险分析 + 论文差距检测（JSON输出）
3. **Phase 3**: 验收方案生成（自动化测试 + 人工清单）

---

## 2. 核心数据模型

### 2.1 PaperContext（论文结构化表示）

```python
@dataclass
class PaperContext:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    sections: List[Section]
    methodology: List[Section]     # 提取的方法论章节
    figures: Dict[str, str]
    tables: Dict[str, Any]
    code_url: Optional[str]
```

### 2.2 ExperimentPlan（最终输出）

```python
@dataclass
class ExperimentPlan:
    paper_id: str
    paper_title: str
    overview: str                    # 执行摘要
    modules: List[Module]            # 模块划分
    architecture_diagram: str        # Mermaid类图
    python_packages: List[str]       # 依赖
    hardware_requirements: Dict
    risks: RiskMatrix                # 风险矩阵
    gaps: GapReport                  # 论文差距
    acceptance_criteria: AcceptanceCriteria
```

### 2.3 RiskMatrix（风险矩阵）

```python
@dataclass
class Risk:
    category: RiskCategory  # IMPLEMENTATION_DIFFICULTY / PAPER_CLARITY / ENVIRONMENT_DEPENDENCY
    severity: Severity      # LOW / MEDIUM / HIGH / CRITICAL
    component: str
    description: str
    mitigation: Optional[str]

@dataclass
class RiskMatrix:
    risks: List[Risk]
    def overall_score(self) -> float  # 1-5分
```

---

## 3. 关键决策（已确认）

| 决策项 | 选择 | 说明 |
|--------|------|------|
| 存储位置 | `agents/reproduction/` | 作为Agent模块，与其他agent协同 |
| 实现策略 | MVP优先 | 先实现端到端流程，再逐步完善 |
| 测试数据 | 先复用后扩展 | 使用Paper2Code数据集 |
| 项目结构 | 轻量化 | models/prompts/output合并到core/ |

---

## 4. MVP实现范围

### MVP功能清单

| 功能 | MVP实现 | 完整版实现 |
|------|---------|------------|
| 论文解析 | 仅支持JSON格式(s2orc) | PDF/JSON/LaTeX |
| 模型调用 | 仅OpenAI | OpenAI/Claude/vLLM |
| 规划阶段 | 3次LLM调用 | 可配置多阶段 |
| 风险分析 | 结构化RiskMatrix | 独立skill模块 |
| 差距检测 | 合并到Phase 2 | 独立模块 |
| 验收方案 | 自动化测试 + 人工清单 | 完整测试生成 |
| 输出 | Markdown + JSON | 文档 + 脚本 + 测试文件 |

---

## 5. 实现进度

### 已完成 (MVP v0.1) ✅

| 模块 | 文件 | 状态 |
|------|------|------|
| 数据模型 | `core/models.py` | ✅ |
| 论文解析器 | `core/parser.py` | ✅ |
| 模型调用 | `core/model_provider.py` | ✅ |
| Prompt模板 | `core/prompts.py` | ✅ |
| 输出模板 | `core/template.py` | ✅ |
| Agent主流程 | `core/agent.py` | ✅ |
| CLI入口 | `cli.py` | ✅ |
| ContextPilot示例 | `examples/contextpilot_example.py` | ✅ |
| 单元测试 | `tests/test_core.py` | ✅ |

### 待实现

- [ ] **v0.2**: 添加Claude支持
- [ ] **v0.3**: PDF解析支持
- [ ] **v0.4**: reproduce.sh脚本生成
- [ ] **v0.5**: 自动化测试代码生成
- [ ] **v1.0**: 完整SKILLS体系

---

## 6. 使用方式

### 方式1: 命令行（完整流程）

```bash
# 设置API Key
export OPENAI_API_KEY="your-key"

# 运行完整流程
cd agents/reproduction
python3 -m reproduction paper.json -o output/

# 仅解析论文（不调用LLM）
python3 -m reproduction paper.json --parse-only

# 使用特定模型
python3 -m reproduction paper.json --model o3-mini -o output/
```

### 方式2: Python API

```python
from core import ReproductionAgent, render_plan_markdown

# 自动生成（需要API Key）
agent = ReproductionAgent(model="gpt-4o")
plan = agent.generate_plan("paper.json", output_dir="output/")

# 手动构建
from core import ExperimentPlan, RiskMatrix, Risk, RiskCategory, Severity

risks = RiskMatrix(risks=[
    Risk(
        category=RiskCategory.IMPLEMENTATION_DIFFICULTY,
        severity=Severity.MEDIUM,
        component="Core Module",
        description="Description",
    )
])

plan = ExperimentPlan(
    paper_id="paper_id",
    paper_title="Paper Title",
    overview="Overview",
    risks=risks,
)

# 渲染输出
markdown = render_plan_markdown(plan)
```

### 方式3: 运行示例

```bash
cd agents/reproduction
python3 examples/contextpilot_example.py
```

---

## 7. 测试

```bash
# 运行单元测试
cd agents/reproduction
python3 -m pytest tests/ -v

# 或手动测试
python3 -c "
from tests.test_core import *
import sys
sys.exit(0)
"
```

---

## 8. 关键参考文件

1. **Paper2Code规划阶段**: `repo/Paper2Code/codes/1_planning_llm.py`
2. **Paper2Code工具函数**: `repo/Paper2Code/codes/utils.py`
3. **s2orc JSON格式**: `repo/Paper2Code/examples/Transformer_cleaned.json`
4. **ContextPilot示例输出**: `examples/output/contextpilot_plan.md`
