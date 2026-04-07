# 复现计划生成器 - MVP 版本规划

## 目标

快速实现端到端可用的最小复现计划生成器，验证核心流程可行性。

**核心交付物**: `paper_to_plan(paper, artifacts) -> ExperimentPlan`

---

## 功能范围

| 功能 | MVP 实现方式 |
|------|-------------|
| 论文解析 | 调用API |
| API调用 | 混合策略：Claude用Anthropic tool_use，Codex用OpenAI Responses API，Kimi用OpenAI-compatible |
| 规划阶段 | 3 次 LLM 调用 (规划 → 风险 → 验收) |
| 风险分析 | 结构化 RiskMatrix，合并到 Phase 2 |
| 差距检测 | 合并到 Phase 2 输出 |
| 验收方案 | 自动化测试 + 人工清单 |
| 输出格式 | Markdown + JSON |
| Skills | Python 内嵌 Prompt 模板 (prompts.py) |

---

## 架构

```
Paper → ReproductionAgent
    → [Agent Mode] build_task_prompt() → AgentProvider.run_task() → ExperimentPlan
    → [Classic Mode] parse → LLM call (JSON) → ExperimentPlan
    → 输出生成(template.py) → Markdown + JSON
```

### 执行流程

- **Agent模式**：单次Agent调用（包含规划+风险+验收）
- **经典模式**：简化的LLM调用（JSON输出，无regex解析）

---

## 数据模型

```python
@dataclass
class PaperContext:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    sections: List[Section]
    methodology: List[Section]
    figures: Dict[str, str]
    tables: Dict[str, Any]
    code_url: Optional[str]

@dataclass
class ExperimentPlan:
    paper_id: str
    paper_title: str
    overview: str
    modules: List[Module]
    architecture_diagram: str        # Mermaid
    python_packages: List[str]
    hardware_requirements: Dict
    risks: RiskMatrix
    gaps: GapReport
    acceptance_criteria: AcceptanceCriteria
```

---

## 目录结构

```
agents/reproduction/
├── __init__.py
├── __main__.py
├── cli.py
├── config.yaml
├── core/
│   ├── models.py            # 数据模型 + from_dict()
│   ├── api_provider.py      # ModelProvider + AgentProvider 体系
│   ├── task_prompt.py        # 统一任务描述
│   ├── template.py          # Markdown 输出
│   ├── agent.py             # 薄编排层
│   └── config.py            # 配置
├── skills/
│   └── tools.py             # 工具集（arXiv/Scholar/GitHub/PwC）
├── docs/
├── examples/
└── tests/
```

---

## 使用方式

```bash
# Agent 模式 — Claude
export ANTHROPIC_API_KEY="your-key"
python3 -m reproduction paper.pdf --api-style agent_claude -o output/

# Agent 模式 — Kimi
export OPENAI_API_KEY="your-kimi-key"
python3 -m reproduction paper.pdf --api-style agent_kimi --base-url https://api.moonshot.cn/v1 -o output/

# 经典模式
export OPENAI_API_KEY="your-key"
cd agents/reproduction
python3 -m reproduction paper.pdf -o output/

# 仅解析
python3 -m reproduction paper.pdf --parse-only

# Python API
from core import ReproductionAgent
agent = ReproductionAgent(api_style="agent_claude")
plan = agent.generate_plan("xxxx.pdf", output_dir="output/")
```

---

## 完成标准

- [ ] 至少在一个真实论文上跑通完整流程
- [ ] 输出包含：模块划分、依赖列表、伪代码、风险点、论文描述不足清单
- [ ] CLI 和 Python API 均可用
- [ ] 有基本的单元测试覆盖

---

## 实现进度

| 模块 | 状态 |
|------|------|
| 数据模型 + from_dict() (models.py) | ✅ |
| API Provider 体系 (api_provider.py) | ✅ |
| 统一任务描述 (task_prompt.py) | ✅ |
| 输出模板 (template.py) | ✅ |
| 薄编排层 (agent.py) | ✅ |
| CLI 入口 (cli.py) | ✅ |
| 工具集 (skills/tools.py) | ✅ |
| 配置模块 (config.py) | ✅ |
| Agent 模式端到端测试 | 🔲 |
| 经典模式端到端测试 | 🔲 |
| mvp 示例更新 | 🔲 |
