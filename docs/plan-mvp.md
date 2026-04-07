# 复现计划生成器 - MVP 版本规划

## 目标

快速实现端到端可用的最小复现计划生成器，验证核心流程可行性。

**核心交付物**: `paper_to_plan(paper, artifacts) -> reproduction_plan.md`

---

## 功能范围

| 功能 | MVP 实现方式 |
|------|-------------|
| 论文解析 | PDF / URL / 文本文件 |
| API调用 | 多Provider支持 (Agent模式 + Classic模式) |
| 规划阶段 | Agent单次调用完成规划+风险+验收 |
| 风险分析 | 结构化 RiskMatrix，嵌入Agent输出 |
| 差距检测 | 嵌入Agent输出 |
| 验收方案 | 测试清单 + 验证指标 |
| 输出格式 | Markdown（直接输出，无JSON中间层） |
| Skills | Python内嵌 + Context文件 |

---

## 架构

### 双模式架构

```
Paper Input
    → ReproductionAgent
        → [Agent Mode] build_task_prompt() → AgentProvider.run_task() → Markdown
        → [Classic Mode] parse → LLM call → Markdown
    → 输出生成(template.py) → Markdown + metadata
```

**Agent模式** (推荐):
- Claude: `agent_claude` - Anthropic原生tool_use
- Codex: `agent_codex` - OpenAI Responses API
- Kimi: `agent_kimi` - OpenAI-compatible + file-extract

**经典模式**:
- OpenAI-compatible: GPT-4o等
- Anthropic: Claude原生API

### 执行流程 (Agent模式)

```
1. 读取论文内容
   └─→ 支持PDF/URL/搜索关键词

2. 构建任务描述
   └─→ build_task_prompt(paper_content, title, authors, tier)
       └─→ 整合paper + context/专家经验 + 输出格式要求

3. 委派给Agent
   └─→ AgentProvider.run_task(task, system_prompt, attachments)
       └─→ Agent自主完成：规划 + 风险 + 差距 + 验收

4. 输出结果
   └─→ Markdown格式复现计划
       └─→ _save_outputs() → plan.md + trajectories.json + cost_summary.json
```

---

## 数据模型

### 核心类

```python
@dataclass
class PaperContext:
    """论文统一结构化表示"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str = ""
    sections: List[Section] = field(default_factory=list)
    methodology: List[Section] = field(default_factory=list)
    hyperparameters: List[Hyperparameter] = field(default_factory=list)
    code_url: Optional[str] = None

class ReproductionAgent:
    """复现计划生成Agent"""
    def generate_plan(
        self,
        paper_source: str,          # 论文来源
        output_dir: Optional[str],  # 输出目录
        tier: str = "both",         # "mvp" / "full" / "both"
    ) -> str:  # 返回Markdown文本
```

### 输出结构 (Markdown)

```markdown
# 复现计划: {论文标题}

## 1. 概述

## 2. 模块划分

### 2.1 {模块名}
- 实现顺序: {order}
- 依赖: {dependencies}
- 伪代码: {pseudo_code}

## 3. 依赖列表

## 4. 硬件要求

## 5. 风险评估

## 6. 差距分析

## 7. 验收标准
```

---

## 目录结构

```
agents/reproduction/
├── __init__.py
├── __main__.py              # 入口
├── cli.py                   # 命令行接口
├── config.yaml              # 配置文件
├── core/                    # 核心逻辑
│   ├── __init__.py
│   ├── agent.py             # ReproductionAgent主流程
│   ├── api_provider.py      # Provider体系
│   ├── config.py            # 配置管理
│   ├── models.py            # 数据模型
│   ├── parser.py            # 论文解析
│   ├── task_prompt.py       # 统一任务描述
│   └── template.py          # 输出模板
├── skills/                  # 工具集
│   ├── __init__.py
│   └── tools.py             # arXiv/Scholar/GitHub/PwC
├── context/                 # 专家经验
│   ├── ml-paper-patterns.md
│   ├── reproduction-expert.md
│   └── risk-taxonomy.md
├── docs/                    # 文档
├── examples/                # 使用示例
└── tests/                   # 测试
```

---

## 使用方式

### CLI

```bash
# Agent 模式 — Claude
export ANTHROPIC_API_KEY="your-key"
python3 -m reproduction paper.pdf --api-style agent_claude -o output/

# Agent 模式 — Kimi
export OPENAI_API_KEY="your-kimi-key"
python3 -m reproduction paper.pdf --api-style agent_kimi --base-url https://api.moonshot.cn/v1 -o output/

# Agent 模式 — Codex
export OPENAI_API_KEY="your-key"
python3 -m reproduction paper.pdf --api-style agent_codex -o output/

# 经典模式
export OPENAI_API_KEY="your-key"
python3 -m reproduction paper.pdf --api-style openai_compatible -o output/

# 从URL
python3 -m reproduction "https://arxiv.org/pdf/2401.xxxxx" -o output/

# 搜索关键词
python3 -m reproduction "attention is all you need" -o output/

# 交互模式
python3 -m reproduction paper.pdf -o output/ --interactive
```

### Python API

```python
from core import ReproductionAgent

# Agent模式
agent = ReproductionAgent(api_style="agent_claude")
plan = agent.generate_plan("xxxx.pdf", output_dir="output/", tier="both")

# Classic模式
agent = ReproductionAgent(
    api_style="openai_compatible",
    model="gpt-4o",
    api_key="your-key"
)
plan = agent.generate_plan("xxxx.pdf", output_dir="output/")
```

---

## 配置

```yaml
# config.yaml
model:
  name: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"
  base_url: null
  api_style: "openai_compatible"  # agent_claude / agent_kimi / agent_codex

output:
  format: "markdown"
  include_trajectory: true
  include_cost: true

planning:
  max_paper_length: 100000

limits:
  max_cost_per_paper: 5.0
```

---

## Agent能力矩阵

| Provider | API Style | tool_use | web_search | PDF理解 | 备注 |
|----------|-----------|----------|------------|---------|------|
| Claude | agent_claude | ✅ Anthropic native | 通过工具 | base64 document | 推荐 |
| Codex | agent_codex | ✅ Responses API | ✅ 内置 | 待实现 | |
| Kimi | agent_kimi | ✅ OpenAI-compat | ✅ $web_search | ✅ file-extract | |
| GPT-4o | openai_compatible | ❌ | ❌ | base64 | Classic模式 |

---

## 完成标准

- [x] 至少在一个真实论文上跑通完整流程
- [x] 输出包含：模块划分、依赖列表、伪代码、风险点、论文描述不足清单
- [x] CLI 和 Python API 均可用
- [x] 支持至少3种Provider (Claude/Codex/Kimi)
- [x] 支持PDF/URL/搜索关键词三种输入
- [ ] 有基本的单元测试覆盖

---

## 实现进度

| 模块 | 状态 |
|------|------|
| 数据模型 (models.py) | ✅ |
| API Provider体系 (api_provider.py) | ✅ |
| 统一任务描述 (task_prompt.py) | ✅ |
| 输出模板 (template.py) | ✅ |
| Agent主流程 (agent.py) | ✅ |
| CLI入口 (cli.py) | ✅ |
| 工具集 (skills/tools.py) | ✅ |
| 配置模块 (config.py) | ✅ |
| Context专家经验 | ✅ |
| Agent模式端到端测试 | 🔲 |
| Classic模式端到端测试 | 🔲 |

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | - | 初始概念验证 |
| v0.2 | - | 添加多Provider支持 |
| v0.3 | - | Agent-centric架构 |
| v0.4 | - | 直接输出Markdown，移除JSON中间层 |
