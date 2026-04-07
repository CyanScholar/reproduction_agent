# Reproduction Agent - 论文复现计划生成器

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 看完论文后，直接得到「先写什么、先跑什么、哪里最容易失败」的完整复现方案。

## 快速开始

### 安装

```bash
cd agents/reproduction
pip install -r requirements.txt
```

### 基础使用

**Agent 模式 — Claude (推荐)**
```bash
export ANTHROPIC_API_KEY="your-key"
python -m reproduction paper.pdf -o output/ --api-style agent_claude
```

**Agent 模式 — Kimi**
```bash
export OPENAI_API_KEY="your-kimi-key"
python -m reproduction paper.pdf -o output/ --api-style agent_kimi --base-url https://api.moonshot.cn/v1
```

**Agent 模式 — Codex/OpenAI**
```bash
export OPENAI_API_KEY="your-key"
python -m reproduction paper.pdf -o output/ --api-style agent_codex
```

**经典模式 — OpenAI-compatible**
```bash
python -m reproduction paper.pdf -o output/ --api-style openai_compatible --model gpt-4o
```

**从 URL 解析**
```bash
python -m reproduction "https://arxiv.org/pdf/2401.xxxxx" -o output/
```

**搜索关键词自动下载**
```bash
python -m reproduction "attention is all you need" -o output/
```

**交互模式**
```bash
python -m reproduction paper.pdf -o output/ --interactive
```

## 项目结构

```
agents/reproduction/
├── README.md                 # 本文件
├── requirements.txt          # Python依赖
├── config.yaml              # 配置文件
├── __main__.py              # 入口: python -m reproduction
├── cli.py                   # 命令行接口
├── core/                    # 核心逻辑
│   ├── agent.py             # ReproductionAgent主流程
│   ├── api_provider.py      # 多Provider支持
│   ├── config.py            # 配置管理
│   ├── models.py            # 数据模型
│   ├── parser.py            # 论文解析
│   ├── task_prompt.py       # 统一任务描述
│   └── template.py          # 输出模板
├── skills/                  # 工具集
│   └── tools.py             # arXiv/Scholar/GitHub/PwC工具
├── context/                 # 专家经验上下文
│   ├── ml-paper-patterns.md
│   ├── reproduction-expert.md
│   └── risk-taxonomy.md
├── docs/                    # 文档
│   ├── plan-mvp.md          # MVP版本规划
│   ├── plan-full.md         # 完整版本规划
│   ├── dev.md               # 开发日志
│   └── demands.md           # 需求文档
├── examples/                # 使用示例
│   ├── mvp_claude_code.sh
│   ├── mvp_codex.sh
│   ├── mvp_kimi.sh
│   └── mvp_example.py
└── tests/                   # 测试
```

## 核心特性

### 1. 双模式架构

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| **Agent模式** | 单次委派给 Agent API，Agent自主完成规划+风险+验收 | Claude/Codex/Kimi |
| **经典模式** | 简化LLM调用管道 | OpenAI-compatible API |

### 2. 三层技能体系

| 层级 | 内容 | 位置 |
|------|------|------|
| **Skills** | Prompt模板/工具链 | `skills/tools.py` |
| **Context** | 专家经验封装 | `context/*.md` |
| **Scripts** | Python可执行脚本 | `core/*.py` |

### 3. 支持的模型

| Provider | API Style | Features |
|----------|-----------|----------|
| Claude | `agent_claude` | tool_use, PDF理解, ReAct循环 |
| Codex | `agent_codex` | Responses API, web_search |
| Kimi | `agent_kimi` | OpenAI-compatible, file-extract, $web_search |
| OpenAI | `openai_compatible` | Chat Completions API |
| Anthropic | `anthropic` | 原生Anthropic API |

### 4. 输出格式

运行后会在输出目录生成：

```
output/
├── reproduction_plan.md      # Markdown格式的复现计划
├── plan.json                 # 结构化数据
├── trajectories.json         # LLM调用轨迹 (可选)
└── cost_summary.json         # 成本统计 (可选)
```

**复现计划包含**：
- 📋 模块划分与实现顺序
- 🔧 依赖列表与环境配置
- 📝 伪代码与关键实现
- ⚠️ 风险点与应对策略
- 📊 验收标准与测试方案
- 🔍 论文描述不足之处清单

## 配置说明

编辑 `config.yaml`：

```yaml
# 模型设置
model:
  name: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"  # 支持环境变量
  base_url: null                 # 自定义API端点
  api_style: "openai_compatible" # 或 agent_claude / agent_kimi

# 输出设置
output:
  format: "markdown"
  include_trajectory: true       # 保存对话轨迹
  include_cost: true             # 保存成本统计

# 规划设置
planning:
  max_paper_length: 100000       # 最大论文字符数
  phases:
    - name: "planning"
      enabled: true
    - name: "risk_analysis"
      enabled: true
    - name: "acceptance"
      enabled: true

# 成本限制 (USD)
limits:
  max_cost_per_paper: 5.0
  warn_cost: 2.0
```

## Python API

```python
from core import ReproductionAgent

# 初始化Agent
agent = ReproductionAgent(api_style="agent_claude")

# 生成复现计划
markdown = agent.generate_plan(
    paper_source="path/to/paper.pdf",
    output_dir="output/",
    tier="both"  # "mvp", "full", or "both"
)

print(markdown)
```

## 版本规划

| 版本 | 状态 | 关键特性 |
|------|------|----------|
| v0.3 | ✅ | Agent-centric架构，支持Claude/Codex/Kimi |
| v0.4 | ✅ | 直接输出Markdown，移除JSON中间层 |
| v0.5 | 🚧 | 独立Risk/Gap分析模块 |
| v0.6 | 📋 | reproduce.sh自动生成 |
| v0.7 | 📋 | 自动化测试代码生成 |
| v1.0 | 📋 | 完整版 + 评测报告 |

详见 [docs/plan-mvp.md](docs/plan-mvp.md) 和 [docs/plan-full.md](docs/plan-full.md)

## 设计哲学

1. **AI-Human协同**：AI负责结构化和初步分析，人类负责审核和关键决策
2. **Skills即Prompt**：将专业知识封装为可复用的Skill模块
3. **Context即知识**：专家经验以Markdown形式沉淀，便于AI和人类共同理解
4. **面向API设计**：充分利用现代LLM的Agent能力（tool_use, file理解, ReAct）

## 参考项目

- [Paper2Code](https://github.com/paper2code/paper2code) - 论文复现工作流参考

## 许可证

MIT License - 详见 LICENSE 文件
