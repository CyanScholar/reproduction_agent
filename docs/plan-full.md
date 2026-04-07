# 复现计划生成器 - 完整版本规划

## 目标

在 MVP 基础上，扩展为完整的论文复现规划系统，支持多输入格式、多模型后端、完整 Skills 体系和自动化脚本生成。

---

## 功能范围

| 功能 | 完整版实现 | 对比 MVP |
|------|-----------|---------|
| 论文解析 | PDF / URL / LaTeX | MVP 仅 PDF |
| API调用 | OpenAI / FastAPI /.. | MVP 仅 OpenAI |
| 规划阶段 | 可配置多阶段 pipeline | MVP 固定 3 阶段 |
| 风险分析 | 独立 skill 模块 | MVP 合并到 Phase 2 |
| 差距检测 | 独立模块，支持交叉验证 | MVP 合并到 Phase 2 |
| 验收方案 | 完整测试代码生成 | MVP 仅清单 |
| 输出格式 | 文档 + 脚本 + 测试文件 | MVP 仅 Markdown + JSON |
| Skills 体系 | 完整 Skills 机制 (loader + context 注入) | MVP 内嵌 Prompt |
| 脚本生成 | reproduce.sh 自动生成 | MVP 无 |
| 人工反馈 | 模块划分编辑、scope 删减、目标锁定 | MVP 无 |

---

## 架构升级

### 多格式输入

```
Paper Input (PDF/URL/LaTeX)
    → Agent → 对应 Parser
    → PaperContext (统一结构)
```


### 多模型后端

```python
ModelProvider (经典)
    - OpenAICompatibleProvider
    - AnthropicProvider
AgentProvider (Agent-centric)
    - ClaudeAgentProvider  # Anthropic tool_use
    - CodexAgentProvider   # OpenAI Responses API
    - KimiAgentProvider    # OpenAI-compatible + file understanding
```

通过 `config.yaml` 配置切换：
```yaml
model:
  provider: agent_claude  # agent_claude / agent_codex / agent_kimi / openai / anthropic
  name: claude-sonnet-4-6
  temperature: 0.2
```

### Skills 体系

Skills 已重构为 Agent-centric 架构：
- 原有 Markdown skill 文件（planning / risk-analysis / acceptance / gap-detection）的知识已编译进 `core/task_prompt.py`，由 `build_task_prompt()` 统一生成任务描述
- 工具函数合并到 `skills/tools.py`，包含 arXiv / Scholar / GitHub / PwC 等外部查询能力
- Agent 在单次调用中自主完成规划、风险评估、验收方案生成

```
core/
├── task_prompt.py       # 编译后的 skill 知识 → 统一任务 prompt
skills/
└── tools.py             # 工具集（arXiv/Scholar/GitHub/PwC）
```

### 自动化输出

```
output/
├── plan.md              # 复现计划文档
├── plan.json            # 结构化数据
├── reproduce.sh         # 一键复现脚本
├── tests/
│   ├── test_smoke.py    # 冒烟测试
│   ├── test_unit.py     # 单元测试
│   └── test_benchmark.py # 基准测试
└── configs/
    └── hyperparams.yaml # 超参数配置
```

---

## 新增模块

### 1. PDF 解析器 (`core/pdf_parser.py`)
- PyMuPDF 提取文本和图表
- GROBID 学术结构解析（可选）
- 图表 OCR 提取关键数值

### 2. LaTeX 解析器 (`core/latex_parser.py`)
- section/subsection 结构提取
- 公式提取与标注
- figure/table 引用解析

### 3. 差距检测 skill (`skills/gap-detection.md`)
- 独立于风险分析
- 交叉引用论文各节寻找矛盾
- 与同领域常见做法对比

### 4. 脚本生成 skill (`skills/script-generation.md`)
- 生成 `reproduce.sh` (环境搭建 + 训练 + 评估)
- 生成测试框架代码
- 生成超参数配置文件

### 5. 人工反馈接口 (`core/feedback.py`)
- 模块划分可编辑
- Scope 删减
- 第一轮目标锁定
- 支持 CLI 交互和 API 回调

---

## 执行流程 (完整版)

### Agent-centric 单次委托流程

```
Paper Input
    → ReproductionAgent.generate_plan()
    → [Agent Mode]
        build_task_prompt(paper_text)   # 构建统一任务描述
        → AgentProvider.run_task()      # 单次 Agent 调用
        → Agent 自主完成：规划 + 风险 + 差距 + 验收
        → 解析返回 → ExperimentPlan
    → [Classic Mode]
        LLM call (JSON 输出)
        → ExperimentPlan.from_dict()
    → [人工反馈点: 模块划分/scope 调整]（可选）
    → 输出生成(template.py) → Markdown + JSON
```

---

## Agent 能力矩阵

| Provider | tool_use | web_search | PDF理解 | ReAct循环 |
|----------|----------|------------|---------|----------|
| ClaudeAgent | ✅ Anthropic native | 通过工具 | base64 document | ✅ |
| CodexAgent | ✅ Responses API | ✅ 内置 | 待实现 | ✅ |
| KimiAgent | ✅ OpenAI-compat | ✅ $web_search | ✅ file-extract | ✅ |

---

## 评测方案

| 评测维度 | 方法 |
|---------|------|
| 计划可执行性 | 人工打分 (1-5)，至少 3 位评估者 |
| Baseline 对比 | 与 Paper2Code 生成结果对比 |
| 标准 benchmark | SciReplicate-Bench / ResearchCodeBench |
| 真实场景验证 | 至少在 1 个真实科研任务中跑通并被同学使用 |
| 失败模式分析 | 记录不可靠场景，说明何时需人工接管 |

---

## 版本路线图

| 版本 | 里程碑 | 关键特性 |
|------|--------|---------|
| v0.2 | 多模型 | 添加 Claude 支持 |
| v0.3 | 多格式 | PDF 解析支持 |
| v0.4 | 脚本生成 | reproduce.sh 自动生成 |
| v0.5 | 测试生成 | 自动化测试代码生成 |
| v0.6 | 差距检测 | 独立 gap-detection skill |
| v0.7 | 人工反馈 | 交互式模块编辑 |
| v1.0 | 完整版 | 全部特性 + 评测报告 |

---

## 完成标准

- [ ] 支持 PDF / JSON / LaTeX 三种输入格式
- [ ] 支持 OpenAI / Claude / vLLM 三种模型后端
- [ ] Skills 体系完整运作（自动发现、context 注入、模板填充）
- [ ] 自动生成 reproduce.sh 和测试代码
- [ ] 有 baseline 对比，不接受仅主观 demo
- [ ] 至少在一个真实科研任务中被同学实际使用
- [ ] 产出失败模式分析，说明不可靠场景和人工接管时机
