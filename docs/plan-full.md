# 复现计划生成器 - 完整版本规划

## 目标

在 MVP 基础上，扩展为完整的论文复现规划系统，支持多输入格式、多模型后端、完整 Skills 体系和自动化脚本生成。

---

## 当前状态 (v0.4)

| 功能 | 当前实现 | 完整版目标 |
|------|----------|-----------|
| 论文解析 | PDF / URL / 搜索关键词 | PDF / URL / LaTeX / arXiv |
| API调用 | Claude / Codex / Kimi / OpenAI | + vLLM / Ollama |
| 规划阶段 | Agent单次调用 (规划+风险+验收) | 可配置多阶段pipeline |
| 风险分析 | Agent输出结构化风险 | 独立Risk模块 + 交叉验证 |
| 差距检测 | Agent输出 | 独立Gap Detection模块 |
| 验收方案 | 测试清单 | 完整测试代码生成 |
| 输出格式 | Markdown + metadata | Markdown + JSON + 脚本 + 配置 |
| Skills体系 | Python内嵌 + Context | 完整Skill机制 (loader + context注入) |
| 脚本生成 | ❌ | reproduce.sh自动生成 |
| 人工反馈 | ❌ | 交互式模块编辑 |

---

## 架构

### 双模式架构 (已稳定)

```
Paper Input
    → ReproductionAgent
        → [Agent Mode] build_task_prompt() → AgentProvider.run_task() → Markdown
        → [Classic Mode] parse → LLM call → Markdown
    → 人工反馈 (可选)
    → 输出生成 → Markdown + metadata + scripts
```

### 多层技能体系

```
Skills (工具链)
    └── tools.py: arXiv / Scholar / GitHub / PwC

Context (专家经验)
    ├── ml-paper-patterns.md
    ├── reproduction-expert.md
    └── risk-taxonomy.md

Scripts (可执行工具)
    └── core/*.py: 核心逻辑
```

---

## 多模型后端

```python
ModelProvider (经典)
    - OpenAICompatibleProvider    # OpenAI / Kimi / DeepSeek / vLLM
    - AnthropicProvider           # 原生Anthropic API

AgentProvider (Agent-centric)
    - ClaudeAgentProvider         # Anthropic tool_use
    - CodexAgentProvider          # OpenAI Responses API
    - KimiAgentProvider           # OpenAI-compatible + file understanding
```

配置切换：
```yaml
model:
  provider: agent_claude  # agent_claude / agent_codex / agent_kimi / openai / anthropic
  name: claude-sonnet-4-6
  temperature: 0.2
```

---

## Agent能力矩阵

| Provider | API Style | tool_use | web_search | PDF理解 | 状态 |
|----------|-----------|----------|------------|---------|------|
| Claude | agent_claude | ✅ Anthropic native | 通过工具 | base64 document | ✅ |
| Codex | agent_codex | ✅ Responses API | ✅ 内置 | 待实现 | ✅ |
| Kimi | agent_kimi | ✅ OpenAI-compat | ✅ $web_search | ✅ file-extract | ✅ |
| vLLM | openai_compatible | ❌ | ❌ | ❌ | 📋 |
| Ollama | openai_compatible | ❌ | ❌ | ❌ | 📋 |

---

## 版本路线图

| 版本 | 里程碑 | 关键特性 | 状态 |
|------|--------|----------|------|
| v0.3 | Agent架构 | 支持Claude/Codex/Kimi Agent模式 | ✅ |
| v0.4 | 直接输出 | 移除JSON中间层，直接输出Markdown | ✅ |
| v0.5 | 独立模块 | Risk/Gap分析模块化 | 🚧 |
| v0.6 | 脚本生成 | reproduce.sh自动生成 | 📋 |
| v0.7 | 测试生成 | 自动化测试代码生成 | 📋 |
| v0.8 | LaTeX支持 | LaTeX源文件解析 | 📋 |
| v0.9 | 人工反馈 | 交互式模块编辑 | 📋 |
| v1.0 | 完整版 | 全部特性 + 评测报告 | 📋 |

---

## 新增模块规划

### v0.5 - 独立分析模块

```python
# core/risk_analyzer.py
class RiskAnalyzer:
    """独立风险分析模块"""
    def analyze(self, paper: PaperContext, plan: str) -> RiskMatrix

# core/gap_detector.py
class GapDetector:
    """独立差距检测模块"""
    def detect(self, paper: PaperContext) -> GapReport
    def cross_validate(self, sections: List[Section]) -> List[Inconsistency]
```

### v0.6 - 脚本生成

```python
# core/script_generator.py
class ScriptGenerator:
    """自动化脚本生成"""
    def generate_reproduce_sh(self, plan: ExperimentPlan) -> str
    def generate_config_yaml(self, plan: ExperimentPlan) -> str
```

输出结构：
```
output/
├── plan.md                   # 复现计划文档
├── plan.json                 # 结构化数据
├── reproduce.sh              # 一键复现脚本
├── configs/
│   └── hyperparams.yaml      # 超参数配置
├── tests/
│   ├── test_smoke.py         # 冒烟测试
│   ├── test_unit.py          # 单元测试
│   └── test_benchmark.py     # 基准测试
├── trajectories.json         # LLM调用轨迹
└── cost_summary.json         # 成本统计
```

### v0.7 - 测试代码生成

```python
# core/test_generator.py
class TestGenerator:
    """自动化测试代码生成"""
    def generate_smoke_tests(self, plan: ExperimentPlan) -> str
    def generate_unit_tests(self, plan: ExperimentPlan) -> str
    def generate_benchmark_tests(self, plan: ExperimentPlan) -> str
```

### v0.8 - LaTeX解析

```python
# core/latex_parser.py
class LaTeXParser:
    """LaTeX源文件解析"""
    def parse(self, latex_path: Path) -> PaperContext
    def extract_equations(self) -> List[Equation]
    def extract_figures(self) -> List[Figure]
    def extract_tables(self) -> List[Table]
```

### v0.9 - 人工反馈接口

```python
# core/feedback.py
class FeedbackInterface:
    """人工反馈接口"""
    def edit_modules(self, plan: ExperimentPlan) -> ExperimentPlan
    def adjust_scope(self, plan: ExperimentPlan, scope: str) -> ExperimentPlan
    def lock_target(self, plan: ExperimentPlan, target: str) -> ExperimentPlan
```

支持：
- CLI交互模式
- API回调
- Web界面 (可选)

---

## 完整执行流程

```
Paper Input (PDF/URL/LaTeX/arXiv)
    ↓
PaperParser
    ↓
PaperContext (统一结构)
    ↓
ReproductionAgent.generate_plan()
    ↓
┌─────────────────────────────────────────────┐
│ Agent Mode                                  │
│   build_task_prompt(paper, context, tier)   │
│   → AgentProvider.run_task()                │
│   → Agent自主完成：规划+风险+差距+验收       │
└─────────────────────────────────────────────┘
    ↓
ExperimentPlan (中间表示)
    ↓
[可选] 人工反馈
    - 模块划分编辑
    - Scope删减
    - 目标锁定
    ↓
ScriptGenerator
    - reproduce.sh
    - configs/hyperparams.yaml
    ↓
TestGenerator
    - tests/test_*.py
    ↓
TemplateRenderer
    ↓
Output/
    ├── plan.md
    ├── plan.json
    ├── reproduce.sh
    ├── configs/
    ├── tests/
    ├── trajectories.json
    └── cost_summary.json
```

---

## 评测方案

| 评测维度 | 方法 | 目标 |
|---------|------|------|
| 计划可执行性 | 人工打分 (1-5)，至少3位评估者 | ≥ 4.0 |
| Baseline对比 | 与Paper2Code生成结果对比 | 显著优于Baseline |
| 标准benchmark | SciReplicate-Bench / ResearchCodeBench | 达到SOTA水平 |
| 真实场景验证 | 至少1个真实科研任务跑通 | 被同学实际使用 |
| 失败模式分析 | 记录不可靠场景 | 文档化人工接管时机 |

---

## 完成标准

- [x] 支持 PDF / URL / 搜索关键词 输入
- [ ] 支持 LaTeX 源文件输入
- [x] 支持 OpenAI / Claude 模型后端
- [ ] 支持 vLLM / Ollama 本地模型
- [ ] Skills 体系完整运作（自动发现、context注入）
- [ ] 自动生成 reproduce.sh
- [ ] 自动生成测试代码
- [ ] 有 baseline 对比
- [ ] 至少在一个真实科研任务中被实际使用
- [ ] 产出失败模式分析文档

---

## 设计哲学

1. **AI-Human协同**: AI负责结构化和初步分析，人类负责审核和关键决策
2. **面向API设计**: 充分利用现代LLM的Agent能力
3. **Skills即知识**: 将专业知识封装为可复用的模块
4. **渐进式完善**: 从MVP开始，逐步添加完整功能

---

## 参考

- [Paper2Code](https://github.com/paper2code/paper2code) - 论文复现工作流参考
- [Claude Code Skills](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/skills) - Skills机制参考
