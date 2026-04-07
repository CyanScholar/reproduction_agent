"""
Task Prompt Builder - Unified prompt compilation for reproduction planning.

Replaces the old prompts.py + skills/loader.py system by compiling all expert
context (reproduction-expert, ml-paper-patterns, risk-taxonomy) into Python
constants and providing a single-call prompt builder.

This file is fully self-contained: no imports from other project modules.
"""



# =============================================================================
# AGENT_SYSTEM_PROMPT
# =============================================================================
# Comprehensive system prompt compiled from three knowledge sources:
#   - context/reproduction-expert.md
#   - context/ml-paper-patterns.md
#   - context/risk-taxonomy.md
# =============================================================================

AGENT_SYSTEM_PROMPT = """\
You are a world-class ML reproduction specialist. Your sole purpose is to \
analyze a research paper and produce a complete, structured reproduction plan \
that another engineer can follow to reimplement and verify the paper's results.

You combine three areas of deep expertise described below.

# ============================================================
# 1. REPRODUCTION EXPERT KNOWLEDGE
# ============================================================

## What Determines Reproduction Difficulty

### Paper-side Factors
- Method completeness: Does the paper describe ALL steps from data \
preprocessing to final evaluation? Papers that say "we follow standard \
practice" without specifying leave critical gaps.
- Hyperparameter disclosure: Are all hyperparameters explicitly stated? \
Missing learning rates, batch sizes, and optimizer settings are the #1 cause \
of reproduction failure.
- Evaluation protocol clarity: Are the metrics, datasets, and evaluation \
scripts clearly defined? Ambiguous evaluation leads to meaningless comparisons.
- Algorithmic novelty: Novel loss functions, custom layers, or non-standard \
training procedures require more effort and have higher failure risk than \
standard architectures.

### Implementation-side Factors
- Framework compatibility: Does the approach assume specific framework \
features (e.g., PyTorch-specific ops)?
- Hardware requirements: GPU memory requirements often determine feasibility.
- Data availability: Proprietary or restricted datasets block reproduction.
- Random seed sensitivity: Some methods are highly sensitive to \
initialization; papers rarely report this.

## Common Failure Modes in Reproduction
1. "Details in the appendix" syndrome: Critical implementation details buried \
in appendices or supplementary material that are easy to miss.
2. Implicit preprocessing: Data preprocessing steps assumed to be "standard" \
but varying significantly between implementations.
3. Training instability: Methods that require careful learning rate scheduling \
or gradient clipping that is not fully documented.
4. Evaluation mismatch: Using different evaluation protocols (e.g., last \
checkpoint vs best checkpoint, different tokenizers).
5. Codebase drift: When official code exists but does not match the paper due \
to post-publication changes.

## MVP vs Full Plan: How to Split

### MVP (Minimum Viable Plan)
- The single most important experiment (usually Table 1 / main result)
- The core algorithm with minimal configuration
- A single dataset (the smallest or most accessible)
- One evaluation metric (the primary one)
- Hardcoded reasonable defaults for unclear parameters
- CPU or single-GPU feasibility

### MVP Should Exclude
- Ablation studies
- Multi-dataset evaluation
- Hyperparameter search
- Distributed training
- Baseline comparisons (unless trivially available)
- Secondary metrics

### Full Plan Adds
- All experiments from the paper
- All datasets and metrics
- Complete hyperparameter configuration
- Baseline implementations or references
- Ablation studies
- Performance optimization (multi-GPU, mixed precision)

# ============================================================
# 2. ML PAPER PATTERNS KNOWLEDGE
# ============================================================

## Paper Structure and Reproduction Relevance

| Section        | Reproduction Relevance |
|----------------|----------------------|
| Introduction   | Low - context only |
| Related Work   | Low - useful for baselines |
| Method / Model | **Critical** - main implementation target |
| Experiments    | **Critical** - evaluation targets |
| Ablation Study | Medium - helps understand which parts matter |
| Conclusion     | Low |
| Appendix       | **Often critical** - frequently contains missing details |

## Common ML Architecture Patterns

### Transformer-based
- Multi-head attention: need attention dimension, number of heads, dropout rate
- Position encoding: need type (sinusoidal/learned), max sequence length
- Layer norm: need pre-norm vs post-norm
- Feed-forward: need hidden dimension, activation function

### CNN-based
- Convolution layers: need kernel sizes, channels, stride, padding
- Pooling: need type, size
- Batch normalization: need momentum, epsilon

### Training Pipeline
- Optimizer: need type (Adam/SGD/AdamW), learning rate, weight decay, betas
- Learning rate schedule: need type (cosine/step/linear), warmup steps
- Batch size: need per-GPU and total (for gradient accumulation)
- Epochs / iterations: need total training steps, evaluation frequency
- Data augmentation: need exact transforms and their parameters

## Common Pitfalls by Paper Type

### NLP Papers
- Tokenizer choice matters enormously (BPE vs WordPiece vs SentencePiece)
- Pre-trained model version and checkpoint source
- Maximum sequence length and truncation strategy
- Label smoothing value

### Computer Vision Papers
- Input resolution and resize strategy (crop vs pad vs stretch)
- Data augmentation pipeline order
- Batch normalization running statistics handling
- Test-time augmentation (TTA) usage

### Reinforcement Learning Papers
- Random seed sensitivity (often need 5+ seeds)
- Environment version and wrapper configuration
- Replay buffer size and sampling strategy
- Exploration schedule

# ============================================================
# 3. RISK TAXONOMY KNOWLEDGE
# ============================================================

## Risk Categories

### 1. implementation_difficulty
Technical complexity of implementing the proposed method.

| Severity | Criteria |
|----------|----------|
| low      | Standard components, well-documented APIs, common patterns |
| medium   | Custom layers or loss functions, moderate algorithmic complexity |
| high     | Novel algorithms, non-standard training procedures, complex data pipelines |
| critical | Requires specialized domain expertise, undocumented tricks, or extremely complex math |

### 2. paper_clarity
How well the paper describes what needs to be implemented.

| Severity | Criteria |
|----------|----------|
| low      | Clear descriptions with pseudocode or equations |
| medium   | Some details missing but inferable from context |
| high     | Significant ambiguity, multiple valid interpretations |
| critical | Core method description is incomplete or contradictory |

### 3. environment_dependency
External dependencies that may block or complicate reproduction.

| Severity | Criteria |
|----------|----------|
| low      | Common packages, no special hardware |
| medium   | Specific package versions, single GPU required |
| high     | Special hardware (multi-GPU, TPU), restricted datasets |
| critical | Proprietary software, unavailable datasets, extreme compute needs |

### 4. data_availability
Access to required datasets and their usability.

| Severity | Criteria |
|----------|----------|
| low      | Public datasets, well-documented formats |
| medium   | Requires registration or agreement, minor preprocessing needed |
| high     | Restricted access, significant preprocessing, or large download |
| critical | Proprietary data, no public alternative, or data no longer available |

### 5. computational_resource
Compute requirements for training and evaluation.

| Severity | Criteria |
|----------|----------|
| low      | Single CPU or modest GPU, trains in hours |
| medium   | Single GPU, trains in 1-3 days |
| high     | Multi-GPU, trains in days to a week |
| critical | Multi-node, trains in weeks, requires 100+ GPU-hours |

## Gap Types

- **missing**: Information that should be in the paper but is completely absent. \
Example: "The paper does not mention the learning rate used for fine-tuning."
- **ambiguous**: Information that is present but open to multiple interpretations. \
Example: "The paper says 'standard augmentation' without specifying which transforms."
- **contradictory**: Information that conflicts between different parts of the paper. \
Example: "Section 3 says batch size 32 but Table 2 footnote says batch size 64."
- **omitted**: Information intentionally left out (common for space reasons). \
Example: "Hyperparameter details are said to be in supplementary material that is not available."

## Severity Levels

| Level    | Score Weight | Action Required |
|----------|-------------|-----------------|
| low      | 1           | Note and proceed with reasonable defaults |
| medium   | 2           | Investigate alternatives, document assumption |
| high     | 3           | Requires explicit decision, may need experimentation |
| critical | 4           | Blocks progress, needs resolution before implementation |

# ============================================================
# WORKING PRINCIPLES
# ============================================================

When analyzing a paper and building a reproduction plan, you MUST:

1. Align with the Paper: Your plan must strictly follow the methods, datasets, \
model configurations, and experimental setups described.
2. Be Clear and Structured: Present the plan in a well-organized format.
3. Prioritize Efficiency: Focus on practical implementation steps, order by \
dependency and risk.
4. Identify Uncertainties: Explicitly mark anything unclear from the paper \
with [UNCLEAR].
5. Design for Modularity: Create clean module boundaries and interfaces.
6. Assess Risks Systematically: Use the risk taxonomy above to classify every \
identified risk.
7. Detect Gaps Thoroughly: Scrutinize every section for missing, ambiguous, \
contradictory, or omitted information.
8. Define Acceptance Criteria: Every plan must include verifiable acceptance \
tests and benchmark targets.
"""


# =============================================================================
# MARKDOWN OUTPUT TEMPLATE
# =============================================================================
# Describes the desired Markdown structure for the reproduction plan output.
# Used in prompt builders to tell the LLM what format to produce.
# =============================================================================

MARKDOWN_OUTPUT_TEMPLATE = """\
Use the following Markdown structure for your output. Output ONLY the Markdown \
content, no extra commentary before or after.

```
# 复现计划: {paper_title}

## 计划概览

本文档包含两套复现计划：
- **Part A: MVP 计划** — 最小可运行版，快速验证核心方法
- **Part B: 完整计划** — 完整复现论文所有实验结果

**切分依据**: (explain your rationale for how you split MVP vs Full)

| 层级 | 复现难度 (1-5) |
|------|---------------|
| MVP  | X.X           |
| Full | X.X           |

---

# Part A: MVP 计划 (最小可运行版)

## 1. 执行摘要

(Executive summary: paper contribution, key innovations, reproduction \
complexity, high-level implementation strategy for MVP)

**整体复现难度**: X.X/5

## 2. 架构概览

### 2.1 模块结构

```mermaid
classDiagram
    (module/class relationships)
```

### 2.2 执行流程

```mermaid
sequenceDiagram
    (training and/or inference flow)
```

### 2.3 模块详情

#### ModuleName
- **文件**: `src/path/to/file.py`
- **描述**: what this module does
- **依赖**: list of other module names
- **接口**: public function/class signatures
- **实现要点**: key algorithms, pseudocode, equations, references to paper sections

(repeat for each module)

## 3. 依赖

### 3.1 Python包

```
torch>=2.0
transformers>=4.30
(etc.)
```

### 3.2 硬件要求

- **gpu**: required or not
- **gpu_memory_gb**: X
- **ram_gb**: X
- **estimated_training_time**: X hours

## 4. 配置参数

```yaml
(ALL hyperparameters, annotate each with [PAPER], [INFERRED], or [UNCLEAR])
```

## 5. 风险评估

### 5.1 风险摘要

| 严重程度 | 数量 |
|----------|------|
| CRITICAL | X    |
| HIGH     | X    |
| MEDIUM   | X    |
| LOW      | X    |

### 5.2 风险详情

(For CRITICAL and HIGH risks, give full detail:)

**ComponentName**
- 类别: (implementation_difficulty | paper_clarity | environment_dependency | \
data_availability | computational_resource)
- 描述: detailed description
- 缓解建议: mitigation strategy

(For MEDIUM and LOW risks, one-liner each:)
- **ComponentName**: description

## 6. 论文描述缺失清单

### 6.1 关键缺失 (Critical)

**[gap_type]** paper_section
- 描述: what is missing or unclear
- 建议: how to potentially resolve

(gap_type is one of: missing, ambiguous, contradictory, omitted)

### 6.2 其他缺失

- **[gap_type]** section: description

## 7. 验收方案

### 7.1 自动化测试

- [ ] **test_name** (unit|integration|benchmark)
  - 描述: what this test validates
  - 容差: tolerance if applicable

### 7.2 人工检查清单

- [ ] **check_name**
  - 检查方法: how to verify
  - 期望结果: what to look for

### 7.3 基准对比目标

| 指标 | 论文报告值 | 最低可接受值 |
|------|-----------|-------------|
| metric_name | X.XX | X.XX |

---

# Part B: 完整复现计划

(Same structure as Part A, but covering ALL experiments, datasets, metrics, \
ablations, baselines, and performance optimizations from the paper)

---

# Part C: 升级路径 (MVP -> Full)

1. Step one to upgrade from MVP to Full
2. Step two...
(ordered list of concrete steps)

---

*此文档由 Reproduction Planner Agent 自动生成*
```
"""

MARKDOWN_SINGLE_TIER_TEMPLATE = """\
Use the following Markdown structure for your output. Output ONLY the Markdown \
content, no extra commentary before or after.

```
# 复现计划: {paper_title}

## 1. 执行摘要

(Executive summary: paper contribution, key innovations, reproduction \
complexity, high-level implementation strategy)

**整体复现难度**: X.X/5

## 2. 架构概览

### 2.1 模块结构

```mermaid
classDiagram
    (module/class relationships)
```

### 2.2 执行流程

```mermaid
sequenceDiagram
    (training and/or inference flow)
```

### 2.3 模块详情

#### ModuleName
- **文件**: `src/path/to/file.py`
- **描述**: what this module does
- **依赖**: list of other module names
- **接口**: public function/class signatures
- **实现要点**: key algorithms, pseudocode, equations, references to paper sections

(repeat for each module)

## 3. 依赖

### 3.1 Python包

```
torch>=2.0
(etc.)
```

### 3.2 硬件要求

- **gpu**: required or not
- **gpu_memory_gb**: X
- **ram_gb**: X

## 4. 配置参数

```yaml
(ALL hyperparameters, annotate each with [PAPER], [INFERRED], or [UNCLEAR])
```

## 5. 风险评估

(same structure as above — summary table + details)

## 6. 论文描述缺失清单

(same structure as above — critical gaps with detail, others as one-liners)

## 7. 验收方案

(same structure as above — automated tests, manual checks, benchmark targets)

---

*此文档由 Reproduction Planner Agent 自动生成*
```
"""


# =============================================================================
# build_task_prompt
# =============================================================================

def build_task_prompt(
    paper_content: str,
    title: str,
    authors: str,
    tier: str = "both",
) -> str:
    """
    Build the complete user prompt for a single agent delegation.

    Args:
        paper_content: Full text of the paper (already extracted / parsed).
        title: Paper title.
        authors: Comma-separated author string.
        tier: One of "mvp", "full", or "both".

    Returns:
        The formatted user prompt string.
    """
    if tier == "both":
        return build_tiered_task_prompt(paper_content, title, authors)

    # Determine tier-specific instructions
    if tier == "mvp":
        tier_instruction = (
            "Generate an **MVP (Minimum Viable Plan)** reproduction plan.\n\n"
            "MVP scope:\n"
            "- Core algorithm only, single dataset (the smallest/most accessible)\n"
            "- Primary metric only\n"
            "- Hardcoded reasonable defaults for unclear parameters\n"
            "- CPU or single-GPU feasibility\n"
            "- Exclude ablations, multi-dataset evaluation, hyperparameter search, "
            "distributed training, and secondary metrics\n"
        )
    else:  # "full"
        tier_instruction = (
            "Generate a **Full** reproduction plan.\n\n"
            "Full scope:\n"
            "- All experiments described in the paper\n"
            "- All datasets and evaluation metrics\n"
            "- Complete hyperparameter configuration\n"
            "- Baseline implementations or references\n"
            "- Ablation studies\n"
            "- Performance optimizations (multi-GPU, mixed precision where applicable)\n"
        )

    prompt = f"""\
## Paper Information
- **Title**: {title}
- **Authors**: {authors}

## Paper Content

{paper_content}

## Task

Analyze the paper above and produce a structured reproduction plan. The authors \
did not release official code, so we are planning our own implementation from \
scratch.

### Tier

{tier_instruction}

### Step-by-step instructions

Follow these steps carefully:

#### Step 1: Paper Analysis
- Read the paper thoroughly, paying special attention to Method, Experiments, \
and Appendix sections.
- Identify the core algorithm, datasets, metrics, and key hyperparameters.
- Note any implicit assumptions, "standard practice" references, or missing \
details.

#### Step 2: Existing Implementation Search
- If you have tool access, search for existing implementations on GitHub and \
Papers With Code.
- Note any reference implementations that could inform the plan, but do NOT \
assume their correctness.

#### Step 3: Reproduction Plan Design
- Design the module breakdown: each module gets a name, file path, description, \
dependencies, public interfaces, and implementation notes with pseudocode.
- Create a Mermaid classDiagram showing module/class relationships.
- Create a Mermaid sequenceDiagram showing the training and/or inference flow.
- List all Python packages with version constraints.
- Specify hardware requirements.
- Extract ALL hyperparameters into a config section, annotating each value with \
[PAPER], [INFERRED], or [UNCLEAR].

#### Step 4: Risk Analysis
- For every identified risk, classify it using the five risk categories \
(implementation_difficulty, paper_clarity, environment_dependency, \
data_availability, computational_resource) and four severity levels (low, \
medium, high, critical).
- Provide a mitigation strategy for each risk.
- Compute an overall reproduction difficulty score (1.0 - 5.0).

#### Step 5: Gap Detection
- Scrutinize the paper for gaps of type: missing, ambiguous, contradictory, \
omitted.
- For each gap, note the paper section, impact (critical/important/minor), and \
a suggested resolution.

#### Step 6: Acceptance Criteria Design
- Design automated tests (unit, integration, benchmark).
- Design manual verification checks.
- Extract reported benchmark targets with acceptable tolerance ranges.

### Output Format

{MARKDOWN_SINGLE_TIER_TEMPLATE}

IMPORTANT: Output ONLY the Markdown document. Do not wrap it in code blocks. \
Do not include any text before or after the Markdown content.
"""
    return prompt


# =============================================================================
# build_tiered_task_prompt
# =============================================================================

def build_tiered_task_prompt(
    paper_content: str,
    title: str,
    authors: str,
) -> str:
    """
    Build the user prompt requesting both MVP and Full plans as Markdown.

    Args:
        paper_content: Full text of the paper.
        title: Paper title.
        authors: Comma-separated author string.

    Returns:
        The formatted user prompt string.
    """
    prompt = f"""\
## Paper Information
- **Title**: {title}
- **Authors**: {authors}

## Paper Content

{paper_content}

## Task

Analyze the paper above and produce a **two-tier** reproduction plan containing \
both an MVP plan and a Full plan. The authors did not release official code, so \
we are planning our own implementation from scratch.

### Tier Definitions

**MVP (Minimum Viable Plan)**:
- Core algorithm + single dataset (the smallest/most accessible) + primary \
metric + hardcoded defaults for unclear parameters.
- Must be feasible on CPU or a single GPU.
- Excludes ablations, multi-dataset evaluation, hyperparameter search, \
distributed training, and secondary metrics.

**Full Plan**:
- All experiments, all datasets, all metrics, all ablations from the paper.
- Complete hyperparameter configuration.
- Baseline implementations or references.
- Performance optimizations (multi-GPU, mixed precision where applicable).

### Step-by-step instructions

Follow these steps carefully:

#### Step 1: Paper Analysis
- Read the paper thoroughly, paying special attention to Method, Experiments, \
and Appendix sections.
- Identify the core algorithm, datasets, metrics, and key hyperparameters.
- Note any implicit assumptions, "standard practice" references, or missing \
details.

#### Step 2: Existing Implementation Search
- If you have tool access, search for existing implementations on GitHub and \
Papers With Code.
- Note any reference implementations that could inform the plan, but do NOT \
assume their correctness.

#### Step 3: Tier Split Decision
- Decide which experiment is the single most important (for MVP).
- Decide which dataset is smallest or most accessible (for MVP).
- Write a clear rationale for how you split MVP vs Full.
- Define an ordered upgrade path (list of steps to go from MVP to Full).

#### Step 4: Reproduction Plan Design (for BOTH tiers)
For each tier (MVP and Full), produce a complete plan:
- Module breakdown: each module gets a name, file path, description, \
dependencies, public interfaces, and implementation notes with pseudocode.
- Mermaid classDiagram showing module/class relationships.
- Mermaid sequenceDiagram showing the training and/or inference flow.
- Python packages with version constraints.
- Hardware requirements.
- Config section with ALL hyperparameters, annotating each value with [PAPER], \
[INFERRED], or [UNCLEAR].

#### Step 5: Risk Analysis (for BOTH tiers)
- For every identified risk, classify using the five risk categories \
(implementation_difficulty, paper_clarity, environment_dependency, \
data_availability, computational_resource) and four severity levels (low, \
medium, high, critical).
- Provide a mitigation strategy for each risk.
- Compute an overall reproduction difficulty score (1.0 - 5.0) per tier.

#### Step 6: Gap Detection (for BOTH tiers)
- Scrutinize the paper for gaps of type: missing, ambiguous, contradictory, \
omitted.
- For each gap, note the paper section, impact (critical/important/minor), and \
a suggested resolution.
- Note: many gaps will be shared between tiers, but the impact rating may \
differ.

#### Step 7: Acceptance Criteria Design (for BOTH tiers)
- Design automated tests (unit, integration, benchmark).
- Design manual verification checks.
- Extract reported benchmark targets with acceptable tolerance ranges.
- MVP acceptance criteria focus on the primary metric; Full includes all \
reported metrics.

### Output Format

{MARKDOWN_OUTPUT_TEMPLATE}

IMPORTANT: Output ONLY the Markdown document. Do not wrap it in code blocks. \
Do not include any text before or after the Markdown content.
"""
    return prompt
