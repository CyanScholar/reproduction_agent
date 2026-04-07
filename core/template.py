"""
Output Template - Markdown输出模板
"""

from .models import ExperimentPlan, TieredExperimentPlan, Severity


def render_plan_markdown(plan: ExperimentPlan) -> str:
    """
    将ExperimentPlan渲染为Markdown文档
    """
    sections = []

    # 标题
    sections.append(f"# 复现计划: {plan.paper_title}")
    sections.append(f"\n> Generated: {plan.generated_at}")
    sections.append(f"> Version: {plan.generator_version}")
    sections.append("")

    # 执行摘要
    sections.append("---\n")
    sections.append("## 1. 执行摘要\n")
    if plan.overview:
        sections.append(plan.overview)
    else:
        sections.append("_待生成_")

    if plan.risks:
        sections.append(f"\n**整体复现难度**: {plan.risks.overall_score():.1f}/5")
    sections.append("")

    # 架构概览
    sections.append("---\n")
    sections.append("## 2. 架构概览\n")

    if plan.architecture_diagram:
        sections.append("### 2.1 模块结构\n")
        sections.append(plan.architecture_diagram)
        sections.append("")

    if plan.call_flow:
        sections.append("### 2.2 执行流程\n")
        sections.append(plan.call_flow)
        sections.append("")

    # 模块详情
    if plan.modules:
        sections.append("### 2.3 模块详情\n")
        for module in plan.modules:
            sections.append(f"#### {module.name}")
            sections.append(f"- **文件**: `{module.file_path}`")
            sections.append(f"- **描述**: {module.description}")
            if module.dependencies:
                sections.append(f"- **依赖**: {', '.join(module.dependencies)}")
            sections.append("")

    # 依赖
    sections.append("---\n")
    sections.append("## 3. 依赖\n")

    if plan.python_packages:
        sections.append("### 3.1 Python包\n")
        sections.append("```")
        for pkg in plan.python_packages:
            sections.append(pkg)
        sections.append("```\n")

    if plan.hardware_requirements:
        sections.append("### 3.2 硬件要求\n")
        for key, value in plan.hardware_requirements.items():
            sections.append(f"- **{key}**: {value}")
        sections.append("")

    # 配置
    sections.append("---\n")
    sections.append("## 4. 配置参数\n")

    if plan.config:
        sections.append("```yaml")
        if isinstance(plan.config, dict) and "raw_yaml" in plan.config:
            sections.append(plan.config["raw_yaml"])
        else:
            import json
            sections.append(json.dumps(plan.config, indent=2))
        sections.append("```\n")

    if plan.hyperparameters:
        sections.append("### 4.1 超参数说明\n")
        for hp in plan.hyperparameters:
            source_icon = {"explicit": "✅", "inferred": "⚠️", "default": "⚙️", "unclear": "❓"}.get(hp.source, "")
            sections.append(f"- **{hp.name}**: {hp.value} {source_icon}")
            if hp.notes:
                sections.append(f"  - 备注: {hp.notes}")
        sections.append("")

    # 风险评估
    sections.append("---\n")
    sections.append("## 5. 风险评估\n")

    if plan.risks and plan.risks.risks:
        # 摘要表格
        sections.append("### 5.1 风险摘要\n")
        sections.append("| 严重程度 | 数量 |")
        sections.append("|----------|------|")
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            count = len(plan.risks.get_by_severity(severity))
            sections.append(f"| {severity.value.upper()} | {count} |")
        sections.append("")

        # 按严重程度列出风险
        sections.append("### 5.2 风险详情\n")

        for severity in [Severity.CRITICAL, Severity.HIGH]:
            risks = plan.risks.get_by_severity(severity)
            if risks:
                sections.append(f"#### {severity.value.upper()} 风险\n")
                for risk in risks:
                    sections.append(f"**{risk.component}**")
                    sections.append(f"- 类别: {risk.category.value}")
                    sections.append(f"- 描述: {risk.description}")
                    if risk.mitigation:
                        sections.append(f"- 缓解建议: {risk.mitigation}")
                    sections.append("")

        # 中低风险简要列出
        for severity in [Severity.MEDIUM, Severity.LOW]:
            risks = plan.risks.get_by_severity(severity)
            if risks:
                sections.append(f"#### {severity.value.upper()} 风险\n")
                for risk in risks:
                    sections.append(f"- **{risk.component}**: {risk.description}")
                sections.append("")
    else:
        sections.append("_暂无风险识别_\n")

    # 论文缺失清单
    sections.append("---\n")
    sections.append("## 6. 论文描述缺失清单\n")

    if plan.gaps and plan.gaps.gaps:
        # 关键缺失
        critical = plan.gaps.critical_gaps()
        if critical:
            sections.append("### 6.1 关键缺失 (Critical)\n")
            for gap in critical:
                sections.append(f"**[{gap.type}]** {gap.section}")
                sections.append(f"- 描述: {gap.description}")
                if gap.suggestion:
                    sections.append(f"- 建议: {gap.suggestion}")
                sections.append("")

        # 其他缺失
        other_gaps = [g for g in plan.gaps.gaps if g not in critical]
        if other_gaps:
            sections.append("### 6.2 其他缺失\n")
            for gap in other_gaps:
                sections.append(f"- **[{gap.type}]** {gap.section}: {gap.description}")
            sections.append("")
    else:
        sections.append("_暂无缺失识别_\n")

    # 验收方案
    sections.append("---\n")
    sections.append("## 7. 验收方案\n")

    if plan.acceptance_criteria:
        # 自动化测试
        if plan.acceptance_criteria.automated_tests:
            sections.append("### 7.1 自动化测试\n")
            for test in plan.acceptance_criteria.automated_tests:
                sections.append(f"- [ ] **{test.name}** ({test.test_type})")
                sections.append(f"  - 描述: {test.description}")
                if test.tolerance:
                    sections.append(f"  - 容差: {test.tolerance}")
            sections.append("")

        # 人工检查清单
        if plan.acceptance_criteria.manual_checks:
            sections.append("### 7.2 人工检查清单\n")
            for check in plan.acceptance_criteria.manual_checks:
                sections.append(f"- [ ] **{check.name}**")
                sections.append(f"  - 检查方法: {check.procedure}")
                sections.append(f"  - 期望结果: {check.expected_outcome}")
            sections.append("")

        # 基准目标
        if plan.acceptance_criteria.benchmark_targets:
            sections.append("### 7.3 基准对比目标\n")
            sections.append("| 指标 | 论文报告值 | 最低可接受值 |")
            sections.append("|------|-----------|-------------|")
            for metric, values in plan.acceptance_criteria.benchmark_targets.items():
                reported = values.get("reported", "N/A")
                min_acceptable = values.get("min_acceptable", "N/A")
                sections.append(f"| {metric} | {reported} | {min_acceptable} |")
            sections.append("")
    else:
        sections.append("_暂无验收方案_\n")

    # 结尾
    sections.append("---\n")
    sections.append("*此文档由 Reproduction Planner Agent 自动生成*\n")

    return "\n".join(sections)


def render_tiered_plan_markdown(plan: TieredExperimentPlan) -> str:
    """
    将 TieredExperimentPlan 渲染为包含 MVP + Full 两层的 Markdown 文档
    """
    sections = []

    # 标题
    sections.append(f"# 复现计划: {plan.paper_title}")
    sections.append(f"\n> Generated: {plan.generated_at}")
    sections.append(f"> Version: {plan.generator_version}")
    sections.append("")

    # 两层切分说明
    sections.append("---\n")
    sections.append("## 计划概览\n")
    sections.append("本文档包含两套复现计划：\n")
    sections.append("- **Part A: MVP 计划** — 最小可运行版，快速验证核心方法")
    sections.append("- **Part B: 完整计划** — 完整复现论文所有实验结果")
    sections.append("")

    if plan.tier_rationale:
        sections.append(f"**切分依据**: {plan.tier_rationale}\n")

    # 难度概览
    mvp_score = plan.mvp_plan.risks.overall_score() if plan.mvp_plan and plan.mvp_plan.risks else None
    full_score = plan.full_plan.risks.overall_score() if plan.full_plan and plan.full_plan.risks else None

    if mvp_score or full_score:
        sections.append("| 层级 | 复现难度 |")
        sections.append("|------|---------|")
        if mvp_score:
            sections.append(f"| MVP | {mvp_score:.1f}/5 |")
        if full_score:
            sections.append(f"| Full | {full_score:.1f}/5 |")
        sections.append("")

    # Part A: MVP 计划
    sections.append("---\n")
    sections.append("# Part A: MVP 计划 (最小可运行版)\n")

    if plan.mvp_plan:
        sections.append(render_plan_markdown(plan.mvp_plan))
    else:
        sections.append("_MVP 计划未生成_\n")

    # Part B: 完整计划
    sections.append("\n---\n")
    sections.append("# Part B: 完整复现计划\n")

    if plan.full_plan:
        sections.append(render_plan_markdown(plan.full_plan))
    else:
        sections.append("_完整计划未生成_\n")

    # Part C: 升级路径
    if plan.upgrade_path:
        sections.append("\n---\n")
        sections.append("# Part C: 升级路径 (MVP -> Full)\n")
        for i, step in enumerate(plan.upgrade_path, 1):
            sections.append(f"{i}. {step}")
        sections.append("")

    # 结尾
    sections.append("---\n")
    sections.append("*此文档由 Reproduction Planner Agent v0.2 自动生成*\n")

    return "\n".join(sections)
