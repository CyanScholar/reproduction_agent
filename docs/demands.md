# Demands
本模块最值得做的改造与优化：
+ 同时给出“最小可运行版”和“完整复现版”两套计划
+ 显式暴露论文中的缺失实现细节与高风险环节
+ 规划结果必须能被 execution agent 和人类直接采用

本模块Agent最小可交付物（MVP）
+ paper_to_plan(paper, artifacts) -> ExperimentPlan
+ 模块划分、依赖列表、伪代码、风险点
+ 论文描述不足之处的清单

接口与人工校对点
+ 输入：PaperDoc + ArtifactLink[]；输出：ExperimentPlan
+ 人工反馈位点：人工可改写模块划分、删减 scope、锁定第一轮目标
+ 要求：接口既要支持 CLI/脚本单独调用，也要支持被总 workflow 编排调用。

建议 benchmark / 评测方法
+ SciReplicate-Bench
+ ResearchCodeBench
+ 人工打分的计划可执行性

完成标准
+ 至少能在一个真实科研任务中跑通，并被同学实际使用一次。
+ 必须有 baseline 对比，不接受只展示“看起来不错”的主观 demo。
+ 必须产出失败模式分析，说明在哪些情况下不可靠、何时需要人工接管

先实现Agent的MVP版本，然后再做完整版本