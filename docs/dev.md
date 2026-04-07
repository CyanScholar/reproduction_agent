# V0
我计划实现一个复现计划生成器Agent：看完 paper 后可直接得到“先写什么、先跑什么、哪里最容易失败”。
主要通过调用主流模型比如gpt的api，配合调好的SKILLS和python脚本来完成对复现方案的制定和实现。
1. 阅读repo/Paper2Code里的代码，梳理它是如何设计的、如何复现论文的。
2. 阅读@agents/reproduction/科研辅助系统—调研&思考 (2).pdf ，这里是我初步的规划；而图片是对这个特性的要求：paper_to_plan(paper, artifacts) -> ExperimentPlan、 文档（模块划分、依赖列表、伪代码、风险点）、论文描述不足之处的清单、复现脚本、验收方案
3. 先充分理解我的需求，然后与我讨论出来逻辑成s型的方案

实现需要用到的工具：
+ skills就是prompt模板/工具链，是类似Claude Code的Skills机制。
+ 专家经验封装（或者说上下文信息）context.md
+ python可执行脚本/工具
但单一的能力模块可能不足够实现需求，协同完成这次工作。但是要注重可扩展性，要简洁易修改。


# V1
我计划实现一个复现计划生成器Agent：看完 paper 后可直接得到“先写什么、先跑什么、哪里最容易失败”，具体要求见@agents/reproduction/docs/demands.md（包含mvp版本和完整版本的要求）。

我目前已经实现了一版，你可以阅读agents/reproduction/docs/plan.md和agents/reproduction/core/* 来理解现有的实现流程。现在的实现里我觉得还缺乏：
1. “最小可运行版”和“完整复现版”两套计划，目前计划是被合并成一个plan的
2. 当前的实现里没体现出skills，仍然是以python脚本为主，我希望的工具包括：
    + skills：就是prompt模板/工具链，是类似Claude Code的Skills机制，文件格式为md。
    + context：专家经验封装（或者说上下文信息），目前已有，但主要是python文件格式，不利于修改
    + python可执行脚本/工具，目前已有

你可以检查当前的实现是否还有其他待完善的地方

# V2
实现理念：通过脚本、skills、context三部分配合，争取尽可能减少人类的模糊输入和没必要的互动，让AI回答更规范更高质量，且能自动实现代码复现流程。

目前的缺陷：
1. 现在的Skills太简单了，几乎和context.md差别不大，需要在Skills里显式调用更多专业的工具和网站。
2. 做一个GUI，降低使用门槛。
3. model_provider的逻辑应该更加简化，本agent主要是调用api的，所以面向api去设计更合理。（但是对于claude code、codex这样的模型，可以设计专门的api脚本文件去调用相关库）
