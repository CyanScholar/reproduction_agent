# Examples - 使用示例

本目录包含复现计划生成器的使用示例。

## ContextPilot 论文复现计划

文件: [contextpilot_example.py](contextpilot_example.py)

### 使用方式

```bash
# 设置API Key
export OPENAI_API_KEY="your-key"

# 运行
cd agents/reproduction
python3 examples/contextpilot_example.py --paper path/to/paper.json

# 指定输出目录和模型
python3 examples/contextpilot_example.py \
    --paper path/to/paper.json \
    --output output/ \
    --model gpt-4o
```

### 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--paper` | 是 | 论文JSON文件路径（s2orc格式） |
| `--output` | 否 | 输出目录，默认 `examples/output/` |
| `--model` | 否 | 模型选择：gpt-4o / gpt-4o-mini / o3-mini |

### 输出

运行成功后会在输出目录生成：
- `reproduction_plan.md` - Markdown格式的复现计划
- `experiment_plan.json` - JSON格式的结构化数据
- `trajectories.json` - LLM调用轨迹
- `cost_summary.json` - 成本统计
