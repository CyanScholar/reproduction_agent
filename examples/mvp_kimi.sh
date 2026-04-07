#!/bin/bash
# Kimi Agent 模式示例

cd "$(dirname "$0")"

export OPENAI_API_KEY='YOUR_API_KEY'

python3 mvp_example.py \
    --paper ContextPilot.pdf \
    --output output/ \
    --api-style agent_kimi \
    --model kimi-k2.5 \
    --base-url https://api.moonshot.cn/v1
