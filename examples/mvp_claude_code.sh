#!/bin/bash
# Claude Agent 模式示例

cd "$(dirname "$0")"

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "请设置 ANTHROPIC_API_KEY:"
    echo "  export ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

python3 mvp_example.py \
    --paper ContextPilot.pdf \
    --output output/ \
    --api-style agent_claude \
    --model claude-sonnet-4-20250514
