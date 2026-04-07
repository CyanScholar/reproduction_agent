#!/bin/bash
# Codex/OpenAI Agent 模式示例

cd "$(dirname "$0")"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "请设置 OPENAI_API_KEY:"
    echo "  export OPENAI_API_KEY='your-key'"
    exit 1
fi

python3 mvp_example.py \
    --paper ContextPilot.pdf \
    --output output/ \
    --api-style agent_codex \
    --model gpt-4o
